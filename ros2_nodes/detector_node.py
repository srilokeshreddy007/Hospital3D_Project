#!/usr/bin/env python3
import os
import math
import time
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException

# --- Project imports (your repo) ---
import torch
from src.model_sota import TwoHeadSOTA

# ---------- simple PointCloud2 helpers ----------
def pc2_to_xyz(msg: PointCloud2):
    """Convert PointCloud2 to Nx3 float32 numpy array. Assumes fields x,y,z present."""
    import struct
    fmt = {}
    offset_map = {}
    for f in msg.fields:
        offset_map[f.name] = f.offset
        if f.datatype == PointField.FLOAT32:
            fmt[f.name] = "f"
        elif f.datatype == PointField.FLOAT64:
            fmt[f.name] = "d"
        elif f.datatype in (PointField.INT16, PointField.UINT16):
            fmt[f.name] = "H"
        elif f.datatype in (PointField.INT32, PointField.UINT32):
            fmt[f.name] = "I"
        elif f.datatype in (PointField.INT8, PointField.UINT8):
            fmt[f.name] = "B"
        else:
            fmt[f.name] = None  # skip unsupported

    step = msg.point_step
    data = msg.data
    npts = msg.height * msg.width
    out = np.empty((npts, 3), dtype=np.float32)

    has64 = (fmt.get("x") == "d")
    unpack_fmt = "<" + ("d" if has64 else "f")
    for i in range(npts):
        base = i * step
        # robust: if x/y/z missing, fill zero
        def _read(name):
            off = offset_map.get(name, None)
            if off is None:
                return 0.0
            val = struct.unpack_from(unpack_fmt, data, base + off)[0]
            return float(val)
        out[i, 0] = _read("x")
        out[i, 1] = _read("y")
        out[i, 2] = _read("z")
    return out

def xyz_to_pc2(points_xyz: np.ndarray, frame_id: str) -> PointCloud2:
    """Build PointCloud2 from Nx3 float32 array."""
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    data = points_xyz.tobytes()
    msg = PointCloud2(
        header=header,
        height=1,
        width=points_xyz.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=12,
        row_step=12 * points_xyz.shape[0],
        data=data,
    )
    return msg

# ---------- geometry helpers ----------
def box_from_pred(center, size, yaw):
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = np.array([
        [-hx,-hy,-hz], [ hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
        [-hx,-hy, hz], [ hx,-hy, hz], [ hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return corners @ R.T + np.array([cx,cy,cz], dtype=np.float32)

def project_corners_xy(corners_3d):
    """Drop z and return 2D polygon (convex hull of projected 8 corners)."""
    from scipy.spatial import ConvexHull
    pts2 = corners_3d[:, :2]
    hull = ConvexHull(pts2)
    poly = pts2[hull.vertices]
    return poly  # (M,2) CCW

def fill_polygon_points(poly_xy, resolution=0.05):
    """Rasterize polygon area into a grid of points (x,y,0)."""
    import shapely.geometry as geom
    import shapely.affinity as aff
    from shapely.ops import triangulate

    P = geom.Polygon(poly_xy)
    if not P.is_valid:
        P = P.buffer(0.0)
    minx, miny, maxx, maxy = P.bounds
    xs = np.arange(minx, maxx + resolution, resolution, dtype=np.float32)
    ys = np.arange(miny, maxy + resolution, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    mask = np.array([P.contains(geom.Point(p[0], p[1])) for p in pts], dtype=bool)
    pts_inside = pts[mask]
    if pts_inside.size == 0:
        # fallback to polygon edges only
        pts_inside = np.array(poly_xy, dtype=np.float32)
    z = np.zeros((pts_inside.shape[0], 1), dtype=np.float32)
    return np.hstack([pts_inside, z])

def transform_points_xy(points_xy, tf: TransformStamped):
    """Apply 2D transform using TF (assumes yaw-only rotation for simplicity)."""
    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    tz = tf.transform.translation.z
    q = tf.transform.rotation
    # yaw from quaternion
    yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    out = (points_xy @ R.T) + np.array([tx, ty], dtype=np.float32)
    return out

# ---------- the ROS 2 node ----------
class DetectorNode(Node):
    def __init__(self):
        super().__init__("hospital3d_detector")

        # ---- parameters (declare + get) ----
        self.declare_parameter("cloud_topic", "/camera/points")
        self.declare_parameter("cloud_frame", "camera_link")
        self.declare_parameter("costmap_frame", "map")
        self.declare_parameter("ckpt_path", "outputs/best_sota_rebal.pt")
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("publish_topic", "/detected_obstacles")
        self.declare_parameter("resolution", 0.05)   # costmap resolution
        self.declare_parameter("roi_xmin", 0.3)      # simple crop ROI in sensor frame
        self.declare_parameter("roi_xmax", 3.0)
        self.declare_parameter("roi_yabs", 1.5)
        self.declare_parameter("roi_zabs", 1.5)
        self.declare_parameter("publish_markers", True)

        self.cloud_topic   = self.get_parameter("cloud_topic").get_parameter_value().string_value
        self.cloud_frame   = self.get_parameter("cloud_frame").get_parameter_value().string_value
        self.costmap_frame = self.get_parameter("costmap_frame").get_parameter_value().string_value
        self.ckpt_path     = self.get_parameter("ckpt_path").get_parameter_value().string_value
        self.num_points    = int(self.get_parameter("num_points").value)
        self.pub_topic     = self.get_parameter("publish_topic").get_parameter_value().string_value
        self.resolution    = float(self.get_parameter("resolution").value)
        self.roi_xmin      = float(self.get_parameter("roi_xmin").value)
        self.roi_xmax      = float(self.get_parameter("roi_xmax").value)
        self.roi_yabs      = float(self.get_parameter("roi_yabs").value)
        self.roi_zabs      = float(self.get_parameter("roi_zabs").value)
        self.pub_markers   = bool(self.get_parameter("publish_markers").value)

        # ---- publishers ----
        qos = QoSProfile(depth=1,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_cloud = self.create_publisher(PointCloud2, self.pub_topic, qos)
        self.pub_mk = self.create_publisher(MarkerArray, "/detector_markers", 10) if self.pub_markers else None

        # ---- TF buffer/listener ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- load model (CPU) ----
        self.device = torch.device("cpu")
        ck = torch.load(self.ckpt_path, map_location="cpu")
        num_classes = ck.get("num_classes", 38)
        self.model = TwoHeadSOTA(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(ck["model"], strict=False)
        self.model.eval()
        self.get_logger().info(f"Loaded checkpoint: {self.ckpt_path}")

        # ---- subscribe ----
        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, qos)

        self.get_logger().info("DetectorNode ready.")

    # ------ callbacks ------
    def on_cloud(self, msg: PointCloud2):
        # 1) get transform from cloud frame to costmap frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.costmap_frame, msg.header.frame_id, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({msg.header.frame_id}->{self.costmap_frame}): {e}")
            return

        # 2) convert cloud to numpy and crop ROI in sensor frame
        pts = pc2_to_xyz(msg)
        if pts.size == 0:
            return
        mask = (
            (pts[:,0] > self.roi_xmin) & (pts[:,0] < self.roi_xmax) &
            (np.abs(pts[:,1]) < self.roi_yabs) &
            (np.abs(pts[:,2]) < self.roi_zabs)
        )
        roi = pts[mask]
        if roi.shape[0] < 64:
            # too few points for a stable prediction
            return

        # 3) downsample to N=1024 (random or farthest; random is fine for CPU)
        N = min(self.num_points, roi.shape[0])
        idx = np.random.choice(roi.shape[0], N, replace=False)
        sample = roi[idx].astype(np.float32)
        # model expects (B,N,3) tensor
        x = torch.from_numpy(sample[None, ...])  # (1,N,3)
        with torch.no_grad():
            logits, box = self.model(x)
            box = box.squeeze(0).cpu().numpy()
            # center (3), size (3), yaw (1)
            center = box[:3]
            size   = np.maximum(box[3:6], 1e-3)
            yaw    = float(box[6])

        # 4) build OBB corners in sensor frame and project to 2D polygon
        corners = box_from_pred(center, size, yaw)
        poly_sensor = project_corners_xy(corners)

        # 5) transform polygon to costmap frame
        poly_map = transform_points_xy(poly_sensor, tf)

        # 6) fill polygon with grid points at costmap resolution and publish as PointCloud2
        filled = fill_polygon_points(poly_map, resolution=self.resolution)  # (M,3) z=0
        cloud_msg = xyz_to_pc2(filled, self.costmap_frame)
        self.pub_cloud.publish(cloud_msg)

        # 7) markers (optional)
        if self.pub_mk:
            self.pub_mk.publish(self.make_markers(poly_map, msg.header.stamp))

    # ------ visualization ------
    def make_markers(self, poly_map, stamp):
        ma = MarkerArray()
        line = Marker()
        line.header.frame_id = self.costmap_frame
        line.header.stamp = stamp
        line.ns = "detector"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.03
        line.color.a = 1.0
        line.color.r = 1.0
        line.color.g = 0.0
        line.color.b = 0.0
        # close polygon
        pts = list(poly_map) + [poly_map[0]]
        from geometry_msgs.msg import Point
        line.points = [Point(x=float(p[0]), y=float(p[1]), z=0.05) for p in pts]
        ma.markers.append(line)
        return ma

def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
