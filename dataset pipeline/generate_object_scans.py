import blenderproc as bproc
import numpy as np
import os
import json
import math
import random
import bpy
import open3d as o3d

# === CONFIG ===
OBJ_PATH = "/home/sri/ultrasonic_sim/OBJ_converted/wrkingcart2.obj"  # ← Update this per object
LABEL_MAP_PATH = "/home/sri/ultrasonic_sim/output/label_map.json"
OUTPUT_DIR = "/home/sri/ultrasonic_sim/output/isolated_scans"
SCANS_PER_OBJECT = 30

CAMERA_HEIGHT_RANGE = (0.4, 0.5)
CAMERA_DIST_RANGE = (0.2, 4.0)
LIGHT_Z_RANGE = (1.5, 3.0)
LIGHT_ENERGY_RANGE = (300, 1000)

# === INIT BlenderProc ===
bproc.init()
bpy.ops.wm.read_factory_settings(use_empty=True)

# Optional: limit Blender to 1 frame
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 1
bpy.context.scene.cycles.samples = 1  # Fast render

# === Create ground plane ===
floor = bproc.object.create_primitive('PLANE', scale=[3, 3, 1])
floor.set_location([0, 0, 0])

# === Load OBJ and position ===
bproc.loader.load_obj(OBJ_PATH)
obj = bproc.object.get_all_mesh_objects()[0]
obj.set_location([0, 0, 0.01])
obj_center = obj.get_location()
obj_class = os.path.splitext(os.path.basename(OBJ_PATH))[0]

# === Load label map ===
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
class_id = label_map.get(obj_class.lower(), -1)
if class_id == -1:
    raise ValueError(f"❌ Class '{obj_class}' not found in label map!")

print(f"📦 Scanning {obj_class} (ID={class_id}) at {obj_center} ...")

# === Setup camera ===
cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

bproc.camera.set_intrinsics_from_blender_params(
    lens=35,
    lens_unit="MILLIMETERS",
    image_width=640,
    image_height=480
)

# === Configure renderer once ===
bproc.renderer.set_output_format("OPEN_EXR")
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# === Prepare output folder ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
LABEL_FILE = os.path.join(OUTPUT_DIR, f"{obj_class}_labels.txt")
label_lines = []

# === Main scan loop ===
for scan_id in range(SCANS_PER_OBJECT):
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(*CAMERA_DIST_RANGE)
    height = np.random.uniform(*CAMERA_HEIGHT_RANGE)
    cam_position = np.array([
        obj_center[0] + radius * math.cos(angle),
        obj_center[1] + radius * math.sin(angle),
        height
    ])

    rotation = bproc.camera.rotation_from_forward_vec(obj_center - cam_position)
    cam2world_matrix = np.eye(4)
    cam2world_matrix[:3, :3] = rotation
    cam2world_matrix[:3, 3] = cam_position
    bproc.camera.add_camera_pose(cam2world_matrix)

    # Random lighting
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([
        obj_center[0] + random.uniform(-1.0, 1.0),
        obj_center[1] + random.uniform(-1.0, 1.0),
        obj_center[2] + random.uniform(*LIGHT_Z_RANGE)
    ])
    light.set_energy(random.uniform(*LIGHT_ENERGY_RANGE))
    light.set_color(np.random.uniform(0.9, 1.0, size=3))

    # === Render
    data = bproc.renderer.render()
    depth_image = np.array(data["depth"][0]) if isinstance(data["depth"], (list, tuple)) else np.array(data["depth"])

    # === Convert to point cloud
    pointcloud = bproc.camera.pointcloud_from_depth(depth_image)
    points = pointcloud.reshape(-1, 3)
    points = points[~np.isnan(points).any(axis=1)]

    # === Save .ply
    scan_name = f"{obj_class}_scan_{scan_id:05d}.ply"
    pcd_path = os.path.join(OUTPUT_DIR, scan_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(pcd_path, pcd)

    label_lines.append(f"{scan_name} {class_id}")
    print(f"✅ Saved: {scan_name}")

# === Save labels
with open(LABEL_FILE, "w") as f:
    f.write("\n".join(label_lines))

print(f"\n🎉 Done! Total scans: {SCANS_PER_OBJECT}")
print(f"📄 Labels saved to: {LABEL_FILE}")

