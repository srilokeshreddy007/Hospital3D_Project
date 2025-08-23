# src/viz_offscreen.py
import os, json, math, argparse, traceback
import numpy as np

# ---------------- helpers ----------------
def box_corners(center, size, yaw):
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

def edges_from_corners(_):
    return np.array([
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ], dtype=np.int32)

def points_in_obb(pts, center, size, yaw):
    """Boolean mask: which points are inside the oriented box."""
    pts0 = pts - np.asarray(center, np.float32)
    c, s = math.cos(-yaw), math.sin(-yaw)  # inverse rot
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    p_local = pts0 @ R.T
    half = np.asarray(size, np.float32) / 2.0
    return (
        (np.abs(p_local[:,0]) <= half[0]) &
        (np.abs(p_local[:,1]) <= half[1]) &
        (np.abs(p_local[:,2]) <= half[2])
    )

# ---------------- Open3D offscreen ----------------
def try_open3d_render(out_png, pts, gt, pred):
    import open3d as o3d
    from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

    # point cloud (uniform light gray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.35, 0.35, 0.35])

    mat_pc = MaterialRecord()
    mat_pc.shader = "defaultUnlit"
    mat_pc.point_size = 2.0

    def make_lineset(corners, color):
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(edges_from_corners(corners)),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * 12)
        return ls

    gt_ls = make_lineset(box_corners(gt["center"], gt["size"], gt["yaw"]), [0.0, 1.0, 0.0])
    pr_ls = make_lineset(box_corners(pred["center"], pred["size"], pred["yaw"]), [1.0, 0.0, 0.0])

    # line material for O3D 0.19
    mat_line = MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 2.5

    W, H = 1280, 960
    renderer = OffscreenRenderer(W, H)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])

    scene.add_geometry("pc", pcd, mat_pc)
    scene.add_geometry("gt", gt_ls, mat_line)
    scene.add_geometry("pred", pr_ls, mat_line)

    # camera fit
    aabb = pcd.get_axis_aligned_bounding_box()
    if aabb.get_volume() == 0:
        aabb = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    aabb = aabb.scale(1.15, aabb.get_center())  # 15% padding for O3D 0.19

    center = aabb.get_center()
    extent = np.linalg.norm(aabb.get_extent())
    cam_dist = 2.0 * extent if extent > 0 else 2.0

    eye = center + np.array([0.0, -cam_dist, cam_dist])
    up  = np.array([0.0, 0.0, 1.0])
    scene.camera.look_at(center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(out_png, img)
    return True

# ---------------- Matplotlib fallback ----------------
def mpl_fallback(out_png, pts, gt, pred, title_text="", class_name=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # highlight inliers (inside predicted OBB)
    mask_in = points_in_obb(pts, pred["center"], pred["size"], pred["yaw"])

    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")

    # scatter outside (light gray) + inside (orange)
    n = len(pts)
    s_out = max(1, min(1.5, 120000 // max(1, n)))
    s_in  = s_out + 0.3
    ax.scatter(pts[~mask_in,0], pts[~mask_in,1], pts[~mask_in,2], s=s_out, alpha=0.6, c="#BFBFBF")
    ax.scatter(pts[ mask_in,0], pts[ mask_in,1], pts[ mask_in,2], s=s_in,  alpha=0.9, c="#ff8800")

    # draw thicker boxes
    def draw_box(corners, color):
        E = edges_from_corners(corners)
        for i, j in E:
            xs = [corners[i,0], corners[j,0]]
            ys = [corners[i,1], corners[j,1]]
            zs = [corners[i,2], corners[j,2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2.5)

    draw_box(box_corners(gt["center"], gt["size"], gt["yaw"]),  "#1ca11c")  # GT green
    draw_box(box_corners(pred["center"], pred["size"], pred["yaw"]), "#e52420")  # Pred red

    # floating label at predicted center (class only, no score)
    ax.text(pred["center"][0], pred["center"][1], pred["center"][2] + pred["size"][2]/2 + 0.02,
            f"{class_name}",
            color="black", fontsize=8, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.7))

    # aesthetics
    ax.grid(False)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=35)      # consistent angle
    ax.set_box_aspect([1, 1, 1])

    # title (class only)
    if title_text:
        fig.suptitle(title_text, fontsize=11, y=0.98)

    # minimal legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color="#1ca11c", lw=2.5, label="GT box"),
        Line2D([0],[0], color="#e52420", lw=2.5, label="Pred box"),
        Line2D([0],[0], marker='o', color='w', label='Inlier points',
               markerfacecolor="#ff8800", markersize=6),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--json", required=True, help="pred JSON from infer_sota.py")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    pred = json.load(open(args.json))
    fname = pred["file"]
    pts = np.load(os.path.join(args.data_root, "pointclouds", fname)).astype(np.float32)[:, :3]

    gt = {"center": np.array(pred["gt_center"], dtype=np.float32),
          "size":   np.array(pred["gt_size"], dtype=np.float32),
          "yaw":    float(pred["gt_yaw"])}

    pr = {"center": np.array(pred["pred_center"], dtype=np.float32),
          "size":   np.array(pred["pred_size"], dtype=np.float32),
          "yaw":    float(pred["pred_yaw"])}

    # title = class name only (no score)
    title = f"{pred.get('pred_class_name','')}"
    class_name = pred.get("pred_class_name", "")

    ok = False
    try:
        ok = try_open3d_render(args.out, pts, gt, pr)
    except Exception:
        print("[viz_offscreen] Open3D failed, falling back to Matplotlib:\n" + traceback.format_exc())

    if not ok:
        mpl_fallback(args.out, pts, gt, pr, title_text=title, class_name=class_name)
        print(f"[viz_offscreen] Saved (Matplotlib): {args.out}")
    else:
        print(f"[viz_offscreen] Saved (Open3D): {args.out}")

if __name__ == "__main__":
    main()
