# src/plots/make_ros2_pipeline.py
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figs/ros2_pipeline.png"
os.makedirs("figs", exist_ok=True)

# ---------- helpers ----------
def add_box(ax, x, y, w, h, text, fc="#E8F1FF", ec="#1E88E5", text_color="#0D47A1"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        fc=fc, ec=ec, lw=2
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, va="center", ha="center",
            fontsize=12, color=text_color)
    return (x, y, w, h)

def add_down_arrow(ax, box_top, box_bottom, label=None, label_dx=0.35, label_dy=0.0):
    """Arrow from center-bottom of top box to center-top of bottom box,
    with label placed slightly to the side (dx, dy) so it doesn't overlap."""
    (x1, y1, w1, h1) = box_top
    (x2, y2, w2, h2) = box_bottom
    p0 = (x1 + w1/2, y1)            # bottom center of top
    p1 = (x2 + w2/2, y2 + h2)       # top center of bottom
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                          lw=2, color="#37474F")
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_dx
        my = (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=10, color="#37474F")

# ---------- layout ----------
fig, ax = plt.subplots(figsize=(6.2, 9))  # portrait for vertical flow

# common sizes
W, H = 4.8, 1.15
X = 0.6

cam    = add_box(ax, X, 7.6, W, H, "Depth Camera\n(PointCloud2)")
det    = add_box(ax, X, 6.0, W, H, "Detector Node\n(PointNeXt‑tiny)")
topic  = add_box(ax, X, 4.4, W, H, r"/detected_obstacles\n(PointCloud2)")
nav    = add_box(ax, X, 2.8, W, H, "Nav2 Navigation\n(Costmap + Planner)")
robot  = add_box(ax, X, 1.2, W, H, "Robot Movement\n(Safe Path Execution)")

# arrows with labels offset to the right
add_down_arrow(ax, cam, det,   "3D point cloud")
add_down_arrow(ax, det, topic, "3D boxes → 2D polygons")
add_down_arrow(ax, topic, nav, "Obstacle data")
add_down_arrow(ax, nav, robot, "Planned path")

# cosmetics
ax.set_xlim(0, 6.2); ax.set_ylim(0.6, 9.2)
ax.axis("off")
fig.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
