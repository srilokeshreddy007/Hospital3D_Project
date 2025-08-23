# src/plots/make_navigation_process.py
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib import patheffects as pe

OUT = "figs/navigation_process.png"
os.makedirs("figs", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))

# hallway background
ax.add_patch(Rectangle((0,0), 10, 5, facecolor="#FAFAFA", edgecolor="#CFD8DC"))
# side walls
ax.add_patch(Rectangle((0,0), 10, 0.2, facecolor="#ECEFF1", edgecolor="#B0BEC5"))
ax.add_patch(Rectangle((0,4.8), 10, 0.2, facecolor="#ECEFF1", edgecolor="#B0BEC5"))

# goal marker
ax.add_patch(Circle((9.5, 2.5), 0.12, color="#43A047"))
ax.text(9.5, 2.8, "Goal", ha="center", va="bottom", fontsize=11, color="#2E7D32")

# robot (simple disc + heading)
ax.add_patch(Circle((1.0, 2.5), 0.25, color="#1976D2"))
ax.add_patch(FancyArrowPatch((1.0, 2.5), (1.4, 2.5), arrowstyle="-|>", mutation_scale=14,
                             lw=2, color="white"))
ax.text(1.0, 2.05, "Robot", ha="center", va="top", fontsize=10, color="#0D47A1",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# obstacles (bed + IV pole) with GT (green) and Pred (red) boxes
# Bed
bed_xy = (4.2, 1.6); bed_wh = (1.8, 1.2)
ax.add_patch(Rectangle(bed_xy, *bed_wh, facecolor="#B0BEC5", edgecolor="#78909C"))
# GT box (green)
ax.add_patch(Rectangle(bed_xy, *bed_wh, fill=False, edgecolor="#2E7D32", lw=2))
# Pred box (slightly offset) (red)
ax.add_patch(Rectangle((4.25, 1.55), bed_wh[0]*0.98, bed_wh[1]*1.03, fill=False, edgecolor="#D32F2F", lw=2))
ax.text(5.1, 2.9, "Bed", ha="center", va="bottom", fontsize=10, color="#263238")

# IV pole
pole_xy = (6.9, 3.4); pole_wh = (0.15, 0.9)
ax.add_patch(Rectangle(pole_xy, *pole_wh, facecolor="#B0BEC5", edgecolor="#78909C"))
ax.add_patch(Rectangle(pole_xy, *pole_wh, fill=False, edgecolor="#2E7D32", lw=2))
ax.add_patch(Rectangle((6.92, 3.35), pole_wh[0]*0.95, pole_wh[1]*1.05, fill=False, edgecolor="#D32F2F", lw=2))
ax.text(6.95, 4.35, "IV Pole", ha="center", va="bottom", fontsize=10, color="#263238")

# planned path (cubic Bezier around obstacles)
t = np.linspace(0, 1, 200)[:, None]  # (200,1)
P0 = np.array([1.1, 2.5])[None, :]
P1 = np.array([2.8, 2.6])[None, :]
P2 = np.array([3.2, 4.4])[None, :]
P3 = np.array([9.5, 2.5])[None, :]
curve = ((1 - t)**3) * P0 + 3*((1 - t)**2) * t * P1 + 3*(1 - t)*(t**2) * P2 + (t**3) * P3

ax.plot(curve[:,0], curve[:,1], lw=3, color="#1976D2")
ax.add_patch(FancyArrowPatch(curve[-2], curve[-1], arrowstyle="-|>", mutation_scale=14,
                             lw=3, color="#1976D2"))

# legend
ax.plot([], [], color="#2E7D32", lw=2, label="GT box")
ax.plot([], [], color="#D32F2F", lw=2, label="Pred box")
ax.plot([], [], color="#1976D2", lw=3, label="Planned path")
ax.legend(loc="lower left", frameon=True)

ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
