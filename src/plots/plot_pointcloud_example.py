# src/plots/plot_pointcloud_example.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pick a sample .npy file
fpath = "data/pointclouds/hospitalbed2_scan_00002.npy"

# load npy
pts = np.load(fpath).astype(np.float32)
print(f"Loaded {pts.shape[0]} points from {fpath}")

# Matplotlib scatter
out_mpl = "figs/point_cloud_example.png"
fig = plt.figure(figsize=(6,6), dpi=150)
ax = fig.add_subplot(111, projection="3d")
s = max(1, min(2, 100000 // max(1, len(pts))))
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, alpha=0.7)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.savefig(out_mpl)
plt.close(fig)
print(f"Saved Matplotlib scatter: {out_mpl}")
