import open3d as o3d
import numpy as np
import os

# === CONFIG ===
INPUT_DIR = "/home/sri/ultrasonic_sim/output/isolated_scans"       # <-- Update this
OUTPUT_DIR = "/home/sri/ultrasonic_sim/output"      # <-- Update this
NUM_POINTS = 2048                           # Points per cloud
MAX_RADIUS = 5.0                            # Max distance to keep

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_point_cloud(pcd_path):
    # Load PLY
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # Filter: remove NaNs and points that are too far
    valid = ~np.isnan(points).any(axis=1)
    points = points[valid]
    distances = np.linalg.norm(points, axis=1)
    points = points[distances < MAX_RADIUS]

    if len(points) < 100:
        print(f"⚠️ Skipped {os.path.basename(pcd_path)}: too few valid points")
        return

    # Downsample
    if len(points) > NUM_POINTS:
        idx = np.random.choice(len(points), NUM_POINTS, replace=False)
        points = points[idx]
    elif len(points) < NUM_POINTS:
        pad = np.random.choice(len(points), NUM_POINTS - len(points), replace=True)
        points = np.vstack([points, points[pad]])

    # Normalize
    points = points - np.mean(points, axis=0)  # Center
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale                    # Unit sphere

    return points.astype(np.float32)

# === Batch Processing ===
ply_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".ply")]

for fname in ply_files:
    ply_path = os.path.join(INPUT_DIR, fname)
    npy_path = os.path.join(OUTPUT_DIR, fname.replace(".ply", ".npy"))

    processed = preprocess_point_cloud(ply_path)
    if processed is not None:
        np.save(npy_path, processed)
        print(f"✅ Saved: {npy_path}")
