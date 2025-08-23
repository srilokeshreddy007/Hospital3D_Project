import os, json, math, random
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

def _rotz(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def canonicalize(points: np.ndarray, center: np.ndarray, size: np.ndarray, heading: float):
    """
    Translate by -center, rotate by -heading (z axis), scale by s=max(size).
    Returns points_can, base_scale s.
    """
    pts = points - center.reshape(1, 3)
    if abs(heading) > 1e-8:
        R = _rotz(-heading).astype(np.float32)
        pts = (R @ pts.T).T
    s = float(np.max(size)) if np.max(size) > 0 else 1.0
    pts = pts / s
    return pts.astype(np.float32), s

def jitter_points(pts, sigma=0.01, clip=0.05):
    N, C = pts.shape
    jitter = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return (pts + jitter).astype(np.float32)

def random_dropout(pts, drop_prob=0.1):
    if drop_prob <= 0: return pts
    mask = np.random.rand(pts.shape[0]) > drop_prob
    if mask.sum() < 16:  # keep at least a few
        return pts
    return pts[mask]

def random_rotate_tilt(pts, max_tilt_deg=15):
    # small tilt around x/y; keep z-up approximately
    ax = np.deg2rad(np.random.uniform(-max_tilt_deg, max_tilt_deg))
    ay = np.deg2rad(np.random.uniform(-max_tilt_deg, max_tilt_deg))
    Rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]], dtype=np.float32)
    Ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]], dtype=np.float32)
    R = (Ry @ Rx).astype(np.float32)
    return (R @ pts.T).T

class SingleObjectPointClouds(Dataset):
    """
    Each item: one isolated object point cloud.
    Targets: class_id (int) and 3D box residuals relative to canonical frame:
      residuals = [dx, dy, dz, dlog_l, dlog_w, dlog_h, dyaw]
    We expect box json entries with {center, size(l,w,h), heading, class_id}.
    """
    def __init__(
        self,
        pc_dir: str,
        list_file: str,
        label_map_file: str,
        box_json_file: str,
        npoints: int = 4096,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.pc_dir = pc_dir
        with open(list_file, 'r') as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        # filename -> class_id map (from labels_combined.txt)
        self.label_map: Dict[str, int] = {}
        with open(label_map_file, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                fn, cid = ln.split()
                self.label_map[fn] = int(cid)
        with open(box_json_file, 'r') as f:
            self.boxes: Dict[str, dict] = json.load(f)

        # keep only those listed that exist in both maps
        self.files = [fn for fn in self.files if (fn in self.label_map and fn in self.boxes)]
        self.npoints = npoints
        self.augment = augment
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return len(self.files)

    def _sample_points(self, pts: np.ndarray, target_n: int) -> np.ndarray:
        n = pts.shape[0]
        if n == target_n:
            return pts
        if n > target_n:
            idx = np.random.choice(n, target_n, replace=False)
            return pts[idx]
        # upsample
        reps = target_n // n + 1
        pts_rep = np.repeat(pts, reps, axis=0)
        return pts_rep[:target_n]

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        full = os.path.join(self.pc_dir, fname)
        pts = np.load(full)  # (N, 3) float32 expected
        meta = self.boxes[fname]
        center = np.array(meta['center'], dtype=np.float32)
        size = np.array(meta['size'], dtype=np.float32)  # (l,w,h)
        heading = float(meta.get('heading', 0.0))
        class_id = int(meta.get('class_id', self.label_map[fname]))

        # canonicalize
        pts_can, s = canonicalize(pts, center, size, heading)

        # optional augmentations in canonical space
        if self.augment:
            # small z-rotation invariance
            ang = np.deg2rad(np.random.uniform(-10, 10))
            Rz = _rotz(ang).astype(np.float32)
            pts_can = (Rz @ pts_can.T).T
            # light scaling
            scale = np.random.uniform(0.95, 1.05)
            pts_can = pts_can * scale
            # jitter/dropout/tilt
            pts_can = random_rotate_tilt(pts_can, max_tilt_deg=10)
            pts_can = jitter_points(pts_can, sigma=0.01, clip=0.03)
            pts_can = random_dropout(pts_can, drop_prob=0.05)

        # fixed-size sampling
        pts_can = self._sample_points(pts_can, self.npoints)

        # residual targets in canonical frame
        # base frame has center=0, yaw=0, and scale s.
        # So normalized gt size is size/s; residuals are:
        dcenter = np.zeros(3, dtype=np.float32)  # after perfect canonicalization
        norm_size = np.maximum(size / s, 1e-6)
        dlog_size = np.log(norm_size).astype(np.float32)  # ~small residuals
        dyaw = np.float32(0.0)  # heading - 0 (already removed)
        box_res = np.concatenate([dcenter, dlog_size, [dyaw]]).astype(np.float32)

        return {
            "points": torch.from_numpy(pts_can).float(),    # (N,3)
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "box_res": torch.from_numpy(box_res).float(),   # (7,)
            "fname": fname,
            "base_scale": torch.tensor(s, dtype=torch.float32),
            "gt_center": torch.from_numpy(center).float(),
            "gt_size": torch.from_numpy(size).float(),
            "gt_heading": torch.tensor(heading, dtype=torch.float32),
        }
