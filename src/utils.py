import os, json, random, math
from pathlib import Path
from typing import Dict, List, Union, Iterable, Tuple

import numpy as np
import torch

# ---------------------------
# Config
# ---------------------------
def load_config(path: Union[str, Path]) -> dict:
    """
    Load a YAML config file into a Python dict.
    """
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Back-compat alias (train.py imports seed_all)
def seed_all(seed: int = 42):
    return set_seed(seed)

# ---------------------------
# Class name/id helpers
# ---------------------------

# Mapping inferred from your labels file.
_CLASS_NAMES = [
    "backpack",        # 0
    "cart",            # 1
    "cart2",           # 2
    "chair2",          # 3
    "curtain",         # 4
    "curtainstand",    # 5
    "dufflebag",       # 6
    "dustbin",         # 7
    "dustbin2",        # 8
    "hospitalbed",     # 9
    "hospitalbed2",    # 10
    "human",           # 11
    "human2",          # 12
    "ivpolewithbag",   # 13
    "ivpolewithbag2",  # 14
    "ivpolewithbag3",  # 15
    "linencart",       # 16
    "linencart2",      # 17
    "linencart3",      # 18
    "medicalcart",     # 19
    "medicalcart2",    # 20
    "overheadtable",   # 21
    "overheadtable2",  # 22
    "pillow",          # 23
    "strecher",        # 24
    "strecher2",       # 25
    "strecher3",       # 26
    "trashbin",        # 27
    "trashbin2",       # 28
    "waitingseats",    # 29
    "waitingseats2",   # 30
    "waitingseats3",   # 31
    "wheelchair",      # 32
    "wheelchair2",     # 33
    "wheelchair3",     # 34
    "workingcart",     # 35
    "workingcart3",    # 36
    "wrkingcart2",     # 37
]

_name_to_id = {n: i for i, n in enumerate(_CLASS_NAMES)}
_id_to_name = {i: n for i, n in enumerate(_CLASS_NAMES)}

def class_id_to_name(cid: int) -> str:
    return _id_to_name[int(cid)]

def name_to_class_id(name: str) -> int:
    return int(_name_to_id[name])

# ---------------------------
# Geometry helpers
# ---------------------------
def canonicalize_points(pts: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Zero-center and scale points to ~unit scale.
    Works with (N,3) numpy or torch.
    """
    if isinstance(pts, np.ndarray):
        c = pts.mean(axis=0, keepdims=True)
        s = pts.std(axis=0, keepdims=True) + 1e-6
        return (pts - c) / s
    elif torch.is_tensor(pts):
        c = pts.mean(dim=0, keepdim=True)
        s = pts.std(dim=0, keepdim=True) + 1e-6
        return (pts - c) / s
    else:
        raise TypeError("pts must be numpy array or torch tensor")

def box_to_tensor(box: dict, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Convert a dict box with keys center(size=3), size(size=3), heading(scalar)
    into a 7D torch tensor: [cx,cy,cz, sx,sy,sz, heading]
    """
    arr = np.asarray(list(box["center"]) + list(box["size"]) + [box["heading"]], dtype=np.float32)
    return torch.from_numpy(arr).to(device)

def apply_residuals(base_box: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """
    Apply residual deltas to a base 7D box tensor.
    base_box: (..., 7)
    residual: (..., 7) where:
        - residual[:3] -> delta center (additive)
        - residual[3:6] -> delta on log-size (multiplicative via exp)
        - residual[6] -> delta heading (additive)
    Returns new box tensor with same shape.
    """
    while residual.dim() < base_box.dim():
        residual = residual.unsqueeze(0)
    out = base_box.clone()
    out[..., :3] = base_box[..., :3] + residual[..., :3]
    out[..., 3:6] = base_box[..., 3:6] * torch.exp(residual[..., 3:6])
    out[..., 6] = base_box[..., 6] + residual[..., 6]
    return out

# Back-compat alias (in case other files import the old name)
def apply_residuals_to_box(base_box: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return apply_residuals(base_box, residual)

# ---------------------------
# Metrics helpers
# ---------------------------

class AverageMeter:
    """Compute and store current value and average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)

def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy for classification logits (N,C) and target labels (N,)."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred == target).float().sum().item()
        return correct / max(target.numel(), 1)
