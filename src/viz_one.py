import os, json, argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d

from model import PointNetTwoHead as PointNetSingle

# --- minimal helpers (mirrors utils.py) ---
def class_id_to_name(idx):
    # Fill this mapping to your 39 classes in the correct order
    # Example placeholder (edit to your list):
    names = [
        "backpack","cart","cart2","chair2","curtain","curtainstand",
        "dufflebag","dustbin","dustbin2","hospitalbed","hospitalbed2",
        "human","human2","ivpolewithbag","ivpolewithbag2","ivpolewithbag3",
        "linencart","linencart2","linencart3","medicalcart","medicalcart2",
        "overheadtable","overheadtable2","pillow","strecher","strecher3",
        "strecher2","trashbin","trashbin2","waitingseats","waitingseats2",
        "waitingseats3","wheelchair","wheelchair2","wheelchair3",
        "workingcart","workingcart3","wrkingcart2"  # ensure length=39
    ]
    return names[idx] if 0 <= idx < len(names) else f"class_{idx}"

def canonicalize_points(points, box):
    # Translate to center, de-rotate yaw (here yaw=0 in your data), scale by max(size)
    c = np.array(box[:3], dtype=np.float32)
    s = np.array(box[3:6], dtype=np.float32)
    maxs = max(1e-6, float(s.max()))
    pts = points - c[None, :]
    pts = pts / maxs
    return pts, (c, s, maxs)

def apply_residuals(box, res):
    # box: [cx,cy,cz,l,w,h,yaw]; res: [dx,dy,dz, dl,dw,dh, dyaw]
    out = np.array(box, dtype=np.float32).copy()
    out[:3] += res[:3]
    out[3:6] = np.maximum(1e-6, out[3:6] * np.exp(res[3:6]))  # log-space like refinement (if trained that way)
    out[6] += res[6]
    return out

def load_box(box_json, filename):
    with open(box_json, "r") as f:
        meta = json.load(f)
    b = meta[Path(filename).name]
    center = np.array(b["center"], dtype=np.float32)
    size   = np.array(b["size"], dtype=np.float32)
    yaw    = float(b.get("heading", 0.0))
    cls_id = int(b["class_id"])
    return np.concatenate([center, size, np.array([yaw], dtype=np.float32)]), cls_id

def to_o3d_box(box, color):
    # axis-aligned box from center/size; yaw ignored (your data yaw≈0)
    center = box[:3]
    size = box[3:6]
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - size/2.0,
                                               max_bound=center + size/2.0)
    bbox.color = color
    return bbox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="Path to a single .npy point cloud")
    ap.add_argument("--box_json", required=True, help="Path to box_labels.json")
    ap.add_argument("--ckpt", required=True, help="Path to outputs/best.pt")
    ap.add_argument("--temperature", default=None, help="Path to outputs/calibration.json (optional)")
    ap.add_argument("--num_points", type=int, default=4096)
    args = ap.parse_args()

    # load data
    pts = np.load(args.npy).astype(np.float32)  # (N,3)
    gt_box, _gt_cls = load_box(args.box_json, args.npy)

    # sample / pad to num_points
    if pts.shape[0] >= args.num_points:
        sel = np.random.choice(pts.shape[0], args.num_points, replace=False)
    else:
        pad = np.random.choice(pts.shape[0], args.num_points - pts.shape[0], replace=True)
        sel = np.concatenate([np.arange(pts.shape[0]), pad])
    pts = pts[sel]

    # canonicalize with GT (like training)
    can_pts, (c, s, maxs) = canonicalize_points(pts, gt_box)
    can_box = np.concatenate([np.zeros(3, dtype=np.float32),  # center≈0
                              gt_box[3:6] / maxs,             # scaled
                              np.array([0.0], dtype=np.float32)])  # yaw≈0

    # model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    num_classes = cfg.get("model", {}).get("num_classes", 39)
    model = PointNetSingle(num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    with torch.no_grad():
        tpts = torch.from_numpy(can_pts[None, ...])  # (1,N,3)
        logits, res = model(tpts)
        logits = logits[0]
        res = res[0].numpy()

    # temperature (optional)
    if args.temperature and Path(args.temperature).exists():
        with open(args.temperature, "r") as f:
            T = float(json.load(f).get("temperature", 1.0))
    else:
        T = 1.0

    probs = torch.softmax(logits / T, dim=0).numpy()
    pred_id = int(probs.argmax())
    conf = float(probs[pred_id])

    # predicted box in canonical → back to original scale
    # First, construct canonical base (zero center, scaled size), then apply residuals,
    # then unscale/translate back.
    pred_can_box = apply_residuals(can_box, res)
    # unscale
    pred_box = pred_can_box.copy()
    pred_box[:3] = pred_can_box[:3] * maxs + c
    pred_box[3:6] = np.maximum(1e-6, pred_can_box[3:6] * maxs)

    # ---- Open3D scene ----
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.55, 0.55, 0.55])

    gt_bbox = to_o3d_box(gt_box, color=[0.0, 0.8, 0.0])     # green
    pr_bbox = to_o3d_box(pred_box, color=[0.9, 0.0, 0.0])   # red

    name = class_id_to_name(pred_id)
    win_title = f"Pred: {name} ({conf:.2f}) | File: {Path(args.npy).name}"
    o3d.visualization.draw_geometries([pcd, gt_bbox, pr_bbox], window_name=win_title)

    # also print to console
    print(json.dumps({
        "file": Path(args.npy).name,
        "pred_class": name,
        "confidence": round(conf, 4),
        "pred_box": pred_box.tolist(),
        "gt_box": gt_box.tolist()
    }, indent=2))

if __name__ == "__main__":
    main()

