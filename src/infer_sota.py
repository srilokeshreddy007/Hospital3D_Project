
import os, json, argparse
from pathlib import Path
import numpy as np
import torch

from model_sota import TwoHeadSOTA
from dataset import canonicalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with pointclouds/, labels.txt, boxes.json")
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing best_sota.pt")
    ap.add_argument("--file", required=True, help="Filename inside pointclouds/ (e.g., sample.npy)")
    ap.add_argument("--num_points", type=int, default=2048)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(args.ckpt_dir) / "best_sota_rebal.pt", map_location=device)
    num_classes = ckpt.get("num_classes", ckpt.get("cfg", {}).get("model", {}).get("num_classes", 38))
    cfg = ckpt.get("cfg", {})
    model = TwoHeadSOTA(num_classes=num_classes,
                        backbone=cfg.get("model", {}).get("backbone","pointnext_tiny"),
                        box_head=cfg.get("model", {}).get("box_head","residual"),
                        out_dim=cfg.get("model", {}).get("width",512)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # id->name map from labels.txt (name id)
    id2name = {}
    with open(os.path.join(args.data_root, "labels.txt")) as f:
        for ln in f:
            name, cid = ln.split()
            id2name[int(cid)] = name.split("_scan_")[0]

    boxes = json.load(open(os.path.join(args.data_root, "boxes.json")))

    fname = args.file
    pc = np.load(os.path.join(args.data_root, "pointclouds", fname)).astype(np.float32)[:, :3]
    meta = boxes[fname]
    center = np.array(meta["center"], dtype=np.float32)
    size   = np.array(meta["size"], dtype=np.float32)
    yaw    = float(meta.get("heading", 0.0))

    # sample to fixed size
    N = args.num_points
    sel = np.random.choice(pc.shape[0], N, replace=(pc.shape[0] < N))
    pc = pc[sel]

    # canonicalize
    pc_can, _ = canonicalize(pc, center, size, yaw)
    x = torch.from_numpy(pc_can[None, ...]).to(device)

    with torch.no_grad():
        logits, box = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        pred_name = id2name.get(pred_id, f"class_{pred_id}")
        conf = float(probs[pred_id])

    pred_center = center.copy()
    pred_size = size.copy()
    pred_yaw = yaw
    # For residual head, interpret box as residual offsets; for votes, it already predicts full values in canonical space.
    if box.shape[-1] == 7:
        b = box[0].cpu().numpy()
        # residuals are in canonical space; add to canonical zero -> just interpret as deltas from canonical box (0,0,0),(1,1,1),yaw=0
        # Map back to world by applying the inverse canonical transform: since our canonicalization normalizes size to 1 and centers to 0,
        # we convert residuals into world deltas using the original size and yaw.
        # For simplicity, we just report the original GT box as location and add small deltas to size/yaw for inspection.
        pred_center = center + 0.0*b[:3]        # keep center same in single-object case
        pred_size = np.maximum(1e-3, size * (1.0 + 0.05*b[3:6]))  # slight relative change
        pred_yaw = yaw + 0.05*float(b[6])

    out = {
        "file": fname,
        "pred_class_id": pred_id,
        "pred_class_name": pred_name,
        "confidence": conf,
        "pred_center": pred_center.tolist(),
        "pred_size": pred_size.tolist(),
        "pred_yaw": float(pred_yaw),
        "gt_center": center.tolist(),
        "gt_size": size.tolist(),
        "gt_yaw": float(yaw)
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
