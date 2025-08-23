
import os, json, argparse, numpy as np, torch
from pathlib import Path
from model import PointNetTwoHead as PointNetSingle
from dataset import canonicalize

def load_calibration(ckpt_dir: Path):
    path = ckpt_dir / "calibration.json"
    if path.exists():
        with open(path, "r") as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root that contains pointclouds/, labels.txt, boxes.json")
    ap.add_argument("--ckpt_dir", required=True, help="Directory that contains best.pt and calibration.json")
    ap.add_argument("--file", required=True, help="Filename inside pointclouds/ to run (e.g., xyz.npy)")
    ap.add_argument("--num_points", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.ckpt_dir) / "best.pt", map_location=device)
    num_classes = ckpt.get("num_classes", ckpt.get("cfg", {}).get("model", {}).get("num_classes", 39))
    model = PointNetSingle(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    boxes = json.load(open(os.path.join(args.data_root, "boxes.json")))
    # id->name map
    id2name = {}
    with open(os.path.join(args.data_root, "labels.txt")) as f:
        for ln in f:
            name, cid = ln.split()
            cid = int(cid)
            if cid not in id2name:
                # trim any trailing _scan_ suffixes if present
                id2name[cid] = name.split("_scan_")[0]

    T = load_calibration(Path(args.ckpt_dir))

    # load points and meta
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
    pc_can, s = canonicalize(pc, center, size, yaw)
    x = torch.from_numpy(pc_can[None, ...]).to(device)

    with torch.no_grad():
        logits, boxres = model(x)
        logits = logits / T
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        pred_name = id2name.get(pred_id, f"class_{pred_id}")
        conf = float(probs[pred_id])

    out = {
        "file": fname,
        "pred_class_id": pred_id,
        "pred_class_name": pred_name,
        "confidence": conf,
        "center": meta["center"],
        "size": meta["size"],
        "yaw": float(meta.get("heading", 0.0)),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
