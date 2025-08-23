# src/eval.py
import os, argparse, yaml, math, torch
from collections import Counter, defaultdict
from torch.utils.data import DataLoader

from dataset import SingleObjectPointClouds
from model_sota import TwoHeadSOTA


def build_loader(cfg, split="val", batch_size=16):
    if split == "test" and "test_list" in cfg["data"] and cfg["data"]["test_list"]:
        list_file = cfg["data"]["test_list"]
    else:
        list_file = cfg["data"]["val_list"]

    ds = SingleObjectPointClouds(
        pc_dir=os.path.join(cfg["data"]["root"], "pointclouds"),
        list_file=list_file,
        label_map_file=cfg["data"]["labels_file"],
        box_json_file=cfg["data"]["box_json"],
        npoints=cfg["data"]["num_points"],
        augment=False,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return ds, dl


@torch.no_grad()
def predict_logits(model, x, tta=False):
    """Return logits with or without simple TTA (±10° around Z)."""
    if not tta:
        return model(x)[0]

    # 3-view TTA: -10°, 0°, +10°
    logits = 0
    device = x.device
    dtype = x.dtype

    for deg in (-10, 0, 10):
        if deg == 0:
            xr = x
        else:
            theta = torch.tensor(deg / 180.0 * math.pi, device=device, dtype=dtype)
            c, s = torch.cos(theta), torch.sin(theta)
            R = torch.tensor([[c, -s, 0.0],
                              [s,  c, 0.0],
                              [0.0, 0.0, 1.0]], device=device, dtype=dtype)
            xr = torch.einsum('ij,bnj->bni', R, x)
        logits = logits + model(xr)[0]

    return logits / 3.0


@torch.no_grad()
def evaluate(model, loader, num_classes, tta=False):
    pred_hist, gt_hist = Counter(), Counter()
    conf = defaultdict(int)
    correct = total = 0

    model.eval()
    for b in loader:
        x = b["points"]
        y = b["class_id"]
        logits = predict_logits(model, x, tta=tta)
        p = logits.argmax(-1)

        for yi, pi in zip(y.tolist(), p.tolist()):
            gt_hist[yi] += 1
            pred_hist[pi] += 1
            conf[(yi, pi)] += 1
            correct += int(yi == pi)
            total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, pred_hist, gt_hist, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to configs/sota.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--tta", action="store_true", help="Enable ±10° Z-rotation TTA")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    _, dl = build_loader(cfg, args.split, args.batch)

    ck = torch.load(args.ckpt, map_location="cpu")
    num_classes = ck.get("num_classes", cfg["model"]["num_classes"])

    model = TwoHeadSOTA(
        num_classes=num_classes,
        backbone=cfg["model"]["backbone"],
        box_head=cfg["model"]["box_head"],
        out_dim=cfg["model"]["width"],
    )
    model.load_state_dict(ck["model"], strict=False)

    acc, pred_hist, gt_hist, conf = evaluate(model, dl, num_classes, tta=args.tta)

    print(f"{args.split} acc ({'TTA' if args.tta else 'no-TTA'}): {acc:.4f}")
    print("top predicted classes:", pred_hist.most_common(5))
    print("top gt classes:", gt_hist.most_common(5))

    print("\nPer-class accuracy:")
    for c in range(num_classes):
        total_c = gt_hist[c]
        acc_c = conf[(c, c)] / total_c if total_c > 0 else 0
        print(f"Class {c}: {acc_c:.3f} ({conf[(c,c)]}/{total_c})")

    # Helpful: show confusions for historically weak classes, if they exist
    for hc in [2, 21, 22, 33]:
        if hc < num_classes:
            pairs = sorted([(k, v) for k, v in conf.items() if k[0] == hc],
                           key=lambda x: -x[1])[:5]
            print(f"\nTop confusions for class {hc}:", pairs)


if __name__ == "__main__":
    main()
