# train_sota.py  —  CPU-friendly, stabilized training loop
# - No augmentation by default (set USE_AUG=True below if you want to re-enable lightly)
# - CrossEntropyLoss (no label smoothing)
# - Balanced sampler
# - Head warmup (freeze backbone first few epochs)

import os, csv, math, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from model_sota import TwoHeadSOTA
from dataset import SingleObjectPointClouds
from utils import set_seed as seed_all, accuracy_top1, load_config, AverageMeter


# -------- toggles (can be promoted to YAML later) --------
USE_AUG = False           # keep False for now (no-tricks). set True to call augment_batch().
HEAD_WARMUP_EPOCHS = 3    # epochs to train heads with backbone frozen
PRINT_EVERY = 1           # epochs


# -------- optional light augmentation (disabled when USE_AUG=False) --------
def augment_batch(points):
    # points: (B,N,3), light augment to reduce drift if enabled
    import math as _math
    B, N, _ = points.shape
    x = points
    x = x + torch.randn_like(x) * 0.002
    # drop ~5% points (mask false -> zero out)
    keep = (torch.rand(B, N, 1, device=x.device) > 0.95)
    x = torch.where(keep, x, torch.zeros_like(x))
    # small z-rotation ±10°
    theta = (torch.rand(B, device=x.device) - 0.5) * (2 * _math.pi / 18)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
        torch.stack([s,  c, torch.zeros_like(c)], dim=-1),
        torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1),
    ], dim=1)  # (B,3,3)
    x = torch.einsum('bij,bnj->bni', R, x)
    return x


def make_dataset(cfg, split):
    if split == "train":
        list_file = cfg["data"]["train_list"]
        augment = False  # IMPORTANT: avoid double aug — we control it in the loop via USE_AUG
    else:
        list_file = cfg["data"]["val_list"]
        augment = False

    ds = SingleObjectPointClouds(
        pc_dir=os.path.join(cfg["data"]["root"], "pointclouds"),
        list_file=list_file,
        label_map_file=cfg["data"]["labels_file"],
        box_json_file=cfg["data"]["box_json"],
        npoints=cfg["data"]["num_points"],
        augment=augment,
        seed=cfg.get("seed", 42),
    )
    return ds


def build_loaders(cfg):
    tr_ds = make_dataset(cfg, "train")
    va_ds = make_dataset(cfg, "val")

    # Balanced sampler (sqrt-inverse frequency)
    import collections
    counts = collections.Counter()
    for f in tr_ds.files:
        counts[int(tr_ds.label_map[f])] += 1
    boost = {2: 1.2, 21: 1.3, 22: 1.2, 33: 1.2}
    ww = []
    for f in tr_ds.files:
        c = int(tr_ds.label_map[f])
        base = (counts[c] ** -0.5)
        ww.append(base * boost.get(c, 1.0))
    weights = torch.tensor(ww, dtype=torch.float)

    sampler = WeightedRandomSampler(weights, num_samples=len(tr_ds), replacement=True)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["train"].get("num_workers", 2),
        drop_last=True,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 2),
    )
    return tr_ds, va_ds, tr_loader, va_loader


def build_scheduler(opt, steps_total, warmup_ratio=0.05):
    warmup = max(10, int(warmup_ratio * steps_total))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, opt, ce, l1, box_w, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        pts = batch["points"].to(device)
        labels = batch["class_id"].to(device)
        target_res = torch.zeros((pts.size(0), 7), device=device)

        if USE_AUG:
            pts = augment_batch(pts)

        logits, pred_res = model(pts)
        loss = ce(logits, labels) + box_w * l1(pred_res, target_res)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_meter.update(float(loss.item()), pts.size(0))
        acc_meter.update(accuracy_top1(logits, labels), pts.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, ce, l1, box_w, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        pts = batch["points"].to(device)
        labels = batch["class_id"].to(device)
        target_res = torch.zeros((pts.size(0), 7), device=device)

        logits, pred_res = model(pts)
        loss = ce(logits, labels) + box_w * l1(pred_res, target_res)

        loss_meter.update(float(loss.item()), pts.size(0))
        acc_meter.update(accuracy_top1(logits, labels), pts.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def main(cfg_path):
    cfg = load_config(cfg_path)
    seed_all(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(cfg["train"].get("runs_dir", "runs")); runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / "metrics_sota.csv"

    tr_ds, va_ds, tr_loader, va_loader = build_loaders(cfg)
    print(f"Train size: {len(tr_ds)} | Val size: {len(va_ds)} | Batches/epoch: {len(tr_loader)}")

    num_classes = cfg["model"]["num_classes"]
    model = TwoHeadSOTA(
        num_classes=num_classes,
        backbone=cfg["model"].get("backbone", "pointnext_tiny"),
        box_head=cfg["model"].get("box_head", "residual"),
        out_dim=cfg["model"].get("width", 512),
        # dropout and ln live inside model_sota.py; we set drop there
    ).to(device)

    # ---- Head warmup: freeze backbone, train heads only ----
    for p in model.backbone.parameters():
        p.requires_grad = False

    # collect head params (cls + box if residual)
    head_params = list(model.cls.parameters())
    if getattr(model, "box_head_type", "residual") == "residual" and hasattr(model, "box"):
        head_params += list(model.box.parameters())

    opt = torch.optim.AdamW(head_params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    steps_total = cfg["train"]["epochs"] * len(tr_loader)
    sched = build_scheduler(opt, steps_total)

    num_classes = cfg["model"]["num_classes"]
    w = torch.ones(num_classes, dtype=torch.float, device=device)
    # gentle, not extreme (you can tune these up/down by 0.2)
    w[2]  = 1.2
    w[21] = 1.3
    w[22] = 1.3
    w[33] = 1.3
    ce = nn.CrossEntropyLoss(weight=w)

    l1 = nn.L1Loss()
    box_w = cfg["loss"]["box_weight"]

    new_file = not csv_path.exists()
    if new_file:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_acc = -1.0
    E = cfg["train"]["epochs"]

    for epoch in range(1, E + 1):
        # Unfreeze backbone after warmup
        if epoch == HEAD_WARMUP_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            # lower LR a bit after unfreezing
            opt = torch.optim.AdamW(model.parameters(), lr=max(1e-5, cfg["train"]["lr"] * 0.5),
                                    weight_decay=cfg["train"]["weight_decay"])
            # rebuild scheduler for remaining steps
            steps_left = (E - epoch + 1) * len(tr_loader)
            sched = build_scheduler(opt, steps_left)

        tr_metrics = train_one_epoch(model, tr_loader, opt, ce, l1, box_w, device)
        va_metrics = evaluate(model, va_loader, ce, l1, box_w, device)
        sched.step()

        if epoch % PRINT_EVERY == 0:
            print(f"[{epoch:03d}/{E}] train {tr_metrics['loss']:.4f}/{tr_metrics['acc']:.4f} | "
                  f"val {va_metrics['loss']:.4f}/{va_metrics['acc']:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{tr_metrics['loss']:.4f}", f"{tr_metrics['acc']:.4f}",
                f"{va_metrics['loss']:.4f}", f"{va_metrics['acc']:.4f}",
                opt.param_groups[0]["lr"],
            ])

        if va_metrics["acc"] > best_acc:
            best_acc = va_metrics["acc"]
            SAVE_NAME = "best_sota_rebal.pt" 
            torch.save(
                {"model": model.state_dict(), "num_classes": num_classes, "cfg": cfg},
                out_dir / SAVE_NAME
            )

    print(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()
    main(args.cfg)
