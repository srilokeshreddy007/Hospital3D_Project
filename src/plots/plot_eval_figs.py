# src/plots/plot_eval_figs.py
import os, json, numpy as np, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SingleObjectPointClouds
from model_sota import TwoHeadSOTA
import yaml

CFG = "configs/sota.yaml"
CKPT = "outputs/best_sota_rebal.pt"   # or your chosen ckpt
BATCH = 32
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

cfg = yaml.safe_load(open(CFG))
device = "cpu"

# dataset/loader
ds = SingleObjectPointClouds(
    pc_dir=os.path.join(cfg["data"]["root"], "pointclouds"),
    list_file=cfg["data"]["val_list"],
    label_map_file=cfg["data"]["labels_file"],
    box_json_file=cfg["data"]["box_json"],
    npoints=cfg["data"]["num_points"],
    augment=False, seed=42
)
dl = DataLoader(ds, batch_size=BATCH, shuffle=False)

# model
ck = torch.load(CKPT, map_location=device)
num_classes = ck.get("num_classes", cfg["model"]["num_classes"])
m = TwoHeadSOTA(num_classes=num_classes,
                backbone=cfg["model"]["backbone"],
                box_head=cfg["model"]["box_head"],
                out_dim=cfg["model"]["width"])
m.load_state_dict(ck["model"], strict=False); m.eval()

# eval
tot = np.zeros(num_classes, int)
cor = np.zeros(num_classes, int)
cm  = np.zeros((num_classes, num_classes), int)

with torch.no_grad():
    for b in dl:
        x = b["points"]
        y = b["class_id"].numpy()
        logits = m(x)[0]
        p = logits.argmax(-1).numpy()
        for yi, pi in zip(y, p):
            tot[yi] += 1
            cor[yi] += int(yi==pi)
            cm[yi, pi] += 1

acc = np.divide(cor, np.maximum(1, tot))

# Per-class bar chart
plt.figure(figsize=(10,4), dpi=200)
plt.bar(np.arange(num_classes), acc*100)
plt.xlabel("Class ID"); plt.ylabel("Accuracy (%)"); plt.title("Per-class Accuracy (Val)")
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "per_class_accuracy.png")
plt.savefig(bar_path)

# Confusion matrix (optional, may be dense)
plt.figure(figsize=(6,5), dpi=200)
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix (Val)")
plt.xlabel("Predicted"); plt.ylabel("Ground Truth")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)

print("Saved:", bar_path, "and", cm_path)
