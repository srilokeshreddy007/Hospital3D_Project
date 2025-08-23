# src/plots/plot_training_curves.py
import os, csv
import matplotlib.pyplot as plt

CSV = os.path.join("runs", "metrics_sota.csv")  # epoch,train_loss,train_acc,val_loss,val_acc,lr
OUT = os.path.join("figs", "training_curves.png")
os.makedirs("figs", exist_ok=True)

ep, tr_loss, va_loss, tr_acc, va_acc = [], [], [], [], []
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        ep.append(int(row["epoch"]))
        tr_loss.append(float(row["train_loss"]))
        va_loss.append(float(row["val_loss"]))
        tr_acc.append(float(row["train_acc"]))
        va_acc.append(float(row["val_acc"]))

plt.figure(figsize=(8,4), dpi=200)
plt.plot(ep, tr_loss, label="Train Loss")
plt.plot(ep, va_loss, label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epochs")
plt.tight_layout(); plt.savefig(os.path.join("figs", "loss_curve.png"))

plt.figure(figsize=(8,4), dpi=200)
plt.plot(ep, tr_acc, label="Train Acc")
plt.plot(ep, va_acc, label="Val Acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epochs")
plt.tight_layout(); plt.savefig(os.path.join("figs", "acc_curve.png"))

# optional combined mosaic
import numpy as np
from PIL import Image
img1, img2 = Image.open("figs/loss_curve.png"), Image.open("figs/acc_curve.png")
w = max(img1.width, img2.width); h = img1.height + img2.height
combo = Image.new("RGB", (w, h), "white")
combo.paste(img1, (0,0)); combo.paste(img2, (0,img1.height))
combo.save(OUT)
print("Saved:", OUT)
