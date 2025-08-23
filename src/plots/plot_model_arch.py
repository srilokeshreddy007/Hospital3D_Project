# src/plots/plot_model_arch.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Layers (simplified PointNeXt-tiny flow)
layers = [
    "Input\n1024 points",
    "Set Abstraction\n(local grouping)",
    "Feature Extraction\n(MLPs + attention)",
    "Global Pooling",
    "Fully Connected\nlayers",
    "Output\nClass + 3D Box"
]

fig, ax = plt.subplots(figsize=(8,4), dpi=150)

# Draw boxes
for i, text in enumerate(layers):
    ax.add_patch(plt.Rectangle((i*1.8, 0), 1.6, 1, fill=True,
                               edgecolor="black", facecolor="#87CEEB"))
    ax.text(i*1.8+0.8, 0.5, text, ha="center", va="center", fontsize=8, wrap=True)

# Arrows between boxes
for i in range(len(layers)-1):
    ax.arrow(i*1.8+1.6, 0.5, 0.2, 0, head_width=0.1, head_length=0.2,
             fc="k", ec="k", length_includes_head=True)

ax.set_xlim(-0.5, len(layers)*1.8)
ax.set_ylim(-0.5, 1.5)
ax.axis("off")
plt.tight_layout()
plt.savefig("figs/model_arch.png")
plt.close(fig)

print("Saved: figs/model_arch.png")
