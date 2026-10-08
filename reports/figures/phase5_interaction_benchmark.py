#!/usr/bin/env python3
"""Phase 5 - interaction benchmark figure: this work vs PLM-interact on the
leakage-free Bernett test set (52,048 pairs). Serif type; no names embedded."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for p in ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 12, "axes.linewidth": 0.9, "axes.edgecolor": "#444444"})

models = ["This work\n(650M)", "PLM-interact\n(650M)"]
colors = ["#2b6cb0", "#9aa5b1"]
panels = [("Interaction AUROC", [0.7174, 0.6224], None),
          ("Interaction AUPR",  [0.7093, 0.6359], 0.69)]  # 0.69 = published baseline (AUPR)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
for ax, (title, vals, baseline), lab in zip(axes, panels, ["(a)", "(b)"]):
    x = np.arange(len(models))
    bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="#333333", linewidth=0.9, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=11.5)
    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=1.2, color="#b7791f", zorder=1)
        ax.text(len(models) - 0.5, baseline + 0.006, f"published baseline ~{baseline:.2f}",
                ha="right", va="bottom", fontsize=9, color="#8a5a12", style="italic")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0, 0.82)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("higher is better", fontsize=9.5, color="#555555", style="italic")
    ax.text(-0.14, 1.05, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

fig.suptitle("Interaction prediction — leakage-free Bernett test set (52,048 pairs)",
             fontsize=13.5, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/mnt/user-data/outputs/reports/figures/phase5_interaction_benchmark.png", dpi=300, bbox_inches="tight")
fig.savefig("/mnt/user-data/outputs/reports/figures/phase5_interaction_benchmark.pdf", bbox_inches="tight")
print("wrote phase5_interaction_benchmark.png / .pdf")
