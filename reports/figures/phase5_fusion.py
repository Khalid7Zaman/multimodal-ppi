#!/usr/bin/env python3
"""Phase 5 B3 figure: gated vs additive fusion (650M, all views, held-out test, 3 seeds).
Honest presentation: bars = mean, error bars = +/- s.d. (real values from agg_gated.py).
Serif type; no names embedded."""
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

conds  = ["additive", "gated"]
colors = ["#2b6cb0", "#9c6b2e"]
P = {
 "Affinity Pearson $r$": dict(hi=True,  mean=[0.439, 0.422], sd=[0.018, 0.009]),
 "Affinity RMSE":        dict(hi=False, mean=[1.575, 1.587], sd=[0.044, 0.092]),
 "Interface AUPR":       dict(hi=True,  mean=[0.281, 0.288], sd=[0.006, 0.016]),
}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.3))
x = np.arange(2)
for ax, (title, d), lab in zip(axes, P.items(), ["(a)", "(b)", "(c)"]):
    ax.bar(x, d["mean"], width=0.58, color=colors, edgecolor="#333333", linewidth=0.9, zorder=2)
    ax.errorbar(x, d["mean"], yerr=d["sd"], fmt="none", ecolor="#222222",
                elinewidth=1.3, capsize=6, capthick=1.3, zorder=4)
    for xi, m, s in zip(x, d["mean"], d["sd"]):
        ax.text(xi, m + s + 0.03 * max(d["mean"]), f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(np.array(d["mean"]) + np.array(d["sd"])) * 1.25)
    ax.set_title(title, fontsize=12.5, pad=9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("higher is better" if d["hi"] else "lower is better",
                  fontsize=9, color="#555555", style="italic")
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")

fig.suptitle("Fusion study - gated vs additive (650M, all views, held-out test, mean $\\pm$ s.d., 3 seeds)",
             fontsize=12.5, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("phase5_fusion.png", dpi=300, bbox_inches="tight")
fig.savefig("phase5_fusion.pdf", bbox_inches="tight")
print("wrote phase5_fusion.png / .pdf")
