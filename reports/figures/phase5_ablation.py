#!/usr/bin/env python3
"""Phase 5 ablation figure (650M, held-out test set, 3 seeds): sequence -> +structure -> +evolution.
Honest presentation: bars show the mean, error bars show +/- s.d. (the real values from
phase5/agg_ablation.py), so the reader sees the conditions overlap within noise.
Serif type; no names embedded. No fabricated per-seed points are drawn."""
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

conds  = ["sequence", "+ structure", "+ evolution"]
colors = ["#9aa5b1", "#6b8cae", "#2b6cb0"]
# real values from agg_ablation.py (mean, s.d. over 3 seeds)
P = {
 "Affinity Pearson $r$": dict(hi=True,  mean=[0.429, 0.441, 0.439], sd=[0.026, 0.014, 0.018]),
 "Affinity RMSE":        dict(hi=False, mean=[1.593, 1.592, 1.575], sd=[0.123, 0.008, 0.044]),
 "Interface AUPR":       dict(hi=True,  mean=[0.286, 0.289, 0.281], sd=[0.007, 0.010, 0.006]),
}

fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3))
x = np.arange(3)
for ax, (title, d), lab in zip(axes, P.items(), ["(a)", "(b)", "(c)"]):
    ax.bar(x, d["mean"], width=0.62, color=colors, edgecolor="#333333", linewidth=0.9, zorder=2)
    ax.errorbar(x, d["mean"], yerr=d["sd"], fmt="none", ecolor="#222222",
                elinewidth=1.3, capsize=6, capthick=1.3, zorder=4)
    for xi, m, s in zip(x, d["mean"], d["sd"]):
        ax.text(xi, m + s + 0.03 * max(d["mean"]), f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    top = max(np.array(d["mean"]) + np.array(d["sd"]))
    ax.set_ylim(0, top * 1.25)
    ax.set_title(title, fontsize=12.5, pad=9)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("higher is better" if d["hi"] else "lower is better",
                  fontsize=9, color="#555555", style="italic")
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")

fig.suptitle("Module ablation — 650M model, held-out test set (mean $\\pm$ s.d., 3 seeds)",
             fontsize=13, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/mnt/user-data/outputs/reports/figures/phase5_ablation.png", dpi=300, bbox_inches="tight")
fig.savefig("/mnt/user-data/outputs/reports/figures/phase5_ablation.pdf", bbox_inches="tight")
print("wrote phase5_ablation.png / .pdf")
