#!/usr/bin/env python3
"""Phase 5 B2 figure: evolution helps where the MSA is deep (affinity, 650M, held-out test).
Grouped bars: deep vs shallow MSA, + structure vs + evolution, with real error bars.
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

groups = ["Deep MSA\n(n=379)", "Shallow MSA\n(n=378)"]
struct_c, evo_c = "#9aa5b1", "#2b6cb0"
# real values from agg_stratified.py
pear = dict(struct_m=[0.512, 0.300], struct_s=[0.041, 0.014],
            evo_m=[0.536, 0.279],   evo_s=[0.008, 0.037])
rmse = dict(struct_m=[1.451, 1.721], struct_s=[0.031, 0.042],
            evo_m=[1.415, 1.720],   evo_s=[0.019, 0.063])

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
x = np.arange(2); w = 0.36
for ax, (title, D, hi) in zip(axes,
        [("Affinity Pearson $r$", pear, True), ("Affinity RMSE", rmse, False)]):
    b1 = ax.bar(x - w/2, D["struct_m"], w, yerr=D["struct_s"], capsize=5,
                color=struct_c, edgecolor="#333", linewidth=0.8, label="+ structure",
                error_kw=dict(elinewidth=1.2, capthick=1.2))
    b2 = ax.bar(x + w/2, D["evo_m"], w, yerr=D["evo_s"], capsize=5,
                color=evo_c, edgecolor="#333", linewidth=0.8, label="+ evolution",
                error_kw=dict(elinewidth=1.2, capthick=1.2))
    for bars, m, s in ((b1, D["struct_m"], D["struct_s"]), (b2, D["evo_m"], D["evo_s"])):
        for rect, mv, sv in zip(bars, m, s):
            ax.text(rect.get_x() + rect.get_width()/2, mv + sv + 0.02 * max(D["evo_m"]),
                    f"{mv:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=12.5, pad=9)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylim(0, max(max(D["struct_m"]), max(D["evo_m"])) * 1.25)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("higher is better" if hi else "lower is better",
                  fontsize=9, color="#555555", style="italic")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=10.5, ncol=2, loc="upper center",
           bbox_to_anchor=(0.5, 0.94))
fig.suptitle("Evolution improves affinity where the alignment is deep - 650M, held-out test (mean $\\pm$ s.d., 3 seeds)",
             fontsize=12.5, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("phase5_stratified.png", dpi=300, bbox_inches="tight")
fig.savefig("phase5_stratified.pdf", bbox_inches="tight")
print("wrote phase5_stratified.png / .pdf")
