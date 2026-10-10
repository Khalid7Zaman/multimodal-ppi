#!/usr/bin/env python
"""Phase 5 Part C - interpretability figures from the .npz files written by interp_extract.py.
For each example complex, one figure with two panels:
  (a) cross-attention map (receptor -> ligand); red ticks mark the TRUE interface residues,
  (b) the complex's C-alpha atoms in 3D, coloured by PREDICTED interface probability, with the
      TRUE interface residues circled in red.
Fast, CPU-only, re-runnable:  python phase5/interp_plot.py
Writes reports/figures/phase5_interp_<PDB>.png/.pdf. Serif type; no names embedded."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import font_manager

for p in ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11})

RED = "#d1495b"
INTERP = "phase5/interp"
OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)


def set_equal_3d(ax, coords):
    c = coords
    ranges = c.max(0) - c.min(0)
    ax.set_box_aspect(tuple(ranges / ranges.max()))


def plot_one(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pdb = str(d["pdb"]); aupr = float(d["aupr"])
    a2b = d["a2b"]                                   # [nr, nl]
    rc, lc = d["rec_coords"], d["lig_coords"]
    rp, lp = d["rec_prob"], d["lig_prob"]
    rt, lt = d["rec_true"], d["lig_true"]
    nr, nl = a2b.shape

    fig = plt.figure(figsize=(11.0, 4.7))

    # --- panel (a): cross-attention heatmap ---
    axh = fig.add_subplot(1, 2, 1)
    im = axh.imshow(a2b, aspect="auto", cmap="viridis", origin="lower")
    axh.set_xlabel("ligand residue", fontsize=10)
    axh.set_ylabel("receptor residue", fontsize=10)
    axh.set_title(f"(a) cross-attention: receptor -> ligand", fontsize=12, pad=8)
    # red ticks at the TRUE interface residues, just outside the map
    for c in np.where(lt == 1)[0]:
        axh.plot([c, c], [nr - 0.5, nr + max(1, nr * 0.02)], color=RED, lw=0.7, clip_on=False)
    for r in np.where(rt == 1)[0]:
        axh.plot([nl - 0.5, nl + max(1, nl * 0.02)], [r, r], color=RED, lw=0.7, clip_on=False)
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    cb.set_label("attention weight", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    axh.tick_params(labelsize=8)

    # --- panel (b): predicted interface on the 3D structure ---
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax3.scatter(rc[:, 0], rc[:, 1], rc[:, 2], c=rp, cmap="viridis", vmin=0, vmax=1,
                     s=16, marker="o", depthshade=True, label="receptor")
    ax3.scatter(lc[:, 0], lc[:, 1], lc[:, 2], c=lp, cmap="viridis", vmin=0, vmax=1,
                s=16, marker="^", depthshade=True, label="ligand")
    ri, li = rc[rt == 1], lc[lt == 1]
    if len(ri):
        ax3.scatter(ri[:, 0], ri[:, 1], ri[:, 2], facecolors="none", edgecolors=RED, s=52, lw=1.1)
    if len(li):
        ax3.scatter(li[:, 0], li[:, 1], li[:, 2], facecolors="none", edgecolors=RED, s=52, lw=1.1)
    set_equal_3d(ax3, np.vstack([rc, lc]))
    ax3.set_title("(b) predicted interface on structure", fontsize=12, pad=8)
    ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
    ax3.tick_params(length=0)
    cb2 = fig.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02)
    cb2.set_label("predicted interface probability", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    ax3.text2D(0.0, -0.02, "red circle = true interface;   receptor = round, ligand = triangle",
               transform=ax3.transAxes, fontsize=8, color="#444")

    fig.suptitle(f"Interpretability example - PDB {pdb}  (interface AUPR {aupr:.2f})",
                 fontsize=12.5, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, f"phase5_interp_{pdb}")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  (receptor {nr}, ligand {nl}, AUPR {aupr:.2f})")


def main():
    files = sorted(glob.glob(os.path.join(INTERP, "*.npz")))
    if not files:
        print(f"no .npz files in {INTERP} - run phase5/interp_extract.sbatch first")
        return
    for f in files:
        plot_one(f)


if __name__ == "__main__":
    main()
