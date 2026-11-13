#!/usr/bin/env python
"""
Phase C: sequence-space coverage analysis, with numerical evidence and a figure.

Using the ESM-C embeddings of the experimental and the synthetic proteins, this shows, with numbers,
that the synthetic set covers the sequence space of the experimental set and extends beyond it along
each dimension. It reduces the embeddings with PCA and reports, per dimension, whether the synthetic
range covers the experimental range and by how much it extends, plus overall coverage and extension
metrics, and it writes a two-panel figure. CPU only.

Run from ~/Projects/multimodal-ppi (esmc env):
    python phase_c/coverage_analysis.py --exp <exp_prefix> --syn <syn_prefix> --out <dir>
reads <prefix>.npy (+ .ids.txt), where the prefixes come from embed_proteins.py.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="prefix of experimental embeddings (<prefix>.npy)")
    ap.add_argument("--syn", required=True, help="prefix of synthetic embeddings (<prefix>.npy)")
    ap.add_argument("--out", default=None, help="output directory (default: the --exp directory)")
    ap.add_argument("--k", type=int, default=10, help="number of PCA dimensions to analyse")
    args = ap.parse_args()

    Xe = np.load(args.exp + ".npy")
    Xs = np.load(args.syn + ".npy")
    out = args.out or (os.path.dirname(args.exp) or ".")
    k = args.k

    # standardise on the pooled set, then PCA on the pooled set, so both sets share one frame
    both = np.vstack([Xe, Xs])
    sc = StandardScaler().fit(both)
    pca = PCA(n_components=k).fit(sc.transform(both))
    Pe = pca.transform(sc.transform(Xe))
    Ps = pca.transform(sc.transform(Xs))

    print("=" * 74)
    print("Phase C: sequence-space coverage (ESM-C embedding, PCA)")
    print("=" * 74)
    print(f"experimental proteins: {len(Xe):,}   synthetic proteins: {len(Xs):,}")
    print(f"PCA dimensions: {k}   variance explained: {pca.explained_variance_ratio_[:k].sum() * 100:.1f}%")
    print()

    print("Per-dimension coverage (does the synthetic range cover and extend the experimental range):")
    print(f"  {'PC':>3} {'exp_min':>9} {'exp_max':>9} {'syn_min':>9} {'syn_max':>9} {'covers':>7} {'extend%':>8}")
    covered = 0
    for d in range(k):
        emin, emax = float(Pe[:, d].min()), float(Pe[:, d].max())
        smin, smax = float(Ps[:, d].min()), float(Ps[:, d].max())
        cov = (smin <= emin) and (smax >= emax)
        erange = (emax - emin) if emax > emin else 1.0
        ext = (max(emin - smin, 0.0) + max(smax - emax, 0.0)) / erange * 100.0
        covered += int(cov)
        print(f"  {d + 1:>3} {emin:>9.2f} {emax:>9.2f} {smin:>9.2f} {smax:>9.2f} {str(cov):>7} {ext:>8.1f}")
    print()

    # robust bounding box (1-99 percentile) volume ratio across the k dimensions
    def box(P):
        return np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)
    ratio = float(np.prod(box(Ps) / np.clip(box(Pe), 1e-9, None)))

    # coverage: fraction of experimental points that fall inside the synthetic range on every dimension
    smin_all = Ps.min(0)
    smax_all = Ps.max(0)
    inside = 100.0 * float(np.mean(np.all((Pe >= smin_all) & (Pe <= smax_all), axis=1)))

    # informative: how far an experimental protein is from its nearest synthetic protein,
    # relative to how far experimental proteins are from each other (smaller = denser coverage)
    nn = NearestNeighbors(n_neighbors=1).fit(Ps)
    d_es, _ = nn.kneighbors(Pe)
    nne = NearestNeighbors(n_neighbors=2).fit(Pe)
    d_ee, _ = nne.kneighbors(Pe)
    nn_ratio = float(np.median(d_es[:, 0]) / max(np.median(d_ee[:, 1]), 1e-9))

    # extension: synthetic points outside the experimental 1-99 percentile box
    lo = np.percentile(Pe, 1, axis=0)
    hi = np.percentile(Pe, 99, axis=0)
    outside = 100.0 * float(np.mean(np.any((Ps < lo) | (Ps > hi), axis=1)))

    print("Summary:")
    print(f"  dimensions where synthetic fully covers the experimental range: {covered}/{k}")
    print(f"  experimental proteins inside the synthetic range on every dimension (coverage): {inside:.1f}%")
    print(f"  bounding-box volume ratio synthetic/experimental (1-99 pct, {k} PCs): {ratio:.2e}")
    print(f"  synthetic proteins outside the experimental 1-99 pct box (extension): {outside:.1f}%")
    print(f"  median experimental-to-nearest-synthetic distance over typical experimental")
    print(f"    spacing: {nn_ratio:.2f} (smaller means denser synthetic coverage)")
    print()

    # figure: PC1-PC2 scatter, and per-dimension range bars
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(Ps[:, 0], Ps[:, 1], s=3, alpha=0.18, c="#4c78a8", label=f"synthetic ({len(Xs):,})")
    ax[0].scatter(Pe[:, 0], Pe[:, 1], s=5, alpha=0.5, c="#e45756", label=f"experimental ({len(Xe):,})")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("Sequence space (ESM-C, PC1 vs PC2)")
    ax[0].legend(markerscale=3, loc="best", fontsize=9)

    d = np.arange(k)
    emin = Pe[:, :k].min(0); emax = Pe[:, :k].max(0)
    smin = Ps[:, :k].min(0); smax = Ps[:, :k].max(0)
    ax[1].barh(d - 0.2, emax - emin, height=0.4, left=emin, color="#e45756", label="experimental")
    ax[1].barh(d + 0.2, smax - smin, height=0.4, left=smin, color="#4c78a8", label="synthetic")
    ax[1].set_yticks(d)
    ax[1].set_yticklabels([f"PC{i + 1}" for i in d])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("range")
    ax[1].set_title("Per-dimension range: synthetic covers and extends")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    figp = os.path.join(out, "coverage_sequence_space.png")
    fig.savefig(figp, dpi=130)
    print("wrote figure:", figp)


if __name__ == "__main__":
    main()
