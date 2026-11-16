#!/usr/bin/env python
"""
Phase C: fast sequence-space coverage analysis (no GPU).

This gives the same kind of numerical evidence as the ESM-C version, but using simple, interpretable
sequence features (amino-acid composition and length) instead of a large model, so it runs in seconds
on a login node with no GPU. It shows, with numbers and a figure, that the synthetic proteins cover the
sequence space of the experimental proteins and extend beyond it along each dimension. The ESM-C version
(embed_coverage.sbatch) adds a richer representation when a GPU is free, as confirmation.

Run from ~/Projects/multimodal-ppi:
    python phase_c/coverage_fast.py
reads the experimental and synthetic protein FASTAs in ~/Projects/ppi-data/phase_c/ and writes
coverage_fast.png there.
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

AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    ids, seqs, name, buf = [], [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    ids.append(name); seqs.append("".join(buf))
                name = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            ids.append(name); seqs.append("".join(buf))
    return ids, seqs


def featurize(seqs):
    """Per-protein features: 20 amino-acid fractions + log10(length). Fast, interpretable, no GPU."""
    F = np.zeros((len(seqs), 21), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = "".join(ch for ch in s.upper() if ch.isalpha())
        n = len(s)
        if n == 0:
            continue
        for j, a in enumerate(AA):
            F[i, j] = s.count(a) / n
        F[i, 20] = np.log10(n)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--k", type=int, default=10, help="number of PCA dimensions to analyse")
    ap.add_argument("--max_syn", type=int, default=None,
                    help="optional cap on synthetic proteins (random sample) to keep it fast")
    args = ap.parse_args()

    pc = os.path.join(os.path.expanduser(args.data), "phase_c")
    _, exp_seqs = read_fasta(os.path.join(pc, "proteins.fasta"))
    syn_ids, syn_seqs = read_fasta(os.path.join(pc, "candidate_proteins.fasta"))
    if args.max_syn and len(syn_seqs) > args.max_syn:
        import random
        idx = list(range(len(syn_seqs))); random.Random(0).shuffle(idx); idx = sorted(idx[:args.max_syn])
        syn_seqs = [syn_seqs[i] for i in idx]

    Xe = featurize(exp_seqs)
    Xs = featurize(syn_seqs)
    k = args.k

    both = np.vstack([Xe, Xs])
    sc = StandardScaler().fit(both)
    pca = PCA(n_components=k).fit(sc.transform(both))
    Pe = pca.transform(sc.transform(Xe))
    Ps = pca.transform(sc.transform(Xs))

    print("=" * 74)
    print("Phase C: sequence-space coverage (amino-acid composition + length, no GPU)")
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

    def box(P):
        return np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)
    ratio = float(np.prod(box(Ps) / np.clip(box(Pe), 1e-9, None)))
    smin_all = Ps.min(0); smax_all = Ps.max(0)
    inside = 100.0 * float(np.mean(np.all((Pe >= smin_all) & (Pe <= smax_all), axis=1)))
    nn = NearestNeighbors(n_neighbors=1).fit(Ps)
    d_es, _ = nn.kneighbors(Pe)
    nne = NearestNeighbors(n_neighbors=2).fit(Pe)
    d_ee, _ = nne.kneighbors(Pe)
    nn_ratio = float(np.median(d_es[:, 0]) / max(np.median(d_ee[:, 1]), 1e-9))
    lo = np.percentile(Pe, 1, axis=0); hi = np.percentile(Pe, 99, axis=0)
    outside = 100.0 * float(np.mean(np.any((Ps < lo) | (Ps > hi), axis=1)))

    print("Summary:")
    print(f"  dimensions where synthetic fully covers the experimental range: {covered}/{k}")
    print(f"  experimental proteins inside the synthetic range on every dimension (coverage): {inside:.1f}%")
    print(f"  bounding-box volume ratio synthetic/experimental (1-99 pct, {k} PCs): {ratio:.2e}")
    print(f"  synthetic proteins outside the experimental 1-99 pct box (extension): {outside:.1f}%")
    print(f"  median experimental-to-nearest-synthetic distance over typical experimental")
    print(f"    spacing: {nn_ratio:.2f} (smaller means denser synthetic coverage)")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(Ps[:, 0], Ps[:, 1], s=3, alpha=0.15, c="#4c78a8", label=f"synthetic ({len(Xs):,})")
    ax[0].scatter(Pe[:, 0], Pe[:, 1], s=5, alpha=0.5, c="#e45756", label=f"experimental ({len(Xe):,})")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].set_title("Sequence space (composition, PC1 vs PC2)")
    ax[0].legend(markerscale=3, loc="best", fontsize=9)
    d = np.arange(k)
    emin = Pe[:, :k].min(0); emax = Pe[:, :k].max(0)
    smin = Ps[:, :k].min(0); smax = Ps[:, :k].max(0)
    ax[1].barh(d - 0.2, emax - emin, height=0.4, left=emin, color="#e45756", label="experimental")
    ax[1].barh(d + 0.2, smax - smin, height=0.4, left=smin, color="#4c78a8", label="synthetic")
    ax[1].set_yticks(d); ax[1].set_yticklabels([f"PC{i + 1}" for i in d]); ax[1].invert_yaxis()
    ax[1].set_xlabel("range"); ax[1].set_title("Per-dimension range: synthetic covers and extends")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    figp = os.path.join(pc, "coverage_fast.png")
    fig.savefig(figp, dpi=130)
    print("wrote figure:", figp)


if __name__ == "__main__":
    main()
