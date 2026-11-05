#!/usr/bin/env python
"""
Phase C: sequence-coverage analysis and seed set for synthetic data generation.

The supervisor asked, before generating synthetic data, to analyse the sequence coverage of the
current experimental data and to prepare a sequence dataset that the synthetic generation will draw
on. He set two targets: the synthetic data should be at least an order of magnitude (ten times)
larger than the experimental data, and it should cover more sequence space than the experimental
data covers. This script measures the current coverage and writes the seed sequence set, so the
targets are concrete and the later generation can be checked against them.

It reads the block-one outputs under <data>/phase_c/ (experimental_manifest.csv, proteins.csv,
proteins.fasta) and the mmseqs2 clustering (clust_id0.3_cluster.tsv). It only reads and summarises,
plus it writes one FASTA of the experimental affinity proteins as the seed set. No GPU.

Run (from ~/Projects/multimodal-ppi, pandas/numpy env active, after block one and the audit):
    python phase_c/sequence_coverage.py
"""
import os
import argparse
import numpy as np
import pandas as pd


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--clusters", default=None,
                    help="mmseqs _cluster.tsv (default <data>/phase_c/clust_id0.3_cluster.tsv)")
    ap.add_argument("--factor", type=int, default=10,
                    help="target multiple of the experimental size for the synthetic data")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    pc = os.path.join(data, "phase_c")
    clusters = args.clusters or os.path.join(pc, "clust_id0.3_cluster.tsv")

    man = pd.read_csv(os.path.join(pc, "experimental_manifest.csv"), low_memory=False,
                      dtype={"protein_a_id": str, "protein_b_id": str})
    cl = pd.read_csv(clusters, sep="\t", header=None, names=["rep", "member"], dtype=str)
    member2cluster = dict(zip(cl["member"], cl["rep"]))

    # sequences, from proteins.fasta
    seq_of = {}
    pid = None
    with open(os.path.join(pc, "proteins.fasta")) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                pid = line[1:]
            elif pid is not None:
                seq_of[pid] = line
                pid = None

    def cluster_of(p):
        return member2cluster.get(p, p)

    print("=" * 70)
    print("Phase C: sequence-coverage analysis")
    print("=" * 70)

    # ---- overall and per-source protein / cluster counts ----
    def prots(df):
        return set(df.protein_a_id) | set(df.protein_b_id)

    all_p = prots(man)
    all_c = set(cluster_of(p) for p in all_p)
    print(f"all experimental data: {len(all_p):,} unique proteins in {len(all_c):,} clusters")
    for source in sorted(man.source.unique()):
        sub = man[man.source == source]
        p = prots(sub)
        c = set(cluster_of(x) for x in p)
        print(f"  {source:20s}: {len(p):,} proteins, {len(c):,} clusters, {len(sub):,} pairs")
    print()

    # ---- cluster-size distribution (redundancy) ----
    sizes = cl.groupby("rep").size()
    sing016 = int((sizes == 1).sum())
    print("Cluster-size distribution (all proteins):")
    print(f"  clusters {len(sizes):,}; singletons {sing016:,} ({100.0*sing016/len(sizes):.1f}%); "
          f"largest {int(sizes.max()):,}; median size {int(sizes.median())}")
    print(f"  {100.0*(1-len(sizes)/max(len(all_p),1)):.1f}% of proteins sit in a non-singleton cluster")
    print()

    # ---- protein length distribution ----
    lens = np.array([len(seq_of[p]) for p in all_p if p in seq_of])
    print("Protein length (residues):")
    print(f"  min {int(lens.min())}, 25% {pct(lens,25):.0f}, median {pct(lens,50):.0f}, "
          f"75% {pct(lens,75):.0f}, max {int(lens.max())}")
    print()

    # ---- affinity-set footprint (the experimental affinity sequence space) ----
    aff = man[man.source == "ppb_affinity"]
    aff_p = prots(aff)
    aff_c = set(cluster_of(p) for p in aff_p)
    aff_pairkeys = set()
    for a, b in zip(aff.protein_a_id, aff.protein_b_id):
        ca, cb = cluster_of(a), cluster_of(b)
        aff_pairkeys.add((ca, cb) if ca <= cb else (cb, ca))
    print("Experimental affinity footprint (what synthetic affinity must exceed):")
    print(f"  complexes: {len(aff):,}")
    print(f"  distinct proteins: {len(aff_p):,}  in {len(aff_c):,} clusters")
    print(f"  distinct cluster-pairs (interaction sequence space covered): {len(aff_pairkeys):,}")
    print()

    # ---- targets from the supervisor's instruction ----
    target_pairs = args.factor * len(aff)
    print(f"Supervisor targets:")
    print(f"  synthetic size >= {args.factor} x experimental = at least {target_pairs:,} pairs")
    print(f"  synthetic sequence coverage > experimental: aim for more than {len(aff_c):,} protein "
          f"clusters and more than {len(aff_pairkeys):,} distinct cluster-pairs")
    print()

    # ---- write the seed set: experimental affinity proteins ----
    out_fa = os.path.join(pc, "affinity_proteins.fasta")
    n = 0
    with open(out_fa, "w") as f:
        for p in sorted(aff_p):
            if p in seq_of:
                f.write(f">{p}\n{seq_of[p]}\n")
                n += 1
    print(f"Wrote seed sequence set: {out_fa} ({n:,} affinity proteins)")
    print("This is the starting pool of real proteins. Block two expands pairing across organisms")
    print("(guided by an interaction database) to reach the size and coverage targets above.")


if __name__ == "__main__":
    main()
