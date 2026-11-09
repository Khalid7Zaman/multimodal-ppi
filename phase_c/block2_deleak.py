#!/usr/bin/env python
"""
Phase C block two, step 3 (apply): remove candidate pairs that leak into the test set, report coverage.

Using the clustering of candidate proteins together with the evaluation test proteins, a candidate
protein "leaks" if it shares a cluster with any test protein. Every candidate pair with a leaking
protein on either side is dropped, so the synthetic data cannot resemble anything in the evaluation.
The de-leaked pairs are written and the coverage is reported against the supervisor's targets. No GPU.
"""
import os
import argparse
import pandas as pd

EXP_CLUSTERS = 1814       # experimental affinity clusters (block one coverage)
EXP_CLUSTERPAIRS = 1428   # experimental affinity distinct cluster-pairs
TARGET_PAIRS = 80450      # 10x the experimental affinity complexes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--clusters", required=True, help="mmseqs _cluster.tsv over candidates + test")
    ap.add_argument("--test_ids", required=True, help="eval_test_ids.txt from the prep step")
    args = ap.parse_args()

    pc = os.path.join(os.path.expanduser(args.data), "phase_c")
    df = pd.read_csv(os.path.join(pc, "candidate_pairs.csv"),
                     dtype={"protein1_id": str, "protein2_id": str})
    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"], dtype=str)
    m2c = dict(zip(cl["member"], cl["rep"]))
    test_ids = set(x.strip() for x in open(args.test_ids) if x.strip())

    def cof(p):
        return m2c.get(p, p)

    test_clusters = set(cof(t) for t in test_ids)
    ca = df["protein1_id"].map(cof)
    cb = df["protein2_id"].map(cof)
    leak = ca.isin(test_clusters) | cb.isin(test_clusters)
    clean = df[~leak].reset_index(drop=True)

    prots = set(clean["protein1_id"]) | set(clean["protein2_id"])
    clusters = set(cof(p) for p in prots)
    cpairs = set()
    for x, y in zip(ca[~leak], cb[~leak]):
        cpairs.add((x, y) if x <= y else (y, x))

    out = os.path.join(pc, "candidate_pairs_clean.csv")
    clean.to_csv(out, index=False)

    print("=" * 70)
    print("Phase C block two step 3: de-leaked candidate set")
    print("=" * 70)
    print(f"candidate pairs: {len(df):,} -> {len(clean):,} "
          f"(removed {int(leak.sum()):,} that share a cluster with a test protein)")
    print(f"unique candidate proteins: {len(prots):,} in {len(clusters):,} clusters")
    print(f"distinct cluster-pairs: {len(cpairs):,}")
    print(f"organisms represented: {clean['taxid'].nunique()}")
    print()
    print("Targets (from the supervisor):")
    print(f"  size >= {TARGET_PAIRS:,}        : {'MET' if len(clean) >= TARGET_PAIRS else 'NOT met'} "
          f"({len(clean):,})")
    print(f"  clusters > {EXP_CLUSTERS:,} (experimental) : "
          f"{'MET' if len(clusters) > EXP_CLUSTERS else 'NOT met'} ({len(clusters):,})")
    print(f"  cluster-pairs > {EXP_CLUSTERPAIRS:,} (experimental): "
          f"{'MET' if len(cpairs) > EXP_CLUSTERPAIRS else 'NOT met'} ({len(cpairs):,})")
    print()
    print("Wrote:", out)
    print("This is the de-leaked candidate interacting-pair set for synthetic generation.")


if __name__ == "__main__":
    main()
