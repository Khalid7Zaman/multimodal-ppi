#!/usr/bin/env python
"""
Phase C block one follow-up: analyse an identity-threshold leakage audit.

The manifest builder already checked exact-match leakage. This goes further: it reads
an mmseqs2 clustering of the unique proteins (where near-duplicate sequences land in
the same cluster) and reports, within each source, how many test proteins and test
pairs share a cluster with a train or validation protein or pair. That is the honest
near-duplicate check, not just exact matches.

Inputs:
  --data      base directory (default ~/Projects/ppi-data); reads phase_c/experimental_manifest.csv
  --clusters  the mmseqs easy-cluster _cluster.tsv (two columns: representative, member)
  --minid     the identity threshold used, for the printout only

It only reads files and prints a summary. Run through phase_c/leakage_audit.sh, which
does the clustering first.
"""
import os
import argparse
import pandas as pd

SPLITS_TRAINVAL = {"train", "val"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--minid", default="?")
    args = ap.parse_args()

    man_path = os.path.join(os.path.expanduser(args.data), "phase_c", "experimental_manifest.csv")
    man = pd.read_csv(man_path, dtype={"protein_a_id": str, "protein_b_id": str})

    # mmseqs easy-cluster writes: representative <tab> member, no header.
    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"],
                     dtype=str)
    member2cluster = dict(zip(cl["member"], cl["rep"]))

    n_proteins = len(set(cl["member"]))
    n_clusters = len(set(cl["rep"]))
    sizes = cl.groupby("rep").size()
    singletons = int((sizes == 1).sum())

    print("=" * 70)
    print(f"Phase C block one: identity-threshold leakage audit (min-seq-id {args.minid})")
    print("=" * 70)
    print(f"proteins clustered : {n_proteins}")
    print(f"clusters           : {n_clusters} "
          f"({singletons} singletons, largest {int(sizes.max())} members)")
    print(f"redundancy         : {100.0 * (1 - n_clusters / max(n_proteins, 1)):.1f}% "
          f"of proteins fall into a cluster with at least one other protein")
    print()

    def cluster_of(pid):
        return member2cluster.get(pid, pid)

    print("Cluster-level leakage (test shares a cluster with train or validation):")
    for source in sorted(man.source.unique()):
        sub = man[man.source == source]
        tv = sub[sub.split.isin(SPLITS_TRAINVAL)]
        te = sub[sub.split == "test"]

        # protein-level
        tv_prot_clusters = set(cluster_of(p) for p in
                               set(tv.protein_a_id) | set(tv.protein_b_id))
        te_prots = set(te.protein_a_id) | set(te.protein_b_id)
        te_prot_leak = sum(1 for p in te_prots if cluster_of(p) in tv_prot_clusters)

        # pair-level: an unordered pair of clusters
        def pair_cluster_keys(df):
            keys = set()
            for a, b in zip(df.protein_a_id, df.protein_b_id):
                ca, cb = cluster_of(a), cluster_of(b)
                keys.add((ca, cb) if ca <= cb else (cb, ca))
            return keys

        tv_keys = pair_cluster_keys(tv)
        te_keys_list = []
        leak_pairs = 0
        seen = set()
        for a, b in zip(te.protein_a_id, te.protein_b_id):
            ca, cb = cluster_of(a), cluster_of(b)
            key = (ca, cb) if ca <= cb else (cb, ca)
            if key in tv_keys:
                leak_pairs += 1
            seen.add(key)
        n_te_pairs = len(te)

        print(f"  {source}:")
        print(f"    test proteins {len(te_prots)}, of which {te_prot_leak} "
              f"({100.0 * te_prot_leak / max(len(te_prots), 1):.1f}%) share a cluster with train/val")
        print(f"    test pairs {n_te_pairs}, of which {leak_pairs} "
              f"({100.0 * leak_pairs / max(n_te_pairs, 1):.1f}%) match a train/val pair at the cluster level")
    print()
    print("Reading: a high cluster-level test/train overlap means the split lets the model")
    print("see near-duplicates of its test proteins during training, which inflates the score.")
    print("A clean fix is a cluster-based re-split that keeps whole clusters within one split.")


if __name__ == "__main__":
    main()
