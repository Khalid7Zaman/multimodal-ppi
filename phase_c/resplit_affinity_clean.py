#!/usr/bin/env python
"""
Phase C: leakage-free re-split of the affinity (PPB) data.

The identity-threshold audit showed that the original PPB train/validation/test split lets a
large share of test proteins share a sequence cluster with training, which biases the affinity
numbers upward. This builds a clean split instead: it groups complexes so that every homolog
cluster (from the mmseqs2 clustering) stays wholly inside one split, so no test protein has a
near-duplicate in training. It is non-destructive: the original CSVs are left untouched and the
clean split is written to a new directory.

Method:
  1. Read the three original ppb_<split>_iface.csv files and the mmseqs2 cluster table.
  2. For each complex, look up the cluster of each of its two proteins.
  3. Union clusters that are linked through a shared complex (union-find), so a connected group
     of clusters is never separated. Each complex belongs to one group.
  4. Assign whole groups to train / validation / test, largest group first, each group going to
     the split that is furthest below its target share. Targets copy the original proportions.
  5. Write the clean split to <out> with the same columns as the originals, and verify that no
     cluster spans two splits in the result.

Run (from ~/Projects/multimodal-ppi, pandas/numpy env active, after the audit has produced the
cluster table):
    python phase_c/resplit_affinity_clean.py
"""
import os
import argparse
import hashlib
import pandas as pd

SPLITS = ["train", "val", "test"]


def clean_seq(x):
    if not isinstance(x, str):
        return ""
    return "".join(x.split()).upper()


def sid(seq, n=12):
    return hashlib.sha1(seq.encode()).hexdigest()[:n]


def read_iface(path):
    head = pd.read_csv(path, nrows=0).columns
    str_cols = {c: str for c in ("iface_query", "iface_text", "pdb",
                                 "receptor_chains", "ligand_chains") if c in head}
    return pd.read_csv(path, dtype=str_cols), list(head)


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--clusters", default=None,
                    help="mmseqs _cluster.tsv (default <data>/phase_c/clust_id0.3_cluster.tsv)")
    ap.add_argument("--out", default=None,
                    help="output directory for the clean split (default <data>/affinity/ppb_clean)")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    ppb = os.path.join(data, "affinity", "ppb")
    clusters = args.clusters or os.path.join(data, "phase_c", "clust_id0.3_cluster.tsv")
    out = args.out or os.path.join(data, "affinity", "ppb_clean")
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("Phase C: leakage-free re-split of the affinity (PPB) data")
    print("=" * 70)
    print("source split :", ppb)
    print("clusters     :", clusters)
    print("clean split  :", out)
    print()

    # ---- load originals ----
    frames = []
    columns = None
    orig_sizes = {}
    for split in SPLITS:
        path = os.path.join(ppb, f"ppb_{split}_iface.csv")
        df, cols = read_iface(path)
        columns = columns or cols
        df = df.copy()
        df["_orig_split"] = split
        frames.append(df)
        orig_sizes[split] = len(df)
    pool = pd.concat(frames, ignore_index=True)

    # drop exact duplicate complexes (same sequences + pdb), keeping the first
    before = len(pool)
    pool = pool.drop_duplicates(subset=["query", "text", "pdb"]).reset_index(drop=True)
    dropped = before - len(pool)
    total = len(pool)

    # ---- cluster lookup ----
    cl = pd.read_csv(clusters, sep="\t", header=None, names=["rep", "member"], dtype=str)
    member2cluster = dict(zip(cl["member"], cl["rep"]))

    def cluster_of(seq):
        return member2cluster.get(sid(clean_seq(seq)), sid(clean_seq(seq)))

    pool["clu_a"] = pool["query"].map(cluster_of)
    pool["clu_b"] = pool["text"].map(cluster_of)

    # ---- union-find over clusters, group complexes ----
    uf = UF()
    for ca, cb in zip(pool["clu_a"], pool["clu_b"]):
        uf.union(ca, cb)
    pool["group"] = pool["clu_a"].map(uf.find)

    groups = pool.groupby("group").size().sort_values(ascending=False)
    print(f"complexes: {total} (dropped {dropped} exact duplicates); "
          f"groups (connected cluster sets): {len(groups)}; largest group {int(groups.max())} complexes")
    print()

    # ---- assign whole groups to splits, largest first, toward the original proportions ----
    frac = {s: orig_sizes[s] / sum(orig_sizes.values()) for s in SPLITS}
    tgt = {s: frac[s] * total for s in SPLITS}
    cnt = {s: 0 for s in SPLITS}
    group_split = {}
    # deterministic order: size desc, then group id
    order = sorted(groups.index, key=lambda g: (-int(groups[g]), g))
    for g in order:
        size = int(groups[g])
        # choose the split that stays most below its target after taking this group
        pick = min(SPLITS, key=lambda s: (cnt[s] + size) / max(tgt[s], 1e-9))
        group_split[g] = pick
        cnt[pick] += size

    pool["split"] = pool["group"].map(group_split)

    # ---- verify no cluster spans two splits ----
    span = 0
    cl2splits = {}
    for _, r in pool.iterrows():
        for c in (r["clu_a"], r["clu_b"]):
            cl2splits.setdefault(c, set()).add(r["split"])
    span = sum(1 for s in cl2splits.values() if len(s) > 1)

    # ---- cluster-level test/train leakage on the NEW split (should be 0) ----
    def prot_clusters(df):
        return set(df["clu_a"]) | set(df["clu_b"])
    tv = pool[pool.split.isin(["train", "val"])]
    te = pool[pool.split == "test"]
    tv_clusters = prot_clusters(tv)
    te_prots = set(te["query"].map(cluster_of)) | set(te["text"].map(cluster_of))
    leak = sum(1 for c in te_prots if c in tv_clusters)

    # ---- write clean split ----
    moved = 0
    for split in SPLITS:
        sub = pool[pool.split == split]
        moved += int((sub["_orig_split"] != split).sum())
        sub[columns].to_csv(os.path.join(out, f"ppb_{split}_iface.csv"), index=False)

    # ---- report ----
    print("Split sizes (original -> clean):")
    for s in SPLITS:
        n_new = int((pool.split == s).sum())
        with_if = int(pool[(pool.split == s)].apply(
            lambda r: isinstance(r["iface_query"], str) and set(str(r["iface_query"])) <= set("01")
            and len(str(r["iface_query"])) > 0, axis=1).sum())
        print(f"  {s:5s}: {orig_sizes[s]:>5d} -> {n_new:>5d}   (target {tgt[s]:.0f}, interface {with_if})")
    print(f"  complexes whose split changed: {moved}/{total}")
    print()
    print(f"Verification: clusters spanning more than one split: {span} (must be 0)")
    print(f"Verification: test proteins sharing a cluster with train/val: {leak} (must be 0)")
    print()
    print("Wrote the clean split to:", out)
    print("Files: ppb_train_iface.csv, ppb_val_iface.csv, ppb_test_iface.csv (same columns as the originals).")
    print()
    print("Next: retrain the affinity model on this split with")
    print("  python phase3/train_phase3.py ... --ppb_dir", out)
    print("The provided job phase3/train_phase3_esmc_clean_seeds.sbatch does this for three seeds.")


if __name__ == "__main__":
    main()
