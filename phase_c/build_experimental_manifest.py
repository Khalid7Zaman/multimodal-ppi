#!/usr/bin/env python
"""
Phase C, block one: consolidate the experimental foundation into one manifest.

This reads the data already assembled in Phase 1 and merges it into a single table
keyed by protein pair, so that every later Phase C and Phase D step draws from one
documented source instead of several loose CSVs. It is deliberately light: it uses
only pandas and numpy, touches no GPU, and finishes in seconds on a login node.

What it reads (all under ~/Projects/ppi-data, produced in Phase 1):
  interaction/bernett/bernett_{train,val,test}.csv   columns: query, text, label
  affinity/ppb/ppb_{train,val,test}_iface.csv        columns: query, text, pKd, kd_M,
                                                      pdb, receptor_chains, ligand_chains,
                                                      iface_query, iface_text
  structures/{PDB}.cif                               cached RCSB structures

What it writes (under ~/Projects/ppi-data/phase_c, kept out of git like the rest of
the raw data):
  experimental_manifest.csv   one row per pair, unified schema (see COLUMNS below)
  proteins.csv                one row per unique protein sequence
  proteins.fasta              the unique proteins, ready for an mmseqs2 identity
                              clustering pass (the later, stricter leakage audit)

It also prints a full summary to the screen: counts per source and split, label
balance, affinity and interface coverage, structure coverage, and an exact-match
leakage audit (whether any test protein or test pair also appears in train or val).
The printed summary is what goes into the written data card.

Run (from ~/Projects/multimodal-ppi, with a pandas/numpy environment active):
    python phase_c/build_experimental_manifest.py
"""
import os
import sys
import argparse
import hashlib
import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]

# Unified manifest columns, in order.
COLUMNS = [
    "pair_id",            # stable id for this row (source + split + ordered sequences)
    "pair_key",           # order-independent key of the two sequences (for dedup and pair leakage)
    "source",             # bernett_interaction or ppb_affinity
    "split",              # train, val, or test (as assigned in Phase 1)
    "protein_a_id",       # stable id of sequence A
    "protein_b_id",       # stable id of sequence B
    "len_a",              # length of sequence A
    "len_b",              # length of sequence B
    "interaction_label",  # 1 interacting, 0 not (all ppb complexes are interacting -> 1)
    "pKd",                # binding-affinity target where measured, else empty
    "has_affinity",       # True where pKd is present
    "pdb",                # structure id for ppb complexes, else empty
    "receptor_chains",
    "ligand_chains",
    "has_structure",      # True where the cached .cif file exists
    "has_interface",      # True where both interface masks are present and 0/1
]


def sid(seq, n=12):
    """A short, stable id for a sequence (first n hex of its sha1)."""
    return hashlib.sha1(seq.encode()).hexdigest()[:n]


def pair_key(a, b):
    """Order-independent key for a pair of sequences."""
    lo, hi = (a, b) if a <= b else (b, a)
    return hashlib.sha1((lo + "|" + hi).encode()).hexdigest()[:16]


def pair_id(source, split, a, b):
    return hashlib.sha1((source + "|" + split + "|" + a + "|" + b).encode()).hexdigest()[:16]


def clean_seq(x):
    if not isinstance(x, str):
        return ""
    return "".join(x.split()).upper()


def read_csv_str_iface(path):
    """Read a ppb iface csv, forcing the interface columns to string.

    The interface masks are long digit-only strings (for example 000110...). Newer
    pandas otherwise infers them as integers and overflows on the large value. Peek
    at the header first so a dtype is set only for columns that exist in this file.
    """
    head = pd.read_csv(path, nrows=0).columns
    str_cols = {c: str for c in ("iface_query", "iface_text") if c in head}
    return pd.read_csv(path, dtype=str_cols)


def is_mask(s):
    return isinstance(s, str) and len(s) > 0 and set(s) <= set("01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"),
                    help="base directory holding interaction/, affinity/, structures/")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <data>/phase_c)")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    out = args.out or os.path.join(data, "phase_c")
    bernett_dir = os.path.join(data, "interaction", "bernett")
    ppb_dir = os.path.join(data, "affinity", "ppb")
    struct_dir = os.path.join(data, "structures")
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("Phase C block one: consolidate the experimental foundation")
    print("=" * 70)
    print("data base :", data)
    print("output    :", out)
    print()

    rows = []
    missing = []

    # ---- interaction source: Bernett pairs ----
    for split in SPLITS:
        path = os.path.join(bernett_dir, f"bernett_{split}.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            a, b = clean_seq(r["query"]), clean_seq(r["text"])
            if not a or not b:
                continue
            rows.append({
                "pair_id": pair_id("bernett_interaction", split, a, b),
                "pair_key": pair_key(a, b),
                "source": "bernett_interaction",
                "split": split,
                "protein_a_id": sid(a), "protein_b_id": sid(b),
                "len_a": len(a), "len_b": len(b),
                "seq_a": a, "seq_b": b,
                "interaction_label": int(r["label"]),
                "pKd": np.nan, "has_affinity": False,
                "pdb": "", "receptor_chains": "", "ligand_chains": "",
                "has_structure": False, "has_interface": False,
            })

    # ---- affinity + interface source: PPB complexes ----
    have_struct_dir = os.path.isdir(struct_dir)
    for split in SPLITS:
        path = os.path.join(ppb_dir, f"ppb_{split}_iface.csv")
        if not os.path.exists(path):
            # fall back to the pre-interface csv if the iface version is absent
            alt = os.path.join(ppb_dir, f"ppb_{split}.csv")
            if os.path.exists(alt):
                path = alt
            else:
                missing.append(path)
                continue
        df = read_csv_str_iface(path)
        for _, r in df.iterrows():
            a, b = clean_seq(r["query"]), clean_seq(r["text"])
            if not a or not b:
                continue
            pdb = str(r["pdb"]).strip() if not pd.isna(r.get("pdb", np.nan)) else ""
            cif = os.path.join(struct_dir, f"{pdb.upper()}.cif")
            has_struct = bool(pdb) and have_struct_dir and os.path.exists(cif) and os.path.getsize(cif) > 0
            pkd = pd.to_numeric(r.get("pKd", np.nan), errors="coerce")
            iq, it = r.get("iface_query", ""), r.get("iface_text", "")
            has_iface = is_mask(iq) and is_mask(it)
            rows.append({
                "pair_id": pair_id("ppb_affinity", split, a, b),
                "pair_key": pair_key(a, b),
                "source": "ppb_affinity",
                "split": split,
                "protein_a_id": sid(a), "protein_b_id": sid(b),
                "len_a": len(a), "len_b": len(b),
                "seq_a": a, "seq_b": b,
                "interaction_label": 1,
                "pKd": float(pkd) if np.isfinite(pkd) else np.nan,
                "has_affinity": bool(np.isfinite(pkd)),
                "pdb": pdb,
                "receptor_chains": str(r.get("receptor_chains", "")),
                "ligand_chains": str(r.get("ligand_chains", "")),
                "has_structure": has_struct,
                "has_interface": has_iface,
            })

    if missing:
        print("WARNING: these expected files were not found and were skipped:")
        for m in missing:
            print("  ", m)
        print()

    if not rows:
        print("ERROR: no data read. Check the --data path and the Phase 1 outputs.")
        sys.exit(1)

    man = pd.DataFrame(rows)

    # ---- de-duplicate within source + split on the order-independent pair key ----
    before = len(man)
    man = man.drop_duplicates(subset=["source", "split", "pair_key"]).reset_index(drop=True)
    dups = before - len(man)

    # ---- per-protein table (unique sequences across everything) ----
    prot = {}
    for _, r in man.iterrows():
        for pid, seq, ln in ((r["protein_a_id"], r["seq_a"], r["len_a"]),
                             (r["protein_b_id"], r["seq_b"], r["len_b"])):
            e = prot.setdefault(pid, {"protein_id": pid, "length": ln,
                                      "sources": set(), "splits": set()})
            e["sources"].add(r["source"])
            e["splits"].add(f"{r['source']}:{r['split']}")
    prot_df = pd.DataFrame([
        {"protein_id": e["protein_id"], "length": e["length"],
         "sources": ",".join(sorted(e["sources"])),
         "splits": ",".join(sorted(e["splits"]))}
        for e in prot.values()
    ])

    # ---- exact-match leakage audit (protein level and pair level) ----
    def proteins_in(source, split):
        sub = man[(man.source == source) & (man.split == split)]
        return set(sub.protein_a_id) | set(sub.protein_b_id)

    def pairs_in(source, split):
        sub = man[(man.source == source) & (man.split == split)]
        return set(sub.pair_key)

    leak_lines = []
    for source in sorted(man.source.unique()):
        tr_p = proteins_in(source, "train") | proteins_in(source, "val")
        te_p = proteins_in(source, "test")
        tr_k = pairs_in(source, "train") | pairs_in(source, "val")
        te_k = pairs_in(source, "test")
        p_overlap = len(te_p & tr_p)
        k_overlap = len(te_k & tr_k)
        leak_lines.append(
            f"  {source}: test proteins {len(te_p)}, of which {p_overlap} also in train/val "
            f"({100.0 * p_overlap / max(len(te_p), 1):.1f}%); "
            f"test pairs {len(te_k)}, of which {k_overlap} also in train/val "
            f"({100.0 * k_overlap / max(len(te_k), 1):.1f}%)")

    # cross-source protein overlap (informational, not leakage within a task)
    bern_p = set(man[man.source == "bernett_interaction"].protein_a_id) | \
             set(man[man.source == "bernett_interaction"].protein_b_id)
    ppb_p = set(man[man.source == "ppb_affinity"].protein_a_id) | \
            set(man[man.source == "ppb_affinity"].protein_b_id)
    cross = len(bern_p & ppb_p)

    # ---- write outputs ----
    man_out = man.drop(columns=["seq_a", "seq_b"])
    man_out = man_out[COLUMNS]
    man_path = os.path.join(out, "experimental_manifest.csv")
    man_out.to_csv(man_path, index=False)

    prot_path = os.path.join(out, "proteins.csv")
    prot_df.to_csv(prot_path, index=False)

    seq_of = {}
    for _, r in man.iterrows():
        seq_of.setdefault(r["protein_a_id"], r["seq_a"])
        seq_of.setdefault(r["protein_b_id"], r["seq_b"])
    fasta_path = os.path.join(out, "proteins.fasta")
    with open(fasta_path, "w") as f:
        for pid, seq in seq_of.items():
            f.write(f">{pid}\n{seq}\n")

    # ---- summary ----
    print("Counts per source and split (after de-duplication):")
    tab = man.groupby(["source", "split"]).size().reset_index(name="pairs")
    for _, r in tab.iterrows():
        print(f"  {r['source']:20s} {r['split']:5s} {r['pairs']:>8d} pairs")
    print(f"  de-duplicated rows removed within source+split: {dups}")
    print()

    bern = man[man.source == "bernett_interaction"]
    if len(bern):
        pos = int((bern.interaction_label == 1).sum())
        neg = int((bern.interaction_label == 0).sum())
        print(f"Interaction (Bernett): {len(bern)} pairs, {pos} positive, {neg} negative "
              f"({100.0 * pos / len(bern):.1f}% positive)")

    ppb = man[man.source == "ppb_affinity"]
    if len(ppb):
        with_aff = int(ppb.has_affinity.sum())
        with_str = int(ppb.has_structure.sum())
        with_if = int(ppb.has_interface.sum())
        pkds = ppb.loc[ppb.has_affinity, "pKd"]
        print(f"Affinity (PPB): {len(ppb)} complexes, {with_aff} with pKd "
              f"(median {pkds.median():.2f}, range {pkds.min():.2f} to {pkds.max():.2f})")
        print(f"  structures present (cached .cif): {with_str}/{len(ppb)} "
              f"({100.0 * with_str / len(ppb):.1f}%)")
        print(f"  interface masks present: {with_if}/{len(ppb)} "
              f"({100.0 * with_if / len(ppb):.1f}%)")
        for split in SPLITS:
            s = ppb[ppb.split == split]
            if len(s):
                print(f"    {split:5s}: {len(s):>5d} complexes | "
                      f"pKd {int(s.has_affinity.sum())} | "
                      f"struct {int(s.has_structure.sum())} | "
                      f"iface {int(s.has_interface.sum())}")
    print()

    print("Unique proteins:", len(prot_df))
    print(f"  cross-source: {cross} proteins appear in both the interaction and the affinity sets")
    print()

    print("Exact-match leakage audit (test against train and val, within each source):")
    for line in leak_lines:
        print(line)
    print("  Note: an identity-threshold audit (near-duplicates, not just exact matches)")
    print("  is the follow-up step, using proteins.fasta with mmseqs2.")
    print()

    print("Wrote:")
    print("  ", man_path)
    print("  ", prot_path)
    print("  ", fasta_path)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
