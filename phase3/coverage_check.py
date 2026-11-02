#!/usr/bin/env python
"""Phase 3 input-coverage check (read-only): how many proteins actually receive a
STRUCTURE (a 1:1-aligned experimental contact graph) and an MSA (a non-empty .a3m),
per split. These are the real coverage numbers of the three-view model's two optional
views, counted exactly the way the data loader looks them up at training time.

No GPU, no model, no writes. Run in the esmc environment, from the repo root:
    python phase3/coverage_check.py --split test
    python phase3/coverage_check.py --split train
"""
import os, argparse
import pandas as pd
from struct_features import featurize_complex, PPB
from collect_ppb_seqs import seq_id

MSA_A3M = os.path.expanduser("~/Projects/ppi-data/msa/full/a3m")


def has_msa(seq):
    """True if a non-empty .a3m exists for this sequence, using the loader's own id."""
    p = os.path.join(MSA_A3M, f"{seq_id(str(seq))}.a3m")
    return os.path.exists(p) and os.path.getsize(p) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(PPB, f"ppb_{args.split}_iface.csv"))
    nC = len(df)
    prot_struct = prot_msa = both_struct = prot_either = 0
    for _, row in df.iterrows():
        feat = featurize_complex(row)
        r_s = bool(feat and feat["receptor"]["aligned"])
        l_s = bool(feat and feat["ligand"]["aligned"])
        r_m = has_msa(row["query"])
        l_m = has_msa(row["text"])
        prot_struct += int(r_s) + int(l_s)
        prot_msa += int(r_m) + int(l_m)
        both_struct += int(r_s and l_s)
        prot_either += int(r_s or r_m) + int(l_s or l_m)
    nP = 2 * nC
    pct = lambda x, tot: f"{x}/{tot} ({100.0 * x / tot:.1f}%)"
    print(f"split: {args.split}   complexes: {nC}   proteins: {nP}")
    print(f"  structure (experimental, aligned 1:1) : {pct(prot_struct, nP)} of proteins")
    print(f"  both proteins have a structure        : {pct(both_struct, nC)} of complexes")
    print(f"  MSA present (non-empty .a3m)          : {pct(prot_msa, nP)} of proteins")
    print(f"  at least one extra view (struct or MSA): {pct(prot_either, nP)} of proteins")
    print("Note: proteins without a view fall back to sequence only; AlphaFold-predicted")
    print("structures (roadmap Phase B/C) would raise the structure coverage above.")


if __name__ == "__main__":
    main()
