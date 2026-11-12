#!/usr/bin/env python
"""
Phase C simulation hand-off, piece 1: export the experimental affinity pairs.

This writes the protein-protein pairs that already have measured binding values (the PPB-Affinity
complexes), with both sequences, the measured pKd, the structure id, and the split label, as one
clean table for the MM/GBSA collaborator. The split column (from the leakage-free re-split) is kept
so that neighbour generation and model training can stay on train and validation and leave the test
set alone. No GPU.

Run from ~/Projects/multimodal-ppi:
    python phase_c/export_experimental_affinity.py
"""
import os
import argparse
import pandas as pd

SPLITS = ["train", "val", "test"]


def read_iface(path):
    head = pd.read_csv(path, nrows=0).columns
    sc = {c: str for c in ("iface_query", "iface_text", "pdb", "receptor_chains", "ligand_chains")
          if c in head}
    return pd.read_csv(path, dtype=sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--ppb_dir", default=None,
                    help="directory of ppb_<split>_iface.csv (default <data>/affinity/ppb_clean)")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    ppb = args.ppb_dir or os.path.join(data, "affinity", "ppb_clean")
    pc = os.path.join(data, "phase_c")
    os.makedirs(pc, exist_ok=True)

    rows = []
    for s in SPLITS:
        p = os.path.join(ppb, f"ppb_{s}_iface.csv")
        if not os.path.exists(p):
            print("missing:", p)
            continue
        df = read_iface(p)
        for _, r in df.iterrows():
            rows.append({
                "complex_pdb": str(r.get("pdb", "")),
                "split": s,
                "seq_receptor": str(r["query"]),
                "seq_ligand": str(r["text"]),
                "pKd": r.get("pKd", ""),
                "receptor_chains": str(r.get("receptor_chains", "")),
                "ligand_chains": str(r.get("ligand_chains", "")),
                "len_receptor": len(str(r["query"])),
                "len_ligand": len(str(r["text"])),
            })

    out = pd.DataFrame(rows, columns=["complex_pdb", "split", "seq_receptor", "seq_ligand", "pKd",
                                      "receptor_chains", "ligand_chains", "len_receptor", "len_ligand"])
    outp = os.path.join(pc, "experimental_affinity_pairs.csv")
    out.to_csv(outp, index=False)
    print(f"wrote {len(out):,} experimental affinity complexes -> {outp}")
    print(out.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
