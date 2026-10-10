#!/usr/bin/env python
"""Phase 5 Part C - export a clean per-residue table for the interpretability examples, for
manual checking (e.g. alongside PyMOL). For every residue we actually modelled it lists the
predicted interface probability and the true experimental interface label.
Run in the mmppi env:  python phase5/export_residue_table.py
Writes phase5/interp/interface_predictions.csv"""
import os, sys, glob, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
import numpy as np
import pandas as pd
from struct_features import load_model, get_residues, chains_of, PPB, AA3TO1

df = pd.read_csv(os.path.join(PPB, "ppb_test_iface.csv"))
out = []
for npz in sorted(glob.glob(os.path.join(HERE, "interp", "*.npz"))):
    d = np.load(npz, allow_pickle=True)
    pdb = str(d["pdb"])
    rows = df[df["pdb"].astype(str).str.upper() == pdb.upper()]
    if len(rows) == 0:
        print("skip (no csv row):", pdb); continue
    row = rows.iloc[0]
    model = load_model(pdb)
    rec = get_residues(model, chains_of(row["receptor_chains"]))
    lig = get_residues(model, chains_of(row["ligand_chains"]))
    for prot, reslist, prob, true in (("receptor", rec, d["rec_prob"], d["rec_true"]),
                                      ("ligand",   lig, d["lig_prob"], d["lig_true"])):
        for r, p, t in zip(reslist, prob, true):
            chain = r.get_parent().id
            resseq = r.id[1]
            aa = AA3TO1.get(r.resname, "X")
            exp = "" if float(t) < 0 else int(round(float(t)))   # 1=interface, 0=not, ""=unknown
            out.append([pdb, prot, chain, int(resseq), aa, round(float(p), 3), exp])

outpath = os.path.join(HERE, "interp", "interface_predictions.csv")
with open(outpath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pdb", "protein", "chain", "resseq", "aa", "predicted_prob", "experimental_interface"])
    w.writerows(out)
print(f"wrote {outpath}  ({len(out)} modelled residues across the example complexes)")
