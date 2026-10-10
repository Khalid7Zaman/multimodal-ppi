#!/usr/bin/env python
"""Phase 5 Part C - PyMOL step 1 (run in the mmppi env, on the cluster).
Writes the per-residue PREDICTED interface probability into the B-factor column of each example
structure, so PyMOL can colour the cartoon by it. Reads the interp .npz files + the cached mmCIF.
Output: phase5/interp/<PDB>_pred.pdb   (B-factor = predicted interface probability, 0..1)."""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO
from struct_features import load_model, get_residues, chains_of, PPB

df = pd.read_csv(os.path.join(PPB, "ppb_test_iface.csv"))
npzs = sorted(glob.glob(os.path.join(HERE, "interp", "*.npz")))
if not npzs:
    sys.exit("no interp/*.npz found - run phase5/interp_extract.sbatch first")

for npz in npzs:
    try:
        d = np.load(npz, allow_pickle=True)
        pdb = str(d["pdb"])
        rows = df[df["pdb"].astype(str).str.upper() == pdb.upper()]
        if len(rows) == 0:
            print("skip (no csv row):", pdb); continue
        row = rows.iloc[0]
        model = load_model(pdb)
        if model is None:
            print("skip (no structure):", pdb); continue
        for chain in model:                      # zero every B-factor first
            for res in chain:
                for atom in res:
                    atom.bfactor = 0.0
        rec = get_residues(model, chains_of(row["receptor_chains"]))
        lig = get_residues(model, chains_of(row["ligand_chains"]))
        for r, p in zip(rec, d["rec_prob"]):
            for atom in r:
                atom.bfactor = float(p)
        for r, p in zip(lig, d["lig_prob"]):
            for atom in r:
                atom.bfactor = float(p)
        io = PDBIO(); io.set_structure(model)
        out = os.path.join(HERE, "interp", f"{pdb}_pred.pdb")
        io.save(out)
        print(f"wrote {out}  (receptor {len(rec)} res, ligand {len(lig)} res)")
    except Exception as e:
        print(f"skip {npz}: {e}")
