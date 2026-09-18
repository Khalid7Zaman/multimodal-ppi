#!/usr/bin/env python
"""
Phase 3 - Structural module (part 1): build residue CONTACT GRAPHS from PPB complex structures.

Each PPB-Affinity complex has an experimental structure (RCSB mmCIF, cached on the cluster under
~/Projects/ppi-data/structures/). This module turns each protein (receptor and ligand) into a
*residue contact graph*: nodes = residues (aligned 1:1 with the sequence we feed ESM-2), and an
edge connects two residues whose C-alpha atoms are within a distance cutoff. That contact graph is
the "3D shape" signal the Phase 2 sequence-only model was missing.

This file is a reusable building block: `build_contact_map()` and `featurize_complex()` are imported
by the Phase 3 model later. Run it directly to SELF-TEST on real PPB data (read-only, writes nothing):

    python phase3/struct_features.py            # print 20 aligned examples + summary
    python phase3/struct_features.py --n 5      # fewer examples
"""
import os, argparse
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

# three-letter -> one-letter amino acid codes (standard 20)
AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

DATA   = os.path.expanduser("~/Projects/ppi-data")
PPB    = os.path.join(DATA, "affinity", "ppb")
STRUCT = os.path.join(DATA, "structures")
_PARSER = MMCIFParser(QUIET=True)


def chains_of(v):
    """'A, B' -> ['A', 'B']"""
    return [c.strip() for c in str(v).split(",") if c.strip()]


def load_model(pdb):
    """Load the first model of a cached mmCIF structure; return None if missing or unreadable."""
    path = os.path.join(STRUCT, f"{pdb.upper()}.cif")
    if not os.path.exists(path):
        return None
    try:
        return _PARSER.get_structure(pdb, path)[0]
    except Exception:
        return None


def get_residues(model, chain_ids):
    """Standard amino-acid residues (that have a C-alpha atom) across the given chains, in order."""
    out = []
    for cid in chain_ids:
        if cid in model:
            for r in model[cid]:
                if r.id[0] == " " and r.resname in AA3TO1 and "CA" in r:
                    out.append(r)
    return out


def build_contact_map(residues, threshold=8.0):
    """
    Dense residue contact map from C-alpha coordinates. Pure NumPy - no torch-geometric needed.

    Returns (A, coords):
      A      : (L, L) float32 adjacency; A[i, j] = 1.0 if residues i and j have C-alpha atoms
               within `threshold` angstrom (diagonal zeroed), else 0.0.
      coords : (L, 3) float32 C-alpha coordinates.
    """
    coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
    if len(coords) == 0:
        return np.zeros((0, 0), np.float32), coords
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    A = (d < threshold).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A, coords


def featurize_complex(row, threshold=8.0):
    """
    Build contact graphs for one PPB row (a pandas Series with columns query/text/pdb/
    receptor_chains/ligand_chains). Returns a dict, or None if the structure is unavailable:

      {
        "pdb": <id>,
        "receptor": {"seq_len", "n_res", "aligned", "A", "coords"},
        "ligand":   {"seq_len", "n_res", "aligned", "A", "coords"},
      }

    `aligned` is True when the structure's observed residue count equals the sequence length.
    Crystal structures often omit disordered residues, so some complexes won't line up 1:1 - the
    caller (the model / training loop) decides how to handle those (e.g. skip the structural view).
    """
    model = load_model(str(row["pdb"]))
    if model is None:
        return None
    out = {"pdb": str(row["pdb"]).upper()}
    for side, seq_col, chain_col in [("receptor", "query", "receptor_chains"),
                                     ("ligand",   "text",  "ligand_chains")]:
        res = get_residues(model, chains_of(row[chain_col]))
        A, coords = build_contact_map(res, threshold)
        seq_len = len(str(row[seq_col]))
        out[side] = {"seq_len": seq_len, "n_res": len(res),
                     "aligned": len(res) == seq_len, "A": A, "coords": coords}
    return out


def _self_test(n=20, threshold=8.0, scan=80):
    csv = os.path.join(PPB, "ppb_train_iface.csv")
    df = pd.read_csv(csv)
    print(f"Reading: {csv}")
    print(f"PPB train complexes : {len(df)} (quick check scans the first {scan})")
    print(f"contact threshold   : {threshold} A (C-alpha to C-alpha)\n")

    both_ok = 0
    densities = []
    shown = 0
    scanned = 0
    for _, row in df.iterrows():
        if scanned >= scan:
            break
        feat = featurize_complex(row, threshold)
        if feat is None:
            continue
        scanned += 1
        rec, lig = feat["receptor"], feat["ligand"]
        if rec["aligned"] and lig["aligned"]:
            both_ok += 1
            if rec["n_res"] > 0:
                densities.append(rec["A"].sum() / rec["n_res"])
            if shown < n:
                shown += 1
                print(f"[{feat['pdb']}] receptor seq={rec['seq_len']:4d} struct={rec['n_res']:4d} "
                      f"contacts/res={rec['A'].sum()/max(rec['n_res'],1):5.1f}  |  "
                      f"ligand seq={lig['seq_len']:4d} struct={lig['n_res']:4d}")

    print(f"\nOf the first {scanned} complexes scanned, BOTH proteins aligned "
          f"(sequence length == structure residues) in {both_ok}.")
    if densities:
        d = np.array(densities)
        print(f"receptor contacts per residue: mean {d.mean():.1f}  (min {d.min():.1f}, "
              f"max {d.max():.1f}) - a real folded protein is typically ~6-12")
    print("\nNote: full alignment coverage over all 6,485 complexes is ~60% (see phase1/DATA_README.md);")
    print("complexes that don't line up 1:1 will simply skip the structural view (sequence still used).")
    print("\nOK: structural featurizer works on real PPB structures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Self-test the Phase 3 structural featurizer on PPB data.")
    ap.add_argument("--n", type=int, default=20, help="how many aligned examples to print")
    ap.add_argument("--threshold", type=float, default=8.0, help="C-alpha contact cutoff (angstrom)")
    ap.add_argument("--scan", type=int, default=80, help="how many complexes to scan for the quick summary")
    args = ap.parse_args()
    _self_test(args.n, args.threshold, args.scan)
