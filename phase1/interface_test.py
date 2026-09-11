#!/usr/bin/env python
# Phase 1 - TEST interface extraction on a few complexes: download the structure,
# find residues whose atoms come within 5 A of the partner protein, and check how
# the structure's residues align with our CSV sequences.
import os, urllib.request, warnings
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
warnings.filterwarnings("ignore")

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
CUTOFF = 5.0
STRUCT_DIR = os.path.expanduser("~/Projects/ppi-data/structures")
os.makedirs(STRUCT_DIR, exist_ok=True)

def fetch(pdb):
    path = os.path.join(STRUCT_DIR, f"{pdb}.cif")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = f"https://files.rcsb.org/download/{pdb}.cif"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    return path

def chains_of(val):
    return [c.strip() for c in str(val).split(",") if c.strip()]

def chain_seq(model, chain_ids):
    s = []
    for cid in chain_ids:
        if cid in model:
            for res in model[cid]:
                if res.id[0] == " " and res.resname in AA3TO1:
                    s.append(AA3TO1[res.resname])
    return "".join(s)

parser = MMCIFParser(QUIET=True)
df = pd.read_csv(os.path.expanduser("~/Projects/ppi-data/affinity/ppb/ppb_test.csv"))

for i in range(3):
    row = df.iloc[i]
    pdb = str(row["pdb"]).upper()
    rec, lig = chains_of(row["receptor_chains"]), chains_of(row["ligand_chains"])
    model = parser.get_structure(pdb, fetch(pdb))[0]
    rec_atoms = [a for c in rec if c in model for a in model[c].get_atoms()]
    lig_atoms = [a for c in lig if c in model for a in model[c].get_atoms()]
    if not rec_atoms or not lig_atoms:
        print(f"{pdb}: chains missing in structure (rec {rec}, lig {lig})", flush=True); continue
    rec_if = {a.get_parent().id for a in rec_atoms if NeighborSearch(lig_atoms).search(a.coord, CUTOFF)}
    lig_if = {a.get_parent().id for a in lig_atoms if NeighborSearch(rec_atoms).search(a.coord, CUTOFF)}
    obs_rec, obs_lig = chain_seq(model, rec), chain_seq(model, lig)
    csv_rec, csv_lig = str(row["query"]), str(row["text"])
    print(f"{pdb}  rec={rec} lig={lig}", flush=True)
    print(f"   receptor: {len(obs_rec)} residues, {len(rec_if)} at interface | csv_len={len(csv_rec)}  obs_in_csv={obs_rec in csv_rec}", flush=True)
    print(f"   ligand:   {len(obs_lig)} residues, {len(lig_if)} at interface | csv_len={len(csv_lig)}  obs_in_csv={obs_lig in csv_lig}", flush=True)
