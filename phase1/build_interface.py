#!/usr/bin/env python
# Phase 1 - derive interface-residue labels for every PPB-Affinity complex.
# For each complex: download the structure (RCSB, cached), find residues whose
# atoms come within 5 A of the partner protein, and write a 0/1 interface mask
# aligned to our CSV sequence. Adds iface_query / iface_text columns per split.
import os, urllib.request, warnings, time
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
warnings.filterwarnings("ignore")

AA3TO1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
CUTOFF = 5.0
DATA = os.path.expanduser("~/Projects/ppi-data")
STRUCT_DIR = os.path.join(DATA, "structures")
PPB = os.path.join(DATA, "affinity", "ppb")
os.makedirs(STRUCT_DIR, exist_ok=True)
parser = MMCIFParser(QUIET=True)

def fetch(pdb):
    path = os.path.join(STRUCT_DIR, f"{pdb}.cif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb}.cif"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r, open(path, "wb") as f:
                f.write(r.read())
            return path
        except Exception:
            time.sleep(2)
    return None

def get_model(pdb):
    p = fetch(pdb)
    if not p:
        return None
    try:
        return parser.get_structure(pdb, p)[0]
    except Exception:
        return None

def chains_of(v):
    return [c.strip() for c in str(v).split(",") if c.strip()]

def residues_of(model, chain_ids):
    res = []
    for cid in chain_ids:
        if cid in model:
            for r in model[cid]:
                if r.id[0] == " " and r.resname in AA3TO1:
                    res.append(r)
    return res

def masks(model, rec_ids, lig_ids):
    rres, lres = residues_of(model, rec_ids), residues_of(model, lig_ids)
    if not rres or not lres:
        return None
    ns_l = NeighborSearch([a for r in lres for a in r])
    ns_r = NeighborSearch([a for r in rres for a in r])
    rmask = "".join("1" if any(ns_l.search(a.coord, CUTOFF) for a in r) else "0" for r in rres)
    lmask = "".join("1" if any(ns_r.search(a.coord, CUTOFF) for a in r) else "0" for r in lres)
    return rmask, lmask

for split in ["train", "val", "test"]:
    fp = os.path.join(PPB, f"ppb_{split}.csv")
    df = pd.read_csv(fp)
    iq, it = [""] * len(df), [""] * len(df)
    ok = mism = fail = 0
    last_pdb, model = None, None
    order = df["pdb"].astype(str).str.upper().sort_values().index.tolist()
    for n, pos in enumerate(order, 1):
        row = df.iloc[pos]
        pdb = str(row["pdb"]).upper()
        if pdb != last_pdb:
            model = get_model(pdb); last_pdb = pdb
        if model is None:
            fail += 1
        else:
            m = masks(model, chains_of(row["receptor_chains"]), chains_of(row["ligand_chains"]))
            if m and len(m[0]) == len(str(row["query"])) and len(m[1]) == len(str(row["text"])):
                iq[pos], it[pos] = m; ok += 1
            else:
                mism += 1
        if n % 200 == 0:
            print(f"  [{split}] {n}/{len(df)} processed", flush=True)
    df["iface_query"], df["iface_text"] = iq, it
    outp = os.path.join(PPB, f"ppb_{split}_iface.csv")
    df.to_csv(outp, index=False)
    print(f"{split}: {ok} labeled, {mism} length-mismatch, {fail} failed -> {outp}", flush=True)
print("ALL DONE", flush=True)
