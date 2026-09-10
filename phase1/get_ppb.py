#!/usr/bin/env python
# Phase 1 - build the protein-protein binding-affinity dataset from PPB-Affinity
# (proteinea/ppb_affinity, filtered.csv) pulled via the hf-mirror. Produces
# query,text,pKd CSVs per split, keeping PDB/chain info for the interface step.
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download
import pandas as pd, numpy as np

OUT = os.path.expanduser("~/Projects/ppi-data/affinity/ppb")
os.makedirs(OUT, exist_ok=True)
path = hf_hub_download("proteinea/ppb_affinity", "filtered.csv",
                       repo_type="dataset", local_dir=OUT)
df = pd.read_csv(path)

def combine(s):                       # multi-chain sides are comma-separated
    if pd.isna(s): return ""
    return "".join(str(s).split(","))  # concatenate chains into one sequence

kd = pd.to_numeric(df["KD(M)"], errors="coerce")
out = pd.DataFrame({
    "query":           df["Receptor Sequences"].map(combine),  # protein A (receptor)
    "text":            df["Ligand Sequences"].map(combine),    # protein B (ligand)
    "pKd":             -np.log10(kd),                          # regression target
    "kd_M":            kd,
    "pdb":             df["PDB"],
    "receptor_chains": df["Receptor Chains"],
    "ligand_chains":   df["Ligand Chains"],
    "split":           df["split"].astype(str),
})
before = len(out)
out = out[(out["query"].str.len() > 0) & (out["text"].str.len() > 0) & np.isfinite(out["pKd"])]
print(f"kept {len(out)}/{before} complexes after cleaning", flush=True)

namemap = {"train": "train", "valid": "val", "validation": "val", "test": "test"}
for sval, sub in out.groupby("split"):
    name = namemap.get(sval, sval)
    p = os.path.join(OUT, f"ppb_{name}.csv")
    sub.drop(columns=["split"]).to_csv(p, index=False)
    print(f"ppb_{name}.csv: {len(sub)} complexes | pKd {sub.pKd.min():.2f}..{sub.pKd.max():.2f} (median {sub.pKd.median():.2f})", flush=True)
print("done ->", OUT)
