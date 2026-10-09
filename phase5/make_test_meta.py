#!/usr/bin/env python
"""Per-complex metadata for the held-out TEST split, for the stratified analysis:
  - has_structure : does the complex have an aligned experimental contact map (either chain)?
  - depth         : MSA depth (mean number of aligned sequences across the two chains)
Writes phase5/test_meta.json, aligned to ppb_test order."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer
from data import PPBComplexData, MSA_A3M
from collect_ppb_seqs import seq_id
from evo_features import parse_a3m


def a3m_depth(seq):
    sid = seq_id(str(seq))
    path = os.path.join(MSA_A3M, f"{sid}.a3m")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path) as f:
        return len(parse_a3m(f.readlines()))


def main():
    esm = os.path.expanduser("~/Projects/PLM-interact/offline/esm2_t33_650M_UR50D")
    tok = AutoTokenizer.from_pretrained(esm)
    ds = PPBComplexData("test", tok, max_length=512, use_struct=True, use_msa=True)

    meta = []
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        item = ds[i]
        rec_s = float(item["rec"]["adj"].sum()) > 0
        lig_s = float(item["lig"]["adj"].sum()) > 0
        rd = a3m_depth(row["query"])
        ld = a3m_depth(row["text"])
        meta.append({"idx": i, "pdb": str(row["pdb"]),
                     "has_structure": bool(rec_s or lig_s),
                     "rec_depth": rd, "lig_depth": ld, "depth": (rd + ld) / 2.0})
        if i % 100 == 0:
            print(f"  {i}/{len(ds)}", flush=True)

    os.makedirs("phase5", exist_ok=True)
    json.dump(meta, open("phase5/test_meta.json", "w"))
    hs = sum(m["has_structure"] for m in meta)
    dep = np.array([m["depth"] for m in meta])
    print(f"wrote phase5/test_meta.json  ({len(meta)} complexes)", flush=True)
    print(f"  with structure: {hs}/{len(meta)}  ({100 * hs / len(meta):.0f}%)", flush=True)
    print(f"  MSA depth (mean of 2 chains): median {np.median(dep):.0f}, mean {dep.mean():.0f}", flush=True)


if __name__ == "__main__":
    main()
