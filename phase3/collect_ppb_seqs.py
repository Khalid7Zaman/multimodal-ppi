#!/usr/bin/env python
"""
Phase 3 (evolutionary module) - collect the unique PPB protein sequences we need MSAs for.

The PPB-Affinity splits list receptor ("query") and ligand ("text") sequences; the same protein
appears in many complexes, so we de-duplicate. Each unique sequence gets a stable id (a short hash
of the sequence itself), so the SAME sequence always maps to the SAME id - which lets us line each
protein up with its MSA / conservation later, from any split.

Outputs (under ~/Projects/ppi-data/):
  ppb_queries.fasta   - one entry per unique protein: ">ppbq_<hash>\\n<sequence>"
  ppb_seq2id.json     - {id: sequence} mapping

Run (read-only w.r.t. the CSVs; writes the two files above):
    python phase3/collect_ppb_seqs.py
"""
import os, json, hashlib
import numpy as np
import pandas as pd

DATA      = os.path.expanduser("~/Projects/ppi-data")
PPB       = os.path.join(DATA, "affinity", "ppb")
OUT_FASTA = os.path.join(DATA, "ppb_queries.fasta")
OUT_MAP   = os.path.join(DATA, "ppb_seq2id.json")

# amino-acid alphabet we accept (20 standard + a few ambiguity codes)
AA_OK = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def seq_id(seq: str) -> str:
    """Stable id for a sequence: same sequence -> same id, always."""
    return "ppbq_" + hashlib.md5(seq.encode()).hexdigest()[:12]


def collect():
    """Return {sequence: id} for every unique receptor/ligand sequence across all splits."""
    seqs = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(PPB, f"ppb_{split}_iface.csv"))
        for col in ["query", "text"]:
            for s in df[col].astype(str):
                s = s.strip().upper()
                if s and set(s) <= AA_OK:
                    seqs.setdefault(s, seq_id(s))
    return seqs


if __name__ == "__main__":
    seqs = collect()
    with open(OUT_FASTA, "w") as f:
        for s, i in seqs.items():
            f.write(f">{i}\n{s}\n")
    json.dump({i: s for s, i in seqs.items()}, open(OUT_MAP, "w"))

    lens = np.array([len(s) for s in seqs])
    print(f"unique PPB proteins : {len(seqs)}")
    print(f"sequence length     : min {lens.min()}, median {int(np.median(lens))}, max {lens.max()}")
    print(f"wrote               : {OUT_FASTA}")
    print(f"wrote               : {OUT_MAP}")
    print("\nOK: PPB query set ready for MSA search against UniRef50.")
