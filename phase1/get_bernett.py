#!/usr/bin/env python
# Phase 1 — download the Bernett gold-standard PPI dataset (leakage-free) from
# Hugging Face (Synthyra/bernett_gold_ppi) via the hf-mirror, and write clean
# query,text,label CSVs for train / val / test into ~/Projects/ppi-data.
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
import pandas as pd

OUT = os.path.expanduser("~/Projects/ppi-data/interaction/bernett")
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("Synthyra/bernett_gold_ppi")
split_out = {"train": "train", "valid": "val", "validation": "val", "test": "test"}
for split in ds:
    df = ds[split].to_pandas()[["SeqA", "SeqB", "labels"]]
    df = df.rename(columns={"SeqA": "query", "SeqB": "text", "labels": "label"})
    name = split_out.get(split, split)
    path = os.path.join(OUT, f"bernett_{name}.csv")
    df.to_csv(path, index=False)
    pos = int((df.label == 1).sum()); neg = int((df.label == 0).sum())
    print(f"bernett_{name}.csv: {len(df)} pairs (pos {pos}, neg {neg})", flush=True)
print("done ->", OUT)
