#!/usr/bin/env python
import pandas as pd
from Bio import SeqIO

FASTA = "D-SCRIPT/data/seqs/human.fasta"
PAIRS = "D-SCRIPT/data/pairs/human_test.tsv"

# 1) load every sequence, keyed by its Ensembl ID
id2seq = {}
for rec in SeqIO.parse(FASTA, "fasta"):
    id2seq[rec.id] = str(rec.seq)
print("sequences loaded:", len(id2seq), flush=True)

# 2) walk the pairs, replacing each ID with its sequence
rows, missing = [], 0
with open(PAIRS) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        a, b, label = parts[0], parts[1], parts[2]
        if a in id2seq and b in id2seq:
            rows.append((id2seq[a], id2seq[b], int(label)))
        else:
            missing += 1

df = pd.DataFrame(rows, columns=["query", "text", "label"])
df.to_csv("human_test.csv", index=False)
pos = int((df.label == 1).sum()); neg = int((df.label == 0).sum())
print(f"human_test.csv written: {len(df)} pairs  (skipped, missing sequence: {missing})", flush=True)
print(f"positives: {pos}   negatives: {neg}", flush=True)

# 3) a small balanced subset for a fast validation run
small = pd.concat([df[df.label == 1].head(300), df[df.label == 0].head(300)]).sample(frac=1, random_state=1)
small.to_csv("human_test_small.csv", index=False)
print(f"human_test_small.csv written: {len(small)} pairs (300 pos + 300 neg)", flush=True)
