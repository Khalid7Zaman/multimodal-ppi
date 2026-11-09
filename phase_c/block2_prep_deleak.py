#!/usr/bin/env python
"""
Phase C block two, step 3 (prep): build the union FASTA and the test-id list for the de-leak.

To make sure the synthetic data never resembles anything in the evaluation, the candidate proteins
are clustered together with the evaluation TEST proteins. Those test proteins are the clean affinity
test split (ppb_clean) plus the Bernett interaction test split. This writes:
  <data>/phase_c/union_deleak.fasta   candidate proteins + test proteins, by stable sequence id
  <data>/phase_c/eval_test_ids.txt    the ids of the test proteins
so the next clustering step can find which candidate proteins fall in a cluster with a test protein.
No GPU.
"""
import os
import argparse
import hashlib
import pandas as pd


def clean(s):
    return "".join(s.split()).upper()


def sid(s, n=12):
    return hashlib.sha1(clean(s).encode()).hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    args = ap.parse_args()
    data = os.path.expanduser(args.data)
    pc = os.path.join(data, "phase_c")

    test = {}

    def add(path, cols):
        if not os.path.exists(path):
            print("WARNING missing:", path)
            return
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c in df.columns:
                for s in df[c].dropna():
                    if isinstance(s, str) and s.strip():
                        test[sid(s)] = clean(s)

    add(os.path.join(data, "interaction", "bernett", "bernett_test.csv"), ["query", "text"])
    add(os.path.join(data, "affinity", "ppb_clean", "ppb_test_iface.csv"), ["query", "text"])

    with open(os.path.join(pc, "eval_test_ids.txt"), "w") as f:
        for k in test:
            f.write(k + "\n")

    # union = candidate proteins + test proteins (keyed by id, so exact duplicates merge)
    union = {}
    name, buf = None, []
    with open(os.path.join(pc, "candidate_proteins.fasta")) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    union[name] = "".join(buf)
                name = line[1:]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            union[name] = "".join(buf)
    for k, v in test.items():
        union.setdefault(k, v)

    with open(os.path.join(pc, "union_deleak.fasta"), "w") as f:
        for k, v in union.items():
            f.write(f">{k}\n{v}\n")

    print(f"prep: test proteins {len(test):,}, union proteins {len(union):,}")
    print("wrote", os.path.join(pc, "eval_test_ids.txt"))
    print("wrote", os.path.join(pc, "union_deleak.fasta"))


if __name__ == "__main__":
    main()
