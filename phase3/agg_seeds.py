#!/usr/bin/env python
"""Aggregate the per-seed test metrics into mean +/- standard deviation (for error bars).
Run after the seeded 650M runs finish:  python phase3/agg_seeds.py
"""
import json, glob, sys
import numpy as np

runs = sorted(glob.glob("phase3/runs/mm_650M_s*/test_metrics.json"))
if not runs:
    sys.exit("No test_metrics.json found yet - the seeded runs are not finished.")

M = [json.load(open(r)) for r in runs]
print(f"held-out TEST metrics over {len(M)} seeds:")
for key, label, hi in (("pearson", "affinity Pearson", True),
                        ("rmse",    "affinity RMSE",    False),
                        ("aupr",    "interface AUPR",   True)):
    v = np.array([m[key] for m in M], dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"  {label:18s}: {v.mean():.3f} +/- {sd:.3f}   (seeds: {[round(x,3) for x in v]})")
