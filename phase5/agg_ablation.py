#!/usr/bin/env python
"""Aggregate the 650M ablation into mean +/- s.d. over seeds, on the held-out TEST split.
Run after the ablation array finishes:  python phase5/agg_ablation.py
Reads the two ablation conditions plus the already-available full model (mm_650M_s*)."""
import json, glob
import numpy as np

CONDS = [
    ("sequence only",        "phase3/runs/abl_seq_s*/test_metrics.json"),
    ("+ structure",          "phase3/runs/abl_seqstruct_s*/test_metrics.json"),
    ("+ evolution (full)",   "phase3/runs/mm_650M_s*/test_metrics.json"),
]

def agg(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    M = [json.load(open(f)) for f in files]
    out = {}
    for key in ("pearson", "rmse", "aupr"):
        v = np.array([m[key] for m in M], dtype=float)
        out[key] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
    out["n"] = len(M)
    return out

print("Held-out TEST ablation (650M, mean +/- s.d. over seeds)")
print(f"{'condition':22s} {'affinity Pearson':>20s} {'affinity RMSE':>18s} {'interface AUPR':>18s}   seeds")
for name, pattern in CONDS:
    a = agg(pattern)
    if a is None:
        print(f"{name:22s}   (no results yet)")
        continue
    p, r, u = a["pearson"], a["rmse"], a["aupr"]
    print(f"{name:22s}   {p[0]:.3f} +/- {p[1]:.3f}      {r[0]:.3f} +/- {r[1]:.3f}     "
          f"{u[0]:.3f} +/- {u[1]:.3f}      {a['n']}")
