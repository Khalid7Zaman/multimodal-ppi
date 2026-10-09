#!/usr/bin/env python
"""Compare the gated-fusion model with the additive model on the held-out TEST split.
Both are the full 650M model (all three views), 3 seeds; only the fusion differs.
Run after the gated runs finish:  python phase5/agg_gated.py"""
import json, glob
import numpy as np

CONDS = [
    ("additive fusion (full)", "phase3/runs/mm_650M_s*/test_metrics.json"),
    ("gated fusion (full)",    "phase3/runs/mm_650M_gated_s*/test_metrics.json"),
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


print("Held-out TEST: gated vs additive fusion (650M, all views, mean +/- s.d. over seeds)")
print(f"{'fusion':26s} {'affinity Pearson':>20s} {'affinity RMSE':>18s} {'interface AUPR':>18s}   seeds")
for name, pattern in CONDS:
    a = agg(pattern)
    if a is None:
        print(f"{name:26s}   (no results yet)")
        continue
    p, r, u = a["pearson"], a["rmse"], a["aupr"]
    print(f"{name:26s}   {p[0]:.3f} +/- {p[1]:.3f}      {r[0]:.3f} +/- {r[1]:.3f}     "
          f"{u[0]:.3f} +/- {u[1]:.3f}      {a['n']}")
