#!/usr/bin/env python
"""Aggregate the per-seed test metrics into mean +/- standard deviation (for error bars).
Run after the seeded runs finish. By default it aggregates the ESM-2 650M runs; pass a run-dir
glob (or prefix) to aggregate a different set, for example the ESM-C runs:
    python phase3/agg_seeds.py                            # ESM-2 650M  (mm_650M_s*)
    python phase3/agg_seeds.py "phase3/runs/mm_esmc_s*"   # ESM-C 600M
"""
import json, glob, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "phase3/runs/mm_650M_s*"
if not pat.endswith("test_metrics.json"):
    pat = pat.rstrip("/") + "/test_metrics.json"
runs = sorted(glob.glob(pat))
if not runs:
    sys.exit(f"No test_metrics.json found for pattern {pat!r} - the seeded runs are not finished.")

M = [json.load(open(r)) for r in runs]
bk = sorted({m.get("backbone", "?") for m in M})
print(f"held-out TEST metrics over {len(M)} seeds  (backbone: {', '.join(bk)}):")
for key, label, hi in (("pearson", "affinity Pearson", True),
                       ("rmse",    "affinity RMSE",    False),
                       ("aupr",    "interface AUPR",   True)):
    v = np.array([m[key] for m in M], dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"  {label:18s}: {v.mean():.3f} +/- {sd:.3f}   (seeds: {[round(x, 3) for x in v]})")
