#!/usr/bin/env python
"""Stratified affinity analysis on the held-out TEST split.
Asks whether each view helps WHERE it applies:
  - STRUCTURE: sequence-only vs +structure, on complexes that HAVE an aligned structure.
  - EVOLUTION: +structure vs +evolution (full), on DEEP-MSA complexes.
Reports mean +/- s.d. over the 3 seeds, and the complementary subset for contrast.
Run after stratified_eval.sbatch finishes:  python phase5/agg_stratified.py"""
import json, glob
import numpy as np

meta = json.load(open("phase5/test_meta.json"))
has_struct = np.array([m["has_structure"] for m in meta], dtype=bool)
depth = np.array([m["depth"] for m in meta], dtype=float)
dep_med = float(np.median(depth))


def load_rows(pattern):
    files = sorted(glob.glob(pattern))
    return [np.array([[r["pkd_true"], r["pkd_pred"]] for r in json.load(open(f))], dtype=float)
            for f in files]


def metric_on(arr, mask):
    t, p = arr[mask, 0], arr[mask, 1]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    pear = float(np.corrcoef(p, t)[0, 1]) if mask.sum() > 1 else float("nan")
    return pear, rmse


def summarize(seed_arrs, mask):
    ps = np.array([metric_on(a, mask)[0] for a in seed_arrs])
    rs = np.array([metric_on(a, mask)[1] for a in seed_arrs])
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
    return ps.mean(), sd(ps), rs.mean(), sd(rs), len(seed_arrs)


def show(title, a_name, a_arrs, b_name, b_arrs, mask):
    print(f"\n{title}  (n = {int(mask.sum())} complexes)")
    for name, arrs in ((a_name, a_arrs), (b_name, b_arrs)):
        pm, ps, rm, rs, k = summarize(arrs, mask)
        print(f"  {name:16s}  Pearson {pm:.3f} +/- {ps:.3f}    RMSE {rm:.3f} +/- {rs:.3f}   ({k} seeds)")


seq  = load_rows("phase3/runs/abl_seq_s*/test_perrow.json")
sst  = load_rows("phase3/runs/abl_seqstruct_s*/test_perrow.json")
full = load_rows("phase3/runs/mm_650M_s*/test_perrow.json")

print(f"TEST complexes: {len(meta)} | with structure: {int(has_struct.sum())} "
      f"({100*has_struct.mean():.0f}%) | MSA depth median {dep_med:.0f}")

show("STRUCTURE effect - on complexes WITH an aligned structure",
     "sequence", seq, "+ structure", sst, has_struct)
show("STRUCTURE effect - on complexes WITHOUT structure (contrast)",
     "sequence", seq, "+ structure", sst, ~has_struct)

deep = depth >= dep_med
show("EVOLUTION effect - on DEEP-MSA complexes (depth >= median)",
     "+ structure", sst, "+ evolution", full, deep)
show("EVOLUTION effect - on SHALLOW-MSA complexes (contrast)",
     "+ structure", sst, "+ evolution", full, ~deep)
