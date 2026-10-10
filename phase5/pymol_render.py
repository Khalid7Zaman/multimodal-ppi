#!/usr/bin/env python
"""Phase 5 Part C - PyMOL step 2 (run with PyMOL).
Renders each phase5/interp/<PDB>_pred.pdb as a cartoon coloured by the PREDICTED interface
probability stored in the B-factor column (blue = low, red = high).
Run from the repo root:   pymol -cq phase5/pymol_render.py
Outputs reports/figures/phase5_pymol_<PDB>.png"""
import glob
import os
from pymol import cmd

os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
files = sorted(glob.glob(os.path.join("phase5", "interp", "*_pred.pdb")))
if not files:
    print("no *_pred.pdb found - run 'python phase5/pymol_prep.py' first")

for f in files:
    pdb = os.path.basename(f)[:-len("_pred.pdb")]
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadows", 0)
    cmd.load(f, pdb)
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.spectrum("b", "blue_white_red", pdb, minimum=0.0, maximum=1.0)
    cmd.orient()
    cmd.ray(1800, 1350)
    out = os.path.join("reports", "figures", f"phase5_pymol_{pdb}.png")
    cmd.png(out, dpi=300)
    print("wrote", out)
