# Phase 1 — Data Assembly: Summary

Author: Khalid Zaman (RA) · Supervisor: Prof. Zhaoxi Sun (SUAT) · 2026-09-11

Phase 1 assembled the training data for the model's three prediction tasks, all from public
sources and fully reproducible from the scripts in phase1/.

## Datasets
| Task (model head)    | Dataset                              | Size (train / val / test)         |
|----------------------|--------------------------------------|-----------------------------------|
| Interaction (yes/no) | Bernett gold standard (leakage-free) | 163,192 / 59,260 / 52,048 pairs   |
| Binding affinity     | PPB-Affinity (filtered)              | 6,485 / 965 / 757 complexes (pKd) |
| Interface residues   | RCSB structures (5 Å contacts)       | ~5,400 complexes labeled          |

- Interaction: the Bernett leakage-free set (via Synthyra/bernett_gold_ppi on Hugging Face),
  balanced 1:1, chosen for honest evaluation.
- Affinity: PPB-Affinity (via proteinea/ppb_affinity), which aggregates PDBbind's protein-protein
  complexes plus SKEMPI and others — a public route to PPI affinities, since PDBbind's PP subset
  is now partly behind PDBbind+. Affinities converted to pKd = -log10(KD).
- Interface residues: computed from each complex's 3D structure (RCSB); residues within 5 Å across
  the interface are labeled. ~40% of train complexes lack a mask because crystal structures omit
  disordered residues; those complexes still keep their affinity and interaction labels. Coverage
  can be improved later via sequence alignment.

Raw data lives on the cluster under ~/Projects/ppi-data/ (kept out of git). Scripts and a full data
description are in phase1/ (see phase1/DATA_README.md).

Next: Phase 2 — core interaction model (sequence + cross-attention).
