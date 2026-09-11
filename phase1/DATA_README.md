# Phase 1 — Data Assembly (multimodal PPI)

Training data for the model's three heads. Raw data lives OUTSIDE this git repo,
under `~/Projects/ppi-data/` (kept out of version control); only the scripts in
`phase1/` are committed. Last updated: 2026-09-11.

## Overview — three tasks, three public sources
| Task (model head)      | Dataset                              | Source                                              |
|------------------------|--------------------------------------|-----------------------------------------------------|
| Interaction (yes/no)   | Bernett gold standard (leakage-free) | Synthyra/bernett_gold_ppi (HF; Bernett et al. 2024) |
| Binding affinity (pKd) | PPB-Affinity (filtered)              | proteinea/ppb_affinity (HF; Liu et al. 2024)        |
| Interface residues     | PPB-Affinity complexes' structures   | RCSB PDB (5 A inter-chain contacts)                 |

## 1. Interaction — ~/Projects/ppi-data/interaction/bernett/
Files: bernett_train.csv, bernett_val.csv, bernett_test.csv
Columns: query, text, label   (two sequences; label 1 = interact, 0 = not)
Sizes (balanced 1:1):
  train 163,192 (81,596 / 81,596)
  val    59,260 (29,630 / 29,630)
  test   52,048 (26,024 / 26,024)
Script: phase1/get_bernett.py

## 2. Affinity + interface — ~/Projects/ppi-data/affinity/ppb/
Files: ppb_train_iface.csv, ppb_val_iface.csv, ppb_test_iface.csv
Columns: query, text, pKd, kd_M, pdb, receptor_chains, ligand_chains,
         iface_query, iface_text
  - query / text : receptor and ligand sequences
  - pKd          : -log10(KD in M)  (regression target; range ~1.4-15.7)
  - iface_query / iface_text : 0/1 string, one char per residue, 1 = interface
    residue (any atom within 5 A of the partner protein). Empty when the
    structure could not be length-aligned to the sequence (see caveat).
Sizes / interface coverage:
  train 6,485 complexes | pKd median 8.16 | 3,882 with interface labels
  val     965 complexes | pKd median 7.30 |   861 with interface labels
  test    757 complexes | pKd median 6.93 |   680 with interface labels
Structures cached in ~/Projects/ppi-data/structures/ (RCSB mmCIF).
Scripts: phase1/get_ppb.py (affinity), phase1/build_interface.py (interface).

## How the data maps to training
- All Bernett pairs (positive AND negative) train the INTERACTION head.
- All PPB complexes are interacting; they train the AFFINITY head (pKd) and,
  where an interface mask exists, the INTERFACE head. Complexes without a mask
  still contribute to the affinity head.

## Caveat (interface coverage)
Interface masks are attached position-by-position, which requires the structure's
observed residues to match the sequence length. Crystal structures often omit
disordered residues, so ~40% of train complexes (fewer in val/test) don't line up
1:1 and are left without an interface mask for now. This can be improved later by
aligning the observed sequence to the full sequence; it does not affect the
affinity or interaction labels.

## Reproduce (env `mmppi`, login node; HF via the hf-mirror)
  export HF_ENDPOINT=https://hf-mirror.com
  python phase1/get_bernett.py          # interaction CSVs
  python phase1/get_ppb.py              # affinity CSVs
  python phase1/build_interface.py      # downloads structures from RCSB, adds interface masks
