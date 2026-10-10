# multimodal-ppi

A multimodal deep-learning model for protein-protein interaction. From the sequences of two
proteins, the model predicts three things at the same time: whether the two proteins interact,
how strongly they bind (binding affinity), and which residues form the contact interface.

Author: Khalid Zaman, Research Assistant Professor
Supervisor: Prof. Zhaoxi Sun

## What the model does

The model combines three views of each protein. A protein language encoder (ESM-2) reads the
sequence. A structural module turns experimental complex structures into residue contact graphs.
An evolutionary module measures conservation from multiple sequence alignments. A cross-attention
layer then lets the two proteins read each other, and three task-specific heads produce the
interaction, the affinity, and the interface predictions. Training uses only public experimental
data, so every result can be reproduced from public sources.

## Main results

- Interaction: on the leakage-free Bernett benchmark (52,048 held-out pairs), the model reaches
  AUROC 0.717 and AUPR 0.709. On the same test set the publicly released comparison model reaches
  0.622 and 0.636, and both of our figures are above the value of about 0.69 reported in the
  literature for this benchmark.
- Affinity and interface: on 757 held-out complexes, repeated over three seeds, the model reaches
  an affinity correlation (Pearson) of 0.439 (plus or minus 0.018) and an interface AUPR of 0.281
  (plus or minus 0.006).
- The evolutionary view helps where its signal exists: on complexes with deep alignments the
  affinity correlation rises from 0.512 to 0.536.
- The interface predictions are interpretable: when mapped back onto the three-dimensional
  structures, the residues the model marks as interface sit on the real contact surfaces.

## Repository layout

- `docs/` : research proposal and model architecture.
- `phase0_baseline/` : environment setup and reproduction of a published baseline.
- `phase1/` : data assembly scripts and the data README.
- `phase2/` : the core interaction model.
- `phase3/` : the structural and evolutionary modules and the multi-task model.
- `phase5/` : benchmarking, ablation studies, and interpretability.
- `reports/` : phase summaries, figures, and a map of where everything lives.
- `PROJECT_STATE.md` : single source of truth for scope, plan, current status, and decisions.

## Data and compute

Training data comes from public sources: the Bernett interaction benchmark and the affinity set
derived from PDBbind. The large data files are kept out of this repository and are regenerated
from public sources by the scripts in `phase1/`. The code runs on a GPU cluster under SLURM, with
PyTorch 2.7.1 and CUDA 12.8.

## Status

Phases 0 through 5 are complete. The remaining work is to prepare the manuscript and the public
release of the code.
