# PROJECT_STATE — Protein–Protein Interaction Model

**This file is the single source of truth for the project.** Both the writing tab and the
implementation tab read and update it. It lives in the repo, so it stays current on the laptop,
on Gitee, and on the cluster. Last updated: 2026-09-11.

## 1. Project summary
A multimodal deep-learning model for protein–protein interaction. From the sequences of two
proteins it predicts: (1) whether they interact, (2) their binding affinity, (3) the interface residues.

## 2. Confirmed scope (with Prof. Zhaoxi Sun)
- **Immediate focus:** protein–protein interaction and binding affinity. Confirmed.
- **Later extension (not now):** ADMET properties, peptide developability such as hydrolytic
  stability, and protein–ligand (small-molecule) binding. Recorded as future work.
- Protein–protein interaction only for this phase. No peptides, no ADMET, no molecular
  simulation as a data engine, and no AlphaFold structure prediction.
- Structural and binding data from **PDBbind** (experimental complex structures and measured
  affinities). Recommended by the supervisor.
- Interaction labels from public PPI datasets (STRING, the PLM-interact release, the Bernett benchmark).
- Evolutionary signal from multiple sequence alignments (MSA).
- Complementary to the group's bound-structure prediction work (DATUM-FACET). We do not predict bound structures.

## 3. Architecture (5 stages)
1. **Input Layer** — Protein A, Protein B → shared encoder.
2. **Multimodal Encoder** — Sequence module (ESM-2 / ESM-3, prefer the newer version) · Structural
   module (PDBbind complexes → residue contact graphs) · Evolutionary module (MSA → conservation encoder).
3. **Cross-Protein Reasoning Engine** — cross-attention transformer → interaction map.
4. **Multi-Task Head** — Interaction · Binding affinity · Interface residues (with interpretability).
5. **Training on experimental data** — supervised on PDBbind affinities and public PPI datasets.

## 4. Infrastructure and environment (confirmed with supervisor)
- Repo: `multimodal-ppi` on GitHub and Gitee, synced to the cluster (`~/Projects/multimodal-ppi`) and the laptop.
- Scheduler: the cluster uses **SLURM**. You submit a job and wait for it to finish.
- Nodes: CPU03, CPU05, CPU06 and GPU03, GPU04, GPU06–GPU10 for general tasks. **GPU05 (RTX 6000D)
  is reserved for model training**, so final training runs there.
- GPU hardware: RTX 5090 (32 GB), RTX 6000D, RTX 5080, and RTX 5070 Ti (16 GB) cards.
- Software: **PyTorch 2.7.1 with CUDA 12.8 (`torch 2.7.1+cu128`)**. Preferred protein language model is the newer ESM (ESM-3).
- Codex is available on the cluster to run commands. Author: Khalid Zaman (RA). Supervisor: Prof. Zhaoxi Sun, SUAT.

## 5. Data and validation (confirmed with supervisor)
- No in-house lab data and no wet-lab validation system at this stage.
- Data comes from public sources: PDBbind (structures and affinities) and public PPI datasets.
- Validation is computational for now. Predictions are checked against PDBbind measurements and,
  where useful, against the group's docking and MM/GBSA calculations.
- The group's own binding methods are docking and MM/GBSA.

## 6. Phased plan
- Phase 0 — Environment and baseline. Install `torch 2.7.1+cu128` and ESM on the cluster. Reproduce
  the PLM-interact baseline on a small dataset. Submit through SLURM. **— DONE (2026-09-09).**
- Phase 1 — Data assembly (public PPI datasets + PDBbind). **— DONE (2026-09-11).**
- Phase 2 — Core interaction model (sequence + cross-attention).
- Phase 3 — Add structural (PDBbind) and evolutionary modules.
- Phase 4 — Add binding-affinity and interface heads.
- Phase 5 — Benchmarking and ablations.
- Phase 6 — Manuscript and public code release.

## 7. Current status
- Proposal written (v10) and shared with supervisor.
- Scope, data source, compute, and software versions confirmed with supervisor.
- Infrastructure set up and tested (repo, cluster, and laptop in sync; mirrored on Gitee and GitHub).
- **Phase 0 COMPLETE (2026-09-09):** environment built on the cluster (conda env `mmppi`,
  Python 3.10, `torch 2.7.1+cu128`) and the PLM-interact baseline reproduced end to end. On the
  D-SCRIPT human test set (52,725 pairs): **AUROC 0.9885, AUPR 0.9174** — well above the D-SCRIPT
  (~0.55) and Topsy-Turvy (~0.58–0.61) AUPR baselines. Scripts under `phase0_baseline/`.
- **Phase 1 COMPLETE (2026-09-11):** training data assembled from public sources.
  Interaction — Bernett gold standard (leakage-free): 163,192 / 59,260 / 52,048 balanced
  train/val/test pairs. Affinity — PPB-Affinity (filtered): 6,485 / 965 / 757 complexes with pKd.
  Interface — ~5,400 complexes with per-residue interface labels (5 Å contacts from RCSB
  structures). Raw data on the cluster under `~/Projects/ppi-data/`; scripts + data README under
  `phase1/` (see phase1/DATA_README.md).
- Next: Phase 2 — core interaction model (sequence + cross-attention).

## 8. Decisions log
- 2026-09-09 — Scope set to protein–protein interaction and binding affinity. PDBbind chosen for
  structure and affinity. Molecular simulation as a data engine, AlphaFold prediction, peptides,
  and ADMET removed from the immediate scope. (Prof. Sun)
- 2026-09-09 — ADMET properties, peptide developability (for example hydrolytic stability), and
  protein–ligand binding recorded as later extensions, not part of the current phase. (Prof. Sun)
- 2026-09-09 — Compute confirmed: SLURM scheduler; GPU05 (RTX 6000D) reserved for model training;
  general tasks on CPU03/05/06 and GPU03/04/06–10. Environment: `torch 2.7.1+cu128`; prefer ESM-3.
  No in-house data or wet-lab validation yet. (Prof. Sun)
- 2026-09-09 — Phase 0 complete. Cluster environment built (conda env `mmppi`, Python 3.10,
  `torch 2.7.1+cu128`; Blackwell RTX 50-series require CUDA 12.8). PLM-interact-650M reproduced on
  the D-SCRIPT human test set: AUROC 0.9885 / AUPR 0.9174. Baseline scripts committed under
  `phase0_baseline/`; repo mirrored on Gitee and GitHub. (K. Zaman)
- 2026-09-11 — Phase 1 complete. Interaction data = Bernett gold standard (leakage-free, via
  Synthyra/bernett_gold_ppi on Hugging Face). Affinity data = PPB-Affinity filtered set (via
  proteinea/ppb_affinity on Hugging Face), which aggregates PDBbind's protein-protein complexes
  plus SKEMPI and others — chosen as the public route to PPI affinities because PDBbind's PP
  subset is now partly behind PDBbind+. Interface residues derived from RCSB structures (5 Å
  inter-chain contacts); ~5,400 complexes labeled. Raw data kept off git under ~/Projects/ppi-data/;
  scripts + DATA_README committed under phase1/. (K. Zaman)
