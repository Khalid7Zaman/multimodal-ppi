# PROJECT_STATE — Protein–Protein Interaction Model

**This file is the single source of truth for the project.** Both the writing tab and the
implementation tab read and update it. It lives in the repo, so it stays current on the laptop,
on Gitee, and on the cluster. Last updated: 2026-09-09.

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
  the PLM-interact baseline on a small dataset. Submit through SLURM.
- Phase 1 — Data assembly (public PPI datasets + PDBbind).
- Phase 2 — Core interaction model (sequence + cross-attention).
- Phase 3 — Add structural (PDBbind) and evolutionary modules.
- Phase 4 — Add binding-affinity and interface heads.
- Phase 5 — Benchmarking and ablations.
- Phase 6 — Manuscript and public code release.

## 7. Current status
- Proposal written (v10) and shared with supervisor.
- Scope, data source, compute, and software versions confirmed with supervisor.
- Infrastructure set up and tested (repo, cluster, and laptop in sync).
- Next: Phase 0 (environment setup) — not started.

## 8. Decisions log
- 2026-09-09 — Scope set to protein–protein interaction and binding affinity. PDBbind chosen for
  structure and affinity. Molecular simulation as a data engine, AlphaFold prediction, peptides,
  and ADMET removed from the immediate scope. (Prof. Sun)
- 2026-09-09 — ADMET properties, peptide developability (for example hydrolytic stability), and
  protein–ligand binding recorded as later extensions, not part of the current phase. (Prof. Sun)
- 2026-09-09 — Compute confirmed: SLURM scheduler; GPU05 (RTX 6000D) reserved for model training;
  general tasks on CPU03/05/06 and GPU03/04/06–10. Environment: `torch 2.7.1+cu128`; prefer ESM-3.
  No in-house data or wet-lab validation yet. (Prof. Sun)
