# PROJECT_STATE — Protein–Protein Interaction Model

**This file is the single source of truth for the project.** Both the writing tab and the
implementation tab read and update it. It lives in the repo, so it stays current on the laptop,
on Gitee, on GitHub, and on the cluster. Last updated: 2026-10-09.

**Related docs (in this repo):** `docs/PROPOSAL.md` (research proposal v10) ·
`docs/ARCHITECTURE.md` (5-stage design + Phase 3/4 plan) · `WORKFLOW.md` (git sync cheat-sheet) ·
`phase1/DATA_README.md` (datasets) · `reports/PHASE0_SUMMARY.md`, `PHASE1_SUMMARY.md`,
`PHASE2_SUMMARY.md`, `PHASE3_SUMMARY.md` (phase write-ups).

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
- Repo: `multimodal-ppi`, synced across four places — cluster (`~/Projects/multimodal-ppi`),
  laptop, **Gitee and GitHub**. `origin` now pushes to BOTH remotes at once (a single `git push`
  updates Gitee and GitHub); see `WORKFLOW.md`.
- Scheduler: the cluster uses **SLURM**. You submit a job and wait for it to finish.
- Nodes: CPU03, CPU05, CPU06 and GPU03, GPU04, GPU06–GPU10 for general tasks. **GPU05 (RTX 6000D)
  is reserved for model training**, so final training runs there. Supervisor guidance: use the
  RTX 6000D or the RTX 5090 for training/heavy work.
- GPU hardware: RTX 5090 (32 GB), RTX 6000D (85 GB), RTX 5080, and RTX 5070 Ti (16 GB) cards.
- Software: **PyTorch 2.7.1 with CUDA 12.8 (`torch 2.7.1+cu128`)**. Preferred protein language model is the newer ESM (ESM-3).
- Author: Khalid Zaman, Research Assistant Professor (RAP). Supervisor: Prof. Zhaoxi Sun, SUAT.

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
- Phase 2 — Core interaction model (sequence + cross-attention). **— DONE (2026-09-17).**
- Phase 3 — Add structural (PDBbind) and evolutionary modules; bring the affinity + interface
  heads forward. **— DONE (2026-09-29).**
- Phase 4 — Affinity + interface heads. (Completed within Phase 3 — see §8.)
- Phase 5 — Benchmarking, ablations, interpretability. **— IN PROGRESS.** Interaction benchmark
  vs PLM-interact complete (see §8); ablation + interpretability remaining.
- Phase 6 — Manuscript and public code release.

## 7. Current status
- Proposal written (v10, in `docs/PROPOSAL.md`) and shared with supervisor.
- Scope, data source, compute, and software versions confirmed with supervisor.
- Infrastructure set up and tested (repo synced across cluster, laptop, Gitee, and GitHub).
- **Phase 0 COMPLETE (2026-09-09):** environment built on the cluster (conda env `mmppi`,
  Python 3.10, `torch 2.7.1+cu128`) and the PLM-interact baseline reproduced end to end. On the
  D-SCRIPT human test set (52,725 pairs): **AUROC 0.9885, AUPR 0.9174.** Scripts under
  `phase0_baseline/`.
- **Phase 1 COMPLETE (2026-09-11):** training data assembled from public sources.
  Interaction — Bernett gold standard (leakage-free): 163,192 / 59,260 / 52,048 balanced
  train/val/test pairs. Affinity — PPB-Affinity (filtered): 6,485 / 965 / 757 complexes with pKd.
  Interface — ~5,400 complexes with per-residue interface labels (5 Å contacts from RCSB
  structures). Raw data on the cluster under `~/Projects/ppi-data/`; scripts + data README under
  `phase1/` (see phase1/DATA_README.md).
- **Phase 2 COMPLETE (2026-09-17):** core interaction model built and evaluated. `CrossAttnPPI` =
  ESM-2 encoder (fine-tuned end-to-end) → 256-d projection → two-way cross-attention (A↔B) →
  symmetric pooling → MLP head. Held-out test set (52,048 pairs): **35M AUROC 0.7033 / AUPR 0.7017;
  650M AUROC 0.7174 / AUPR 0.7093** — both above the ~0.69 published Bernett AUPR, 650M best. Code
  under `phase2/`; full write-up in `reports/PHASE2_SUMMARY.md`.
- **Phase 3 COMPLETE (2026-09-29):** structural and evolutionary modules built, MSAs generated for
  all 9,516 proteins, and the multimodal multi-task model trained and evaluated. Full write-up in
  `reports/PHASE3_SUMMARY.md`. Highlights:
  - **Structural module** — `phase3/struct_features.py` (C-alpha contact graphs at 8 Å) +
    `phase3/struct_module.py` (plain-PyTorch GCN, no torch-geometric).
  - **Evolutionary module** — MMseqs2 search of all 9,516 PPB proteins vs UniRef50 (58.9M seqs);
    **844,884** unique homologs; MSAs for all 9,516 proteins, **median depth 945** (mean 638),
    100% real alignments. Conservation features (22-dim) in `phase3/evo_features.py`. Verification in
    `reports/PHASE3_MSA_VERIFY.txt`.
  - **Multimodal model** — `phase3/model.py` (MultiModalPPI: sequence + structure + evolution →
    cross-attention → affinity + interface heads) and `phase3/data.py` (multi-task dataset).
  - **Ablation + definitive run (validation set).** 35M ablation: adding the evolutionary module
    improved every task (affinity RMSE 2.032 → 1.820, Pearson 0.293 → 0.343, interface AUPR
    0.256 → 0.271). The definitive **650M** run (best at epoch 2) improved them substantially
    further: **affinity RMSE 1.555, Pearson 0.473 (+38% over 35M), interface AUPR 0.293.** Full
    progression and figure in `reports/PHASE3_SUMMARY.md` / `reports/figures/phase3_results.png`.
  - **Held-out test set (3 seeds, definitive 650M).** Repeated with seeds 0/1/2 and evaluated on the
    757-complex test split: **affinity Pearson 0.439 ± 0.018, RMSE 1.575 ± 0.044, interface AUPR
    0.281 ± 0.006** (mean ± s.d.). Stable across seeds; these are the manuscript numbers. Error-bar
    figure at `reports/figures/phase3_test_results.png`.
  - **Next:** full sequence-only → +structure → +evolution ablation, and Phase 5 benchmarking +
    interpretability.

## 8. Decisions log
- 2026-09-09 — Scope set to protein–protein interaction and binding affinity. PDBbind chosen for
  structure and affinity. Molecular simulation as a data engine, AlphaFold prediction, peptides,
  and ADMET removed from the immediate scope. (Prof. Sun)
- 2026-09-09 — ADMET properties, peptide developability, and protein–ligand binding recorded as
  later extensions, not part of the current phase. (Prof. Sun)
- 2026-09-09 — Compute confirmed: SLURM scheduler; GPU05 (RTX 6000D) reserved for model training;
  general tasks on CPU03/05/06 and GPU03/04/06–10. Environment: `torch 2.7.1+cu128`; prefer ESM-3.
  No in-house data or wet-lab validation yet. (Prof. Sun)
- 2026-09-09 — Phase 0 complete. PLM-interact-650M reproduced on the D-SCRIPT human test set:
  AUROC 0.9885 / AUPR 0.9174. (K. Zaman)
- 2026-09-11 — Phase 1 complete. Interaction = Bernett gold standard (via Synthyra/bernett_gold_ppi).
  Affinity = PPB-Affinity filtered set (via proteinea/ppb_affinity), which aggregates PDBbind's
  protein-protein complexes plus SKEMPI and others. Interface residues from RCSB structures (5 Å
  inter-chain contacts). (K. Zaman)
- 2026-09-17 — Phase 2 complete. Built CrossAttnPPI; trained 35M and 650M encoders on the Bernett
  gold standard (GPU05). Test AUPR 0.7017 / 0.7093 — above the ~0.69 published baseline; 650M best.
  (K. Zaman)
- 2026-09-18 — Phase 3 kickoff. Build the structural + evolutionary modules on the PPB-Affinity
  complexes and bring the affinity + interface heads forward. Structural module built WITHOUT
  torch-geometric (dense C-alpha contact map + plain-PyTorch GCN) to avoid fragile compiled-extension
  installs on CUDA 12.8 / Blackwell. Evolutionary module via MMseqs2 + local UniRef50. (K. Zaman)
- 2026-09-29 — Phase 3 complete. UniRef50 database built; MSAs generated for all 9,516 PPB proteins
  (844,884 unique homologs; median depth 945). Reproducible MSA pipeline: search full UniRef50 once,
  reduce to hit sequences, rebuild with compact indexing, assemble MSAs (the assembly step needs
  ~240 GB, run single-threaded on gpu05). Multimodal multi-task model built and trained. First
  controlled ablation (35M, 5 epochs): adding the evolutionary module improved every task
  (affinity RMSE 2.032 → 1.820, Pearson 0.293 → 0.343, interface AUPR 0.256 → 0.271). (K. Zaman)
- 2026-09-30 — Definitive 650M run complete (all three views, 8 epochs; best at epoch 2, fast
  convergence as in Phase 2). Validation: affinity RMSE 1.555, Pearson 0.473, interface AUPR 0.293
  — substantially above the 35M model. Results figure at `reports/figures/phase3_results.png`.
  Remaining for Phase 5: repeated seeds for error bars, held-out test-set evaluation, full
  sequence→+structure→+evolution ablation, and benchmarking + interpretability. (K. Zaman)
- 2026-10-04 — Error bars + held-out test-set evaluation complete. The definitive 650M model was
  retrained with three seeds (0, 1, 2) and scored on the 757-complex test split: affinity
  Pearson 0.439 ± 0.018, RMSE 1.575 ± 0.044, interface AUPR 0.281 ± 0.006 (mean ± s.d.). Results
  are stable across seeds and track the validation figures closely; adopted as the manuscript
  numbers. Error-bar figure at `reports/figures/phase3_test_results.png`. Remaining for Phase 5:
  full sequence-only → +structure → +evolution ablation, benchmarking, and interpretability. (K. Zaman)
- 2026-10-08 — Phase 5 interaction benchmark complete. CrossAttnPPI-650M and the published
  PLM-interact-650M (as released) were scored on the identical leakage-free Bernett test set
  (52,048 pairs, same metrics). Our model: AUROC 0.717 / AUPR 0.709; PLM-interact: AUROC 0.622 /
  AUPR 0.636; above the ~0.69 published AUPR baselines. Our model trained on the Bernett train
  split; PLM-interact used as released (not re-tuned) — part of the gap is task-matched training,
  noted honestly. Job `phase5/bench_interaction.sbatch`; figure `reports/figures/phase5_interaction_benchmark.png`;
  write-up `reports/PHASE5_SUMMARY.md`. Remaining in Phase 5: ablation and interpretability. (K. Zaman)
- 2026-10-09 — Phase 5 module ablation complete (650M, 3 seeds, held-out test split). sequence
  0.429/1.593/0.286; +structure 0.441/1.592/0.289; +evolution(full) 0.439/1.575/0.281
  (Pearson/RMSE/AUPR). The three conditions are statistically tied — all pairwise differences fall
  within the error bars — so NO significant benefit from structure or evolution is claimed at 650M
  on this test set. This honestly refines the earlier single-run 35M validation signal, which did
  not survive the rigorous (3-seed, test, error-bar) check. Figure
  `reports/figures/phase5_ablation.png`; jobs `phase5/ablation_650M.sbatch` + `agg_ablation.py`;
  view switches added to `phase3/train_phase3.py` / `data.py`. Follow-ups in progress: stratified
  analysis (structure where structure exists; evolution on deep MSAs) and a stronger-fusion study.
  Robust wins unaffected: interaction benchmark beats PLM-interact; reproducible multimodal
  pipeline. (K. Zaman)
