# Reports & Project Map

A quick guide to where everything lives (for reviewers / supervisor).

## Status
- ../PROJECT_STATE.md — single source of truth: scope, phased plan, current status, decisions log.

## Phase reports (this folder)
- PHASE0_SUMMARY.md — environment setup + PLM-interact baseline reproduction (with results).
- PHASE1_SUMMARY.md — training-data assembly (interaction, affinity, interface).

## Code
- ../phase0_baseline/ — Phase 0 scripts (env setup, smoke test, benchmark) + README.
- ../phase1/ — Phase 1 data scripts (get_bernett.py, get_ppb.py, build_interface.py) + DATA_README.md.

## Data (NOT in git — lives on the cluster)
- ~/Projects/ppi-data/ on the cluster holds the large datasets and downloaded structures.
  Kept out of git on purpose (multi-GB); the scripts above regenerate it from public sources.

## Where the code runs
- University GPU cluster (SLURM), conda env `mmppi`, PyTorch 2.7.1+cu128.
