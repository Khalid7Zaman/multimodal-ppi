# Phase 0 — Environment setup & PLM-interact baseline

Author: Khalid Zaman (RA) · Supervisor: Prof. Zhaoxi Sun (SUAT) · 2026-09-09

## Goal
Set up the GPU-cluster software environment and reproduce the published
PLM-interact baseline (Liu et al., Nature Communications 2025) before building
our own multimodal PPI model. No training here — we RUN the released model.

## Environment (cluster: ~/Projects/PLM-interact, conda env `mmppi`, Python 3.10)
- torch 2.7.1+cu128  (required for the Blackwell RTX 50-series GPUs)
- PLM-interact (github.com/liudan111/PLM-interact) installed via `pip install -e .`
  -> transformers==4.40.1, numpy, pandas, biopython, scikit-learn, ...
- Models cached under offline/: ESM-2 650M (facebook/esm2_t33_650M_UR50D) and the
  checkpoint danliu1226/PLM-interact-650M-humanV11 (downloaded via hf-mirror.com).

## Result — D-SCRIPT human test set (52,725 pairs, ~1:10 positive:negative)
| Metric | This reproduction | D-SCRIPT | Topsy-Turvy |
|--------|-------------------|----------|-------------|
| AUROC  | 0.9885            | –        | –           |
| AUPR   | 0.9174            | ~0.55    | ~0.58-0.61  |

Smoke test (single protein pair): interaction probability 0.9919.
Note: this is the in-distribution human benchmark (high score). Harder,
leakage-free benchmarks are lower (the paper reports Bernett AUPR 0.69).
Inference used bf16; fp32 would shift the numbers only marginally.

## Files in this folder
- toy_smoke.py            — single-pair smoke test
- phase0_toy_smoke.sbatch — SLURM job for the smoke test
- build_test_csv.py       — builds the query,text,label CSV from the D-SCRIPT data
- benchmark_infer.py      — scores a CSV and prints AUROC/AUPR
- phase0_benchmark.sbatch — SLURM job for the benchmark (pass the CSV as argument 1)

## Reproduce (env `mmppi` active, from ~/Projects/PLM-interact)
1. Download the two models into offline/ (via hf-mirror.com).
2. git clone --depth 1 https://github.com/samsledje/D-SCRIPT.git
3. python build_test_csv.py
4. sbatch phase0_benchmark.sbatch human_test.csv
5. cat phase0_bench_<jobid>.out   # read AUROC / AUPR

GPU note: use partitions gpu_5070ti / gpu_5080 / gpu_5090; NOT gpu_6000D (GPU05,
reserved for training). GPU jobs set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
