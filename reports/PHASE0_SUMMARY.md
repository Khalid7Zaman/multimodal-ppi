# Phase 0 — Environment and Baseline: Summary

Author: Khalid Zaman (RA) · Supervisor: Prof. Zhaoxi Sun (SUAT) · 2026-09-09

Phase 0 settled one question before any model-building began: could we stand up a complete,
reproducible deep-learning environment on the group's GPU cluster and prove it by reproducing
a strong published baseline end to end? Yes.

Working on the cluster under SLURM, we built a clean, isolated environment around PyTorch 2.7.1
with CUDA 12.8 — the build required by the new RTX 50-series "Blackwell" GPUs, on which older
CUDA versions fail to run. We installed PLM-interact (Nature Communications 2025), which adapts
the ESM-2 protein language model to predict protein–protein interactions, together with its
pre-trained 650M-parameter human model.

To confirm the pipeline works from sequence to prediction, we ran a single protein pair as a
sanity check, then evaluated the full D-SCRIPT human benchmark of 52,725 protein pairs.

## Result — D-SCRIPT human test set (52,725 pairs, ~1:10 positive:negative)
| Metric | This reproduction | D-SCRIPT | Topsy-Turvy |
|--------|-------------------|----------|-------------|
| AUROC  | 0.9885            | –        | –           |
| AUPR   | 0.9174            | ~0.55    | ~0.58–0.61  |

This reproduces PLM-interact's central finding: a fine-tuned language model substantially
outperforms prior sequence-based methods. Note this is the in-distribution human benchmark
(scores are naturally high); on harder, leakage-free benchmarks the published numbers are lower
(~0.69 AUPR on the Bernett set).

Code: phase0_baseline/. Cluster environment: conda env `mmppi`, PyTorch 2.7.1+cu128.
