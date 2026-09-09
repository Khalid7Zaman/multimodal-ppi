# Phase 0 — Environment and Baseline: Summary

Phase 0 was about settling one question before any model-building began: could we stand up a
complete, reproducible deep-learning environment on the group's GPU cluster and prove it by
reproducing a strong published baseline from end to end? That has now been done.

Working on the cluster under SLURM, I built a clean, isolated environment around PyTorch 2.7.1
with CUDA 12.8 — the build required by our new RTX 50-series "Blackwell" GPUs, on which older
CUDA versions simply fail to run. Into it I installed PLM-interact, the recent *Nature
Communications* method that adapts a protein language model (ESM-2) to predict protein–protein
interactions, together with its pre-trained 650M-parameter human model.

To confirm the pipeline works from raw sequence to prediction, I first ran a single protein pair
through the model as a quick check, then evaluated the full D-SCRIPT human benchmark of 52,725
protein pairs at the standard 1:10 positive-to-negative ratio. The model achieved an **AUROC of
0.9885 and an AUPR of 0.9174**. For perspective, the earlier sequence-based methods this benchmark
was built around reach roughly 0.55 (D-SCRIPT) and 0.58–0.61 (Topsy-Turvy) in AUPR — so the result
cleanly reproduces PLM-interact's central finding that a fine-tuned language model substantially
outperforms prior approaches. One honest caveat worth carrying forward: this is the in-distribution
human benchmark, where scores are naturally high; on harder, leakage-free evaluations the published
numbers are lower (around 0.69 AUPR on the Bernett set), and we'll keep that in mind when designing
our own evaluation.

All setup and benchmarking code is committed to the `multimodal-ppi` repository and mirrored across
the cluster, Gitee, and GitHub, so the work is fully reproducible and backed up. With a working
environment and a credible baseline in place, the project is ready to move into Phase 1 — assembling
the training data from public PPI datasets and PDBbind — where the design of our own multimodal
model begins in earnest.
