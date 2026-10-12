#!/usr/bin/env bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=phase5/logs/eval_metrics_%j.out

# Phase 5 - full-metric evaluation of our interaction model on the leakage-free Bernett test set.
# Reports AUPR, AUROC, Precision, Recall, F1 (0.5 threshold), Accuracy and MCC, and saves the
# per-pair predictions. Inference only (no training). The AUROC/AUPR should match the earlier
# benchmark (0.717 / 0.709), confirming the same checkpoint and test set.

eval "$(conda shell.bash hook)"; conda activate mmppi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase5/logs phase5/metrics_full

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
BTEST=$HOME/Projects/ppi-data/interaction/bernett/bernett_test.csv

nvidia-smi
python phase2/eval_ppi_full.py \
  --esm "$ESM" \
  --ckpt phase2/runs/ppi_650M_v1/best.pt \
  --test "$BTEST" \
  --out phase5/metrics_full \
  --tag ours_650M \
  --max_length 512 --batch 8

# Show the columns of the saved PLM-interact scored file (from the earlier benchmark) so the
# same five metrics can be computed for the comparison model in the next step.
PLM=$HOME/Projects/ppi-data/phase5_bench/plm_bernett_scored.csv
echo "----- PLM-interact scored file header (for the comparison, if present) -----"
[ -f "$PLM" ] && head -n 2 "$PLM" || echo "not found: $PLM"
echo "----- done -----"
