#!/usr/bin/env bash
#SBATCH --job-name=gated_smoke
#SBATCH --partition=gpu_5070ti,gpu_5080,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=phase5/logs/gated_smoke_%j.out

# Quick correctness check for the gated-fusion path, BEFORE the big 650M run.
# 35M encoder, all views, fusion=gated, overfit 8 complexes for 1 epoch. If it finishes
# cleanly (and the log shows "fusion: gated"), the gated module and its gradients work.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs

ESM35=$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D
nvidia-smi
echo "=== smoke test: gated fusion, all views, 35M, overfit 8, 1 epoch ==="
python phase3/train_phase3.py --esm "$ESM35" \
  --fusion gated --overfit --limit 8 --epochs 1 --batch 2 \
  --out phase3/runs/gated_smoke
echo "----- gated smoke test finished -----"
