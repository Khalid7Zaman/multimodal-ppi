#!/usr/bin/env bash
#SBATCH --job-name=mmCfD_seeds
#SBATCH --partition=gpu_3080ti,gpu_5080,gpu_5070ti,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmCfD_seed%a_%A.out

# Phase B capacity follow-up: FROZEN backbone, but a WIDER fitting network.
# The first frozen runs tied the fine-tuned model on interface and trailed it by about 0.03 Pearson on
# affinity, while training only 0.2% of the parameters (a 1.35M fitting network at d_model=256). The
# question this job answers: is that small affinity gap just under-capacity in the fitting network? We
# double the width to d_model=512 (about four times the trainable parameters, still roughly 1% of the
# 576M backbone) and keep everything else identical to the frozen baseline. If the affinity gap shrinks,
# freezing is confirmed as the right design; if it does not, the gap is intrinsic to frozen features.
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
SEED=$SLURM_ARRAY_TASK_ID
echo "=== ESM-C multimodal, FROZEN backbone, d_model=512, seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --backbone esmc --esm "$HOME/Projects/esm-c/ESMplusplus_large" \
  --freeze_backbone --d_model 512 \
  --epochs 8 --batch 2 --lr 1e-4 --seed "$SEED" \
  --out "phase3/runs/mm_esmc_frozen_d512_s$SEED"
echo "----- done ESM-C FROZEN d512 seed $SEED -----"
