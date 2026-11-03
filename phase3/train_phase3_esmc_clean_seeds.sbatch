#!/usr/bin/env bash
#SBATCH --job-name=mmCclean
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmCclean_seed%a_%A.out

# Phase C: the affinity model retrained on the LEAKAGE-FREE re-split, to replace the
# optimistically biased affinity numbers measured on the original PPB split. Everything matches
# the fine-tuned ESM-C run that produced the current numbers (train_phase3_esmc_seeds.sbatch):
# ESM-C backbone fine-tuned end to end, all three views, 8 epochs, batch 2, lr 1e-5, three seeds,
# best-validation checkpoint scored on the held-out TEST split. The ONLY change is --ppb_dir, which
# points the data loader at the clean split in ~/Projects/ppi-data/affinity/ppb_clean. So the old
# and new numbers are directly comparable and the only difference is the leakage-free split.
#
# Build the clean split first (CPU, seconds): python phase_c/resplit_affinity_clean.py
# Fine-tuning the 600M backbone needs a large training card, so this uses the same partitions as the
# original run; it may queue until gpu_6000D or gpu_5090 is free.
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
SEED=$SLURM_ARRAY_TASK_ID
echo "=== ESM-C multimodal, CLEAN re-split, seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --backbone esmc --esm "$HOME/Projects/esm-c/ESMplusplus_large" \
  --ppb_dir "$HOME/Projects/ppi-data/affinity/ppb_clean" \
  --epochs 8 --batch 2 --lr 1e-5 --seed "$SEED" \
  --out "phase3/runs/mm_esmc_clean_s$SEED"
echo "----- done ESM-C CLEAN seed $SEED -----"
