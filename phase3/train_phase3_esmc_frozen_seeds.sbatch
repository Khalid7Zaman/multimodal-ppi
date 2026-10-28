#!/usr/bin/env bash
#SBATCH --job-name=mmCf_seeds
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmCf_seed%a_%A.out

# Phase B: FROZEN-backbone counterpart of the ESM-C multimodal runs.
# The pretrained ESM-C backbone is used as a fixed feature extractor (--freeze_backbone); only the
# fitting network (projection, structure, evolution, cross-attention and the three heads) is trained.
# Everything else matches the fine-tuned runs (ESM-C, all three views, 8 epochs, batch 2, 3 seeds,
# same held-out TEST scoring) so the only difference is freeze vs fine-tune.
#
# Learning rate: 1e-4 here, ten times the fine-tuned 1e-5. The low 1e-5 exists to protect the
# pretrained backbone weights during fine-tuning; with the backbone frozen, only the randomly
# initialised fitting network trains, and it learns better at a higher rate. This gives the frozen
# model a fair chance rather than handicapping it with a rate chosen for a different purpose.
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
SEED=$SLURM_ARRAY_TASK_ID
echo "=== ESM-C multimodal, FROZEN backbone, seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --backbone esmc --esm "$HOME/Projects/esm-c/ESMplusplus_large" \
  --freeze_backbone \
  --epochs 8 --batch 2 --lr 1e-4 --seed "$SEED" \
  --out "phase3/runs/mm_esmc_frozen_s$SEED"
echo "----- done ESM-C FROZEN seed $SEED -----"
