#!/usr/bin/env bash
#SBATCH --job-name=mmCf_seeds
#SBATCH --partition=gpu_3080ti,gpu_5080,gpu_5070ti,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmCf_seed%a_%A.out

# Phase B: FROZEN-backbone counterpart of the ESM-C multimodal runs.
# The pretrained ESM-C backbone is used as a fixed feature extractor (--freeze_backbone); only the
# fitting network (projection, structure, evolution, cross-attention and the three heads) is trained.
# Everything else matches the fine-tuned runs (ESM-C, all three views, 8 epochs, batch 2, 3 seeds,
# same held-out TEST scoring) so the only difference is freeze vs fine-tune.
#
# A frozen run keeps no backbone gradients or optimizer state, so it needs only a few GB of GPU memory
# and finishes in a few hours. It therefore runs well on the smaller, more-available cards (3080 Ti),
# which is why the partition list is broad and the memory and walltime are modest. The large training
# cards (gpu_6000D / gpu_5090) stay in the list as a fallback.
#
# Learning rate: 1e-4, ten times the fine-tuned 1e-5. The low 1e-5 exists to protect the pretrained
# backbone weights during fine-tuning; with the backbone frozen, only the randomly initialised fitting
# network trains, and it learns better at a higher rate.
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
