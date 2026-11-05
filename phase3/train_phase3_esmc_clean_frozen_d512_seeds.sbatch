#!/usr/bin/env bash
#SBATCH --job-name=mmCcleanF
#SBATCH --partition=gpu_3080ti,gpu_5080,gpu_5070ti,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmCcleanF_seed%a_%A.out

# Phase C: honest affinity numbers on the LEAKAGE-FREE re-split, with the backbone FROZEN.
# The supervisor confirmed that ESM-C is the representation foundation and is frozen in the
# downstream tasks, so this is the adopted design. It matches the Phase B frozen width-512 run
# exactly (frozen backbone, d_model 512, lr 1e-4, all three views, 8 epochs, batch 2, three seeds),
# with the ONLY change being --ppb_dir, which points at the clean re-split. So the result is
# directly comparable to the Phase B frozen width-512 numbers on the original split
# (affinity Pearson 0.411, RMSE 1.574, interface AUPR 0.299): same model, only the leakage-free
# split differs, which isolates the effect of removing the leakage.
#
# A frozen run is light (a few GB of VRAM, a few hours), so it uses the broad partition set and runs
# on the smaller, usually-free cards without waiting for a large training GPU.
# Build the clean split first (CPU, seconds): python phase_c/resplit_affinity_clean.py
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
SEED=$SLURM_ARRAY_TASK_ID
echo "=== ESM-C multimodal, FROZEN backbone, CLEAN re-split, width 512, seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --backbone esmc --esm "$HOME/Projects/esm-c/ESMplusplus_large" \
  --freeze_backbone --d_model 512 \
  --ppb_dir "$HOME/Projects/ppi-data/affinity/ppb_clean" \
  --epochs 8 --batch 2 --lr 1e-4 --seed "$SEED" \
  --out "phase3/runs/mm_esmc_clean_frozen_d512_s$SEED"
echo "----- done ESM-C FROZEN CLEAN d512 seed $SEED -----"
