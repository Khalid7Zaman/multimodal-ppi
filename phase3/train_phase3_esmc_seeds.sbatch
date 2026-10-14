#!/usr/bin/env bash
#SBATCH --job-name=mmC_seeds
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mmC_seed%a_%A.out

# Three-seed runs of the multimodal model with the SELECTED ESM-C backbone (all three views),
# matching the ESM-2 650M settings exactly (8 epochs, batch 2, lr 1e-5) so the two backbones are
# directly comparable. Each seed trains, keeps the best-validation checkpoint, and scores it on
# the held-out TEST split (the manuscript number, saved to test_metrics.json). Runs in the
# dedicated esmc environment. 2-day limit so the clock never cuts a run off.
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
SEED=$SLURM_ARRAY_TASK_ID
echo "=== ESM-C multimodal, seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --backbone esmc --esm "$HOME/Projects/esm-c/ESMplusplus_large" \
  --epochs 8 --batch 2 --lr 1e-5 --seed "$SEED" \
  --out "phase3/runs/mm_esmc_s$SEED"
echo "----- done ESM-C seed $SEED -----"
