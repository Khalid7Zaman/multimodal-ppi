#!/usr/bin/env bash
#SBATCH --job-name=mm650_seeds
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=phase3/logs/mm650_seed%a_%A.out

# Repeated-seed runs of the definitive 650M multimodal model (all three views), for error bars.
# The array runs three independent seeds (0, 1, 2). Each trains, keeps the best-validation
# checkpoint, and finally scores that checkpoint on the held-out TEST split (the manuscript
# number, saved to test_metrics.json). 2-day limit so the clock never cuts a run off.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SEED=$SLURM_ARRAY_TASK_ID
echo "=== seed $SEED ==="
nvidia-smi
python phase3/train_phase3.py \
  --esm "$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D" \
  --epochs 8 --batch 2 --lr 1e-5 --seed "$SEED" \
  --out "phase3/runs/mm_650M_s$SEED"
echo "----- done seed $SEED -----"
