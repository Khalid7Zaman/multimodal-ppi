#!/usr/bin/env bash
#SBATCH --job-name=gated650
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=phase5/logs/gated650_%a_%A.out

# Phase 5 Part B3 - the gated-fusion model at 650M, all three views, 3 seeds, on the held-out
# test split. Same hyper-parameters as the definitive additive run (8 epochs, batch 2, lr 1e-5),
# so the ONLY difference from mm_650M is the fusion mechanism. Compare with agg_gated.py.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
SEED=$SLURM_ARRAY_TASK_ID

nvidia-smi
echo "=== gated fusion, all views, 650M, seed $SEED ==="
python phase3/train_phase3.py \
  --esm "$ESM" --epochs 8 --batch 2 --lr 1e-5 --seed "$SEED" \
  --fusion gated --out "phase3/runs/mm_650M_gated_s$SEED"
echo "----- done: gated seed $SEED -----"
