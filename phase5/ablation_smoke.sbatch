#!/usr/bin/env bash
#SBATCH --job-name=abl_smoke
#SBATCH --partition=gpu_5070ti,gpu_5080,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=phase5/logs/abl_smoke_%j.out

# Quick correctness check for the new ablation switches, BEFORE the big 650M run.
# Uses the small 35M encoder, sequence-only (structure + evolution disabled), overfits
# 8 complexes for 1 epoch. If this finishes cleanly it proves the --no_struct / --no_evo
# flags, the dataset switches, and the forward/backward all work on real data.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs

ESM35=$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D
nvidia-smi
echo "=== smoke test: sequence-only (no structure, no evolution), 35M, overfit 8, 1 epoch ==="
python phase3/train_phase3.py --esm "$ESM35" \
  --no_struct --no_evo --overfit --limit 8 --epochs 1 --batch 2 \
  --out phase3/runs/abl_smoke
echo "----- smoke test finished -----"
