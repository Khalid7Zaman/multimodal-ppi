#!/usr/bin/env bash
#SBATCH --job-name=mm_train_650M
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=phase3/logs/mm_train_650M_%j.out

# Definitive Phase 3 training: the full multimodal, multi-task model with the larger 650M ESM-2
# encoder and all three views active (sequence + structure + evolution). Trains the binding-affinity
# (pKd) and interface-residue heads; reports RMSE / Pearson / AUPR on the held-out validation set.
# Long 2-day limit so the clock never cuts it off. Runs on the RTX 6000D (or 5090) per plan.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
python phase3/train_phase3.py \
  --esm "$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D" \
  --epochs 8 --batch 2 --lr 1e-5 \
  --out phase3/runs/mm_650M_v1
echo "----- done -----"
