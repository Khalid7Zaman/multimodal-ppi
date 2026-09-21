#!/usr/bin/env bash
#SBATCH --job-name=mm_train
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=phase3/logs/mm_train_%j.out

# Full Phase 3 training: the multimodal, multi-task model on all PPB complexes.
# Trains the binding-affinity (pKd) and interface-residue heads; reports RMSE / Pearson / AUPR.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
python phase3/train_phase3.py \
  --esm "$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D" \
  --epochs 5 --batch 2 --lr 1e-5 \
  --out phase3/runs/mm_35M_v1
echo "----- done -----"
