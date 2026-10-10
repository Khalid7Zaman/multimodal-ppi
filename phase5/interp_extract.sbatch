#!/usr/bin/env bash
#SBATCH --job-name=interp
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=phase5/logs/interp_%j.out

# Phase 5 Part C - extract interpretability data (cross-attention + interface probs + coords)
# for a few example test complexes, using the definitive additive 650M checkpoint (seed 0).
# Inference only. Saves small .npz files to phase5/interp/ ; then run phase5/interp_plot.py.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs phase5/interp

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
nvidia-smi
python phase5/interp_extract.py \
  --esm "$ESM" --ckpt phase3/runs/mm_650M_s0/best.pt --k 3 --max_res 400
echo "----- interpretability extract done; now run: python phase5/interp_plot.py -----"
