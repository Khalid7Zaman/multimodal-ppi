#!/usr/bin/env bash
#SBATCH --job-name=mm_sanity
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=phase3/logs/mm_sanity_%j.out

# Sanity check: can the multimodal model learn? Overfit a tiny 16-complex set (loss should fall,
# metrics should climb). Proves the whole pipeline (data + model + losses) is wired correctly,
# exactly like the Phase 2 overfit test.
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
python phase3/train_phase3.py \
  --esm "$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D" \
  --overfit --limit 16 --epochs 60 --batch 2 --lr 5e-5 \
  --out phase3/runs/mm_overfit
echo "----- done -----"
