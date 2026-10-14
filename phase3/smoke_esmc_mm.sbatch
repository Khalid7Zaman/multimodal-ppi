#!/usr/bin/env bash
#SBATCH --job-name=mmC_smoke
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=phase3/logs/mmC_smoke_%j.out

# De-risk the ESM-C multimodal run: verify the per-residue alignment contract holds with the
# ESM-C tokenizer and that MultiModalPPI trains end-to-end through the ESM-C backbone. Runs in
# the dedicated esmc environment.
eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase3/logs

python phase3/smoke_esmc_mm.py --esm "$HOME/Projects/esm-c/ESMplusplus_large" --n 4

echo "----- ESM-C multimodal smoke done -----"
