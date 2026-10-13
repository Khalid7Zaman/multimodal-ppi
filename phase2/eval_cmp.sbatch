#!/usr/bin/env bash
#SBATCH --job-name=eval_cmp
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=phase2/logs/eval_cmp_%j.out

# Score the ESM-2 and ESM-C interaction models on the leakage-free Bernett test set (52,048
# pairs) with the full metric set (AUPR, AUROC, Precision, Recall, F1, Accuracy, MCC). Inference
# only. Saves per-pair predictions and a metrics JSON per model. Runs in the esmc environment.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs phase5/metrics_full

ESM2=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
ESMC=$HOME/Projects/esm-c/ESMplusplus_large
BTEST=$HOME/Projects/ppi-data/interaction/bernett/bernett_test.csv

echo "========== EVAL: ESM-2 (cmp_esm2_s0) =========="
python phase2/eval_ppi_full.py --backbone esm2 --esm "$ESM2" \
  --ckpt phase2/runs/cmp_esm2_s0/best.pt --test "$BTEST" \
  --out phase5/metrics_full --tag cmp_esm2_s0 --batch 16

echo "========== EVAL: ESM-C (cmp_esmc_s0) =========="
python phase2/eval_ppi_full.py --backbone esmc --esm "$ESMC" \
  --ckpt phase2/runs/cmp_esmc_s0/best.pt --test "$BTEST" \
  --out phase5/metrics_full --tag cmp_esmc_s0 --batch 16

echo "----- evaluation complete: metrics in phase5/metrics_full/ -----"
