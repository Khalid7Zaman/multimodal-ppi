#!/usr/bin/env bash
#SBATCH --job-name=bench_ppi
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=phase5/logs/bench_ppi_%j.out

# Phase 5 - interaction benchmark on the leakage-free Bernett test set (52,048 pairs).
# Two models are scored on the IDENTICAL test set, and each reports AUROC / AUPR:
#   (1) our CrossAttnPPI-650M            (phase2/runs/ppi_650M_v1/best.pt)
#   (2) the published PLM-interact-650M, used as released (external reference point)
# Same test pairs + same two metrics = a fair, apples-to-apples head-to-head.
# Inference only (no training). Generous 1-day limit so nothing is cut off.

eval "$(conda shell.bash hook)"; conda activate mmppi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
BTEST=$HOME/Projects/ppi-data/interaction/bernett/bernett_test.csv
RES=$HOME/Projects/ppi-data/phase5_bench
mkdir -p "$RES" "$HOME/Projects/multimodal-ppi/phase5/logs"

nvidia-smi
echo "=================================================================="
echo " (1) OUR MODEL  -  CrossAttnPPI-650M  on Bernett test"
echo "=================================================================="
cd "$HOME/Projects/multimodal-ppi"
python phase2/eval_ppi.py \
  --esm "$ESM" \
  --ckpt phase2/runs/ppi_650M_v1/best.pt \
  --test "$BTEST" \
  --max_length 512 --batch 8 2>&1 | tee "$RES/ours_bernett_test.txt"

echo "=================================================================="
echo " (2) COMPARISON  -  PLM-interact-650M (as released)  on the SAME test"
echo "=================================================================="
cd "$HOME/Projects/PLM-interact"
python benchmark_infer.py \
  --test "$BTEST" \
  --out "$RES/plm_bernett_scored.csv" \
  --batch 4 2>&1 | tee "$RES/plm_bernett_test.txt"

echo "----- interaction benchmark complete -----"
echo "summaries: $RES/ours_bernett_test.txt  and  $RES/plm_bernett_test.txt"
