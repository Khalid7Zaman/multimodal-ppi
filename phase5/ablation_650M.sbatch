#!/usr/bin/env bash
#SBATCH --job-name=abl650
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=phase5/logs/abl650_%a_%A.out

# Phase 5 ablation at the definitive 650M scale, on the held-out PPB-Affinity test split.
# Two conditions x three seeds = 6 runs (the third condition, the full sequence+structure+
# evolution model, is already available from phase3/runs/mm_650M_s0/1/2 and is reused by
# agg_ablation.py). Same hyper-parameters as the definitive run (8 epochs, batch 2, lr 1e-5).
#   array index 0,1,2 -> sequence only        (--no_struct --no_evo)  seeds 0,1,2
#   array index 3,4,5 -> sequence + structure (--no_evo)              seeds 0,1,2
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
COND=$(( SLURM_ARRAY_TASK_ID / 3 ))
if [ "$COND" -eq 0 ]; then
  FLAGS="--no_struct --no_evo"; NAME="seq"
else
  FLAGS="--no_evo";             NAME="seqstruct"
fi
OUT="phase3/runs/abl_${NAME}_s${SEED}"

nvidia-smi
echo "=== ablation condition '$NAME'  seed $SEED  ->  $OUT ==="
python phase3/train_phase3.py \
  --esm "$ESM" --epochs 8 --batch 2 --lr 1e-5 --seed "$SEED" \
  $FLAGS --out "$OUT"
echo "----- done: $NAME seed $SEED -----"
