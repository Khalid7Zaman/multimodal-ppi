#!/usr/bin/env bash
#SBATCH --job-name=strat_eval
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=phase5/logs/strat_%j.out

# Phase 5 Part B2 - stratified analysis (inference only; reuses the trained checkpoints).
# 1) build per-complex metadata for the test split (has_structure, MSA depth)
# 2) write per-complex affinity predictions for all 9 checkpoints (3 conditions x 3 seeds),
#    each evaluated with the SAME view switches it was trained with.
# Then on the login node:  python phase5/agg_stratified.py
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
nvidia-smi

echo "=================== (1) test-split metadata ==================="
python phase5/make_test_meta.py

echo "=================== (2) per-complex predictions ==============="
for S in 0 1 2; do
  echo "--- seed $S : sequence only ---"
  python phase5/eval_perrow.py --esm "$ESM" --ckpt phase3/runs/abl_seq_s$S/best.pt \
     --no_struct --no_evo --out phase3/runs/abl_seq_s$S/test_perrow.json
  echo "--- seed $S : + structure ---"
  python phase5/eval_perrow.py --esm "$ESM" --ckpt phase3/runs/abl_seqstruct_s$S/best.pt \
     --no_evo --out phase3/runs/abl_seqstruct_s$S/test_perrow.json
  echo "--- seed $S : + evolution (full) ---"
  python phase5/eval_perrow.py --esm "$ESM" --ckpt phase3/runs/mm_650M_s$S/best.pt \
     --out phase3/runs/mm_650M_s$S/test_perrow.json
done

echo "=================== done ==================="
echo "now run on the login node:  python phase5/agg_stratified.py"
