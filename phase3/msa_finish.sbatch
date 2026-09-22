#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa_finish
#SBATCH --partition=gpu_6000D
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=phase3/logs/ppb_msa_finish_%j.out

# Resume the MSA build from the SAVED alignment (res.*). Do NOT re-run the 2h search.
# Steps 3-4 only: result2msa (pack alignments -> MSAs) + unpackdb (one .a3m per protein).
# Runs on gpu05 (RTX 6000D, 515 GB) with 480 GB - the headroom the packaging step needed.
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
DB="$DATA/uniref50_db"
THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "node: $(hostname)   threads: $THREADS"
echo "--- free memory at start (GB) ---"; free -g
mkdir -p "$A3M" phase3/logs

# Remove the 0-byte leftovers from the failed attempt so this starts clean.
rm -f "$WORK"/msadb.*

echo "[3/4] result2msa  (from the saved res files) ..."
mmseqs result2msa "$WORK/qdb" "$DB" "$WORK/res" "$WORK/msadb" \
  --db-load-mode 2 --threads "$THREADS"

echo "[4/4] unpack per-protein a3m ..."
mmseqs unpackdb "$WORK/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
