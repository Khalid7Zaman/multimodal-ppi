#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa_finish
#SBATCH --partition=gpu_6000D,cpu05_384
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=phase3/logs/ppb_msa_finish_%j.out

# ---------------------------------------------------------------------------
# Finish the MSA build from the SAVED alignment (res.*). Do NOT re-run the 2h search.
#
# WHY THIS VERSION: running result2msa against the FULL 58.9M-sequence UniRef50
# needs ~500+ GB RAM (it OOM-killed at both 320G and 480G). Fix: the saved res.*
# already record exactly which target sequences were hit, so we build a REDUCED
# target DB of only those sequences and run result2msa against it (~1/6 the size,
# comfortably under 360G). This is the standard memory-reduction approach.
#
# Steps:  A) reduced target DB from hit keys   3) result2msa   4) unpack a3m
# Resumable: step A is skipped if the reduced DB already exists.
# Memory is now modest, so it can run on gpu05 (RTX 6000D) OR the big CPU node cpu05.
# ---------------------------------------------------------------------------
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
DB="$DATA/uniref50_db"
SUB="$WORK/hitdb"                 # reduced target DB: only the hit sequences
THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "node: $(hostname)   threads: $THREADS"
echo "--- free memory at start (GB) ---"; free -g
mkdir -p "$A3M" phase3/logs

# ---- Step A: build the reduced target DB (only sequences actually hit) ----
if [ ! -s "${SUB}.dbtype" ]; then
  echo "[A] collecting unique target keys from saved alignments (res.*) ..."
  # Each res data file holds, per query, NUL-separated blocks of alignment lines;
  # column 1 of each line is the target sequence's internal DB key.
  cat "$WORK"/res.[0-9]* | tr '\0' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -un > "$WORK/hit_keys.txt"
  NKEYS=$(wc -l < "$WORK/hit_keys.txt")
  echo "    unique hit sequences: $NKEYS"
  if [ "$NKEYS" -lt 1000 ]; then
    echo "    ERROR: too few hit keys ($NKEYS) - res.* may be unreadable. Aborting."; exit 1
  fi
  echo "[A] building reduced sequence DB ..."
  mmseqs createsubdb "$WORK/hit_keys.txt" "$DB"     "$SUB"
  echo "[A] building reduced header DB ..."
  mmseqs createsubdb "$WORK/hit_keys.txt" "${DB}_h" "${SUB}_h"
  echo "[A] reduced DB ready:"; ls -lh "$WORK"/hitdb* | head
else
  echo "[A] reduced target DB already present -> skipping rebuild"
fi

# ---- Step 3: result2msa against the SMALL reduced DB (now fits in memory) ----
echo "[3/4] result2msa (reduced DB) ..."
rm -f "$WORK"/msadb.*
mmseqs result2msa "$WORK/qdb" "$SUB" "$WORK/res" "$WORK/msadb" \
  --db-load-mode 2 --threads "$THREADS"

# ---- Step 4: unpack per-protein a3m ----
echo "[4/4] unpack per-protein a3m ..."
mmseqs unpackdb "$WORK/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
