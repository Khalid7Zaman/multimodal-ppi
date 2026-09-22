#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa_finish
#SBATCH --partition=gpu_6000D
#SBATCH --cpus-per-task=8
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=phase3/logs/ppb_msa_finish_%j.out

# ---------------------------------------------------------------------------
# Finish the MSA build from the SAVED alignment (res.*). Do NOT re-run the 2h search.
#
# ROOT CAUSE (finally pinned down, job 290755):
#   result2msa allocates a fixed ~240 GB block PER THREAD (sized from UniRef50's key
#   space, not from how many queries or how big the target DB is). So memory = threads x 240GB.
#   With many threads it needed terabytes and always died.
# FIX: run result2msa SINGLE-THREADED -> one 240 GB block -> fits on one big-memory node
#   (cpu05 = 380 GB, or gpu05 = 515 GB). Slower, but it completes. No batching needed
#   (the 240 GB is fixed regardless of query count), so one clean pass.
# Resumable: step A (reduced DB) is skipped if already built; if result2msa finishes but
#   unpack is interrupted, re-running redoes only the cheap unpack.
# ---------------------------------------------------------------------------
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
DB="$DATA/uniref50_db"
SUB="$WORK/hitdb"           # reduced target DB (already built; smaller to memory-map)
RTHREADS=1                  # result2msa: ONE thread (each thread wants ~240 GB)

mkdir -p "$A3M" phase3/logs
echo "node: $(hostname)   result2msa threads: $RTHREADS"
echo "--- free memory at start (GB) ---"; free -g

# ---- Step A: reduced target DB (only sequences hit) -- skip if already present ----
if [ ! -s "${SUB}.dbtype" ]; then
  echo "[A] collecting unique target keys ..."
  cat "$WORK"/res.[0-9]* | tr '\0' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -un > "$WORK/hit_keys.txt"
  NK=$(wc -l < "$WORK/hit_keys.txt"); echo "    unique hit sequences: $NK"
  [ "$NK" -ge 1000 ] || { echo "    ERROR: too few hit keys ($NK)."; exit 1; }
  mmseqs createsubdb "$WORK/hit_keys.txt" "$DB"     "$SUB"     >/dev/null
  mmseqs createsubdb "$WORK/hit_keys.txt" "${DB}_h" "${SUB}_h" >/dev/null
  echo "[A] reduced DB ready"
else
  echo "[A] reduced target DB already present -> skip rebuild"
fi

# ---- Step 3: result2msa, SINGLE-THREADED (one 240 GB block, fits one big node) ----
echo "[3/4] result2msa (single-threaded) ... this is the slow step; please be patient."
rm -f "$WORK"/msadb.*
/usr/bin/time -v mmseqs result2msa "$WORK/qdb" "$SUB" "$WORK/res" "$WORK/msadb" \
  --db-load-mode 2 --threads "$RTHREADS" 2> "$WORK/result2msa_time.txt" || {
    echo "result2msa failed; tail of its resource log:"; tail -n 20 "$WORK/result2msa_time.txt"; exit 1; }
echo "    result2msa done. Peak memory (from time -v):"
grep "Maximum resident" "$WORK/result2msa_time.txt" || true

# ---- Step 4: unpack per-protein a3m ----
echo "[4/4] unpack per-protein a3m ..."
mmseqs unpackdb "$WORK/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m >/dev/null

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
