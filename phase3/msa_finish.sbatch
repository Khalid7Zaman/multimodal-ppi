#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa_finish
#SBATCH --partition=gpu_6000D,cpu05_384
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=phase3/logs/ppb_msa_finish_%j.out

# ---------------------------------------------------------------------------
# Finish the MSA build from the SAVED alignment (res.*). Do NOT re-run the 2h search.
#
# LESSON (learned across jobs 289477 / 290129 / 290164):
#   result2msa's memory is driven by HOW MANY QUERIES are packaged at once, not by
#   the target-DB size. Running all 9,516 proteins in one call OOM'd even against a
#   594 MB reduced DB. FIX: process the queries in small CHUNKS, so only a few hundred
#   MSAs live in memory at a time.
#
# Pipeline:
#   [A] reduced target DB (only hit sequences)     -- built once, then skipped
#   [B] for each chunk of CHUNK queries:
#         result2msa (chunk vs reduced DB) -> unpack -> name each a3m by its seq_id
#   Resumable: a per-chunk "done" marker means a re-submit continues where it left off.
#   Memory is now modest, so it schedules on gpu05 (RTX 6000D) or the big CPU node cpu05.
# ---------------------------------------------------------------------------
set -euo pipefail
shopt -s nullglob
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
DB="$DATA/uniref50_db"
SUB="$WORK/hitdb"                 # reduced target DB (only sequences that were hit)
B="$WORK/batch"                   # scratch for per-chunk sub-DBs + progress markers
THREADS="${SLURM_CPUS_PER_TASK:-32}"
CHUNK=500                         # queries packaged per result2msa call

mkdir -p "$A3M" "$B" phase3/logs
echo "node: $(hostname)   threads: $THREADS   chunk: $CHUNK"
echo "--- free memory at start (GB) ---"; free -g

# ---- Step A: reduced target DB (only the sequences actually hit) ----
if [ ! -s "${SUB}.dbtype" ]; then
  echo "[A] collecting unique target keys from saved alignments (res.*) ..."
  cat "$WORK"/res.[0-9]* | tr '\0' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -un > "$WORK/hit_keys.txt"
  NK=$(wc -l < "$WORK/hit_keys.txt"); echo "    unique hit sequences: $NK"
  [ "$NK" -ge 1000 ] || { echo "    ERROR: too few hit keys ($NK). Aborting."; exit 1; }
  mmseqs createsubdb "$WORK/hit_keys.txt" "$DB"     "$SUB"     >/dev/null
  mmseqs createsubdb "$WORK/hit_keys.txt" "${DB}_h" "${SUB}_h" >/dev/null
  echo "[A] reduced DB ready ($(du -sh "$SUB" | cut -f1))"
else
  echo "[A] reduced target DB already present -> skip rebuild"
fi

# ---- key -> seq_id map (qdb.lookup: col1=key, col2=accession=seq_id) ----
declare -A NAME
while IFS=$' \t' read -r k acc _; do NAME["$k"]="$acc"; done < "$WORK/qdb.lookup"

# ---- ordered list of all query keys (robust; does not assume 0..N-1) ----
awk '{print $1}' "$WORK/qdb.index" | sort -n > "$B/all_qkeys.txt"
NQ=$(wc -l < "$B/all_qkeys.txt")
echo "[B] queries to process: $NQ   (chunks of $CHUNK)"

# ---- Step B: process queries in chunks ----
i=0
while [ "$i" -lt "$NQ" ]; do
  s=$(( i + 1 )); e=$(( i + CHUNK )); [ "$e" -gt "$NQ" ] && e="$NQ"
  tag=$(printf "%06d" "$i")
  marker="$B/done_$tag"
  if [ -f "$marker" ]; then echo "  chunk $tag already done -> skip"; i="$e"; continue; fi

  echo "  [$(date +%H:%M:%S)] chunk $tag : queries $s..$e"
  sed -n "${s},${e}p" "$B/all_qkeys.txt" > "$B/keys_$tag"

  mmseqs createsubdb "$B/keys_$tag" "$WORK/qdb"   "$B/q_$tag"     >/dev/null
  mmseqs createsubdb "$B/keys_$tag" "$WORK/qdb_h" "$B/q_${tag}_h" >/dev/null
  mmseqs result2msa  "$B/q_$tag" "$SUB" "$WORK/res" "$B/msa_$tag" \
        --db-load-mode 2 --threads "$THREADS" >/dev/null
  mmseqs unpackdb    "$B/msa_$tag" "$B/out_$tag" --unpack-name-mode 0 --unpack-suffix .a3m >/dev/null

  # name each a3m by its protein's seq_id (files come out named by numeric key)
  for f in "$B/out_$tag"/*.a3m; do
    k=$(basename "$f" .a3m); sid="${NAME[$k]:-}"
    [ -n "$sid" ] && mv -f "$f" "$A3M/$sid.a3m"
  done

  rm -rf "$B/q_$tag"* "$B/msa_$tag"* "$B/out_$tag" "$B/keys_$tag"
  touch "$marker"
  echo "      a3m files so far: $(ls "$A3M" | wc -l)"
  i="$e"
done

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
