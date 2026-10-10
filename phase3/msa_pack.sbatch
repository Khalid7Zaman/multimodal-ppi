#!/usr/bin/env bash
#SBATCH --job-name=ppb_pack
#SBATCH --partition=gpu_6000D,cpu05_384,cpu03_192,cpu06_192,gpu_3080ti,gpu_5070ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=phase3/logs/ppb_pack_%j.out

# ---------------------------------------------------------------------------
# Build the MSAs from the SAVED search (res2) - in small BATCHES so result2msa
# never over-allocates memory. This means it runs on ANY free node (modest 96 GB),
# is multi-threaded (fast), and is RESUMABLE (a per-batch marker; re-submitting
# continues where it stopped). Long 3-day limit so the clock never cuts it off.
#
# Requires (already on disk from the weekend run): res2 (saved search) + hitdb_compact.
# ---------------------------------------------------------------------------
set -euo pipefail
shopt -s nullglob
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
CDB="$WORK/hitdb_compact"
B="$WORK/pack"                    # scratch + progress markers
T="${SLURM_CPUS_PER_TASK:-8}"
CHUNK=500                         # proteins per result2msa call

mkdir -p "$A3M" "$B" phase3/logs
echo "node: $(hostname)   threads: $T   chunk: $CHUNK"
echo "--- free memory (GB) ---"; free -g

[ -s "$WORK/res2.dbtype" ] || { echo "ERROR: res2 (saved search) missing"; exit 1; }
[ -s "${CDB}.dbtype" ]     || { echo "ERROR: compact DB missing"; exit 1; }

# key -> seq_id map (qdb.lookup: col1 numeric key, col2 accession = seq_id)
declare -A NAME
while IFS=$' \t' read -r k acc _; do NAME["$k"]="$acc"; done < "$WORK/qdb.lookup"

# ordered list of all query keys
awk '{print $1}' "$WORK/qdb.index" | sort -n > "$B/allkeys.txt"
NQ=$(wc -l < "$B/allkeys.txt")
echo "queries to build: $NQ   (batches of $CHUNK)"

i=0
while [ "$i" -lt "$NQ" ]; do
  s=$((i+1)); e=$((i+CHUNK)); [ "$e" -gt "$NQ" ] && e="$NQ"
  tag=$(printf "%06d" "$i")
  if [ -f "$B/done_$tag" ]; then echo "batch $tag already done -> skip"; i="$e"; continue; fi

  echo "[$(date +%H:%M:%S)] batch $tag : proteins $s..$e"
  sed -n "${s},${e}p" "$B/allkeys.txt" > "$B/k_$tag"
  mmseqs createsubdb "$B/k_$tag" "$WORK/qdb"   "$B/q_$tag"     >/dev/null
  mmseqs createsubdb "$B/k_$tag" "$WORK/qdb_h" "$B/q_${tag}_h" >/dev/null
  mmseqs result2msa  "$B/q_$tag" "$CDB" "$WORK/res2" "$B/m_$tag" --threads "$T" >/dev/null
  mmseqs unpackdb    "$B/m_$tag" "$B/o_$tag" --unpack-name-mode 0 --unpack-suffix .a3m >/dev/null

  for f in "$B/o_$tag"/*.a3m; do
    k=$(basename "$f" .a3m); sid="${NAME[$k]:-}"
    [ -n "$sid" ] && mv -f "$f" "$A3M/$sid.a3m"
  done
  rm -rf "$B/q_$tag"* "$B/m_$tag"* "$B/o_$tag" "$B/k_$tag"
  touch "$B/done_$tag"
  echo "   a3m files so far: $(ls "$A3M" | wc -l)"
  i="$e"
done

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
