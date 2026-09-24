#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa_test
#SBATCH --partition=cpu03_192,cpu06_192,gpu_3080ti,gpu_5070ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=phase3/logs/ppb_msa_test_%j.out

# ---------------------------------------------------------------------------
# SMALL PROOF TEST of the compact-DB fix, on just 30 proteins.
# Goal: confirm (1) memory stays low (the fix works) and (2) the a3m come out correct,
# BEFORE committing to the full 9,516-protein run. Writes to a separate test folder so
# it never touches the real outputs.
# ---------------------------------------------------------------------------
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
DB="$DATA/uniref50_db"
SUB="$WORK/hitdb"
CDB="$WORK/hitdb_compact"
TEST="$WORK/_test"                 # everything for the test lives here
T="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$TEST" phase3/logs
echo "node: $(hostname)   threads: $T"
echo "--- free memory at start (GB) ---"; free -g

# reduced DB (from hit keys) — build if missing
if [ ! -s "${SUB}.dbtype" ]; then
  echo "[A] building reduced DB ..."
  cat "$WORK"/res.[0-9]* | tr '\0' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -un > "$WORK/hit_keys.txt"
  mmseqs createsubdb "$WORK/hit_keys.txt" "$DB"     "$SUB"     >/dev/null
  mmseqs createsubdb "$WORK/hit_keys.txt" "${DB}_h" "${SUB}_h" >/dev/null
fi

# compact-key DB — the fix. build if missing.
if [ ! -s "${CDB}.dbtype" ]; then
  echo "[B] building compact-key DB (convert2fasta -> createdb) ..."
  mmseqs convert2fasta "$SUB" "$WORK/hitdb.fasta" >/dev/null
  mmseqs createdb "$WORK/hitdb.fasta" "$CDB" >/dev/null
fi
echo "    compact DB entries: $(wc -l < "${CDB}.index")"

# make a 30-query sub-DB from the first 30 query keys
awk '{print $1}' "$WORK/qdb.index" | sort -n | head -30 > "$TEST/keys30.txt"
mmseqs createsubdb "$TEST/keys30.txt" "$WORK/qdb"   "$TEST/q30"    >/dev/null
mmseqs createsubdb "$TEST/keys30.txt" "$WORK/qdb_h" "$TEST/q30_h"  >/dev/null

echo "[C] search 30 queries vs compact DB ..."
rm -rf "$TEST/tmp"; mkdir -p "$TEST/tmp"
mmseqs search "$TEST/q30" "$CDB" "$TEST/res" "$TEST/tmp" \
  --threads "$T" -s 5.7 --max-seqs 1000 -e 0.001

echo "[D] result2msa (watch: no 240GB allocation, no Killed) ..."
rm -f "$TEST"/msadb.*
mmseqs result2msa "$TEST/q30" "$CDB" "$TEST/res" "$TEST/msadb" --threads "$T"

echo "[E] unpack ..."
rm -rf "$TEST/a3m"; mkdir -p "$TEST/a3m"
mmseqs unpackdb "$TEST/msadb" "$TEST/a3m" --unpack-name-mode 1 --unpack-suffix .a3m

echo "--- free memory at end (GB) ---"; free -g
N=$(ls "$TEST/a3m" | wc -l)
echo "RESULT: produced $N / 30 a3m files in $TEST/a3m"
one=$(ls "$TEST/a3m"/*.a3m 2>/dev/null | head -1)
if [ -n "$one" ]; then
  echo "Example $one : $(grep -c '^>' "$one") sequences in the alignment"
  echo "First 4 lines:"; head -4 "$one"
fi
echo "----- test done -----"
