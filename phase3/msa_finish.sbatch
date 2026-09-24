#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa
#SBATCH --partition=cpu03_192,cpu06_192,gpu_3080ti,gpu_5070ti
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=phase3/logs/ppb_msa_finish_%j.out

# ---------------------------------------------------------------------------
# Build the MSAs — FAST version that runs on ordinary (non-giant) nodes.
#
# ROOT CAUSE of every earlier failure: result2msa allocates a fixed ~240 GB block
# PER THREAD, sized from UniRef50's full 58.9M key space — even against a reduced DB,
# because the alignment results still reference that huge key range. That forced
# single-thread + a >380 GB node, and both big nodes are now saturated by other users.
#
# FIX: rebuild the hit sequences into a COMPACT database (fresh createdb -> keys 0..844k),
# re-search the queries against it (tiny + fast), then result2msa against the compact DB.
# The 240 GB block shrinks to a few GB, so this runs MULTI-THREADED with modest memory
# on any ordinary free node, in well under an hour.
#
# Resumable: the compact DB (step B) is skipped if already built.
# ---------------------------------------------------------------------------
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
DB="$DATA/uniref50_db"
SUB="$WORK/hitdb"            # reduced DB (only hit sequences; original 58.9M-scale keys)
CDB="$WORK/hitdb_compact"    # compact-key DB (keys 0..844k) -> shrinks the allocation
T="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$A3M" phase3/logs
echo "node: $(hostname)   threads: $T"
echo "--- free memory at start (GB) ---"; free -g

# ---- Step A: reduced DB from the saved alignment's hit keys (skip if present) ----
if [ ! -s "${SUB}.dbtype" ]; then
  echo "[A] collecting hit keys + building reduced DB ..."
  cat "$WORK"/res.[0-9]* | tr '\0' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -un > "$WORK/hit_keys.txt"
  mmseqs createsubdb "$WORK/hit_keys.txt" "$DB"     "$SUB"     >/dev/null
  mmseqs createsubdb "$WORK/hit_keys.txt" "${DB}_h" "${SUB}_h" >/dev/null
else
  echo "[A] reduced DB present -> skip"
fi

# ---- Step B: COMPACT-key DB (FASTA -> fresh createdb; keys renumbered 0..N-1) ----
if [ ! -s "${CDB}.dbtype" ]; then
  echo "[B] building compact-key DB (this is the memory fix) ..."
  mmseqs convert2fasta "$SUB" "$WORK/hitdb.fasta" >/dev/null
  mmseqs createdb "$WORK/hitdb.fasta" "$CDB" >/dev/null
  echo "    compact DB entries: $(wc -l < "${CDB}.index")"
else
  echo "[B] compact DB present -> skip"
fi

# ---- Step C: re-search queries vs the small compact DB (fast, low memory, threaded) ----
echo "[C] search 9,516 queries vs compact DB ..."
rm -rf "$WORK/tmp2"; mkdir -p "$WORK/tmp2"
mmseqs search "$WORK/qdb" "$CDB" "$WORK/res2" "$WORK/tmp2" \
  --threads "$T" -s 5.7 --max-seqs 1000 -e 0.001

# ---- Step D: result2msa (now low memory, multi-threaded) ----
echo "[D] result2msa ..."
rm -f "$WORK"/msadb2.*
mmseqs result2msa "$WORK/qdb" "$CDB" "$WORK/res2" "$WORK/msadb2" --threads "$T"

# ---- Step E: unpack one a3m per protein ----
echo "[E] unpack per-protein a3m ..."
mmseqs unpackdb "$WORK/msadb2" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
