#!/usr/bin/env bash
# Phase C block two, step 1: download STRING physical-interaction data per organism.
#
# STRING is the reported-interaction database that guides pair selection (the supervisor's point 3).
# We take the PHYSICAL subnetwork (proteins that physically interact, not merely co-mentioned) and,
# later, keep only high-confidence edges as the positive pairs. For each organism this fetches two
# files into ~/Projects/ppi-data/string/:
#   <taxid>.protein.physical.links.v12.0.txt.gz   edges: protein1 protein2 combined_score
#   <taxid>.protein.sequences.v12.0.fa.gz         the protein sequences for that organism
# Network only, no GPU.
#
# FIRST run it on ONE small organism to confirm the cluster can reach STRING and to see the format:
#   bash phase_c/block2_fetch_string.sh 511145
# Then, once that works, run it with no arguments to fetch the default diverse organism set:
#   bash phase_c/block2_fetch_string.sh
#
# If the default host is unreachable from the cluster, point it at a mirror and retry, e.g.:
#   STRING_BASE=https://cn.string-db.org/download bash phase_c/block2_fetch_string.sh 511145
set -uo pipefail

VER="12.0"
BASE="${STRING_BASE:-https://stringdb-downloads.org/download}"
OUT="$HOME/Projects/ppi-data/string"
mkdir -p "$OUT"

# A diverse set across the tree of life, for broad sequence-space coverage:
# animals (human 9606, mouse 10090, zebrafish 7955, fly 7227, worm 6239), plant (Arabidopsis 3702),
# fungi (budding yeast 4932, fission yeast 284812), bacteria (E. coli 511145, B. subtilis 224308,
# M. tuberculosis 83332, P. aeruginosa 208964, Synechocystis 1148), archaeon (M. jannaschii 243232),
# amoeba (Dictyostelium 44689).
DEFAULT_TAXA="9606 10090 7955 7227 6239 3702 4932 284812 511145 224308 83332 208964 1148 243232 44689"
TAXA="${*:-$DEFAULT_TAXA}"

ok=0; fail=0
for t in $TAXA; do
  for kind in "protein.physical.links" "protein.sequences"; do
    if [ "$kind" = "protein.sequences" ]; then ext="fa.gz"; else ext="txt.gz"; fi
    url="$BASE/$kind.v$VER/$t.$kind.v$VER.$ext"
    dst="$OUT/$t.$kind.v$VER.$ext"
    if [ -s "$dst" ]; then echo "have   $dst"; ok=$((ok+1)); continue; fi
    echo "fetch  $url"
    if curl -fSL --retry 3 --max-time 600 -o "$dst" "$url"; then
      ok=$((ok+1))
    else
      echo "FAILED $url" >&2; rm -f "$dst"; fail=$((fail+1))
    fi
  done
done

echo
echo "downloaded ok: $ok, failed: $fail  -> $OUT"
ls -lh "$OUT" 2>/dev/null | head -20
echo
echo "To inspect what one organism's files contain (format check):"
echo "  zcat $OUT/511145.protein.physical.links.v12.0.txt.gz | head -4"
echo "  zcat $OUT/511145.protein.sequences.v12.0.fa.gz | head -4"
