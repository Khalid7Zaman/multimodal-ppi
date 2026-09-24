#!/usr/bin/env bash
#SBATCH --job-name=ppb_verify
#SBATCH --partition=gpu_6000D,cpu03_192,cpu06_192
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=phase3/logs/ppb_verify_%j.out

# ---------------------------------------------------------------------------
# Phase 3 - verify the MSAs and validate the data loader on the REAL MSAs.
# Runs automatically after the MSA job (via --dependency=afterok). CPU-only, backfills.
#   1) count a3m files (expect 9,516) and report the depth distribution (rigor check),
#   2) run data.py self-test on real MSAs (train + val) to confirm the loader assembles
#      sequence + structure + evolution correctly.
# Writes a report to reports/PHASE3_MSA_VERIFY.txt. Exits non-zero (stopping the chain)
# only if the MSAs are clearly incomplete, so we never train on broken data.
# ---------------------------------------------------------------------------
set -euo pipefail
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p reports phase3/logs

A3M="$HOME/Projects/ppi-data/msa/full/a3m"
REP="reports/PHASE3_MSA_VERIFY.txt"

echo "=== Phase 3 MSA verification  ($(date)) ===" | tee "$REP"
N=$(ls "$A3M" 2>/dev/null | wc -l)
echo "a3m files present: $N / 9516" | tee -a "$REP"

# depth distribution over a random 300-file sample (sequences per MSA = count of '>' lines)
ls "$A3M"/*.a3m | shuf -n 300 | while read -r f; do
  awk '/^>/{n++} END{print n+0}' "$f"
done > /tmp/depths_$$.txt
python - "/tmp/depths_$$.txt" <<'PY' | tee -a "$REP"
import sys, numpy as np
d = np.loadtxt(sys.argv[1])
print(f"sampled {len(d)} MSAs | depth  min {int(d.min())}  median {int(np.median(d))}  mean {d.mean():.0f}  max {int(d.max())}")
print(f"MSAs with >= 2 sequences (a real alignment): {(d>=2).mean()*100:.1f}%")
PY
rm -f /tmp/depths_$$.txt

echo ""                                              | tee -a "$REP"
echo "=== data.py self-test on REAL MSAs (train) ===" | tee -a "$REP"
python phase3/data.py --split train --n 5           2>&1 | tee -a "$REP"
echo "=== data.py self-test on REAL MSAs (val) ==="   | tee -a "$REP"
python phase3/data.py --split val --n 3             2>&1 | tee -a "$REP"

echo ""                                              | tee -a "$REP"
if [ "$N" -ge 9000 ]; then
  echo "VERIFY OK: $N/9516 a3m present; loader assembles real MSAs. Proceeding to training." | tee -a "$REP"
else
  echo "VERIFY FAIL: only $N a3m files (<9000). Stopping chain so we don't train on partial data." | tee -a "$REP"
  exit 1
fi
echo "----- verify done -----"
