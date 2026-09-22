#!/usr/bin/env bash
# Quick progress check for the Phase 3 MSA build.
# Usage:  bash phase3/progress.sh
# Shows: how many a3m files are done, the job's state, and the latest log lines.

DATA="$HOME/Projects/ppi-data"
A3M="$DATA/msa/full/a3m"
LOGDIR="$HOME/Projects/multimodal-ppi/phase3/logs"
TOTAL=9516

DONE=$(ls "$A3M" 2>/dev/null | wc -l)
PCT=$(( DONE * 100 / TOTAL ))

echo "============================================"
echo "  MSA PROGRESS"
echo "  a3m files done : $DONE / $TOTAL   ($PCT%)"
echo "============================================"
echo
echo "--- your job(s) right now ---"
squeue --me
echo
echo "--- latest log lines ---"
LOG=$(ls -t "$LOGDIR"/ppb_msa_finish_*.out 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "(from: $(basename "$LOG"))"
  tail -n 10 "$LOG"
else
  echo "(no log file found yet)"
fi
echo
if grep -q -- "----- done -----" "$LOG" 2>/dev/null; then
  echo ">>> STATUS: FINISHED. All chunks complete."
elif [ "$DONE" -ge "$TOTAL" ]; then
  echo ">>> STATUS: all $TOTAL files present."
else
  echo ">>> STATUS: in progress (or waiting in queue). Re-run this anytime to refresh."
fi
