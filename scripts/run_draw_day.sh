#!/usr/bin/env bash
# run_draw_day.sh — called by cron on 1st and 16th. SPEC §9.
# This orchestrates the full draw-day pipeline.
# Phase 2+ will make this meaningful; for now it documents the sequence.

set -euo pipefail

REPO="$HOME/projects/fortuna"
TODAY=$(date +%F)
LOG="$REPO/logs/cron-$TODAY.log"

mkdir -p "$REPO/logs"

echo "$(date '+%Y-%m-%d %H:%M:%S') run_draw_day.sh START for $TODAY" | tee -a "$LOG"

# Phase 2+ commands (uncomment when implemented):
# cd "$REPO" && .venv/bin/python -m fortuna scrape --catch-up >> "$LOG" 2>&1
# cd "$REPO" && .venv/bin/python -m fortuna features --target "$TODAY" >> "$LOG" 2>&1
# cd "$REPO" && .venv/bin/python -m fortuna predict --freeze --target "$TODAY" >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') run_draw_day.sh: Phase 2 pipeline not yet implemented" | tee -a "$LOG"
