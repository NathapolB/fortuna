#!/usr/bin/env bash
# install_cron.sh — Print crontab lines for Nash to add manually.
# Do NOT auto-install. Nash reviews and runs `crontab -e` manually.
# SPEC §9 cron schedule — updated v2.2 (Enhancement-2).
#
# New schedule (v2.2):
#   Day 2  of month @ 07:00 BKK — settle 1st, predict for 16th (~14 days ahead)
#   Day 17 of month @ 07:00 BKK — settle 16th, predict for 1st of next month
#
# One cron line calls `fortuna run-scheduled` which handles all logic automatically.

cat <<'EOF'
# ==========================================================================
# Project Fortuna v2.2 — add these lines to crontab (run: crontab -e)
# All times are Asia/Bangkok (Mac mini must have TZ=Asia/Bangkok)
# ==========================================================================

# --- Primary schedule: day 2 and 17 @ 07:00 (settle + train + predict) ---
# Day  2: settle draw from 1st, predict for 16th (~14 days ahead)
# Day 17: settle draw from 16th, predict for 1st of next month (~14 days ahead)
0 7 2,17 * * cd ~/projects/fortuna && .venv/bin/python -m fortuna run-scheduled >> ~/projects/fortuna/logs/cron.log 2>&1

# --- Post-draw result scrape + journal: still on draw days (1st and 16th) ---
# Scrape results after draw (~17:00) and update journal
30 17 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna scrape --catch-up >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
0  18 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna journal --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
0  19 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna evolve --post-draw >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# --- Weekly feature proposer — Sundays 03:00 ---
0  3  * * 0     cd ~/projects/fortuna && .venv/bin/python -m fortuna propose-features >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# --- Monthly tournament — 1st of month 02:00 (before draw activity) ---
0  2  1   * *   cd ~/projects/fortuna && .venv/bin/python -m fortuna tournament --month $(date -v-1m +\%Y-\%m) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# --- Semi-annual breeding — Jan 1 and Jul 1, 04:00 ---
0  4  1   1,7 * cd ~/projects/fortuna && .venv/bin/python -m fortuna breed >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# --- Daily ground-truth backup — 03:00 to non-iCloud path (SPEC §9 / §11) ---
0  3  * * *     mkdir -p ~/backups/fortuna && cp ~/projects/fortuna/data/raw/draws.jsonl ~/backups/fortuna/draws-$(date +\%F).jsonl && cp ~/projects/fortuna/data/raw/draws_corrections.jsonl ~/backups/fortuna/draws_corrections-$(date +\%F).jsonl 2>/dev/null || true

# ==========================================================================
# To install: crontab -e  (then paste the lines above)
# To verify:  crontab -l
# Timezone:   ensure Mac mini system TZ is Asia/Bangkok
#             sudo systemsetup -settimezone Asia/Bangkok
#
# Key cron line (run-scheduled handles everything on day 2 and 17):
#   0 7 2,17 * * cd ~/projects/fortuna && .venv/bin/python -m fortuna run-scheduled >> logs/cron.log 2>&1
# ==========================================================================
EOF
