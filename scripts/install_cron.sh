#!/usr/bin/env bash
# install_cron.sh — Print crontab lines for Nash to add manually.
# Do NOT auto-install. Nash reviews and runs `crontab -e` manually.
# SPEC §9 cron schedule.

cat <<'EOF'
# ==========================================================================
# Project Fortuna — add these lines to crontab (run: crontab -e)
# All times are Asia/Bangkok (Mac mini must have TZ=Asia/Bangkok)
# ==========================================================================

# Draw days = 1st and 16th of every month
0  6  1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna scrape --catch-up >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
0  7  1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna features --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
30 7  1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna predict --freeze --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
30 17 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna scrape --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
0  18 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna settle --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
30 18 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna journal --target $(date +\%F) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1
0  19 1,16 * *  cd ~/projects/fortuna && .venv/bin/python -m fortuna evolve --post-draw >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# Weekly feature proposer — Sundays 03:00
0  3  * * 0     cd ~/projects/fortuna && .venv/bin/python -m fortuna propose-features >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# Monthly tournament — 1st of month 02:00 (before draw activity)
0  2  1   * *   cd ~/projects/fortuna && .venv/bin/python -m fortuna tournament --month $(date -v-1m +\%Y-\%m) >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# Semi-annual breeding — Jan 1 and Jul 1, 04:00
0  4  1   1,7 * cd ~/projects/fortuna && .venv/bin/python -m fortuna breed >> ~/projects/fortuna/logs/cron-$(date +\%F).log 2>&1

# Daily ground-truth backup — 03:00 to non-iCloud path (SPEC §9 / §11)
0  3  * * *     mkdir -p ~/backups/fortuna && cp ~/projects/fortuna/data/raw/draws.jsonl ~/backups/fortuna/draws-$(date +\%F).jsonl && cp ~/projects/fortuna/data/raw/draws_corrections.jsonl ~/backups/fortuna/draws_corrections-$(date +\%F).jsonl 2>/dev/null || true

# ==========================================================================
# To install: crontab -e  (then paste the lines above)
# To verify:  crontab -l
# Timezone:   ensure Mac mini system TZ is Asia/Bangkok
#             sudo systemsetup -settimezone Asia/Bangkok
# ==========================================================================
EOF
