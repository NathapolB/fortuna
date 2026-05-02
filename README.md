# Project Fortuna

**Private personal ML lab — entertainment + learning.**

Self-learning AI ensemble for Thai Government Lottery (สลากกินแบ่งรัฐบาล).

## Honest framing

This is **not** a profit system. Starting assumption: Thai Government Lottery draws are uniform random. The goal is to build a rigorous ML lab on a high-noise problem, maintain statistical transparency, and have a little fun for 800 THB/draw.

Every prediction is committed to this private GitHub repo **before** the draw. Tamper-evidence rests on GitHub branch protection (force-push blocked). See SPEC §0 and §6.1.

## Spec

Full spec: `/Users/nathapolbuddhamongkol/Library/Mobile Documents/com~apple~CloudDocs/jarvis/projects/fortuna/SPEC.md` (v2.1)

## Status

![Phase 0](https://img.shields.io/badge/Phase%200-complete-brightgreen)
![Phase 1](https://img.shields.io/badge/Phase%201-in%20progress-yellow)

## How to reproduce

```bash
# 1. Clone
git clone git@github.com:<nash>/fortuna.git ~/projects/fortuna
cd ~/projects/fortuna

# 2. Create venv and install deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy env
cp .env.example .env
# Edit .env — add GLO_USER_AGENT, NOTION_TOKEN

# 4. Run backfill (fetches 20+ years of draw data)
python scripts/backfill.py --start 2005-01-01 --end 2026-04-30

# 5. Open EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

## Pick split (locked)

2 × first6 + 3 × three_back + 5 × two_back = 10 tickets/draw  
800 THB/draw × 2 draws/month = 1,600 THB/month

## Data

- `data/raw/draws.jsonl` — append-only ground truth, tracked in git
- `data/raw/draws_corrections.jsonl` — corrections referencing draw_id, tracked in git
- `data/lab.db` — SQLite working store (gitignored)
- `data/exports/` — frozen pre-draw prediction JSON files (tracked in git)

## Warning

`data/` MUST NOT live inside iCloud Drive. The repo is at `~/projects/fortuna/` (outside iCloud). A sentinel file `data/.nosync` is checked at startup — if it ends up under iCloud sync, the pipeline aborts.

## Branch protection

Nash must enable: **Settings → Branches → Require force-push protection** (and "Restrict deletions"). Without this, the audit trail is only as strong as local commit history.
