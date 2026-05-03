# Project Fortuna — Master Spec

> Self-learning AI ensemble for Thai Government Lottery (สลากกินแบ่งรัฐบาล)
> Author: Architect (Dev Squad), reviewed by Nash
> Status: Draft v2.3 — 2026-05-02 (Notion schema corrected to real Claude Scheduler DB)
> Owner: Nash (principal), Builder agent (implementer)

**Mirror of:** `~/Library/Mobile Documents/com~apple~CloudDocs/jarvis/projects/fortuna/SPEC.md`
**Source of truth:** The jarvis copy. This mirror is for in-repo reference.

---

## Changelog

### v2.3 — 2026-05-02
- **Notion schema corrected**: `notion_publisher.py` now targets Nash's existing
  **Claude Scheduler DB** (DB ID `343d185571598079963ccfeb61f197c6`) which is shared
  across JARVIS scheduled outputs (MCC, Stock Guru, Content Radar, etc.).
  The DB has only 3 real properties: `Schedule Name` (title), `Run Date` (date),
  `Archive` (checkbox). All previous phantom properties (`Draw Date`, `Total Cost`,
  `Status`, `Freeze Commit`) have been removed from the API payload — they do not
  exist and the Notion API would reject any request that sets them.
  All pick data, model metadata, tamper timestamp, and honest framing now live in the
  **page body** as Notion blocks. `settle_prediction_page()` now only calls
  `PATCH /v1/blocks/{page_id}/children` (no property updates, as none track status).
- **Tests added**: `tests/test_notion_publisher.py` (20 tests covering properties,
  body content, settle append behaviour, and graceful degradation).
- **`.env.example` updated**: clarified shared-DB approach; hardcoded verified DB ID.

### v2.2 — 2026-05-02
- **Enhancement-1**: Notion integration — `fortuna/pipeline/notion_publisher.py`.
  After git freeze, `run_predict()` creates a Notion page with picks + metadata.
  After settlement, `run_settle()` updates page status + appends results.
  Configure via `NOTION_TOKEN` and `NOTION_FORTUNA_DB_ID` env vars.
  Schema: added `predictions.notion_page_id TEXT` column (migration: `scripts/migrate_add_notion_page_id.py`).
- **Enhancement-2**: Cron schedule changed to **day 2 and 17** at 07:00 BKK.
  New CLI command `fortuna run-scheduled` handles settle + train + predict automatically.
  Predict now 14 days before draw (safer leakage margin).
- **Enhancement-3**: Leakage guard in `predict.py` upgraded from warning to **hard ValueError**.
  New `--allow-leak` CLI flag (and `allow_leak=True` param) downgrades to warning for testing/backtest.
- **Fix**: `tests/test_metrics.py::test_bh_fdr_correction` — numpy `bool_` `is True` comparison fixed.
- **Fix**: `tests/test_picker.py::test_select_picks_empty_input` — fallback generator now
  checks diversity against its own accumulating list, not just the static `existing` param.

### v2.1 — 2026-05-02
- Post-review corrections: SPEC §2.2 both UNIQUE constraints, draw schedule clarification.

---

## 0. กรอบความคิดและกติกาสำคัญ (Honest Framing)

โปรเจกต์นี้ **ไม่ใช่** ระบบหากำไร — สมมติฐานเริ่มต้นคือ **หวยรัฐบาลไทยสุ่มแบบ uniform** ตามเป้าหมายเชิงสถิติของเครื่อง draw รางวัลที่ 1 hit rate ที่ "เก่งกว่าสุ่ม" มากที่สุดที่นักวิจัยจริงจังเคยรายงานในลอตเตอรี่กลไกคือระดับ 0.5–2% edge และมักหายไปเมื่อเครื่องถูกเปลี่ยน

เป้าหมายของ Fortuna คือ:

1. **เป็นห้องทดลอง ML จริง** — ฝึก ensemble + meta-learner + evolution loop บนปัญหาที่ noise สูงมาก (signal-to-noise ใกล้ศูนย์)
2. **ความบันเทิง** — งวดละ 800 บาท ลุ้นทุกวันที่ 1 และ 16
3. **โปร่งใสทางสถิติ** — ทุก prediction บันทึกไว้ก่อน draw, AI vs random baseline เทียบด้วย binomial / chi-square test, ไม่ตกแต่งผลย้อนหลัง

**Anti-goals (สิ่งที่จะไม่ทำ):**
- ไม่อ้างว่าทำกำไรได้ในระยะยาว
- ไม่ใช้เลขเด็ดจากเพจ/พระ/ความฝัน (จะเป็น dataset แยก ใช้เป็น sanity-check baseline เท่านั้น)
- ไม่ scale งบ ถ้า model "ดูเหมือน" ชนะใน 1–2 งวด — ต้องผ่าน statistical significance test ที่กำหนดล่วงหน้า

**Tamper-evidence model (updated v2):** Pre-draw prediction freeze relies on **git-commit-as-anchor** rather than self-signed JSON. The frozen prediction file is committed to a private GitHub repo before draw open time. Tamper-evidence rests on the git remote being append-only — force-pushes to `main` MUST be blocked at GitHub repo settings. Nash to enable branch protection: **Settings → Branches → Require force-push protection** (and "Restrict deletions"). If branch protection is off, the audit trail is only as strong as Nash's local commit history.

---

## 1. Repo Structure

**Project lives at `~/projects/fortuna/` — a SEPARATE git repo, NOT inside `~/jarvis/`.** The `~/jarvis/projects/fortuna/SPEC.md` you are reading is ONLY the spec; the actual implementation repo is created via:

```bash
mkdir -p ~/projects/fortuna && cd ~/projects/fortuna && git init && gh repo create fortuna --private --source=. --push
```

**WARNING — `data/` MUST NEVER live inside iCloud Drive.** JSONL append + SQLite write under iCloud sync = corruption risk (partial fsync, conflict copies, lock contention). Belt-and-suspenders defense: a sentinel file `data/.nosync` is created at repo init; if the repo is ever moved and that sentinel ends up at an iCloud-synced path, the pipeline aborts on startup.

```
~/projects/fortuna/                    # OUTSIDE iCloud, separate git repo
├── README.md                          # quick start + status badge
├── SPEC.md                            # mirror of jarvis/projects/fortuna/SPEC.md (this file)
├── requirements.txt                   # pinned deps
├── .env.example                       # GLO_USER_AGENT, NOTION_TOKEN, NOTION_FORTUNA_DB_ID
├── .gitignore                         # data/raw/cache/, __pycache__, *.pyc
├── pyproject.toml                     # ruff/black config, project metadata
│
├── data/
│   ├── .nosync                        # sentinel — pipeline aborts if path is under iCloud
│   ├── raw/
│   │   ├── draws.jsonl                # one draw per line, append-only — TRACKED IN GIT (small)
│   │   ├── draws_corrections.jsonl    # corrections referencing draw_id — TRACKED IN GIT
│   │   ├── draws.checksum             # sha256 of draws.jsonl after each scrape
│   │   ├── cache/                     # gzipped raw HTML — gitignored (large, regenerable)
│   │   └── scrape_log.jsonl           # every fetch: url, status, bytes, ts
│   ├── lab.db                         # SQLite — features, predictions, outcomes, runs
│   ├── reports/
│   │   ├── journal-YYYY-MM-DD.md      # AI-authored meta-cognition per draw
│   │   ├── monthly-YYYY-MM.md         # Beat-Random Tournament leaderboard
│   │   └── hall-of-fame.md            # top-10 models all-time
│   └── exports/
│       └── YYYY-MM-DD-prediction.json # frozen pre-draw snapshot {prediction, sha256, git_sha_at_freeze}
│
├── models/
│   ├── registry.json                  # active models, weights, parents, fitness
│   ├── graveyard.json                 # retired models + reason
│   ├── artifacts/
│   │   └── {model_id}/
│   │       ├── weights.pkl            # or .pt for torch
│   │       ├── meta.json              # hyperparams, train range, git sha
│   │       └── feature_spec.json
│   └── breeding_log.jsonl             # crossover events
│
├── fortuna/                           # python package
│   ├── __init__.py
│   ├── config.py                      # paths, draw schedule, PAYOUTS dict (§2.5)
│   │
│   ├── data/
│   │   ├── scraper.py                 # GLO + sanook + kapook fetchers
│   │   ├── parser.py                  # html → Draw dataclass
│   │   ├── validator.py               # cross-source consistency checks
│   │   └── store.py                   # JSONL append + dedup + checksum
│   │
│   ├── features/
│   │   ├── base.py                    # FeatureSpec abstract
│   │   ├── library.py                 # 30+ named features (digit_freq_30d, ...)
│   │   ├── proposer.py                # auto feature engineering (genetic search)
│   │   └── registry.py                # accepted features + acceptance ts
│   │
│   ├── models/
│   │   ├── base.py                    # BaseModel ABC (see §4)
│   │   ├── frequency.py               # FrequencyBayesian
│   │   ├── markov.py                  # Markov chain on digit transitions
│   │   ├── neural.py                  # LSTM / small Transformer
│   │   ├── rl.py                      # Q-learning agent
│   │   ├── meta.py                    # MetaLearner (logistic stacker)
│   │   └── ensemble.py                # weighted aggregation
│   │
│   ├── evolution/
│   │   ├── ga.py                      # genetic algorithm engine
│   │   ├── breeding.py                # 6-month crossover
│   │   ├── graveyard.py               # retire underperformers
│   │   └── optuna_search.py           # hyperparam tuning
│   │
│   ├── eval/
│   │   ├── metrics.py                 # brier, log_loss, hit_rate, p_and_l
│   │   ├── stats.py                   # binomial, chi-square, BH-FDR, walk-forward CV
│   │   └── leaderboard.py             # tournament aggregator
│   │
│   ├── pipeline/
│   │   ├── train.py                   # nightly + post-draw retrain
│   │   ├── predict.py                 # produces 10 picks for next draw + git-commit freeze + Notion publish
│   │   ├── settle.py                  # post-draw outcome reconciliation + Notion settle update
│   │   ├── notion_publisher.py        # Notion REST API integration (Enhancement-1, schema v2.3)
│   │   ├── verify.py                  # `fortuna verify --date` re-hashes + checks remote git history
│   │   └── journal.py                 # generates AI meta-cognition .md
│   │
│   └── cli.py                         # `python -m fortuna <cmd>` incl. run-scheduled
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Phase 1 exploratory analysis
│   ├── 02_baselines.ipynb             # frequency / Markov sanity checks
│   └── 03_walkforward_cv.ipynb        # CV harness validation
│
├── scripts/
│   ├── backfill.py                    # one-shot bulk backfill (20+ years)
│   ├── eda.py                         # EDA report generator
│   ├── install_cron.sh                # prints crontab lines for Nash to add
│   └── migrate_add_notion_page_id.py  # one-time DB migration for v2.2 notion_page_id column
│
└── tests/
    ├── test_parser.py
    ├── test_store_dedup.py
    ├── test_metrics.py                # Phase 2 — stubbed in Phase 1
    ├── test_walkforward_no_leakage.py # Phase 2 — stubbed in Phase 1
    └── test_notion_publisher.py       # v2.3 — Notion schema + body + settle + degradation
```

---

## 2. Data Schema

### 2.1 `data/raw/draws.jsonl` — append-only ground truth

หนึ่งบรรทัดต่อหนึ่งงวด JSON object schema:

```json
{
  "draw_date": "2026-05-16",
  "draw_id": "2026-05-16",
  "first_prize": "123456",
  "first_prize_near": ["123455", "123457"],
  "three_digit_front": ["123", "456"],
  "three_digit_back": ["789", "012"],
  "two_digit_back": "34",
  "bonus_prizes": {
    "second": ["..."],
    "third": ["..."],
    "fourth": ["..."],
    "fifth": ["..."]
  },
  "source": "glo.or.th",
  "source_url": "https://www.glo.or.th/...",
  "scraped_at": "2026-05-16T16:45:12+07:00",
  "raw_html_sha256": "abc123...",
  "verified_against": ["news.sanook.com", "kapook.com"],
  "schema_version": 1
}
```

Rules:
- Append-only. Never mutate. Corrections go in a sibling `draws_corrections.jsonl` referencing `draw_id`.
- `draw_id` is the unique key. Duplicate `draw_id` rejected by `store.py`.
- Every write triggers re-compute of `data/raw/draws.checksum`.

### 2.2 `data/lab.db` — SQLite working store

See `fortuna/schema.py` DDL for full CREATE TABLE statements including both UNIQUE constraints on `predictions`:

```sql
UNIQUE (draw_id, model_id, prize_type, pick_rank),
UNIQUE (draw_id, model_id, prize_type, pick_value)
```

Both constraints coexist intentionally per SPEC v2 fix #1.

**v2.2 addition:** `predictions` table has a new column `notion_page_id TEXT` (nullable).
Existing databases: run `scripts/migrate_add_notion_page_id.py` once.

### 2.5 Payout Constants

| prize_type   | payout_thb | break_even_hit_rate (vs 80 THB cost) |
|--------------|-----------:|-------------------------------------:|
| `first6`     |  6,000,000 | 1.33e-5 |
| `three_back` |      4,000 | 0.020   |
| `two_back`   |      2,000 | 0.040   |

```python
# fortuna/config.py
PAYOUTS = {
    "first6":     6_000_000,
    "three_back":     4_000,
    "two_back":       2_000,
}
TICKET_COST_THB = 80   # Pao Tang official wholesale price
```

---

## 6. Prediction Pipeline — Notion Integration (v2.3)

`fortuna/pipeline/notion_publisher.py` publishes predictions to Notion using Nash's
existing **Claude Scheduler DB** (shared with other JARVIS scheduled outputs such as
MCC, Stock Guru, and Content Radar). A dedicated Fortuna DB is not created or required.
Filter pages by `Schedule Name contains "Fortuna"` in Notion to see only Fortuna pages.

### Real DB schema (verified live)

**DB ID:** `343d185571598079963ccfeb61f197c6`

| Property | Type | Notes |
|---|---|---|
| Schedule Name | title | Pattern: `🎰 Fortuna — งวด DD MMM BE` (BE = Buddhist Era year) |
| Run Date | date | The draw date in ISO YYYY-MM-DD; used for calendar sorting |
| Archive | checkbox | false by default; true to hide from active views |

No other properties exist. The API will reject any payload that sets unknown properties.

### Content strategy

All pick data lives in the **page body** (Notion blocks):
- Top callout (yellow): AI's Picks header with Thai date and buy-via-Pao-Tang note
- Heading 2 + numbered list: รางวัลที่ 1 (2 picks, code-styled + confidence %)
- Heading 2 + numbered list: เลขท้าย 3 ตัว (3 picks)
- Heading 2 + paragraph: เลขท้าย 2 ตัว (5 picks, code-styled, space-dot separated)
- Divider
- Heading 3 + bullets: Model Contributions (one bullet per model with weight)
- Heading 3 + paragraph: Verifiable Timestamp (GitHub commit link + push time)
- Bottom callout (gray): Honest framing / disclaimer

### Settlement

`settle_prediction_page(page_id, settlement_summary)` appends result blocks to the
same page via `PATCH /v1/blocks/{page_id}/children`. It does NOT attempt to update
any properties (the DB has no status or payout properties). Appended blocks include:
a divider, ผลรางวัล heading, actual winning numbers paragraph, a 4-column results
table (one row per ticket), and a P&L summary callout.

---

## 9. Cron Schedule (v2.2)

**Updated schedule: day 2 and 17 of each month at 07:00 BKK.**

| Day | Action |
|-----|--------|
| Day 2  @ 07:00 | Settle draw from 1st, train, predict for 16th (~14 days ahead) |
| Day 17 @ 07:00 | Settle draw from 16th, train, predict for 1st of next month (~14 days ahead) |
| Draw day (1st/16th) @ 17:30 | Scrape result + journal + evolve |

**Single cron line (v2.2):**
```
0 7 2,17 * * cd ~/projects/fortuna && .venv/bin/python -m fortuna run-scheduled >> logs/cron.log 2>&1
```

`run-scheduled` automatically resolves settle_date and predict_date from the current day.
For manual overrides: `fortuna predict --date YYYY-MM-DD` and `fortuna settle --date YYYY-MM-DD`.

**Leakage safety:** Predicting 14 days before draw means `predict_started_at` is always
well before `draw_cutoff(target_draw_id)`. The guard in `predict.py` enforces this with
a hard `ValueError` (unless `--allow-leak` is passed for testing).

**draw_cutoff semantics (v2.2):**
- Live predict: cutoff = `predict_started_at` (recorded in prediction payload)
- Backtest / walk-forward CV: cutoff = `draw_cutoff(T)` = 06:00 BKK on draw date T

---

## 13. Open Questions for Nash

All resolved as of v2.3:

1. เลขท้าย 3 ตัว = 3 distinct 3-digit numbers, 1 ticket each (3 tickets total). Hamming distance ≥ 1.
2. 80 THB fixed via Pao Tang (เป๋าตัง). DB CHECK enforces this. No 100-baht code path.
3. Notion-only output, no Slack. Failures → Notion failures DB via Notion MCP.
4. Private repo. Revisit after 3 months.
5. (v2.3) Notion DB = shared Claude Scheduler DB, NOT a separate Fortuna DB.

**Pick allocation (locked v2.1):** 2 × first6 + 3 × three_back + 5 × two_back = 10 tickets/draw.

**v2.3 Notion setup for Nash:**
1. Create a Notion integration at https://www.notion.so/my-integrations (if not already done for JARVIS)
2. Add the integration to the existing Claude Scheduler DB (Share → Invite)
3. Set in `.env`:
   ```
   NOTION_TOKEN=secret_xxx
   NOTION_FORTUNA_DB_ID=343d185571598079963ccfeb61f197c6
   ```
   The DB ID is already prefilled in `.env.example`.

---

_See jarvis SPEC for full content. This is a reference mirror._
