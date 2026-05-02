# Project Fortuna — Master Spec

> Self-learning AI ensemble for Thai Government Lottery (สลากกินแบ่งรัฐบาล)
> Author: Architect (Dev Squad), reviewed by Nash
> Status: Draft v2.1 — 2026-05-02 (post-review correction)
> Owner: Nash (principal), Builder agent (implementer)

**Mirror of:** `~/Library/Mobile Documents/com~apple~CloudDocs/jarvis/projects/fortuna/SPEC.md`
**Source of truth:** The jarvis copy. This mirror is for in-repo reference.

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
├── .env.example                       # GLO_USER_AGENT, NOTION_TOKEN (no Slack)
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
│   │   ├── predict.py                 # produces 10 picks for next draw + git-commit freeze
│   │   ├── settle.py                  # post-draw outcome reconciliation
│   │   ├── verify.py                  # `fortuna verify --date` re-hashes + checks remote git history
│   │   └── journal.py                 # generates AI meta-cognition .md
│   │
│   └── cli.py                         # `python -m fortuna <cmd>`
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Phase 1 exploratory analysis
│   ├── 02_baselines.ipynb             # frequency / Markov sanity checks
│   └── 03_walkforward_cv.ipynb        # CV harness validation
│
├── scripts/
│   ├── backfill.py                    # one-shot bulk backfill (20+ years)
│   ├── eda.py                         # EDA report generator
│   └── install_cron.sh                # prints crontab lines for Nash to add
│
└── tests/
    ├── test_parser.py
    ├── test_store_dedup.py
    ├── test_metrics.py                # Phase 2 — stubbed in Phase 1
    └── test_walkforward_no_leakage.py # Phase 2 — stubbed in Phase 1
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

## 13. Open Questions for Nash

All resolved as of v2.1:

1. เลขท้าย 3 ตัว = 3 distinct 3-digit numbers, 1 ticket each (3 tickets total). Hamming distance ≥ 1.
2. 80 THB fixed via Pao Tang (เป๋าตัง). DB CHECK enforces this. No 100-baht code path.
3. Notion-only output, no Slack. Failures → Notion failures DB via Notion MCP.
4. Private repo. Revisit after 3 months.

**Pick allocation (locked v2.1):** 2 × first6 + 3 × three_back + 5 × two_back = 10 tickets/draw.

---

_See jarvis SPEC for full content. This is a reference mirror._
