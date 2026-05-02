# Phase 1 Retrospective — Project Fortuna

> Written by Builder agent per SPEC §12 Phase 1 acceptance criterion 7.
> Date: 2026-05-02 (bootstrap day)

---

## What was built

Phase 0 (Bootstrap):
- Repo initialized at `~/projects/fortuna/` (outside iCloud — sentinel `data/.nosync` enforced)
- Full directory tree per SPEC §1
- `requirements.txt` with pinned deps per SPEC §10
- `.env.example` with `GLO_USER_AGENT` and `NOTION_TOKEN` (no Slack webhook — resolved Q3)
- `.gitignore` correctly tracking `draws.jsonl` and `draws_corrections.jsonl`, ignoring cache/models/lab.db
- `pyproject.toml` with ruff config
- `README.md` with "How to reproduce" section

Phase 1 (Data Foundation):
- `fortuna/schema.py` — Pydantic `Draw` + `DrawCorrection` models + full SQLite DDL with both UNIQUE constraints on `predictions` table (SPEC §2.2 fix #1)
- `fortuna/store.py` — `DrawStore` (JSONL append-only, dedup by draw_id, checksum), `get_or_init_db`, idempotent `insert_prediction`/`insert_outcome`/`insert_feature`
- `fortuna/scraper.py` — `SanookScraper`, `KapookScraper`, `GLOScraper` with caching, politeness (3 sec/req), exponential backoff, scrape log
- `fortuna/parser.py` — `SanookParser`, `KapookParser`, `GLOParser` with multi-pattern HTML extraction, Thai date parsing (BE ↔ CE conversion), fallback strategies
- `fortuna/validator.py` — 2-of-3 quorum validation, digit format checks, shifted date support, discrepancies.jsonl logging
- `fortuna/config.py` — all paths, PAYOUTS dict, PICK_SPLIT (2/3/5 locked), iCloud guard
- `scripts/backfill.py` — CLI: `--start` / `--end`, Sanook + Kapook cross-check loop, summary report, checksum verification
- `scripts/eda.py` — produces `phase1-eda.md` + 3 figures (digit frequency, two-digit histogram, ACF plot)
- `scripts/install_cron.sh` — prints full cron schedule for Nash to add manually
- `tests/test_store_dedup.py` — 10 tests covering JSONL dedup, both UNIQUE constraints, cost_thb validation, persistence across instances
- `tests/test_parser.py` — fixture-based golden tests (skip until fixtures created via `create_fixtures.py`)
- `tests/test_walkforward_no_leakage.py` — Phase 2 stubs with `@pytest.mark.skip`
- `tests/test_metrics.py` — Phase 2 stubs with `@pytest.mark.skip`
- `tests/create_fixtures.py` — script to fetch and save HTML fixtures from each source

---

## What worked

- Schema design was clean — Pydantic validators catch bad data early, DDL maps cleanly from spec
- Keeping scraper/parser/validator separated makes each testable in isolation
- The iCloud guard in `config.py` + `.nosync` sentinel provides defense-in-depth
- JSONL append-only + SQLite dual store pattern is simple and correct for Phase 1

---

## What surprised (or will likely surprise during actual execution)

1. **Source HTML structure** — Sanook and Kapook HTML layouts have likely changed over 20 years. The parser uses multi-pattern matching + fallback strategies, but some draws may require manual fixture inspection and parser tuning.

2. **Sanook archive URL structure** — The assumed URL pattern `/lotto/{YYYY}/{MM}/{DD}/` and archive index `/lotto/archive/{year}/` need validation against actual live pages. If Sanook's archive is paginated differently, `extract_draw_urls` will need adjustment.

3. **Date coverage** — 2005–2026 = ~504 expected draws (2/month × ~21 years). If Sanook's archive starts later (e.g., 2008), the `--start` date may need adjustment.

4. **GLO scraping difficulty** — GLO's site is JS-heavy. The static HTML scraper may not get results from glo.or.th for recent draws. Sanook + Kapook are the reliable 2-of-2 for backfill; GLO is best effort for current draws.

---

## Data quality hypotheses for Phase 2

1. **Uniformity hypothesis** — EDA chi-square tests will likely confirm digits are uniform. Any position showing p < 0.05 needs BH-FDR correction (6 tests on positions alone).

2. **Serial dependence** — Autocorrelation of digit sums at lag 1–30 expected to be within 95% CI. Any lag outside CI is noise artifact given N ≈ 500.

3. **No edge available at Phase 1** — With 500 draws, even a 1% edge requires n ≥ 50 draws per cell and BH-corrected significance before any claim. Phase 2 baseline models will establish the random benchmark.

---

## Next actions for Phase 2

1. Run `python scripts/backfill.py --start 2005-01-01 --end 2026-04-30`
2. Run `python tests/create_fixtures.py` to build parser golden files
3. Run `pytest tests/test_parser.py tests/test_store_dedup.py -v` — must be green
4. Run `python scripts/eda.py` — inspect figures and chi-square results
5. Start Phase 2: implement `FrequencyBayesian`, `MarkovChain`, features library

---

_This retrospective will be updated when backfill completes and actual draw counts + EDA results are available._
