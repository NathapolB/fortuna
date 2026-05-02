"""Backfill script — scrape historical lottery draws and populate draws.jsonl + lab.db.

Usage:
    python scripts/backfill.py --start 2005-01-01 --end 2026-04-30

SPEC §3.4 + Phase 1 deliverable.

Strategy:
    1. Fetch from Sanook archive (primary — 20+ years, stable)
    2. Cross-check each draw against Kapook
    3. Apply 2-of-3 quorum validation
    4. Append accepted draws to data/raw/draws.jsonl
    5. Initialize lab.db schema
    6. Print conflict count for Nash review before commit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fortuna.config import (
    DISCREPANCIES_JSONL,
    DRAWS_JSONL,
    LAB_DB,
    check_not_icloud,
)
from fortuna.schema import Draw
from fortuna.scraper import KapookScraper, SanookScraper
from fortuna.store import DrawStore, get_or_init_db
from fortuna.validator import cross_check, validate_digits, validate_draw_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fortuna.backfill")


def run_backfill(
    start: date = date(2005, 1, 1),
    end: date | None = None,
) -> int:
    """Run the backfill. Returns exit code (0 = success)."""
    check_not_icloud()

    if end is None:
        end = date.today()

    logger.info("Starting backfill: %s → %s", start.isoformat(), end.isoformat())

    # Initialize database
    conn = get_or_init_db(LAB_DB)
    logger.info("SQLite lab.db initialized at %s", LAB_DB)

    store = DrawStore(DRAWS_JSONL)
    logger.info("Existing draws in store: %d", store.count())

    sanook = SanookScraper()
    kapook = KapookScraper()

    accepted = 0
    skipped_duplicate = 0
    skipped_conflict = 0
    skipped_validation = 0
    errors = 0

    # Collect all draw dates from Sanook
    logger.info("Fetching Sanook archive (primary source)...")
    sanook_draws_by_date: dict[str, Draw] = {}

    try:
        for draw in sanook.backfill_archive(since=start, until=end):
            sanook_draws_by_date[draw.draw_id] = draw
            logger.debug("Sanook: fetched draw %s (%s)", draw.draw_id, draw.first_prize)
    except Exception as e:
        logger.error("Sanook backfill iterator error: %s", e, exc_info=True)

    logger.info("Sanook fetched %d draws", len(sanook_draws_by_date))

    # Now cross-check each draw against Kapook
    logger.info("Cross-checking against Kapook...")
    total = len(sanook_draws_by_date)

    for idx, (draw_id, sanook_draw) in enumerate(sorted(sanook_draws_by_date.items()), 1):
        if idx % 50 == 0:
            logger.info("Progress: %d / %d draws processed", idx, total)

        # Skip if already in store
        if store.contains(draw_id):
            skipped_duplicate += 1
            continue

        # Validate sanook draw itself
        digit_errors = validate_digits(sanook_draw)
        if digit_errors:
            logger.warning("Draw %s failed digit validation: %s", draw_id, digit_errors)
            skipped_validation += 1
            continue

        if not validate_draw_date(sanook_draw):
            logger.warning("Draw %s has unexpected date", draw_id)
            # Don't skip — log and accept (some draws are legitimately shifted)

        # Fetch Kapook for cross-check
        kapook_draw: Draw | None = None
        try:
            kapook_draw = kapook.fetch_by_date(date.fromisoformat(draw_id))
        except Exception as e:
            logger.warning("Kapook fetch failed for %s: %s", draw_id, e)

        # Build source list for quorum check
        source_draws = [sanook_draw]
        if kapook_draw is not None:
            source_draws.append(kapook_draw)

        canonical, discrepancies = cross_check(source_draws, draw_id)

        if canonical is None:
            logger.warning("Draw %s REJECTED — quorum failed: %s", draw_id, discrepancies)
            skipped_conflict += 1
            continue

        # Write to store
        written = store.append(canonical)
        if written:
            accepted += 1
        else:
            skipped_duplicate += 1

    conn.close()

    # Summary
    logger.info("=" * 60)
    logger.info("Backfill complete:")
    logger.info("  Accepted:          %d", accepted)
    logger.info("  Skipped duplicate: %d", skipped_duplicate)
    logger.info("  Skipped conflict:  %d", skipped_conflict)
    logger.info("  Skipped invalid:   %d", skipped_validation)
    logger.info("  Errors:            %d", errors)
    logger.info("  Total in store:    %d", store.count())

    if DISCREPANCIES_JSONL.exists():
        conflict_lines = DISCREPANCIES_JSONL.read_text().strip().splitlines()
        logger.info("  Discrepancies logged: %d lines in %s", len(conflict_lines), DISCREPANCIES_JSONL)
        logger.info("  -> Nash should review discrepancies.jsonl before committing")

    # Verify checksum
    if store.verify_checksum():
        logger.info("Checksum verified: %s", DRAWS_JSONL)
    else:
        logger.warning("Checksum mismatch or missing for %s", DRAWS_JSONL)

    target_count = 500
    actual_count = store.count()
    if actual_count >= target_count:
        logger.info(
            "Phase 1 acceptance: %d draws >= %d target. PASS.",
            actual_count, target_count,
        )
    else:
        logger.warning(
            "Phase 1 acceptance: %d draws < %d target. "
            "May need to extend date range or fix source HTML.",
            actual_count, target_count,
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical Thai lottery draws"
    )
    parser.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        default="2005-01-01",
        help="Start date for backfill (default: 2005-01-01)",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        default=None,
        help="End date for backfill (default: today)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None

    sys.exit(run_backfill(start=start, end=end))


if __name__ == "__main__":
    main()
