"""Create HTML fixtures for parser golden-file tests.

Run this script once after backfill has populated draws.jsonl,
or run it standalone to fetch specific pages.

Usage:
    cd ~/projects/fortuna
    source .venv/bin/activate
    python tests/create_fixtures.py

This fetches sample pages from each source, saves HTML to tests/fixtures/{source}/,
and saves expected JSON alongside each HTML file.

Expected JSON format matches the fields asserted in test_parser.py:
{
    "draw_id": "2024-01-01",
    "first_prize": "123456",
    "two_digit_back": "56",
    "three_digit_back": ["789", "012"]
}

After running this script, test_parser.py will stop skipping those tests.
Nash can also hand-create fixtures if source URLs return unexpected HTML.

Sanook URL pattern updated 2026-05-02:
    Old (broken): /lotto/{YYYY}/{MM}/{DD}/
    New (working): /lotto/check/{DDMMYYYY_BE}/  (4-digit Buddhist Era year)
    Example: 16 Jan 2024 CE → BE 2567 → /check/16012567/
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fortuna.scraper import KapookScraper, SanookScraper, GLOScraper, fetch_url
from fortuna.parser import KapookParser, SanookParser, GLOParser
from fortuna.store import DrawStore
from fortuna.config import DRAWS_JSONL

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Dates to use as fixtures — spread across years for good coverage
SAMPLE_DATES = {
    "sanook": [
        "2026-04-16",  # Most recent known draw (16 Apr 2026 CE = 16 Apr 2569 BE)
        "2026-03-16",
        "2026-03-01",
        "2026-02-16",
        "2026-01-16",
    ],
    # Kapook and GLO are currently stubbed; fixtures left for future use
    "kapook": [],
    "glo": [],
}


def _sanook_url(d: date) -> str:
    """Build correct Sanook /check/ URL with 4-digit Buddhist Era year."""
    be_year = d.year + 543
    ddmmyy_be = f"{d.day:02d}{d.month:02d}{be_year}"
    return f"https://news.sanook.com/lotto/check/{ddmmyy_be}/"


def save_fixture_sanook(date_str: str, draws_from_store: dict) -> None:
    d = date.fromisoformat(date_str)
    url = _sanook_url(d)

    fixture_dir = FIXTURES_DIR / "sanook"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    html_file = fixture_dir / f"{date_str}.html"
    expected_file = fixture_dir / f"{date_str}.expected.json"

    # Fetch HTML (allow_404 so we can probe gracefully)
    try:
        html, from_cache = fetch_url(url, "sanook", use_cache=True, allow_404=True)
    except Exception as e:
        print(f"  SKIP sanook/{date_str}: fetch failed: {e}")
        return

    if html is None:
        print(f"  SKIP sanook/{date_str}: 404 at {url}")
        return

    html_file.write_bytes(html)
    print(f"  Saved HTML: {html_file} ({len(html)} bytes, cache={from_cache})")

    parser = SanookParser()
    draw = parser.parse(html, url)

    if draw is not None:
        expected = {
            "draw_id": draw.draw_id,
            "first_prize": draw.first_prize,
            "two_digit_back": draw.two_digit_back,
            "three_digit_back": draw.three_digit_back,
        }
        print(f"  Parsed: first_prize={draw.first_prize}  two_back={draw.two_digit_back}  three_back={draw.three_digit_back}")
    elif date_str in draws_from_store:
        store_draw = draws_from_store[date_str]
        expected = {
            "draw_id": store_draw.draw_id,
            "first_prize": store_draw.first_prize,
            "two_digit_back": store_draw.two_digit_back,
            "three_digit_back": store_draw.three_digit_back,
        }
        print(f"  Used store data for expected values (parser returned None)")
    else:
        print(f"  WARNING: parser returned None and no store data for {date_str}")
        expected = {
            "draw_id": date_str,
            "first_prize": "UNKNOWN",
            "two_digit_back": "XX",
            "three_digit_back": [],
        }

    expected_file.write_text(json.dumps(expected, indent=2, ensure_ascii=False))
    print(f"  Saved expected: {expected_file}")


def main() -> None:
    print("Creating parser fixtures...")

    # Load existing store data for fallback
    store = DrawStore(DRAWS_JSONL)
    draws_by_id = {d.draw_id: d for d in store.iter_draws()}
    print(f"Loaded {len(draws_by_id)} draws from store for fallback")

    print(f"\n{'=' * 40}")
    print("Source: sanook")
    for date_str in SAMPLE_DATES["sanook"]:
        print(f"  Processing {date_str}...")
        save_fixture_sanook(date_str, draws_by_id)

    print("\nFixtures created. Run pytest tests/test_parser.py -v to verify.")


if __name__ == "__main__":
    main()
