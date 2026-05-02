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
        "2024-01-01",
        "2023-06-16",
        "2022-12-01",
        "2010-07-16",
        "2007-03-01",
    ],
    "kapook": [
        "2024-01-01",
        "2023-06-16",
        "2022-12-01",
        "2019-09-16",
        "2015-05-01",
    ],
    "glo": [
        "2024-01-01",
        "2023-12-16",
        "2023-06-01",
        "2022-11-16",
        "2021-05-01",
    ],
}

SANOOK_URL_TMPL = "https://news.sanook.com/lotto/{y}/{m:02d}/{d:02d}/"
KAPOOK_URL_TMPL = "https://horoscope.kapook.com/lottery/{y}/{m:02d}/{d:02d}/"
GLO_URL_TMPL = "https://www.glo.or.th/result/{y}{m:02d}{d:02d}.html"


def _date_to_url(source: str, date_str: str) -> str:
    d = date.fromisoformat(date_str)
    if source == "sanook":
        return SANOOK_URL_TMPL.format(y=d.year, m=d.month, d=d.day)
    elif source == "kapook":
        return KAPOOK_URL_TMPL.format(y=d.year, m=d.month, d=d.day)
    elif source == "glo":
        return GLO_URL_TMPL.format(y=d.year, m=d.month, d=d.day)
    raise ValueError(f"Unknown source: {source}")


def save_fixture(source: str, date_str: str, draws_from_store: dict) -> None:
    url = _date_to_url(source, date_str)
    fixture_dir = FIXTURES_DIR / source
    fixture_dir.mkdir(parents=True, exist_ok=True)

    html_file = fixture_dir / f"{date_str}.html"
    expected_file = fixture_dir / f"{date_str}.expected.json"

    # Fetch HTML
    try:
        html, from_cache = fetch_url(url, source, use_cache=True)
    except Exception as e:
        print(f"  SKIP {source}/{date_str}: fetch failed: {e}")
        return

    html_file.write_bytes(html)
    print(f"  Saved HTML: {html_file} ({len(html)} bytes, cache={from_cache})")

    # Parse and extract expected values
    if source == "sanook":
        parser = SanookParser()
        draw = parser.parse(html, url)
    elif source == "kapook":
        parser = KapookParser()
        draw = parser.parse(html, url)
    elif source == "glo":
        parser = GLOParser()
        draw = parser.parse(html, url)
    else:
        return

    if draw is not None:
        expected = {
            "draw_id": draw.draw_id,
            "first_prize": draw.first_prize,
            "two_digit_back": draw.two_digit_back,
            "three_digit_back": draw.three_digit_back,
        }
    elif date_str in draws_from_store:
        # Fallback to store data
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
        expected = {"draw_id": date_str, "first_prize": "UNKNOWN", "two_digit_back": "XX", "three_digit_back": []}

    expected_file.write_text(json.dumps(expected, indent=2, ensure_ascii=False))
    print(f"  Saved expected: {expected_file}")


def main() -> None:
    print("Creating parser fixtures...")

    # Load existing store data for fallback
    store = DrawStore(DRAWS_JSONL)
    draws_by_id = {d.draw_id: d for d in store.iter_draws()}
    print(f"Loaded {len(draws_by_id)} draws from store for fallback")

    for source, dates in SAMPLE_DATES.items():
        print(f"\n{'=' * 40}")
        print(f"Source: {source}")
        for date_str in dates:
            print(f"  Processing {date_str}...")
            save_fixture(source, date_str, draws_by_id)

    print("\nFixtures created. Run pytest tests/test_parser.py -v to verify.")


if __name__ == "__main__":
    main()
