"""HTTP scrapers for three lottery sources.

SPEC §3.1, §3.2, §3.4.

Sources:
    1. news.sanook.com  — primary backfill (20+ years, stable paginated archive)
    2. kapook.com       — cross-check secondary
    3. glo.or.th        — primary for current/live draws

Politeness: 1 req / 3 sec, random UA, exponential backoff on 429/5xx, max 3 retries.
Cache: raw HTML gzipped to data/raw/cache/{source}/{date}.html.gz keyed by URL hash + date.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urljoin, urlparse

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from fortuna.config import (
    BACKOFF_BASE,
    BKK,
    CACHE_DIR,
    MAX_RETRIES,
    REQUEST_DELAY_SEC,
    SCRAPE_LOG_JSONL,
    USER_AGENTS,
)
from fortuna.parser import GLOParser, KapookParser, SanookParser
from fortuna.schema import Draw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _random_ua() -> str:
    return random.choice(USER_AGENTS)


def _cache_path(source_label: str, url: str) -> Path:
    """Deterministic cache path keyed by URL hash."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / source_label / f"{url_hash}.html.gz"


def _load_cached(cache_file: Path) -> bytes | None:
    if cache_file.exists():
        with gzip.open(cache_file, "rb") as f:
            return f.read()
    return None


def _save_cache(cache_file: Path, content: bytes) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file, "wb") as f:
        f.write(content)


def _log_fetch(url: str, status: int, byte_count: int, from_cache: bool) -> None:
    entry = {
        "url": url,
        "status": status,
        "bytes": byte_count,
        "from_cache": from_cache,
        "ts": datetime.now(BKK).isoformat(),
    }
    SCRAPE_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with SCRAPE_LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fetch_url(
    url: str,
    source_label: str,
    *,
    use_cache: bool = True,
    session: requests.Session | None = None,
) -> tuple[bytes, bool]:
    """Fetch URL with caching and politeness. Returns (html_bytes, from_cache).

    Raises RequestException on permanent failure after retries.
    """
    cache_file = _cache_path(source_label, url)

    if use_cache:
        cached = _load_cached(cache_file)
        if cached is not None:
            _log_fetch(url, 200, len(cached), from_cache=True)
            return cached, True

    s = session or requests.Session()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            sleep_time = REQUEST_DELAY_SEC * (BACKOFF_BASE ** attempt)
            logger.debug("Backoff %.1fs before retry %d for %s", sleep_time, attempt, url)
            time.sleep(sleep_time)
        else:
            time.sleep(REQUEST_DELAY_SEC)

        headers = {
            "User-Agent": _random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        try:
            resp = s.get(url, headers=headers, timeout=30)
        except (ConnectionError, Timeout) as e:
            logger.warning("Network error on attempt %d for %s: %s", attempt + 1, url, e)
            last_exc = e
            continue

        _log_fetch(url, resp.status_code, len(resp.content), from_cache=False)

        if resp.status_code == 200:
            content = resp.content
            _save_cache(cache_file, content)
            return content, False

        if resp.status_code in (429, 500, 502, 503, 504):
            logger.warning("HTTP %d on attempt %d for %s", resp.status_code, attempt + 1, url)
            last_exc = RequestException(f"HTTP {resp.status_code}")
            continue

        # Non-retryable (404, etc.)
        raise RequestException(f"HTTP {resp.status_code} for {url} (not retryable)")

    raise RequestException(
        f"All {MAX_RETRIES} attempts failed for {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Sanook scraper — primary backfill
# ---------------------------------------------------------------------------

SANOOK_BASE = "https://news.sanook.com/lotto"
# Sanook archive URL pattern: /lotto/{YYYY}/{MM}/{DD}/
# The index pages are paginated by year/month.


class SanookScraper:
    """Scrape news.sanook.com lottery archive. SPEC §3.1."""

    ARCHIVE_URL_TEMPLATE = "https://news.sanook.com/lotto/archive/{year}/"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = SanookParser()

    def backfill_archive(
        self, since: date, until: date | None = None
    ) -> Iterator[Draw]:
        """Yield Draw objects from the Sanook archive.

        Iterates year-by-year from `since` to `until` (default: today).
        Each draw page is fetched, parsed, and yielded.
        """
        if until is None:
            until = date.today()

        current_year = since.year
        end_year = until.year

        while current_year <= end_year:
            archive_url = self.ARCHIVE_URL_TEMPLATE.format(year=current_year)
            logger.info("Fetching Sanook archive index for year %d: %s", current_year, archive_url)

            try:
                html, from_cache = fetch_url(archive_url, "sanook", session=self._session)
            except RequestException as e:
                logger.error("Failed to fetch Sanook archive for %d: %s", current_year, e)
                current_year += 1
                continue

            draw_urls = self._parser.extract_draw_urls(html, archive_url)
            logger.info("Found %d draw links for year %d", len(draw_urls), current_year)

            for draw_url in draw_urls:
                # Filter by date range
                draw_date_str = self._parser.extract_date_from_url(draw_url)
                if draw_date_str is None:
                    continue
                try:
                    draw_date = date.fromisoformat(draw_date_str)
                except ValueError:
                    continue
                if draw_date < since or draw_date > until:
                    continue

                try:
                    draw_html, _ = fetch_url(draw_url, "sanook", session=self._session)
                except RequestException as e:
                    logger.warning("Failed to fetch Sanook draw %s: %s", draw_url, e)
                    continue

                draw = self._parser.parse(draw_html, draw_url)
                if draw is not None:
                    yield draw

            current_year += 1

    def fetch_by_date(self, d: date) -> Draw | None:
        """Fetch a specific draw date from Sanook."""
        # Try common URL format: /YYYY/MM/DD/
        url = f"{SANOOK_BASE}/{d.year}/{d.month:02d}/{d.day:02d}/"
        try:
            html, _ = fetch_url(url, "sanook", session=self._session)
            return self._parser.parse(html, url)
        except RequestException as e:
            logger.warning("Sanook fetch_by_date failed for %s: %s", d, e)
            return None


# ---------------------------------------------------------------------------
# Kapook scraper — cross-check
# ---------------------------------------------------------------------------

KAPOOK_BASE = "https://horoscope.kapook.com/lottery"


class KapookScraper:
    """Scrape kapook.com lottery archive for cross-checking. SPEC §3.1."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = KapookParser()

    def fetch_by_date(self, d: date) -> Draw | None:
        """Fetch a specific draw date from Kapook."""
        # Kapook URL pattern: /lottery/{YYYY}/{MM}/{DD}/
        url = f"{KAPOOK_BASE}/{d.year}/{d.month:02d}/{d.day:02d}/"
        try:
            html, _ = fetch_url(url, "kapook", session=self._session)
            return self._parser.parse(html, url)
        except RequestException as e:
            logger.warning("Kapook fetch_by_date failed for %s: %s", d, e)
            return None


# ---------------------------------------------------------------------------
# GLO official scraper — current/live draws
# ---------------------------------------------------------------------------

GLO_BASE = "https://www.glo.or.th"


class GLOScraper:
    """Scrape glo.or.th for current draws. SPEC §3.1."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = GLOParser()

    def fetch_latest(self) -> Draw | None:
        """Fetch the most recent draw result from glo.or.th."""
        url = f"{GLO_BASE}/result/reward-header.html"
        try:
            html, _ = fetch_url(url, "glo", use_cache=False, session=self._session)
            return self._parser.parse(html, url)
        except RequestException as e:
            logger.warning("GLO fetch_latest failed: %s", e)
            return None

    def fetch_by_date(self, d: date) -> Draw | None:
        """Fetch a specific draw date from GLO."""
        # GLO archive URL pattern varies; try common format
        url = f"{GLO_BASE}/result/{d.year}{d.month:02d}{d.day:02d}.html"
        try:
            html, _ = fetch_url(url, "glo", session=self._session)
            return self._parser.parse(html, url)
        except RequestException as e:
            logger.warning("GLO fetch_by_date failed for %s: %s", d, e)
            return None
