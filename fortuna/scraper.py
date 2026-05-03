"""HTTP scrapers for three lottery sources.

SPEC §3.1, §3.2, §3.4.

Sources:
    1. news.sanook.com  — primary backfill (20+ years, date-based /check/ URL)
    2. kapook.com       — cross-check secondary (stubbed — URL pattern unverified)
    3. glo.or.th        — primary for current/live draws (stubbed — JS-heavy page)

Politeness: 1 req / 3 sec, random UA, exponential backoff on 429/5xx, max 3 retries.
Cache: raw HTML gzipped to data/raw/cache/{source}/{url_hash}.html.gz.

Sanook URL fix (2026-05-02):
    Old (broken): /lotto/archive/{year}/  +  /lotto/{YYYY}/{MM}/{DD}/
    New (working): /lotto/check/{DDMMYY_BE}/
    where DDMMYY_BE = day-month-(BE year % 100), zero-padded.
    Example: 16 Apr 2026 CE = 16 Apr 2569 BE → /check/16042569/

    There is no archive index — candidate dates are generated from the Thai
    government lottery schedule (1st and 16th of every month) with ±2 day
    holiday-shift probing.
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
    allow_404: bool = False,
) -> tuple[bytes | None, bool]:
    """Fetch URL with caching and politeness. Returns (html_bytes, from_cache).

    If allow_404=True, a 404 response returns (None, False) instead of raising.
    Raises RequestException on permanent non-404 failure after retries.
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

        if resp.status_code == 404 and allow_404:
            logger.debug("HTTP 404 (allowed) for %s", url)
            return None, False

        if resp.status_code in (429, 500, 502, 503, 504):
            logger.warning("HTTP %d on attempt %d for %s", resp.status_code, attempt + 1, url)
            last_exc = RequestException(f"HTTP {resp.status_code}")
            continue

        # Non-retryable (404 not allowed, etc.)
        raise RequestException(f"HTTP {resp.status_code} for {url} (not retryable)")

    raise RequestException(
        f"All {MAX_RETRIES} attempts failed for {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Sanook scraper — primary backfill
# ---------------------------------------------------------------------------

SANOOK_BASE = "https://news.sanook.com/lotto"


class SanookScraper:
    """Scrape news.sanook.com lottery results by date. SPEC §3.1.

    URL pattern: https://news.sanook.com/lotto/check/{DDMMYY_BE}/
    where DDMMYY_BE = zero-padded day + month + (BE year, 4-digit).
    Example: 16 April 2026 CE → BE 2569 → /check/16042569/

    No archive index exists — candidate draw dates are generated from the Thai
    government schedule (1st and 16th monthly) with ±2-day holiday-shift probing.
    """

    DRAW_URL_TEMPLATE = "https://news.sanook.com/lotto/check/{ddmmyy_be}/"

    # Holiday-shift probe order: try standard date first, then common offsets
    PROBE_OFFSETS = [0, 1, -1, 2, -2]

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = SanookParser()

    # ------------------------------------------------------------------
    # Public API (keeps backfill.py interface intact)
    # ------------------------------------------------------------------

    def backfill_archive(
        self, since: date, until: date | None = None
    ) -> Iterator[Draw]:
        """Yield Draw objects for all draws between since and until (inclusive).

        Generates candidate dates (1st + 16th of every month), probes each with
        ±2 day offsets to handle Thai public holiday shifts.
        """
        if until is None:
            until = date.today()

        logger.info(
            "SanookScraper.backfill_archive: scanning %s → %s",
            since.isoformat(),
            until.isoformat(),
        )

        for candidate in self._generate_candidate_dates(since, until):
            actual_date, html = self._probe_draw(candidate)
            if html is None:
                logger.debug(
                    "No draw found near candidate %s (±2 days) — skipping",
                    candidate.isoformat(),
                )
                continue

            url = self.DRAW_URL_TEMPLATE.format(
                ddmmyy_be=self._format_be(actual_date)
            )
            draw = self._parser.parse(html, url)
            if draw is not None:
                yield draw
            else:
                logger.warning(
                    "SanookParser returned None for draw near %s (actual %s)",
                    candidate.isoformat(),
                    actual_date.isoformat(),
                )

    def fetch_by_date(self, d: date) -> Draw | None:
        """Fetch a specific draw date from Sanook (probes ±2 days for holiday shifts)."""
        actual_date, html = self._probe_draw(d)
        if html is None:
            logger.warning("Sanook fetch_by_date: no draw found near %s", d.isoformat())
            return None
        url = self.DRAW_URL_TEMPLATE.format(ddmmyy_be=self._format_be(actual_date))
        return self._parser.parse(html, url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_draw(self, candidate: date) -> tuple[date, bytes] | tuple[None, None]:
        """Try candidate date and offsets. Return (actual_date, html) or (None, None)."""
        for offset in self.PROBE_OFFSETS:
            try_date = candidate + timedelta(days=offset)
            ddmmyy_be = self._format_be(try_date)
            url = self.DRAW_URL_TEMPLATE.format(ddmmyy_be=ddmmyy_be)
            logger.debug("Probing Sanook URL: %s", url)

            try:
                html, from_cache = fetch_url(
                    url, "sanook", session=self._session, allow_404=True
                )
            except RequestException as e:
                logger.warning("Sanook fetch error for %s: %s", url, e)
                continue

            if html is not None and self._looks_like_draw_page(html):
                logger.info(
                    "Found draw at %s (offset %+d from candidate %s, cache=%s)",
                    try_date.isoformat(),
                    offset,
                    candidate.isoformat(),
                    from_cache,
                )
                return try_date, html

        return None, None

    @staticmethod
    def _generate_candidate_dates(start: date, end: date) -> Iterator[date]:
        """Yield 1st and 16th of each month between start and end inclusive."""
        # Align to the first day of start's month
        d = date(start.year, start.month, 1)
        while d <= end:
            for day in (1, 16):
                candidate = date(d.year, d.month, day)
                if start <= candidate <= end:
                    yield candidate
            # Advance to next month
            if d.month == 12:
                d = date(d.year + 1, 1, 1)
            else:
                d = date(d.year, d.month + 1, 1)

    @staticmethod
    def _format_be(d: date) -> str:
        """Format date as DDMMYYYY in Buddhist Era (4-digit BE year).

        Example: 16 April 2026 CE → BE 2569 → '16042569'
        Example: 2 May 2026 CE → BE 2569 → '02052569'
        """
        be_year = d.year + 543
        return f"{d.day:02d}{d.month:02d}{be_year}"

    @staticmethod
    def _looks_like_draw_page(html: bytes) -> bool:
        """Quick heuristic: page must contain lottocheck CSS classes and Thai prize label."""
        try:
            text = html.decode("utf-8", errors="ignore")
        except Exception:
            return False
        return "lottocheck__" in text and "รางวัลที่ 1" in text


# ---------------------------------------------------------------------------
# Kapook scraper — cross-check (STUBBED — URL pattern unverified)
# ---------------------------------------------------------------------------

# TODO: Kapook URL pattern has not been verified against live site.
# The original pattern /lottery/{YYYY}/{MM}/{DD}/ may return 404 or redirect.
# Stub returns None for all fetches until the correct URL is confirmed.
# Once verified, replace this stub with a proper date-based implementation
# similar to SanookScraper. (SPEC §3.1 fallback: Sanook-only is acceptable Phase 1)

KAPOOK_BASE = "https://horoscope.kapook.com/lottery"


class KapookScraper:
    """Scrape kapook.com lottery archive for cross-checking. SPEC §3.1.

    STUBBED: URL pattern unverified. Returns None for all requests.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = KapookParser()

    def fetch_by_date(self, d: date) -> Draw | None:
        """STUBBED — always returns None until Kapook URL pattern is confirmed."""
        logger.debug(
            "KapookScraper.fetch_by_date: stubbed, skipping %s", d.isoformat()
        )
        return None


# ---------------------------------------------------------------------------
# GLO official scraper — current/live draws (STUBBED — JS-heavy page)
# ---------------------------------------------------------------------------

# TODO: glo.or.th result pages are JavaScript-rendered (React/Angular SPA).
# A plain requests.get() returns the shell HTML without lottery data.
# Needs Selenium or Playwright with a headless browser for Phase 2.
# Stub returns None to avoid silent empty-parse errors. (SPEC §3.1 fallback)

GLO_BASE = "https://www.glo.or.th"


class GLOScraper:
    """Scrape glo.or.th for current draws. SPEC §3.1.

    STUBBED: glo.or.th is a JS-heavy SPA — needs Selenium/Playwright (Phase 2).
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._parser = GLOParser()

    def fetch_latest(self) -> Draw | None:
        """STUBBED — returns None until Selenium integration is added."""
        logger.debug("GLOScraper.fetch_latest: stubbed (JS-heavy page)")
        return None

    def fetch_by_date(self, d: date) -> Draw | None:
        """STUBBED — returns None until Selenium integration is added."""
        logger.debug(
            "GLOScraper.fetch_by_date: stubbed, skipping %s", d.isoformat()
        )
        return None
