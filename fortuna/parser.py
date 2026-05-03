"""HTML parsers for each lottery source.

SPEC §3 — extract draw fields from raw HTML.

Each parser returns a Draw | None. If the HTML structure has changed (source
broke), it logs a warning and returns None rather than raising — the scraper
layer will log it and the validator will catch the gap.

Tested with golden HTML fixtures in tests/test_parser.py.

Sanook parser updated 2026-05-02:
    - New URL pattern: /lotto/check/{DDMMYY_BE}/
      where DDMMYY_BE = day-month-(BE year, 4-digit), zero-padded.
    - HTML structure uses CSS prefix lottocheck__ for all prize sections.
    - Parsing is anchored on lottocheck__column / lottocheck__sec--nearby
      DOM containers, not raw text walking, to avoid JSON-LD / meta tag
      false-positive matches that caused wrong two_back and three_back values.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from fortuna.config import BKK
from fortuna.schema import Draw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_digits(s: str) -> str:
    """Strip non-digit characters from a string."""
    return re.sub(r"\D", "", s)


def _normalize_digits(s: str, expected_len: int) -> str | None:
    """Extract digits and zero-pad to expected_len. Return None if still wrong."""
    digits = _clean_digits(s)
    if not digits:
        return None
    digits = digits.zfill(expected_len)
    return digits if len(digits) == expected_len else None


def _scraped_at() -> str:
    return datetime.now(BKK).isoformat()


def _extract_date_from_thai_text(text: str) -> str | None:
    """Extract draw date from Thai text that may contain Buddhist year.

    Handles both CE (2566) and BE year formats.
    Returns YYYY-MM-DD in CE (Gregorian).
    """
    # Thai month names to number
    thai_months = {
        "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3,
        "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
        "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9,
        "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
        # Short forms
        "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4,
        "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8,
        "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    }

    for month_name, month_num in thai_months.items():
        # Pattern: DD MonthName YYYY (Thai or CE year)
        pattern = rf"(\d{{1,2}})\s+{re.escape(month_name)}\s+(\d{{4}})"
        m = re.search(pattern, text)
        if m:
            day = int(m.group(1))
            year = int(m.group(2))
            # Convert Buddhist Era to CE if needed (BE = CE + 543)
            if year > 2500:
                year -= 543
            try:
                d = date(year, month_num, day)
                return d.isoformat()
            except ValueError:
                continue

    # Try ISO format in text
    m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y > 2500:
            y -= 543
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# SanookParser
# ---------------------------------------------------------------------------


class SanookParser:
    """Parse news.sanook.com lottery result pages.

    Supports both old-style URLs (/lotto/YYYY/MM/DD/) and new-style
    /lotto/check/{DDMMYY_BE}/ URLs introduced in the 2026-05-02 scraper fix.
    """

    SOURCE = "news.sanook.com"

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def parse(self, html: bytes, source_url: str) -> Draw | None:
        """Parse a Sanook draw result page into a Draw object."""
        soup = BeautifulSoup(html, "lxml")
        raw_sha = _sha256_bytes(html)

        try:
            draw_date = self._extract_date(soup, source_url)
            if draw_date is None:
                logger.warning("SanookParser: could not extract date from %s", source_url)
                return None

            first_prize = self._extract_first_prize(soup)
            if first_prize is None:
                logger.warning(
                    "SanookParser: could not extract first_prize from %s", source_url
                )
                return None

            two_digit_back = self._extract_two_digit_back(soup)
            if two_digit_back is None:
                logger.warning(
                    "SanookParser: could not extract two_digit_back from %s", source_url
                )
                return None

            three_digit_back = self._extract_three_digit_back(soup)
            three_digit_front = self._extract_three_digit_front(soup)
            first_prize_near = self._extract_near_prizes(soup)

            return Draw(
                draw_date=draw_date,
                draw_id=draw_date,
                first_prize=first_prize,
                first_prize_near=first_prize_near,
                three_digit_front=three_digit_front,
                three_digit_back=three_digit_back,
                two_digit_back=two_digit_back,
                bonus_prizes={},
                source=self.SOURCE,
                source_url=source_url,
                scraped_at=_scraped_at(),
                raw_html_sha256=raw_sha,
                verified_against=[],
                schema_version=1,
            )
        except Exception as e:
            logger.error(
                "SanookParser error for %s: %s", source_url, e, exc_info=True
            )
            return None

    # -----------------------------------------------------------------------
    # URL helpers (kept for backward compat with tests + create_fixtures.py)
    # -----------------------------------------------------------------------

    def extract_draw_urls(self, html: bytes, base_url: str) -> list[str]:
        """Extract individual draw page URLs from an archive index page.

        Legacy method — Sanook archive index no longer works. Returns empty list.
        Kept for backward compatibility with tests/create_fixtures.py.
        """
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if re.search(r"/lotto/\d{4}/\d{2}/\d{2}", href):
                if href.startswith("http"):
                    urls.append(href)
                else:
                    urls.append(f"https://news.sanook.com{href}")

        seen: set[str] = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def extract_date_from_url(self, url: str) -> str | None:
        """Extract YYYY-MM-DD from URL.

        Supports both formats:
          - Old: .../lotto/2024/01/16/...
          - New: .../lotto/check/16012567/  (DDMMYYYY_BE)
        """
        # Old format: /lotto/YYYY/MM/DD/
        m = re.search(r"/lotto/(\d{4})/(\d{2})/(\d{2})", url)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            try:
                return date(int(y), int(mo), int(d)).isoformat()
            except ValueError:
                return None

        # New format: /lotto/check/DDMMYYYY/ where year is BE (4-digit)
        m = re.search(r"/lotto/check/(\d{2})(\d{2})(\d{4})/", url)
        if m:
            dd, mm, be_year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ce_year = be_year - 543
            try:
                return date(ce_year, mm, dd).isoformat()
            except ValueError:
                return None

        return None

    # -----------------------------------------------------------------------
    # Date extraction
    # -----------------------------------------------------------------------

    def _extract_date(self, soup: BeautifulSoup, source_url: str) -> str | None:
        # 1. URL-embedded date (most reliable)
        date_from_url = self.extract_date_from_url(source_url)
        if date_from_url:
            return date_from_url

        # 2. Page title or h1/h2
        for tag in soup.find_all(["h1", "h2", "title"]):
            text = tag.get_text(strip=True)
            d = _extract_date_from_thai_text(text)
            if d:
                return d

        # 3. Meta tags and prominent text elements
        for tag in soup.find_all(["meta", "p", "span", "div"]):
            content = tag.get("content", "") or tag.get_text(strip=True)
            d = _extract_date_from_thai_text(content)
            if d:
                return d

        return None

    # -----------------------------------------------------------------------
    # Prize extraction — anchored on lottocheck DOM structure
    #
    # Primary strategy: find div.lottocheck__column (or div.lottocheck__sec--nearby)
    # whose span.default-font--reward contains the Thai prize label, then collect
    # lotto__number tags inside that container.
    #
    # This avoids the JSON-LD articleBody <script> tag and breadcrumb text that
    # caused the old _find_numbers_after_label to match date digits (e.g. "16"
    # for the draw day) instead of actual prize numbers.
    #
    # Fallback: _find_numbers_after_label for old-layout pages that predate the
    # lottocheck__ CSS class convention.
    # -----------------------------------------------------------------------

    def _extract_from_lottocheck_column(
        self,
        soup: BeautifulSoup,
        label: str,
        digit_len: int,
        max_count: int = 1,
        container_classes: tuple[str, ...] = ("lottocheck__column", "lottocheck__sec--nearby"),
    ) -> list[str]:
        """Extract prize numbers from a lottocheck__column (or --nearby) container.

        Finds the first container div whose span.default-font--reward text
        contains `label`, then returns up to `max_count` strings of exactly
        `digit_len` digits from lotto__number tags within that container.

        Returns an empty list if no matching container is found.
        """
        digit_pattern = re.compile(r"^\d+$")

        for css_class in container_classes:
            for container in soup.find_all("div", class_=css_class):
                # Check if the prize label lives inside this container
                label_span = container.find("span", class_="default-font--reward")
                if label_span is None:
                    continue
                span_text = label_span.get_text(strip=True)
                if label not in span_text:
                    continue

                # Collect lotto__number tags whose text is exactly digit_len digits
                results: list[str] = []
                for tag in container.find_all(class_="lotto__number"):
                    text = tag.get_text(strip=True)
                    if digit_pattern.match(text) and len(text) == digit_len:
                        if text not in results:
                            results.append(text)
                        if len(results) >= max_count:
                            return results

                if results:
                    return results

        return []

    def _find_numbers_after_label(
        self,
        soup: BeautifulSoup,
        labels: list[str],
        digit_len: int,
        max_count: int = 1,
        search_limit: int = 30,
    ) -> list[str]:
        """Fallback: find numbers of digit_len after any of the given Thai labels.

        Walks forward through text nodes from the label's parent element.
        Used for old Sanook page layouts that predate lottocheck__ CSS classes.
        Returns a list; empty if nothing found.

        NOTE: This method is unreliable on the current Sanook layout because
        the labels appear first inside the JSON-LD <script> articleBody and
        breadcrumb text — always try _extract_from_lottocheck_column first.
        """
        for label in labels:
            # Find the NavigableString or element that contains the label text
            elem = soup.find(string=lambda s, lbl=label: s and lbl in s)
            if elem is None:
                # Try finding an element whose get_text() contains the label
                elem = soup.find(
                    lambda tag: tag.name and label in tag.get_text()
                )
                if elem is None:
                    continue
                # Use the element itself as the starting point
                start_node: Tag = elem  # type: ignore[assignment]
            else:
                start_node = elem.parent  # type: ignore[assignment]

            results: list[str] = []
            pattern = re.compile(rf"(?<!\d)(\d{{{digit_len}}})(?!\d)")

            # First check the parent element's own text
            parent_text = start_node.get_text()
            for m in pattern.finditer(parent_text):
                candidate = m.group(1)
                if candidate not in results:
                    results.append(candidate)
                if len(results) >= max_count:
                    break

            if len(results) >= max_count:
                return results

            # Walk forward through subsequent text nodes
            for sibling_text in start_node.find_all_next(string=True, limit=search_limit):
                text = str(sibling_text).strip()
                if not text:
                    continue
                for m in pattern.finditer(text):
                    candidate = m.group(1)
                    if candidate not in results:
                        results.append(candidate)
                    if len(results) >= max_count:
                        return results

            if results:
                return results

        return []

    def _extract_first_prize(self, soup: BeautifulSoup) -> str | None:
        """Extract the 6-digit first prize number.

        Priority order:
        1. lotto__number--first tag (most specific — the first-prize <strong>)
        2. lottocheck__column anchored on "รางวัลที่ 1" label
        3. Old-layout label-walking fallback
        4. Last resort: first standalone 6-digit in page body text
        """
        # Strategy 1: explicit lotto__number--first class (new Sanook layout)
        tag = soup.find(class_="lotto__number--first")
        if tag:
            text = tag.get_text(strip=True)
            if re.fullmatch(r"\d{6}", text):
                return text

        # Strategy 2: lottocheck__column containing "รางวัลที่ 1" label
        for label in ["รางวัลที่ 1", "รางวัลที่1"]:
            results = self._extract_from_lottocheck_column(
                soup, label=label, digit_len=6, max_count=1
            )
            if results:
                return results[0]

        # Strategy 3: old-layout label walking (pre-lottocheck pages)
        results = self._find_numbers_after_label(
            soup,
            labels=["รางวัลที่ 1", "รางวัลที่1", "รางวัล 1", "ที่ 1"],
            digit_len=6,
            max_count=1,
        )
        if results:
            return results[0]

        # Strategy 4: lottocheck CSS class patterns (other sites / old layouts)
        for cls_pattern in ["first-prize", "reward-1", "reward1", "prize-1", "prize1", "firstprize"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        # Strategy 5: first standalone 6-digit number in page body text
        # (fallback for very old layouts — least reliable)
        body = soup.find("body")
        search_root = body if body else soup
        all_text = search_root.get_text()
        m = re.search(r"(?<!\d)(\d{6})(?!\d)", all_text)
        if m:
            return m.group(1)

        return None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        """Extract the 2-digit back prize.

        Primary: lottocheck__column anchored on Thai label.
        Fallback: old-layout label walking.
        """
        # Primary: lottocheck__column structure (current Sanook layout)
        for label in ["เลขท้าย 2 ตัว", "เลขท้าย2ตัว", "สองตัวท้าย", "2 ตัวท้าย", "ท้าย 2 ตัว"]:
            results = self._extract_from_lottocheck_column(
                soup, label=label, digit_len=2, max_count=1
            )
            if results:
                return results[0]

        # Fallback: old-layout label walking
        results = self._find_numbers_after_label(
            soup,
            labels=["เลขท้าย 2 ตัว", "เลขท้าย2ตัว", "สองตัวท้าย", "2 ตัวท้าย", "ท้าย 2"],
            digit_len=2,
            max_count=1,
        )
        if results:
            return results[0]
        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        """Extract the 3-digit back prize(s). Thai lottery awards 2 numbers.

        Primary: lottocheck__column anchored on Thai label.
        Fallback: old-layout label walking.
        """
        # Primary: lottocheck__column structure
        for label in ["เลขท้าย 3 ตัว", "เลขท้าย3ตัว", "สามตัวท้าย", "3 ตัวท้าย", "ท้าย 3 ตัว"]:
            results = self._extract_from_lottocheck_column(
                soup, label=label, digit_len=3, max_count=2
            )
            if results:
                return results[:2]

        # Fallback: old-layout label walking
        results = self._find_numbers_after_label(
            soup,
            labels=["เลขท้าย 3 ตัว", "เลขท้าย3ตัว", "สามตัวท้าย", "3 ตัวท้าย", "ท้าย 3"],
            digit_len=3,
            max_count=2,
        )
        return results[:2]

    def _extract_three_digit_front(self, soup: BeautifulSoup) -> list[str]:
        """Extract the 3-digit front prize(s). Thai lottery awards 2 numbers.

        Primary: lottocheck__column anchored on Thai label.
        Fallback: old-layout label walking.
        """
        # Primary: lottocheck__column structure
        for label in ["เลขหน้า 3 ตัว", "เลขหน้า3ตัว", "สามตัวหน้า", "3 ตัวหน้า", "หน้า 3 ตัว"]:
            results = self._extract_from_lottocheck_column(
                soup, label=label, digit_len=3, max_count=2
            )
            if results:
                return results[:2]

        # Fallback: old-layout label walking
        results = self._find_numbers_after_label(
            soup,
            labels=["เลขหน้า 3 ตัว", "เลขหน้า3ตัว", "สามตัวหน้า", "3 ตัวหน้า", "หน้า 3"],
            digit_len=3,
            max_count=2,
        )
        return results[:2]

    def _extract_near_prizes(self, soup: BeautifulSoup) -> list[str]:
        """Extract the two near-first-prize numbers (6 digits each).

        Primary: lottocheck__sec--nearby container.
        Fallback: old-layout label walking.
        """
        # Primary: lottocheck__sec--nearby container
        for label in ["รางวัลข้างเคียงรางวัลที่ 1", "ข้างเคียงรางวัลที่ 1"]:
            results = self._extract_from_lottocheck_column(
                soup,
                label=label,
                digit_len=6,
                max_count=2,
                container_classes=("lottocheck__sec--nearby",),
            )
            if results:
                return results[:2]

        # Fallback: old-layout label walking
        results = self._find_numbers_after_label(
            soup,
            labels=["รางวัลข้างเคียงรางวัลที่ 1", "ข้างเคียงรางวัลที่ 1", "ข้างเคียง", "ใกล้เคียง"],
            digit_len=6,
            max_count=2,
        )
        return results[:2]


# ---------------------------------------------------------------------------
# KapookParser
# ---------------------------------------------------------------------------


class KapookParser:
    """Parse kapook.com lottery result pages.

    NOTE: KapookScraper is currently stubbed (URL pattern unverified).
    This parser is kept intact for when the scraper is re-enabled.
    """

    SOURCE = "kapook.com"

    def parse(self, html: bytes, source_url: str) -> Draw | None:
        soup = BeautifulSoup(html, "lxml")
        raw_sha = _sha256_bytes(html)

        try:
            draw_date = self._extract_date(soup, source_url)
            if draw_date is None:
                logger.warning("KapookParser: could not extract date from %s", source_url)
                return None

            first_prize = self._extract_first_prize(soup)
            if first_prize is None:
                logger.warning("KapookParser: could not extract first_prize from %s", source_url)
                return None

            two_digit_back = self._extract_two_digit_back(soup)
            if two_digit_back is None:
                logger.warning("KapookParser: could not extract two_digit_back from %s", source_url)
                return None

            three_digit_back = self._extract_three_digit_back(soup)

            return Draw(
                draw_date=draw_date,
                draw_id=draw_date,
                first_prize=first_prize,
                first_prize_near=[],
                three_digit_front=[],
                three_digit_back=three_digit_back,
                two_digit_back=two_digit_back,
                bonus_prizes={},
                source=self.SOURCE,
                source_url=source_url,
                scraped_at=_scraped_at(),
                raw_html_sha256=raw_sha,
                verified_against=[],
                schema_version=1,
            )
        except Exception as e:
            logger.error("KapookParser error for %s: %s", source_url, e, exc_info=True)
            return None

    def _extract_date(self, soup: BeautifulSoup, source_url: str) -> str | None:
        # URL: /lottery/YYYY/MM/DD/
        m = re.search(r"/lottery/(\d{4})/(\d{2})/(\d{2})", source_url)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
            except ValueError:
                pass

        for tag in soup.find_all(["h1", "h2", "h3", "title"]):
            text = tag.get_text(strip=True)
            d = _extract_date_from_thai_text(text)
            if d:
                return d
        return None

    def _extract_first_prize(self, soup: BeautifulSoup) -> str | None:
        for cls_pattern in ["first-prize", "prize-1", "reward-1", "lotto-result"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        # Fallback: first 6-digit number in page
        six_digit = re.search(r"(?<!\d)(\d{6})(?!\d)", soup.get_text())
        return six_digit.group(1) if six_digit else None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        for keyword in ["เลขท้าย 2 ตัว", "2 ตัวท้าย", "ท้าย 2"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:5]:
                    m = re.search(r"(?<!\d)(\d{2})(?!\d)", node.get_text())
                    if m:
                        return m.group(1)
        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        for keyword in ["เลขท้าย 3 ตัว", "3 ตัวท้าย", "ท้าย 3"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:10]:
                    for m in re.finditer(r"(?<!\d)(\d{3})(?!\d)", node.get_text()):
                        if m.group(1) not in results:
                            results.append(m.group(1))
                    if len(results) >= 2:
                        break
        return results[:2]


# ---------------------------------------------------------------------------
# GLOParser
# ---------------------------------------------------------------------------


class GLOParser:
    """Parse glo.or.th lottery result pages.

    NOTE: GLOScraper is currently stubbed (JS-heavy SPA, needs Selenium Phase 2).
    This parser is kept intact for future use.
    """

    SOURCE = "glo.or.th"

    def parse(self, html: bytes, source_url: str) -> Draw | None:
        soup = BeautifulSoup(html, "lxml")
        raw_sha = _sha256_bytes(html)

        try:
            draw_date = self._extract_date(soup, source_url)
            if draw_date is None:
                logger.warning("GLOParser: could not extract date from %s", source_url)
                return None

            first_prize = self._extract_first_prize(soup)
            if first_prize is None:
                logger.warning("GLOParser: could not extract first_prize from %s", source_url)
                return None

            two_digit_back = self._extract_two_digit_back(soup)
            if two_digit_back is None:
                logger.warning("GLOParser: could not extract two_digit_back from %s", source_url)
                return None

            three_digit_back = self._extract_three_digit_back(soup)

            return Draw(
                draw_date=draw_date,
                draw_id=draw_date,
                first_prize=first_prize,
                first_prize_near=[],
                three_digit_front=[],
                three_digit_back=three_digit_back,
                two_digit_back=two_digit_back,
                bonus_prizes={},
                source=self.SOURCE,
                source_url=source_url,
                scraped_at=_scraped_at(),
                raw_html_sha256=raw_sha,
                verified_against=[],
                schema_version=1,
            )
        except Exception as e:
            logger.error("GLOParser error for %s: %s", source_url, e, exc_info=True)
            return None

    def _extract_date(self, soup: BeautifulSoup, source_url: str) -> str | None:
        # GLO URL: /result/YYYYMMDD.html
        m = re.search(r"/result/(\d{4})(\d{2})(\d{2})\.html", source_url)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
            except ValueError:
                pass

        for tag in soup.find_all(["h1", "h2", "title"]):
            text = tag.get_text(strip=True)
            d = _extract_date_from_thai_text(text)
            if d:
                return d
        return None

    def _extract_first_prize(self, soup: BeautifulSoup) -> str | None:
        for cls_pattern in ["reward-1", "first-prize", "prize1"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        six_digit = re.search(r"(?<!\d)(\d{6})(?!\d)", soup.get_text())
        return six_digit.group(1) if six_digit else None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        for keyword in ["เลขท้าย 2 ตัว", "2 ตัวท้าย"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:5]:
                    m = re.search(r"(?<!\d)(\d{2})(?!\d)", node.get_text())
                    if m:
                        return m.group(1)
        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        for keyword in ["เลขท้าย 3 ตัว", "3 ตัวท้าย"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:10]:
                    for m in re.finditer(r"(?<!\d)(\d{3})(?!\d)", node.get_text()):
                        if m.group(1) not in results:
                            results.append(m.group(1))
                    if len(results) >= 2:
                        break
        return results[:2]
