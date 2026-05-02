"""HTML parsers for each lottery source.

SPEC §3 — extract draw fields from raw HTML.

Each parser returns a Draw | None. If the HTML structure has changed (source
broke), it logs a warning and returns None rather than raising — the scraper
layer will log it and the validator will catch the gap.

Tested with golden HTML fixtures in tests/test_parser.py.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

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
    """Parse news.sanook.com lottery result pages."""

    SOURCE = "news.sanook.com"

    def extract_draw_urls(self, html: bytes, base_url: str) -> list[str]:
        """Extract individual draw page URLs from an archive index page."""
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []

        # Sanook archive: links to individual draw result pages
        # Pattern: href contains /lotto/ and a date-like path
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Look for links that look like lottery result pages
            if re.search(r"/lotto/\d{4}/\d{2}/\d{2}", href):
                if href.startswith("http"):
                    urls.append(href)
                else:
                    urls.append(f"https://news.sanook.com{href}")

        # Deduplicate preserving order
        seen: set[str] = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def extract_date_from_url(self, url: str) -> str | None:
        """Extract YYYY-MM-DD from URL like .../lotto/2024/01/16/..."""
        m = re.search(r"/lotto/(\d{4})/(\d{2})/(\d{2})", url)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            try:
                return date(int(y), int(mo), int(d)).isoformat()
            except ValueError:
                return None
        return None

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
                logger.warning("SanookParser: could not extract first_prize from %s", source_url)
                return None

            two_digit_back = self._extract_two_digit_back(soup)
            if two_digit_back is None:
                logger.warning("SanookParser: could not extract two_digit_back from %s", source_url)
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
            logger.error("SanookParser error for %s: %s", source_url, e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _extract_date(self, soup: BeautifulSoup, source_url: str) -> str | None:
        # 1. Try URL-embedded date first (most reliable)
        date_from_url = self.extract_date_from_url(source_url)
        if date_from_url:
            return date_from_url

        # 2. Try page title or h1/h2
        for tag in soup.find_all(["h1", "h2", "title"]):
            text = tag.get_text(strip=True)
            d = _extract_date_from_thai_text(text)
            if d:
                return d

        # 3. Try meta og:description or any element with date-like text
        for tag in soup.find_all(["meta", "p", "span", "div"]):
            content = tag.get("content", "") or tag.get_text(strip=True)
            d = _extract_date_from_thai_text(content)
            if d:
                return d

        return None

    def _extract_first_prize(self, soup: BeautifulSoup) -> str | None:
        # Multiple layout patterns across years

        # Pattern 1: element with class containing 'first' or 'prize1' or 'reward1'
        for cls_pattern in ["first-prize", "reward-1", "reward1", "prize-1", "prize1", "firstprize"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        # Pattern 2: Look for a standalone 6-digit number near prize keywords
        prize_keywords = ["รางวัลที่ 1", "รางวัลที่1", "รางวัล 1", "ที่ 1"]
        for keyword in prize_keywords:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el:
                # Look at the next sibling or parent's next sibling
                parent = el.parent
                if parent:
                    text = parent.get_text()
                    m = re.search(r"\b(\d{6})\b", text)
                    if m:
                        return m.group(1)
                    # Try next few siblings
                    for sibling in parent.find_next_siblings():
                        m = re.search(r"\b(\d{6})\b", sibling.get_text())
                        if m:
                            return m.group(1)
                        if len(sibling.get_text()) > 200:
                            break

        # Pattern 3: Find any 6-digit sequence that appears once prominently
        # (fallback for old layouts)
        all_text = soup.get_text()
        six_digit_numbers = re.findall(r"\b(\d{6})\b", all_text)
        if six_digit_numbers:
            # The first prominent 6-digit number is likely the first prize
            # in older Sanook layouts
            return six_digit_numbers[0]

        return None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        keywords = ["เลขท้าย 2 ตัว", "เลขท้าย2ตัว", "สองตัวท้าย", "2 ตัวท้าย", "ท้าย 2"]
        for keyword in keywords:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el:
                parent = el.parent
                if parent:
                    # Look in parent and siblings for a 2-digit number
                    for node in [parent] + list(parent.find_next_siblings())[:5]:
                        m = re.search(r"\b(\d{2})\b", node.get_text())
                        if m:
                            return m.group(1)

        # Fallback: find all 2-digit standalone numbers near end of page
        # and take the one closest to 2-digit-back label
        two_digit_matches = re.findall(r"\b(\d{2})\b", soup.get_text())
        if two_digit_matches:
            # Return the last 2-digit number found (usually the lottery suffix)
            return two_digit_matches[-1]

        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        keywords = ["เลขท้าย 3 ตัว", "เลขท้าย3ตัว", "สามตัวท้าย", "3 ตัวท้าย", "ท้าย 3"]
        for keyword in keywords:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el:
                parent = el.parent
                if parent:
                    # Collect 3-digit numbers in parent and next siblings
                    for node in [parent] + list(parent.find_next_siblings())[:10]:
                        for m in re.finditer(r"\b(\d{3})\b", node.get_text()):
                            if m.group(1) not in results:
                                results.append(m.group(1))
                        if len(results) >= 2:
                            break
        return results[:2]  # Typically 2 three-digit back numbers

    def _extract_three_digit_front(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        keywords = ["เลขหน้า 3 ตัว", "เลขหน้า3ตัว", "สามตัวหน้า", "3 ตัวหน้า", "หน้า 3"]
        for keyword in keywords:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el:
                parent = el.parent
                if parent:
                    for node in [parent] + list(parent.find_next_siblings())[:10]:
                        for m in re.finditer(r"\b(\d{3})\b", node.get_text()):
                            if m.group(1) not in results:
                                results.append(m.group(1))
                        if len(results) >= 2:
                            break
        return results[:2]

    def _extract_near_prizes(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        keywords = ["ข้างเคียง", "ใกล้เคียง", "near"]
        for keyword in keywords:
            el = soup.find(string=re.compile(keyword, re.I))
            if el:
                parent = el.parent
                if parent:
                    for m in re.finditer(r"\b(\d{6})\b", parent.get_text()):
                        if m.group(1) not in results:
                            results.append(m.group(1))
        return results[:2]


# ---------------------------------------------------------------------------
# KapookParser
# ---------------------------------------------------------------------------


class KapookParser:
    """Parse kapook.com lottery result pages."""

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
        # Kapook uses class "lottery-result" or similar
        for cls_pattern in ["first-prize", "prize-1", "reward-1", "lotto-result"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        # Fallback: first 6-digit number in page
        six_digit = re.search(r"\b(\d{6})\b", soup.get_text())
        return six_digit.group(1) if six_digit else None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        for keyword in ["เลขท้าย 2 ตัว", "2 ตัวท้าย", "ท้าย 2"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:5]:
                    m = re.search(r"\b(\d{2})\b", node.get_text())
                    if m:
                        return m.group(1)
        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        for keyword in ["เลขท้าย 3 ตัว", "3 ตัวท้าย", "ท้าย 3"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:10]:
                    for m in re.finditer(r"\b(\d{3})\b", node.get_text()):
                        if m.group(1) not in results:
                            results.append(m.group(1))
                    if len(results) >= 2:
                        break
        return results[:2]


# ---------------------------------------------------------------------------
# GLOParser
# ---------------------------------------------------------------------------


class GLOParser:
    """Parse glo.or.th lottery result pages."""

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
        # GLO uses div/span with specific class names in their reward page
        for cls_pattern in ["reward-1", "first-prize", "prize1"]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el:
                digits = _normalize_digits(el.get_text(), 6)
                if digits:
                    return digits

        six_digit = re.search(r"\b(\d{6})\b", soup.get_text())
        return six_digit.group(1) if six_digit else None

    def _extract_two_digit_back(self, soup: BeautifulSoup) -> str | None:
        for keyword in ["เลขท้าย 2 ตัว", "2 ตัวท้าย"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:5]:
                    m = re.search(r"\b(\d{2})\b", node.get_text())
                    if m:
                        return m.group(1)
        return None

    def _extract_three_digit_back(self, soup: BeautifulSoup) -> list[str]:
        results: list[str] = []
        for keyword in ["เลขท้าย 3 ตัว", "3 ตัวท้าย"]:
            el = soup.find(string=re.compile(re.escape(keyword)))
            if el and el.parent:
                for node in [el.parent] + list(el.parent.find_next_siblings())[:10]:
                    for m in re.finditer(r"\b(\d{3})\b", node.get_text()):
                        if m.group(1) not in results:
                            results.append(m.group(1))
                    if len(results) >= 2:
                        break
        return results[:2]
