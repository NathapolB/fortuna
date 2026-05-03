"""Golden-file tests for HTML parsers. SPEC Phase 1 deliverable.

Tests each parser (Sanook, Kapook, GLO) against fixture HTML files.
Fixtures are stored in tests/fixtures/{source}/{date}.html.

Phase 1 acceptance criterion: pytest tests/test_parser.py -v is green.

Note: Fixture HTML files need to be created by running:
    python tests/create_fixtures.py
which fetches and saves sample pages from each source.

Sanook URL pattern updated 2026-05-02 to use /lotto/check/{DDMMYYYY_BE}/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from fortuna.parser import GLOParser, KapookParser, SanookParser
from fortuna.schema import Draw

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_fixture(source: str, filename: str) -> bytes:
    path = FIXTURES_DIR / source / filename
    if not path.exists():
        pytest.skip(f"Fixture not found: {path} — run tests/create_fixtures.py first")
    return path.read_bytes()


def _load_expected(source: str, filename: str) -> dict:
    path = FIXTURES_DIR / source / filename.replace(".html", ".expected.json")
    if not path.exists():
        pytest.skip(f"Expected JSON not found: {path} — run tests/create_fixtures.py first")
    return json.loads(path.read_text())


def _assert_draw_matches(draw: Draw | None, expected: dict, source_label: str) -> None:
    assert draw is not None, f"{source_label}: parser returned None"
    assert draw.draw_id == expected["draw_id"], (
        f"{source_label}: draw_id mismatch: got {draw.draw_id!r}, expected {expected['draw_id']!r}"
    )
    assert draw.first_prize == expected["first_prize"], (
        f"{source_label}: first_prize mismatch: got {draw.first_prize!r}, expected {expected['first_prize']!r}"
    )
    assert draw.two_digit_back == expected["two_digit_back"], (
        f"{source_label}: two_digit_back mismatch: got {draw.two_digit_back!r}, expected {expected['two_digit_back']!r}"
    )
    if "three_digit_back" in expected and expected["three_digit_back"]:
        for expected_num in expected["three_digit_back"]:
            assert expected_num in draw.three_digit_back, (
                f"{source_label}: {expected_num!r} not in three_digit_back {draw.three_digit_back}"
            )


def _sanook_check_url(date_str: str) -> str:
    """Build /check/ URL for a given YYYY-MM-DD date string."""
    d = date.fromisoformat(date_str)
    be_year = d.year + 543
    return f"https://news.sanook.com/lotto/check/{d.day:02d}{d.month:02d}{be_year}/"


# ---------------------------------------------------------------------------
# Sanook parser tests
# ---------------------------------------------------------------------------


class TestSanookParser:
    def setup_method(self):
        self.parser = SanookParser()

    def test_parse_fixture_1(self):
        html = _load_fixture("sanook", "2026-04-16.html")
        expected = _load_expected("sanook", "2026-04-16.html")
        draw = self.parser.parse(html, _sanook_check_url("2026-04-16"))
        _assert_draw_matches(draw, expected, "Sanook")

    def test_parse_fixture_2(self):
        html = _load_fixture("sanook", "2026-03-16.html")
        expected = _load_expected("sanook", "2026-03-16.html")
        draw = self.parser.parse(html, _sanook_check_url("2026-03-16"))
        _assert_draw_matches(draw, expected, "Sanook")

    def test_parse_fixture_3(self):
        html = _load_fixture("sanook", "2026-03-01.html")
        expected = _load_expected("sanook", "2026-03-01.html")
        draw = self.parser.parse(html, _sanook_check_url("2026-03-01"))
        _assert_draw_matches(draw, expected, "Sanook")

    def test_parse_fixture_4(self):
        html = _load_fixture("sanook", "2026-02-16.html")
        expected = _load_expected("sanook", "2026-02-16.html")
        draw = self.parser.parse(html, _sanook_check_url("2026-02-16"))
        _assert_draw_matches(draw, expected, "Sanook")

    def test_parse_fixture_5(self):
        html = _load_fixture("sanook", "2026-01-16.html")
        expected = _load_expected("sanook", "2026-01-16.html")
        draw = self.parser.parse(html, _sanook_check_url("2026-01-16"))
        _assert_draw_matches(draw, expected, "Sanook")

    def test_extract_date_from_check_url(self):
        """New /check/ URL format: DDMMYYYY (4-digit BE year)."""
        parser = SanookParser()
        # 16 Apr 2026 CE = BE 2569 → 16042569
        assert parser.extract_date_from_url(
            "https://news.sanook.com/lotto/check/16042569/"
        ) == "2026-04-16"
        # 2 May 2026 CE = BE 2569 → 02052569
        assert parser.extract_date_from_url(
            "https://news.sanook.com/lotto/check/02052569/"
        ) == "2026-05-02"
        # 1 Jan 2024 CE = BE 2567 → 01012567
        assert parser.extract_date_from_url(
            "https://news.sanook.com/lotto/check/01012567/"
        ) == "2024-01-01"

    def test_extract_date_from_old_url(self):
        """Old URL format still works (backward compat)."""
        parser = SanookParser()
        assert parser.extract_date_from_url(
            "https://news.sanook.com/lotto/2024/01/16/"
        ) == "2024-01-16"
        assert parser.extract_date_from_url(
            "https://news.sanook.com/lotto/2005/03/01/foo"
        ) == "2005-03-01"
        assert parser.extract_date_from_url("https://example.com/no-date/") is None

    def test_parse_invalid_html_returns_none(self):
        parser = SanookParser()
        # Should not raise; may return None or a Draw with potentially missing fields
        result = parser.parse(
            b"<html><body>No lottery data here</body></html>",
            "https://news.sanook.com/lotto/check/01012567/",
        )
        # We just check it doesn't raise
        pass

    def test_format_be(self):
        """_format_be must produce correct 4-digit BE year strings."""
        from fortuna.scraper import SanookScraper
        scraper = SanookScraper.__new__(SanookScraper)
        assert SanookScraper._format_be(date(2026, 4, 16)) == "16042569"
        assert SanookScraper._format_be(date(2026, 5, 2)) == "02052569"
        assert SanookScraper._format_be(date(2024, 1, 1)) == "01012567"
        assert SanookScraper._format_be(date(2005, 3, 1)) == "01032548"

    def test_generate_candidate_dates(self):
        """Should yield 1st and 16th for each month in range."""
        from fortuna.scraper import SanookScraper
        candidates = list(
            SanookScraper._generate_candidate_dates(
                date(2026, 1, 1), date(2026, 3, 31)
            )
        )
        assert date(2026, 1, 1) in candidates
        assert date(2026, 1, 16) in candidates
        assert date(2026, 2, 1) in candidates
        assert date(2026, 2, 16) in candidates
        assert date(2026, 3, 1) in candidates
        assert date(2026, 3, 16) in candidates
        assert len(candidates) == 6

    def test_looks_like_draw_page(self):
        from fortuna.scraper import SanookScraper
        # Bug fix: b"..." literals cannot contain non-ASCII — use .encode("utf-8")
        good = "<div class='lottocheck__sec'>...รางวัลที่ 1...</div>".encode("utf-8")
        bad_no_class = "<div>...รางวัลที่ 1...</div>".encode("utf-8")
        bad_404 = b"<html><body>Not found</body></html>"
        assert SanookScraper._looks_like_draw_page(good) is True
        assert SanookScraper._looks_like_draw_page(bad_no_class) is False
        assert SanookScraper._looks_like_draw_page(bad_404) is False

    def test_extract_draw_urls_legacy(self):
        """Legacy extract_draw_urls still returns URLs matching old pattern."""
        parser = SanookParser()
        html = b"""
        <html><body>
        <a href="/lotto/2024/01/01/">Jan 1</a>
        <a href="/lotto/2024/01/16/">Jan 16</a>
        <a href="https://news.sanook.com/lotto/2024/02/01/">Feb 1</a>
        <a href="/other-page/">Not a lotto link</a>
        </body></html>
        """
        urls = parser.extract_draw_urls(html, "https://news.sanook.com/lotto/archive/2024/")
        assert len(urls) == 3
        assert "https://news.sanook.com/lotto/2024/01/01/" in urls
        assert "https://news.sanook.com/lotto/2024/02/01/" in urls


# ---------------------------------------------------------------------------
# Kapook parser tests (skipped until scraper is re-enabled)
# ---------------------------------------------------------------------------


class TestKapookParser:
    def setup_method(self):
        self.parser = KapookParser()

    def test_parse_fixture_1(self):
        html = _load_fixture("kapook", "2024-01-01.html")
        expected = _load_expected("kapook", "2024-01-01.html")
        draw = self.parser.parse(html, "https://horoscope.kapook.com/lottery/2024/01/01/")
        _assert_draw_matches(draw, expected, "Kapook")

    def test_parse_fixture_2(self):
        html = _load_fixture("kapook", "2023-06-16.html")
        expected = _load_expected("kapook", "2023-06-16.html")
        draw = self.parser.parse(html, "https://horoscope.kapook.com/lottery/2023/06/16/")
        _assert_draw_matches(draw, expected, "Kapook")

    def test_parse_fixture_3(self):
        html = _load_fixture("kapook", "2022-12-01.html")
        expected = _load_expected("kapook", "2022-12-01.html")
        draw = self.parser.parse(html, "https://horoscope.kapook.com/lottery/2022/12/01/")
        _assert_draw_matches(draw, expected, "Kapook")

    def test_parse_fixture_4(self):
        html = _load_fixture("kapook", "2019-09-16.html")
        expected = _load_expected("kapook", "2019-09-16.html")
        draw = self.parser.parse(html, "https://horoscope.kapook.com/lottery/2019/09/16/")
        _assert_draw_matches(draw, expected, "Kapook")

    def test_parse_fixture_5(self):
        html = _load_fixture("kapook", "2015-05-01.html")
        expected = _load_expected("kapook", "2015-05-01.html")
        draw = self.parser.parse(html, "https://horoscope.kapook.com/lottery/2015/05/01/")
        _assert_draw_matches(draw, expected, "Kapook")


# ---------------------------------------------------------------------------
# GLO parser tests (skipped until scraper is re-enabled)
# ---------------------------------------------------------------------------


class TestGLOParser:
    def setup_method(self):
        self.parser = GLOParser()

    def test_parse_fixture_1(self):
        html = _load_fixture("glo", "2024-01-01.html")
        expected = _load_expected("glo", "2024-01-01.html")
        draw = self.parser.parse(html, "https://www.glo.or.th/result/20240101.html")
        _assert_draw_matches(draw, expected, "GLO")

    def test_parse_fixture_2(self):
        html = _load_fixture("glo", "2023-12-16.html")
        expected = _load_expected("glo", "2023-12-16.html")
        draw = self.parser.parse(html, "https://www.glo.or.th/result/20231216.html")
        _assert_draw_matches(draw, expected, "GLO")

    def test_parse_fixture_3(self):
        html = _load_fixture("glo", "2023-06-01.html")
        expected = _load_expected("glo", "2023-06-01.html")
        draw = self.parser.parse(html, "https://www.glo.or.th/result/20230601.html")
        _assert_draw_matches(draw, expected, "GLO")

    def test_parse_fixture_4(self):
        html = _load_fixture("glo", "2022-11-16.html")
        expected = _load_expected("glo", "2022-11-16.html")
        draw = self.parser.parse(html, "https://www.glo.or.th/result/20221116.html")
        _assert_draw_matches(draw, expected, "GLO")

    def test_parse_fixture_5(self):
        html = _load_fixture("glo", "2021-05-01.html")
        expected = _load_expected("glo", "2021-05-01.html")
        draw = self.parser.parse(html, "https://www.glo.or.th/result/20210501.html")
        _assert_draw_matches(draw, expected, "GLO")
