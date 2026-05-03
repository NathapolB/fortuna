"""Picker tests — SPEC §6.1. Verify 2/3/5 split and Hamming diversity."""

from __future__ import annotations

import pytest

from fortuna.models.base import Pick
from fortuna.pipeline.picker import hamming_distance, select_picks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_picks(values: list[str], base_conf: float = 0.9) -> list[Pick]:
    """Create sorted list of Picks from values."""
    return [
        Pick(value=v, confidence=base_conf - i * 0.05, rationale="test")
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------


def test_hamming_same():
    assert hamming_distance("123", "123") == 0


def test_hamming_all_different():
    assert hamming_distance("000", "111") == 3


def test_hamming_one_diff():
    assert hamming_distance("123", "124") == 1


def test_hamming_length_mismatch():
    with pytest.raises(ValueError):
        hamming_distance("12", "123")


# ---------------------------------------------------------------------------
# 2/3/5 split
# ---------------------------------------------------------------------------


def test_select_picks_counts():
    """select_picks must return exactly 2/3/5 picks."""
    ensemble = {
        "first6": _make_picks(["123456", "234567", "345678", "456789", "567890"]),
        "three_back": _make_picks(["123", "456", "789", "012", "345"]),
        "two_back": _make_picks([str(i).zfill(2) for i in range(20)]),
    }
    result = select_picks(ensemble)

    assert len(result["first6"]) == 2
    assert len(result["three_back"]) == 3
    assert len(result["two_back"]) == 5
    assert sum(len(v) for v in result.values()) == 10


# ---------------------------------------------------------------------------
# Hamming diversity
# ---------------------------------------------------------------------------


def test_first6_hamming_diversity():
    """first6 picks must have Hamming distance >= 2."""
    # Provide picks where first two differ by only 1 digit to test filtering
    ensemble = {
        "first6": _make_picks([
            "123456",  # pick 1
            "123457",  # Hamming=1 vs pick 1 — should be skipped
            "223456",  # Hamming=1 vs pick 1 — should be skipped
            "133456",  # Hamming=1 vs pick 1 — should be skipped
            "223457",  # Hamming=2 vs "123456" — should be selected
        ]),
        "three_back": _make_picks(["123", "456", "789"]),
        "two_back": _make_picks([str(i).zfill(2) for i in range(20)]),
    }
    result = select_picks(ensemble)
    first6 = result["first6"]
    assert len(first6) == 2
    hd = hamming_distance(first6[0], first6[1])
    assert hd >= 2, f"first6 Hamming distance {hd} < 2: {first6}"


def test_three_back_hamming_diversity():
    """three_back picks must all be distinct (Hamming >= 1)."""
    ensemble = {
        "first6": _make_picks(["123456", "987654"]),
        "three_back": _make_picks(["123", "456", "789", "012"]),
        "two_back": _make_picks([str(i).zfill(2) for i in range(20)]),
    }
    result = select_picks(ensemble)
    three_back = result["three_back"]
    assert len(three_back) == 3
    assert len(set(three_back)) == 3, f"Duplicate picks in three_back: {three_back}"


def test_two_back_no_duplicates():
    """two_back must have 5 distinct picks."""
    ensemble = {
        "first6": _make_picks(["123456", "987654"]),
        "three_back": _make_picks(["123", "456", "789"]),
        "two_back": _make_picks([str(i).zfill(2) for i in range(20)]),
    }
    result = select_picks(ensemble)
    two_back = result["two_back"]
    assert len(two_back) == 5
    assert len(set(two_back)) == 5, f"Duplicate picks in two_back: {two_back}"


# ---------------------------------------------------------------------------
# Fallback generation
# ---------------------------------------------------------------------------


def test_fallback_when_insufficient_candidates():
    """Picker should generate fallbacks when model provides fewer candidates than needed."""
    ensemble = {
        "first6": _make_picks(["123456"]),  # only 1 candidate for 2 needed
        "three_back": _make_picks(["123"]),  # only 1 candidate for 3 needed
        "two_back": _make_picks(["01"]),     # only 1 candidate for 5 needed
    }
    result = select_picks(ensemble)

    assert len(result["first6"]) == 2
    assert len(result["three_back"]) == 3
    assert len(result["two_back"]) == 5
    # Still satisfy diversity
    first6 = result["first6"]
    assert hamming_distance(first6[0], first6[1]) >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_select_picks_empty_input():
    """Empty ensemble input should produce valid fallback picks."""
    result = select_picks({
        "first6": [],
        "three_back": [],
        "two_back": [],
    })
    assert len(result["first6"]) == 2
    assert len(result["three_back"]) == 3
    assert len(result["two_back"]) == 5
