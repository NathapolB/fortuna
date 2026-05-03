"""Metrics tests — Phase 2. SPEC §7.1.

Tests for fortuna/eval/metrics.py and fortuna/eval/stats.py.
"""

from __future__ import annotations

import math

import pytest

from fortuna.eval.metrics import brier_score, log_loss, hit_rate, lift, sharpe_pnl


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score():
    """Brier score = mean((p - y)^2). SPEC §7.1."""
    # Perfect predictions
    assert brier_score([1.0, 1.0, 0.0], [1, 1, 0]) == pytest.approx(0.0, abs=1e-9)
    # Worst predictions
    assert brier_score([0.0, 0.0, 1.0], [1, 1, 0]) == pytest.approx(1.0, abs=1e-9)
    # Mixed: (0.5-1)^2 + (0.9-1)^2 + (0.1-0)^2 = 0.25 + 0.01 + 0.01 = 0.27
    # mean = 0.09
    assert brier_score([0.5, 0.9, 0.1], [1, 1, 0]) == pytest.approx(0.09, abs=1e-9)


def test_brier_score_length_mismatch():
    """Mismatched lengths should raise ValueError."""
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1, 0, 1])


def test_brier_score_single():
    """Single element: (0.3 - 0)^2 = 0.09."""
    assert brier_score([0.3], [0]) == pytest.approx(0.09, abs=1e-9)


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def test_log_loss():
    """Log loss = -mean(y log p + (1-y) log(1-p)). SPEC §7.1."""
    # Perfect predictions (clamped to avoid log(0))
    score = log_loss([1.0, 0.0], [1, 0])
    assert score < 0.001  # near zero (eps clamping makes it not exactly 0)

    # Uniform predictions: -mean(1*log(0.5) + 0) = log(2) ≈ 0.693
    score = log_loss([0.5, 0.5], [1, 0])
    assert score == pytest.approx(math.log(2), abs=1e-6)


def test_log_loss_length_mismatch():
    with pytest.raises(ValueError):
        log_loss([0.5], [1, 0])


def test_log_loss_all_positive():
    """All positive outcomes."""
    score = log_loss([0.9, 0.8, 0.7], [1, 1, 1])
    expected = -sum(math.log(p) for p in [0.9, 0.8, 0.7]) / 3
    assert score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


def test_hit_rate():
    """hit_rate = hits / tickets. SPEC §7.1."""
    assert hit_rate(3, 10) == pytest.approx(0.3)
    assert hit_rate(0, 100) == pytest.approx(0.0)
    assert hit_rate(10, 10) == pytest.approx(1.0)


def test_hit_rate_zero_tickets():
    """Zero tickets should return 0.0, not divide by zero."""
    assert hit_rate(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


def test_lift():
    """lift = hit_rate / random_hit_rate. SPEC §7.1."""
    assert lift(0.05, 0.01) == pytest.approx(5.0)
    assert lift(0.01, 0.01) == pytest.approx(1.0)
    assert lift(0.0, 0.01) == pytest.approx(0.0)


def test_lift_zero_random():
    """Zero random hit rate should return inf."""
    assert lift(0.05, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------


def test_bh_fdr_correction():
    """BH-FDR correction across (model x prize_type) cells. SPEC §7.2."""
    from fortuna.eval.stats import bh_fdr_correction

    p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
    reject, corrected = bh_fdr_correction(p_values, alpha=0.05)

    assert len(reject) == len(p_values)
    assert len(corrected) == len(p_values)
    # At alpha=0.05, 0.001 and 0.01 should typically be rejected
    # Use bool() cast to handle numpy.bool_ vs Python bool identity comparison
    assert bool(reject[0]) is True   # p=0.001 — clearly significant
    assert bool(reject[-1]) is False  # p=0.5 — not significant


def test_bh_fdr_empty():
    """Empty p-value list."""
    from fortuna.eval.stats import bh_fdr_correction
    reject, corrected = bh_fdr_correction([])
    assert reject == []
    assert corrected == []


# ---------------------------------------------------------------------------
# Binomial test — minimum n requirement
# ---------------------------------------------------------------------------


def test_binomial_minimum_n():
    """Binomial test requires n >= 50 draws per cell before any claim. SPEC §7.2."""
    from fortuna.eval.stats import binomial_test

    # n < 50: should return valid=False
    p_value, valid = binomial_test(hits=5, n=20, p_null=0.01)
    assert valid is False
    assert p_value == 1.0

    # n >= 50: should return valid=True
    p_value, valid = binomial_test(hits=5, n=50, p_null=0.01)
    assert valid is True
    assert 0.0 <= p_value <= 1.0


# ---------------------------------------------------------------------------
# Chi-square uniformity
# ---------------------------------------------------------------------------


def test_chi_square_uniformity():
    """Chi-square goodness-of-fit on digit positions. SPEC §7.2."""
    # Perfectly uniform distribution across 10 digits should have large p-value
    from scipy.stats import chisquare  # type: ignore

    uniform_counts = [100] * 10
    stat, p_value = chisquare(uniform_counts)
    assert p_value > 0.05, f"Uniform distribution should not reject chi-square: p={p_value}"

    # Very non-uniform (all in one digit) should have tiny p-value
    skewed_counts = [1000] + [0] * 9
    stat, p_value = chisquare(skewed_counts)
    assert p_value < 0.001, f"Skewed distribution should reject chi-square: p={p_value}"
