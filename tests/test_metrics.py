"""Metrics tests — Phase 2 stub. SPEC §7.1.

Tests for fortuna/eval/metrics.py and fortuna/eval/stats.py.
Stubbed with @pytest.mark.skip(reason='Phase 2') per SPEC Phase 1 criterion 6.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2")
def test_brier_score():
    """Brier score = mean((p - y)^2). SPEC §7.1."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_log_loss():
    """Log loss = -mean(y log p + (1-y) log(1-p)). SPEC §7.1."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_hit_rate():
    """hit_rate = hits / tickets. SPEC §7.1."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_lift():
    """lift = hit_rate / random_hit_rate. SPEC §7.1."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_bh_fdr_correction():
    """BH-FDR correction across (model x prize_type) cells. SPEC §7.2."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_binomial_minimum_n():
    """Binomial test requires n >= 50 draws per cell before any claim. SPEC §7.2."""
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_chi_square_uniformity():
    """Chi-square goodness-of-fit on digit positions. SPEC §7.2."""
    raise NotImplementedError("Phase 2")
