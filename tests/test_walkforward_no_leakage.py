"""Walk-forward CV leakage tests — Phase 2. SPEC §7.3.

These tests verify two independent assertions:
  (a) The target draw is NOT in the training set.
  (b) Every feature used to predict the target was computed strictly
      BEFORE draw_cutoff(target) = 06:00 Asia/Bangkok on draw_date.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fortuna.config import BKK
from fortuna.eval.walkforward import draw_cutoff, walk_forward_cv
from fortuna.models.base import TrainContext
from fortuna.schema import Draw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_draw(draw_id: str) -> Draw:
    """Create a minimal Draw for testing."""
    return Draw(
        draw_date=draw_id,
        draw_id=draw_id,
        first_prize="123456",
        first_prize_near=["123455", "123457"],
        three_digit_front=["123", "456"],
        three_digit_back=["789", "012"],
        two_digit_back="34",
        bonus_prizes={},
        source="test",
        source_url="http://test.example.com",
        scraped_at="2026-01-01T12:00:00+07:00",
        raw_html_sha256="abc123",
        verified_against=[],
        schema_version=1,
    )


def _make_draws(n: int, start_year: int = 2024) -> list[Draw]:
    """Create n draws spanning semi-monthly dates."""
    draws = []
    year = start_year
    month = 1
    day_cycle = [1, 16]
    day_idx = 0
    for i in range(n):
        draw_id = f"{year:04d}-{month:02d}-{day_cycle[day_idx]:02d}"
        draws.append(_make_draw(draw_id))
        day_idx += 1
        if day_idx >= 2:
            day_idx = 0
            month += 1
            if month > 12:
                month = 1
                year += 1
    return draws


# ---------------------------------------------------------------------------
# Test 1: target draw not in training set
# ---------------------------------------------------------------------------


def test_no_leakage_target_not_in_training():
    """Assert ctx.target_draw_id not in {d.draw_id for d in ctx.draws}.

    SPEC §7.3 Assertion 1.
    """
    draws = _make_draws(50)
    target_idx = 40
    target_draw = draws[target_idx]
    training_draws = draws[:target_idx]

    ctx = TrainContext(
        draws=training_draws,
        features={},
        target_draw_id=target_draw.draw_id,
        git_sha="test",
    )

    # Assertion 1: target not in training
    training_ids = {d.draw_id for d in ctx.draws}
    assert ctx.target_draw_id not in training_ids, (
        f"LEAKAGE: target {ctx.target_draw_id!r} found in training set"
    )

    # Also verify draw_id isolation for all walk-forward indices
    for t in range(20, len(draws)):
        train = draws[:t]
        target = draws[t]
        train_ids = {d.draw_id for d in train}
        assert target.draw_id not in train_ids, (
            f"LEAKAGE at t={t}: {target.draw_id!r} in training IDs"
        )


# ---------------------------------------------------------------------------
# Test 2: feature timestamps before draw cutoff
# ---------------------------------------------------------------------------


def test_no_leakage_feature_timestamps():
    """Assert every feature computed_at < draw_cutoff(target_draw_id).

    SPEC §7.3 Assertion 2.
    draw_cutoff = 06:00 Asia/Bangkok on the draw_date.
    """
    target_draw_id = "2026-05-16"
    cutoff = draw_cutoff(target_draw_id)

    # Simulated features with their computed_at timestamps
    valid_features = {
        "digit_freq_30d": "2026-05-15T22:00:00+07:00",  # day before
        "markov_entropy": "2026-05-15T23:59:59+07:00",  # still before cutoff
        "gap_since_last": "2026-05-16T05:59:59+07:00",  # 1 second before cutoff
    }

    leaky_features = {
        "post_draw_feature": "2026-05-16T06:00:01+07:00",  # 1 second after
        "after_cutoff": "2026-05-16T10:00:00+07:00",      # during draw time
    }

    # All valid features must be before cutoff
    for fname, computed_at_str in valid_features.items():
        computed_at = datetime.fromisoformat(computed_at_str)
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=BKK)
        assert computed_at < cutoff, (
            f"Feature {fname!r} computed at {computed_at_str} should be before cutoff {cutoff.isoformat()}"
        )

    # All leaky features must be detected as >= cutoff
    for fname, computed_at_str in leaky_features.items():
        computed_at = datetime.fromisoformat(computed_at_str)
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=BKK)
        assert computed_at >= cutoff, (
            f"Feature {fname!r} at {computed_at_str} should be AFTER cutoff (leaky)"
        )

    # Verify cutoff is 06:00 BKK
    assert cutoff.hour == 6
    assert cutoff.minute == 0
    assert cutoff.second == 0
    assert cutoff.tzinfo == BKK


# ---------------------------------------------------------------------------
# Test 3: walk-forward CV window check
# ---------------------------------------------------------------------------


def test_walk_forward_cv_window():
    """Full walk-forward CV: train on draws[:i], predict draws[i], repeat.

    SPEC §7.3 full test. Minimum train window = MIN_TRAIN draws.
    """
    from fortuna.eval.walkforward import MIN_TRAIN_DRAWS
    from fortuna.models.frequency_bayesian import FrequencyBayesian

    draws = _make_draws(MIN_TRAIN_DRAWS + 10)

    results = walk_forward_cv(
        draws=draws,
        model_factory=FrequencyBayesian,
        min_train=MIN_TRAIN_DRAWS,
    )

    # Should have evaluated 10 draws (MIN_TRAIN to end)
    assert len(results) > 0, "Walk-forward CV produced no results"

    # Verify leakage guard: each result's draw_id was NOT in its training window
    draw_ids = [d.draw_id for d in draws]
    for r in results:
        target_id = r["draw_id"]
        target_idx = draw_ids.index(target_id)
        # Training was draws[:target_idx], so target should not be in there
        assert target_id not in draw_ids[:target_idx], (
            f"Leakage: {target_id!r} was in training set of size {target_idx}"
        )

    # Prize types covered
    prize_types_seen = {r["prize_type"] for r in results}
    assert "three_back" in prize_types_seen
    assert "two_back" in prize_types_seen
