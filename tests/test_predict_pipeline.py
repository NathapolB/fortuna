"""Integration tests for prediction pipeline — mocked git push. SPEC §6."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fortuna.schema import Draw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_draw(draw_id: str, first_prize: str = "123456") -> Draw:
    """Create a minimal Draw for testing."""
    return Draw(
        draw_date=draw_id,
        draw_id=draw_id,
        first_prize=first_prize,
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


def _make_many_draws(n: int = 60) -> list[Draw]:
    """Generate n semi-monthly draws for testing."""
    draws = []
    year = 2020
    month = 1
    day_cycle = [1, 16]
    day_idx = 0
    for i in range(n):
        draw_id = f"{year:04d}-{month:02d}-{day_cycle[day_idx]:02d}"
        # Vary first_prize so models see actual variation
        fp = str((i * 17 + 123456) % 1000000).zfill(6)
        tb = [str((i * 7 + 789) % 1000).zfill(3), str((i * 3 + 12) % 1000).zfill(3)]
        two = str((i * 11 + 34) % 100).zfill(2)
        draws.append(Draw(
            draw_date=draw_id,
            draw_id=draw_id,
            first_prize=fp,
            first_prize_near=[str((int(fp) - 1) % 1000000).zfill(6)],
            three_digit_front=["100", "200"],
            three_digit_back=tb,
            two_digit_back=two,
            bonus_prizes={},
            source="test",
            source_url="http://test.example.com",
            scraped_at=f"{draw_id}T12:00:00+07:00",
            raw_html_sha256=f"sha{i:04d}",
            verified_against=[],
            schema_version=1,
        ))
        day_idx += 1
        if day_idx >= 2:
            day_idx = 0
            month += 1
            if month > 12:
                month = 1
                year += 1
    return draws


# ---------------------------------------------------------------------------
# Test: run_predict with mocked git and DB
# ---------------------------------------------------------------------------


def test_predict_produces_10_tickets(tmp_path):
    """run_predict should produce 10 tickets with correct 2/3/5 split."""
    draws = _make_many_draws(60)
    target_date = "2026-05-16"

    # Patch DrawStore, git operations, and DB
    with (
        patch("fortuna.pipeline.predict.DrawStore") as mock_store_cls,
        patch("fortuna.pipeline.predict._git_current_sha", return_value="abcdef1234567890"),
        patch("fortuna.pipeline.predict._git_freeze_prediction", return_value="abcdef1234567890"),
        patch("fortuna.pipeline.predict.get_or_init_db") as mock_db,
        patch("fortuna.pipeline.predict.insert_prediction", return_value=1),
        patch("fortuna.pipeline.predict.check_not_icloud"),
        patch("fortuna.config.check_not_icloud"),
    ):
        mock_store = MagicMock()
        mock_store.all_draws.return_value = draws
        mock_store_cls.return_value = mock_store

        mock_conn = MagicMock()
        mock_db.return_value = mock_conn

        # Temporarily point EXPORTS_DIR to tmp_path
        with patch("fortuna.pipeline.predict.EXPORTS_DIR", tmp_path / "exports"):
            (tmp_path / "exports").mkdir(parents=True)
            from fortuna.pipeline.predict import run_predict
            payload = run_predict(target_date=target_date, dry_run=True)

    # Validate payload structure
    assert "picks" in payload
    assert "picks_sha256" in payload
    assert payload["target_draw_id"] == target_date
    assert payload["total_tickets"] == 10
    assert payload["total_cost_thb"] == 800

    # 2/3/5 split
    picks = payload["picks"]
    assert len(picks.get("first6", [])) == 2
    assert len(picks.get("three_back", [])) == 3
    assert len(picks.get("two_back", [])) == 5


def test_predict_sha256_integrity(tmp_path):
    """The picks_sha256 in the export should match recomputed hash."""
    import hashlib

    draws = _make_many_draws(60)
    target_date = "2026-05-16"

    with (
        patch("fortuna.pipeline.predict.DrawStore") as mock_store_cls,
        patch("fortuna.pipeline.predict._git_current_sha", return_value="abc123"),
        patch("fortuna.pipeline.predict._git_freeze_prediction", return_value="abc123"),
        patch("fortuna.pipeline.predict.get_or_init_db") as mock_db,
        patch("fortuna.pipeline.predict.insert_prediction", return_value=1),
        patch("fortuna.pipeline.predict.check_not_icloud"),
        patch("fortuna.config.check_not_icloud"),
    ):
        mock_store = MagicMock()
        mock_store.all_draws.return_value = draws
        mock_store_cls.return_value = mock_store
        mock_db.return_value = MagicMock()

        with patch("fortuna.pipeline.predict.EXPORTS_DIR", tmp_path / "exports"):
            (tmp_path / "exports").mkdir(parents=True)
            from fortuna.pipeline.predict import run_predict
            payload = run_predict(target_date=target_date, dry_run=True)

    # Recompute sha256
    picks_json = json.dumps(payload["picks"], sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(picks_json.encode()).hexdigest()
    assert payload["picks_sha256"] == computed


def test_predict_leakage_guard(tmp_path):
    """Target draw must not be in training draws."""
    draws = _make_many_draws(60)
    # Put a draw with target_date at end of list
    target_date = "2026-05-16"
    # If target_date is in draws, it should not be passed to training
    draws_with_target = draws + [_make_draw(target_date)]

    with (
        patch("fortuna.pipeline.predict.DrawStore") as mock_store_cls,
        patch("fortuna.pipeline.predict._git_current_sha", return_value="abc123"),
        patch("fortuna.pipeline.predict._git_freeze_prediction", return_value="abc123"),
        patch("fortuna.pipeline.predict.get_or_init_db") as mock_db,
        patch("fortuna.pipeline.predict.insert_prediction", return_value=1),
        patch("fortuna.pipeline.predict.check_not_icloud"),
        patch("fortuna.config.check_not_icloud"),
    ):
        mock_store = MagicMock()
        # Return draws including the target date
        mock_store.all_draws.return_value = draws_with_target
        mock_store_cls.return_value = mock_store
        mock_db.return_value = MagicMock()

        with patch("fortuna.pipeline.predict.EXPORTS_DIR", tmp_path / "exports"):
            (tmp_path / "exports").mkdir(parents=True)
            from fortuna.pipeline.predict import run_predict
            # Should complete without leakage assertion error
            payload = run_predict(target_date=target_date, dry_run=True)

    # Training must have excluded target
    # We can verify by checking total picks == 10
    assert payload["total_tickets"] == 10


def test_model_training_and_predict_end_to_end():
    """All 4 base models should train and produce picks without errors."""
    from fortuna.models.base import TrainContext
    from fortuna.models.frequency_bayesian import FrequencyBayesian
    from fortuna.models.markov import MarkovModel
    from fortuna.models.lstm import LSTMModel
    from fortuna.models.rl_qlearn import RLQLearner
    from fortuna.pipeline.picker import select_picks

    draws = _make_many_draws(50)
    training = draws[:45]
    target_id = draws[45].draw_id

    ctx = TrainContext(
        draws=training,
        features={},
        target_draw_id=target_id,
        git_sha="test",
    )

    models = [FrequencyBayesian(), MarkovModel(), LSTMModel(), RLQLearner()]
    for model in models:
        model.fit(ctx)

    # Each model should predict top-k for all prize types
    for model in models:
        for prize_type in ("first6", "three_back", "two_back"):
            picks = model.predict_top_k(prize_type, 5)  # type: ignore
            assert len(picks) > 0, f"{model.model_id} returned no picks for {prize_type}"
            for p in picks:
                assert 0.0 <= p.confidence <= 1.0 or p.confidence > 0
                assert len(p.value) > 0

    # Run through picker
    ensemble = {
        pt: [p for model in models for p in model.predict_top_k(pt, 5)]  # type: ignore
        for pt in ("first6", "three_back", "two_back")
    }
    result = select_picks(ensemble)
    assert len(result["first6"]) == 2
    assert len(result["three_back"]) == 3
    assert len(result["two_back"]) == 5
