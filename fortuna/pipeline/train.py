"""Training pipeline — fit all models, register in models/registry.json. SPEC §8."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from fortuna.config import BKK, MODELS_DIR, check_not_icloud
from fortuna.models.base import TrainContext
from fortuna.store import DrawStore

logger = logging.getLogger(__name__)


def _train_data_hash(draws: list) -> str:
    """Stable hash of draw_ids in training set."""
    ids = sorted(d.draw_id for d in draws)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def _git_sha() -> str:
    """Get current HEAD commit SHA."""
    import subprocess
    from fortuna.config import REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_train(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Train all 4 base models on draws within [start_date, end_date].

    Saves artifacts to models/artifacts/{model_id}/ and updates models/registry.json.
    Returns summary of trained models.
    """
    check_not_icloud()

    # Load draw history
    store = DrawStore()
    all_draws = store.all_draws()

    # Filter by date range
    training_draws = all_draws
    if start_date:
        training_draws = [d for d in training_draws if d.draw_id >= start_date]
    if end_date:
        training_draws = [d for d in training_draws if d.draw_id <= end_date]

    training_draws = sorted(training_draws, key=lambda d: d.draw_id)

    logger.info(
        "Training on %d draws (range: %s to %s)",
        len(training_draws),
        training_draws[0].draw_id if training_draws else "none",
        training_draws[-1].draw_id if training_draws else "none",
    )

    if len(training_draws) < 10:
        raise ValueError(f"Not enough training draws: {len(training_draws)}")

    # "Target" is next draw after training window (conceptual)
    target_draw_id = "train-only"
    git_sha = _git_sha()
    data_hash = _train_data_hash(training_draws)

    ctx = TrainContext(
        draws=training_draws,
        features={},
        target_draw_id=target_draw_id,
        git_sha=git_sha,
    )

    # Instantiate all 4 models
    from fortuna.models.frequency_bayesian import FrequencyBayesian
    from fortuna.models.markov import MarkovModel
    from fortuna.models.lstm import LSTMModel
    from fortuna.models.rl_qlearn import RLQLearner

    models = [
        FrequencyBayesian(),
        MarkovModel(),
        LSTMModel(),
        RLQLearner(),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = MODELS_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trained_entries = []
    created_at = datetime.now(BKK).isoformat()

    for model in models:
        logger.info("Training %s...", model.model_id)
        try:
            model.fit(ctx)

            # Quick holdout score (last 10% of training data)
            holdout_start = max(0, len(training_draws) - max(10, len(training_draws) // 10))
            holdout = training_draws[holdout_start:]
            scores = model.score(holdout) if holdout else {"brier": None, "log_loss": None, "hit_rate": None}

            # Save artifact
            artifact_path = artifacts_dir / model.model_id
            model.serialize(artifact_path)

            entry = {
                "model_id": model.model_id,
                "version": "1.0",
                "fingerprint": model.fingerprint(),
                "train_data_hash": data_hash,
                "train_start": training_draws[0].draw_id if training_draws else None,
                "train_end": training_draws[-1].draw_id if training_draws else None,
                "n_draws": len(training_draws),
                "holdout_brier": scores.get("brier"),
                "holdout_hit_rate": scores.get("hit_rate"),
                "git_sha": git_sha,
                "created_at": created_at,
            }
            trained_entries.append(entry)
            logger.info(
                "  %s: brier=%.4f, hit_rate=%.4f",
                model.model_id,
                scores.get("brier") or 0.0,
                scores.get("hit_rate") or 0.0,
            )

        except Exception as e:
            logger.error("  Training failed for %s: %s", model.model_id, e)

    # Update registry
    registry = {
        "_comment": "Active model registry. Updated by training pipeline.",
        "schema_version": 1,
        "last_updated": created_at,
        "models": trained_entries,
    }
    registry_path = MODELS_DIR / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    logger.info("Registry updated: %d models registered", len(trained_entries))

    return {
        "n_models_trained": len(trained_entries),
        "n_draws": len(training_draws),
        "train_range": f"{training_draws[0].draw_id if training_draws else 'N/A'} to {training_draws[-1].draw_id if training_draws else 'N/A'}",
        "models": trained_entries,
    }
