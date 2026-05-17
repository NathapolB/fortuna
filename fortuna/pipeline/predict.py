"""Prediction pipeline — main entrypoint for producing frozen predictions. SPEC §6.

Steps:
  1. Load all models from models/registry.json
  2. Compute features for target draw (with leakage guards)
  3. Run all 4 base models -> meta-learner -> picker
  4. Write data/exports/<date>-prediction.json with picks, sha256, git_sha_at_freeze
  5. git add + commit + push (tamper-evidence anchor)
  6. Capture commit SHA -> store in DB predictions.freeze_commit_sha
  7. Publish prediction page to Notion (non-blocking; skipped if token absent)

Leakage guard (Enhancement-3):
  In live mode (allow_leak=False, default), raises ValueError if called after
  draw cutoff. Pass allow_leak=True only in test/backtest scenarios.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import cast

from fortuna.config import (
    BKK,
    EXPORTS_DIR,
    MODELS_DIR,
    PICK_SPLIT,
    REPO_ROOT,
    check_not_icloud,
)
from fortuna.models.base import Pick, PrizeType, TrainContext
from fortuna.pipeline.picker import select_picks
from fortuna.store import DrawStore, get_or_init_db, insert_prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_class(model_id: str):
    """Dynamically load model class by model_id prefix."""
    if model_id.startswith("frequency-bayes"):
        from fortuna.models.frequency_bayesian import FrequencyBayesian
        return FrequencyBayesian
    elif model_id.startswith("markov"):
        from fortuna.models.markov import MarkovModel
        return MarkovModel
    elif model_id.startswith("lstm"):
        from fortuna.models.lstm import LSTMModel
        return LSTMModel
    elif model_id.startswith("rl-qlearn"):
        from fortuna.models.rl_qlearn import RLQLearner
        return RLQLearner
    else:
        raise ValueError(f"Unknown model_id prefix: {model_id!r}")


def load_all_models(registry_path: Path = MODELS_DIR / "registry.json") -> list:
    """Load all active models from registry. Returns list[BaseModel]."""
    with open(registry_path) as f:
        registry = json.load(f)

    models = []
    for entry in registry.get("models", []):
        model_id = entry["model_id"]
        artifact_path = MODELS_DIR / "artifacts" / model_id
        if not artifact_path.exists():
            logger.warning("Artifact path missing for %s — skipping", model_id)
            continue
        cls = _load_model_class(model_id)
        model = cls.load(artifact_path)
        models.append(model)
        logger.info("Loaded model: %s", model_id)

    return models


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_current_sha() -> str:
    """Get current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_freeze_prediction(export_path: Path) -> str:
    """Commit and push prediction file. Returns commit SHA.

    SPEC §6: push must succeed BEFORE 07:00 Asia/Bangkok on predict day
    (day 2 or 17 of month — 14 days before draw).
    """
    rel_path = export_path.relative_to(REPO_ROOT)

    subprocess.run(
        ["git", "add", str(rel_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"freeze: prediction {export_path.stem}"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=REPO_ROOT,
        check=True,
    )
    return _git_current_sha()


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


def run_predict(
    target_date: str,
    dry_run: bool = False,
    allow_leak: bool = False,
) -> dict:
    """Run prediction pipeline for target_date (YYYY-MM-DD).

    Args:
        target_date: Draw date to predict for (YYYY-MM-DD).
        dry_run: If True, skip git push. Useful for testing.
        allow_leak: If True, downgrade the post-cutoff leakage guard from a
                    hard ValueError to a warning. Intended for backtest/test
                    scenarios ONLY. Default False (hard block in live mode).

    Returns:
        Prediction dict with picks + metadata.

    Raises:
        ValueError: If current time is >= draw cutoff AND allow_leak=False.
    """
    check_not_icloud()

    from fortuna.eval.walkforward import draw_cutoff

    # --- Enhancement-3: Leakage guard (hard block unless allow_leak) ---
    # For live predict, cutoff = predict_started_at (now). We predict ~14 days
    # early (day 2/17), so this guard is normally never triggered.
    # For backtest, cutoff = T - 14 days (enforced by caller; allow_leak=True).
    predict_started_at = datetime.now(BKK)
    cutoff = draw_cutoff(target_date)
    if predict_started_at >= cutoff:
        msg = (
            f"LEAKAGE GUARD: Current time {predict_started_at.isoformat()} >= "
            f"draw cutoff {cutoff.isoformat()} for draw {target_date}. "
            "Features computed after cutoff are LEAKED. "
            "Re-run earlier, or pass allow_leak=True (testing only)."
        )
        if allow_leak:
            logger.warning(msg)
        else:
            raise ValueError(msg)

    # Load draw history
    store = DrawStore()
    all_draws = store.all_draws()
    # Filter: only draws strictly before target_date
    training_draws = [d for d in all_draws if d.draw_id < target_date]

    logger.info(
        "Loaded %d training draws (excluding target %s)", len(training_draws), target_date
    )

    if len(training_draws) < 10:
        raise ValueError(f"Not enough training draws: {len(training_draws)}")

    # Load or instantiate models
    registry_path = MODELS_DIR / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    registered_models = registry.get("models", [])
    if not registered_models:
        logger.warning("No models in registry — using freshly-initialized defaults")
        from fortuna.models.frequency_bayesian import FrequencyBayesian
        from fortuna.models.markov import MarkovModel
        from fortuna.models.lstm import LSTMModel
        from fortuna.models.rl_qlearn import RLQLearner
        base_models = [
            FrequencyBayesian(),
            MarkovModel(),
            LSTMModel(),
            RLQLearner(),
        ]
    else:
        try:
            base_models = load_all_models(registry_path)
        except Exception as e:
            logger.warning("Registry load failed (%s) — using defaults", e)
            from fortuna.models.frequency_bayesian import FrequencyBayesian
            from fortuna.models.markov import MarkovModel
            from fortuna.models.lstm import LSTMModel
            from fortuna.models.rl_qlearn import RLQLearner
            base_models = [
                FrequencyBayesian(),
                MarkovModel(),
                LSTMModel(),
                RLQLearner(),
            ]

    ctx = TrainContext(
        draws=training_draws,
        features={},
        target_draw_id=target_date,
        git_sha=_git_current_sha(),
    )

    # Leakage guard 1
    assert target_date not in {d.draw_id for d in training_draws}, "LEAKAGE: target in training"

    # Train all models
    logger.info("Training %d base models...", len(base_models))
    for model in base_models:
        try:
            model.fit(ctx)
            logger.info("  Trained: %s", model.model_id)
        except Exception as e:
            logger.error("  Failed to train %s: %s", model.model_id, e)

    # Get predictions from each model
    from fortuna.models.meta_stacker import MetaStacker

    stacker = MetaStacker(base_models=base_models)
    # Simple average mode (no walk-forward calibration at predict time)
    stacker._use_average = {pt: True for pt in ("first6", "three_back", "two_back")}
    n = len(base_models)
    stacker._weights = {pt: [1.0 / n] * n for pt in ("first6", "three_back", "two_back")}
    stacker._fitted = True

    # Collect candidate picks per prize type
    ensemble_picks: dict[str, list[Pick]] = {}
    model_versions: dict[str, str] = {}

    for prize_type_str in ("first6", "three_back", "two_back"):
        pt = cast(PrizeType, prize_type_str)
        all_candidates: list[list[Pick]] = []
        top_k = max(PICK_SPLIT[prize_type_str] * 5, 20)

        for model in base_models:
            try:
                picks = model.predict_top_k(pt, top_k)
                all_candidates.append(picks)
                model_versions[model.model_id] = model.fingerprint()
            except Exception as e:
                logger.warning("predict_top_k failed for %s/%s: %s", model.model_id, prize_type_str, e)
                all_candidates.append([])

        merged = stacker.predict_ensemble(pt, all_candidates, top_k * 2)
        ensemble_picks[prize_type_str] = merged

    # Run picker to select final tickets
    final_picks = select_picks(ensemble_picks)

    # Validate counts
    total_tickets = sum(len(v) for v in final_picks.values())
    assert total_tickets == 10, f"Expected 10 tickets, got {total_tickets}"

    # Build export payload
    frozen_at = datetime.now(BKK).isoformat()
    git_sha = _git_current_sha()

    payload = {
        "target_draw_id": target_date,
        "frozen_at": frozen_at,
        "predict_started_at": predict_started_at.isoformat(),
        "git_sha_at_freeze": git_sha,
        "model_versions": model_versions,
        "ensemble_method": "simple_average",
        "picks": {
            prize_type: [
                {"value": v, "rank": i + 1}
                for i, v in enumerate(values)
            ]
            for prize_type, values in final_picks.items()
        },
        "total_tickets": total_tickets,
        "total_cost_thb": total_tickets * 80,
    }

    # Compute sha256 of the picks payload
    picks_json = json.dumps(payload["picks"], sort_keys=True, separators=(",", ":"))
    payload["picks_sha256"] = hashlib.sha256(picks_json.encode()).hexdigest()

    # Write export file
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / f"{target_date}-prediction.json"
    with open(export_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote prediction to %s", export_path)

    # Git freeze (tamper-evidence)
    if not dry_run:
        try:
            freeze_sha = _git_freeze_prediction(export_path)
            payload["freeze_commit_sha"] = freeze_sha
            # Update file with freeze SHA
            with open(export_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("Frozen prediction committed: %s", freeze_sha)
        except subprocess.CalledProcessError as e:
            logger.error("Git push failed: %s", e)
            payload["freeze_commit_sha"] = None
    else:
        payload["freeze_commit_sha"] = "dry-run"
        logger.info("Dry run — skipping git push")

    # Store predictions in DB
    conn = get_or_init_db()
    for prize_type, picks_list in final_picks.items():
        for rank, value in enumerate(picks_list, start=1):
            insert_prediction(
                conn=conn,
                draw_id=target_date,
                model_id="ensemble",
                prize_type=prize_type,
                pick_value=value,
                pick_rank=rank,
                confidence=None,
                purchased=False,
                frozen_at=frozen_at,
                freeze_commit_sha=payload.get("freeze_commit_sha"),
            )

    # --- Enhancement-1: Notion publish (non-blocking) ---
    notion_page_url: str | None = None
    try:
        from fortuna.pipeline.notion_publisher import publish_prediction
        notion_page_url = publish_prediction(payload)
        if notion_page_url:
            payload["notion_page_url"] = notion_page_url
            # Persist page URL to export file
            with open(export_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            # Store page ID in DB for later settlement update
            # URL ends with "<slug>-<32hex>"; we want only the trailing 32-hex
            # UUID (formatted with dashes) since Notion API requires UUID form.
            try:
                import re
                page_id_raw = notion_page_url.rstrip("/").split("/")[-1]
                m = re.search(r"([0-9a-f]{32})$", page_id_raw)
                if m:
                    h = m.group(1)
                    formatted = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
                else:
                    formatted = page_id_raw  # fallback — will likely fail Notion validation
                conn.execute(
                    "UPDATE predictions SET notion_page_id = ? WHERE draw_id = ? AND model_id = 'ensemble'",
                    (formatted, target_date),
                )
                conn.commit()
                logger.info("Stored Notion page ID %s for draw %s", formatted, target_date)
            except Exception as e:
                logger.warning("Could not store Notion page ID: %s", e)
    except Exception as e:
        logger.warning("Notion publish failed (non-blocking): %s", e)

    logger.info(
        "Prediction complete: %d tickets for draw %s", total_tickets, target_date
    )
    return payload
