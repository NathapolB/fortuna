#!/usr/bin/env python3
"""Quick sanity check — verify Phase 2 models can be instantiated and fit.

Run from repo root:
    cd ~/projects/fortuna && .venv/bin/python scripts/check_phase2.py
"""

import sys
from pathlib import Path

# Ensure repo root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fortuna.schema import Draw
from fortuna.models.base import TrainContext


def make_draw(draw_id: str, i: int = 0) -> Draw:
    fp = str((i * 17 + 100000) % 1000000).zfill(6)
    tb = [str((i * 7 + 100) % 1000).zfill(3), str((i * 3 + 200) % 1000).zfill(3)]
    two = str((i * 11 + 34) % 100).zfill(2)
    return Draw(
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
    )


def make_draws(n: int = 60) -> list:
    draws = []
    year = 2020
    month = 1
    day_cycle = [1, 16]
    day_idx = 0
    for i in range(n):
        draw_id = f"{year:04d}-{month:02d}-{day_cycle[day_idx]:02d}"
        draws.append(make_draw(draw_id, i))
        day_idx += 1
        if day_idx >= 2:
            day_idx = 0
            month += 1
            if month > 12:
                month = 1
                year += 1
    return draws


def main():
    print("Phase 2 sanity check")
    print("=" * 50)

    draws = make_draws(60)
    training = draws[:55]
    target_id = draws[55].draw_id

    ctx = TrainContext(
        draws=training,
        features={},
        target_draw_id=target_id,
        git_sha="check",
    )

    from fortuna.models.frequency_bayesian import FrequencyBayesian
    from fortuna.models.markov import MarkovModel
    from fortuna.models.lstm import LSTMModel
    from fortuna.models.rl_qlearn import RLQLearner
    from fortuna.pipeline.picker import select_picks

    models = [
        ("FrequencyBayesian", FrequencyBayesian()),
        ("MarkovModel", MarkovModel()),
        ("LSTMModel", LSTMModel(epochs=5)),  # quick
        ("RLQLearner", RLQLearner(episodes=1)),  # quick
    ]

    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(ctx)
        print(f"  {name} fitted ✓")
        for pt in ("first6", "three_back", "two_back"):
            picks = model.predict_top_k(pt, 3)
            print(f"  {pt}: {[p.value for p in picks[:3]]}")

    print("\nMeta-stacker + picker...")
    from fortuna.models.meta_stacker import MetaStacker
    base_models = [m for _, m in models]
    stacker = MetaStacker(base_models=base_models)
    n = len(base_models)
    stacker._use_average = {pt: True for pt in ("first6", "three_back", "two_back")}
    stacker._weights = {pt: [1.0/n]*n for pt in ("first6", "three_back", "two_back")}
    stacker._fitted = True

    ensemble_picks = {}
    for pt in ("first6", "three_back", "two_back"):
        candidates = [m.predict_top_k(pt, 10) for m in base_models]  # type: ignore
        merged = stacker.predict_ensemble(pt, candidates, 20)
        ensemble_picks[pt] = merged

    final = select_picks(ensemble_picks)
    print("\nFinal picks:")
    for pt, vals in final.items():
        print(f"  {pt}: {vals}")
    total = sum(len(v) for v in final.values())
    print(f"\nTotal tickets: {total} (expected 10)")
    assert total == 10, f"Expected 10, got {total}"
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
