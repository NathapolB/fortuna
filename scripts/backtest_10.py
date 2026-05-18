"""Backtest: rerun prediction over the last 10 actual draws, measure accuracy.

Accuracy definition (per Nash):
  - A ticket "wins" if it hits any of the 5 tracked prize types.
  - Per-draw accuracy = wins / 10 tickets.
  - Overall = mean of per-draw accuracy.

Two strategies compared:
  A) NEW (v2.4): 10× 6-digit tickets, multi-prize scoring.
  B) OLD (2/3/5): 2 first6 + 3 three_back + 5 two_back, multi-prize scoring
     (uses same scoring rule — so the comparison isolates picker strategy only).

For each target draw:
  - Train base models on draws strictly before target (no leakage).
  - Generate picks per strategy.
  - Score against actual draw using _score_ticket_multi_prize.
  - Count wins.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# silence the noisy training logs
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from fortuna.config import PICK_SPLIT
from fortuna.models.frequency_bayesian import FrequencyBayesian
from fortuna.models.markov import MarkovModel
from fortuna.models.lstm import LSTMModel
from fortuna.models.rl_qlearn import RLQLearner
from fortuna.models.meta_stacker import MetaStacker
from fortuna.models.base import TrainContext
from fortuna.pipeline.picker import select_picks
from fortuna.pipeline.settle import _score_ticket_multi_prize
from fortuna.store import DrawStore


def _train_and_predict(
    train_draws: list,
    split: dict[str, int],
    recency_guard: bool = True,
) -> dict[str, list[str]]:
    """Train fresh models, return picks per the given split."""
    target = train_draws[-1].draw_id + "_next"  # placeholder, not used in fit
    ctx = TrainContext(
        draws=train_draws,
        features={},
        target_draw_id=target,
        git_sha="backtest",
    )
    models = [FrequencyBayesian(), MarkovModel(), LSTMModel(), RLQLearner()]
    for m in models:
        try:
            m.fit(ctx)
        except Exception as e:
            print(f"  fit fail {m.model_id}: {e}", file=sys.stderr)

    stacker = MetaStacker(base_models=models)
    stacker._use_average = {pt: True for pt in ("first6", "three_back", "two_back")}
    n = len(models)
    stacker._weights = {pt: [1.0 / n] * n for pt in ("first6", "three_back", "two_back")}
    stacker._fitted = True

    ensemble = {}
    for pt_str in ("first6", "three_back", "two_back"):
        top_k = max(split.get(pt_str, 0) * 5, 20)
        all_c = []
        for m in models:
            try:
                all_c.append(m.predict_top_k(pt_str, top_k))
            except Exception:
                all_c.append([])
        merged = stacker.predict_ensemble(pt_str, all_c, top_k * 2)
        ensemble[pt_str] = merged

    # Temporarily override PICK_SPLIT for select_picks
    import fortuna.config as cfg
    orig = dict(cfg.PICK_SPLIT)
    cfg.PICK_SPLIT.clear()
    cfg.PICK_SPLIT.update(split)
    recent: set[str] = set()
    if recency_guard:
        recent = {d.first_prize for d in sorted(train_draws, key=lambda x: x.draw_id)[-100:]}
    try:
        return select_picks(ensemble, recent_winners=recent)
    finally:
        cfg.PICK_SPLIT.clear()
        cfg.PICK_SPLIT.update(orig)


def _score(picks: dict[str, list[str]], actual) -> tuple[int, int, int]:
    """Returns (n_winning_tickets, n_total_tickets, total_payout_thb)."""
    wins = 0
    total = 0
    payout = 0
    for _pt, values in picks.items():
        for v in values:
            total += 1
            hits = _score_ticket_multi_prize(v, actual)
            if hits:
                wins += 1
                payout += sum(hits.values())
    return wins, total, payout


def main() -> None:
    store = DrawStore()
    all_draws = sorted(store.all_draws(), key=lambda d: d.draw_id)
    n_backtest = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    targets = all_draws[-n_backtest:]

    NEW_SPLIT = {"first6": 10, "three_back": 0, "two_back": 0}
    OLD_SPLIT = {"first6": 2, "three_back": 3, "two_back": 5}

    rows = []
    new_total_wins, new_total_tix, new_total_payout = 0, 0, 0
    old_total_wins, old_total_tix, old_total_payout = 0, 0, 0

    for i, target in enumerate(targets, 1):
        train = [d for d in all_draws if d.draw_id < target.draw_id]
        print(f"[{i}/{len(targets)}] {target.draw_id} (train n={len(train)}) ...", flush=True)

        new_picks = _train_and_predict(train, NEW_SPLIT)
        old_picks = _train_and_predict(train, OLD_SPLIT)

        nw, nt, np_ = _score(new_picks, target)
        ow, ot, op_ = _score(old_picks, target)

        new_total_wins += nw
        new_total_tix += nt
        new_total_payout += np_
        old_total_wins += ow
        old_total_tix += ot
        old_total_payout += op_

        rows.append({
            "draw": target.draw_id,
            "actual_first": target.first_prize,
            "actual_back2": target.two_digit_back,
            "new_wins": f"{nw}/{nt}",
            "new_acc": f"{nw/nt*100:.0f}%",
            "new_pnl": np_ - nt * 80,
            "old_wins": f"{ow}/{ot}",
            "old_acc": f"{ow/ot*100:.0f}%",
            "old_pnl": op_ - ot * 80,
        })

    print()
    print(f"{'งวด':<12} {'รางวัล1':<8} {'2-tail':<6} | "
          f"{'NEW':<10} {'acc':<5} {'P&L':<8} | "
          f"{'OLD':<10} {'acc':<5} {'P&L':<8}")
    print("-" * 95)
    for r in rows:
        print(f"{r['draw']:<12} {r['actual_first']:<8} {r['actual_back2']:<6} | "
              f"{r['new_wins']:<10} {r['new_acc']:<5} {r['new_pnl']:>+7,} | "
              f"{r['old_wins']:<10} {r['old_acc']:<5} {r['old_pnl']:>+7,}")
    print("-" * 95)

    new_acc_pct = new_total_wins / new_total_tix * 100
    old_acc_pct = old_total_wins / old_total_tix * 100
    new_net = new_total_payout - new_total_tix * 80
    old_net = old_total_payout - old_total_tix * 80

    print(f"{'TOTAL':<12} {'':<8} {'':<6} | "
          f"{new_total_wins:>2}/{new_total_tix:<7} {new_acc_pct:>4.1f}% {new_net:>+7,} | "
          f"{old_total_wins:>2}/{old_total_tix:<7} {old_acc_pct:>4.1f}% {old_net:>+7,}")

    delta_acc = new_acc_pct - old_acc_pct
    delta_net = new_net - old_net
    print()
    print(f"Δ accuracy: {delta_acc:+.1f}pp   Δ P&L: {delta_net:+,} บาท (10 งวด รวม)")

    out = Path(__file__).parent.parent / "data" / "reports" / "backtest-10.json"
    out.write_text(json.dumps({
        "rows": rows,
        "summary": {
            "new": {"wins": new_total_wins, "total": new_total_tix, "acc_pct": new_acc_pct, "net_pnl": new_net},
            "old": {"wins": old_total_wins, "total": old_total_tix, "acc_pct": old_acc_pct, "net_pnl": old_net},
            "delta_acc_pp": delta_acc,
            "delta_net_pnl": delta_net,
        },
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
