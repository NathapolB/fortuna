"""Backtest strategy 5/3/2 vs old 10×first6 over the last N draws.

Per target draw: train 4 base models on draws strictly before it, build the
4-tier ensemble once, then construct BOTH pick sets and score each via
_score_ticket_multi_prize (Pao Tang multi-prize). Reports per-draw + totals.

Usage: PYTHONPATH=. .venv/bin/python scripts/backtest_532.py 24
"""
from __future__ import annotations

import logging
import sys

logging.disable(logging.WARNING)

from fortuna.models.frequency_bayesian import FrequencyBayesian
from fortuna.models.markov import MarkovModel
from fortuna.models.lstm import LSTMModel
from fortuna.models.rl_qlearn import RLQLearner
from fortuna.models.meta_stacker import MetaStacker
from fortuna.models.base import TrainContext
from fortuna.pipeline.picker import select_picks_532, _top_values
from fortuna.pipeline.settle import _score_ticket_multi_prize
from fortuna.store import DrawStore

PRIZE_TYPES = ("first6", "three_front", "three_back", "two_back")
POOL = {"first6": 7, "three_front": 3, "three_back": 3, "two_back": 5}


def _ensemble(train):
    ctx = TrainContext(draws=train, features={}, target_draw_id="next", git_sha="bt")
    models = [FrequencyBayesian(), MarkovModel(), LSTMModel(), RLQLearner()]
    for m in models:
        try:
            m.fit(ctx)
        except Exception:
            pass
    st = MetaStacker(base_models=models)
    st._use_average = {pt: True for pt in PRIZE_TYPES}
    st._weights = {pt: [1.0 / len(models)] * len(models) for pt in PRIZE_TYPES}
    st._fitted = True
    ens = {}
    for pt in PRIZE_TYPES:
        k = max(POOL[pt] * 5, 20)
        cands = []
        for m in models:
            try:
                cands.append(m.predict_top_k(pt, k))
            except Exception:
                cands.append([])
        ens[pt] = st.predict_ensemble(pt, cands, k * 2)
    return ens


def _score(values, actual):
    wins = payout = 0
    for v in values:
        hits = _score_ticket_multi_prize(v, actual)
        if hits:
            wins += 1
            payout += sum(hits.values())
    return wins, payout


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    draws = sorted(DrawStore().all_draws(), key=lambda d: d.draw_id)
    targets = draws[-n:]
    tot = {"new": [0, 0], "old": [0, 0]}  # [wins, payout]
    new_hit_draws = old_hit_draws = 0

    print(f"{'งวด':<12}{'ออก(ท้าย2)':<11}{'NEW w/pay':<16}{'OLD w/pay':<16}")
    print("-" * 55)
    for t in targets:
        train = [d for d in draws if d.draw_id < t.draw_id]
        if len(train) < 30:
            continue
        recent = {d.first_prize for d in train[-100:]}
        ens = _ensemble(train)
        new_vals = [x["value"] for x in select_picks_532(ens, recent_winners=recent)]
        old_vals = _top_values(ens, "first6", 10, exclude=recent)
        nw, npay = _score(new_vals, t)
        ow, opay = _score(old_vals, t)
        tot["new"][0] += nw; tot["new"][1] += npay
        tot["old"][0] += ow; tot["old"][1] += opay
        new_hit_draws += nw > 0
        old_hit_draws += ow > 0
        print(f"{t.draw_id:<12}{t.two_digit_back:<11}{nw}ใบ/+{npay:<11,}{ow}ใบ/+{opay:<11,}")

    nd = len([t for t in targets if len([d for d in draws if d.draw_id < t.draw_id]) >= 30])
    print("-" * 55)
    print(f"\nงวดที่ทดสอบ: {nd}")
    print(f"NEW 5/3/2 : ถูกอย่างน้อย 1 ใบ {new_hit_draws}/{nd} งวด ({new_hit_draws/nd*100:.0f}%) | "
          f"จ่ายรวม +{tot['new'][1]:,} | ลงทุน {nd*800:,} | สุทธิ {tot['new'][1]-nd*800:+,}")
    print(f"OLD 10×6  : ถูกอย่างน้อย 1 ใบ {old_hit_draws}/{nd} งวด ({old_hit_draws/nd*100:.0f}%) | "
          f"จ่ายรวม +{tot['old'][1]:,} | ลงทุน {nd*800:,} | สุทธิ {tot['old'][1]-nd*800:+,}")


if __name__ == "__main__":
    main()
