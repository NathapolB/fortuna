"""Markov chain model — 1st and 2nd order on digit sequences. SPEC §4.

Per-position Markov chains on digit transitions with Laplace smoothing.
Also models transitions on digit sums.
Supports all three prize types: first6, three_back, two_back.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import cast

from fortuna.models.base import BaseModel, Pick, PrizeType, TrainContext


class MarkovModel(BaseModel):
    """1st and 2nd order Markov chains on digit sequences.

    For each position p, we build:
      - 1st order: P(d_t | d_{t-1}) with Laplace smoothing
      - 2nd order: P(d_t | d_{t-2}, d_{t-1}) with Laplace smoothing

    Combined prediction uses interpolated estimate:
      P(d_t | ...) = lambda_2nd * P_2nd + (1-lambda_2nd) * P_1st
    """

    model_id = "markov-v1"
    feature_spec: list[str] = []
    hyperparams: dict = {"order": 2, "laplace": 1.0, "lambda_2nd": 0.6}

    def __init__(
        self,
        order: int = 2,
        laplace: float = 1.0,
        lambda_2nd: float = 0.6,
    ) -> None:
        self.hyperparams = {"order": order, "laplace": laplace, "lambda_2nd": lambda_2nd}
        self.feature_spec = []
        self.order = order
        self.laplace = laplace
        self.lambda_2nd = lambda_2nd
        self._trans1: dict[str, list[dict]] = {}
        self._trans2: dict[str, list[dict]] = {}
        self._sum_trans1: dict[str, dict] = {}
        self._sum_trans2: dict[str, dict] = {}
        self._last_values: dict[str, list[str]] = {}
        self._last_sums: dict[str, list[int]] = {}
        self._fitted = False

    @staticmethod
    def _positions_for(prize_type: PrizeType) -> int:
        return {"first6": 6, "three_back": 3, "two_back": 2}[prize_type]

    @staticmethod
    def _values_for(draw, prize_type: PrizeType) -> list[str]:
        if prize_type == "first6":
            return [draw.first_prize]
        elif prize_type == "three_back":
            return draw.three_digit_back
        elif prize_type == "two_back":
            return [draw.two_digit_back]
        return []

    def fit(self, ctx: TrainContext) -> None:
        """Build transition tables from ctx.draws."""
        self._trans1 = {}
        self._trans2 = {}
        self._sum_trans1 = {}
        self._sum_trans2 = {}
        self._last_values = {}
        self._last_sums = {}

        for prize_type in ("first6", "three_back", "two_back"):
            pt = cast(PrizeType, prize_type)
            n_pos = self._positions_for(pt)

            trans1: list[dict] = [{} for _ in range(n_pos)]
            trans2: list[dict] = [{} for _ in range(n_pos)]
            sum_trans1: dict = {}
            sum_trans2: dict = {}

            # Collect per-draw primary values
            history: list[str] = []
            for draw in ctx.draws:
                vals = self._values_for(draw, pt)
                if vals and len(vals[0]) == n_pos:
                    history.append(vals[0])

            # Build per-position 1st-order transitions
            for t in range(1, len(history)):
                cur = history[t]
                prev = history[t - 1]
                prev2 = history[t - 2] if t >= 2 else None

                for pos in range(n_pos):
                    if cur[pos].isdigit() and prev[pos].isdigit():
                        key1 = (int(prev[pos]),)
                        if key1 not in trans1[pos]:
                            trans1[pos][key1] = {}
                        d = int(cur[pos])
                        trans1[pos][key1][d] = trans1[pos][key1].get(d, 0) + 1

                    if prev2 is not None and cur[pos].isdigit() and prev[pos].isdigit() and prev2[pos].isdigit():
                        key2 = (int(prev2[pos]), int(prev[pos]))
                        if key2 not in trans2[pos]:
                            trans2[pos][key2] = {}
                        d = int(cur[pos])
                        trans2[pos][key2][d] = trans2[pos][key2].get(d, 0) + 1

                # Sum transitions
                try:
                    cur_sum = sum(int(c) for c in cur)
                    prev_sum = sum(int(c) for c in prev)
                    k1 = (prev_sum,)
                    if k1 not in sum_trans1:
                        sum_trans1[k1] = {}
                    sum_trans1[k1][cur_sum] = sum_trans1[k1].get(cur_sum, 0) + 1

                    if prev2 is not None:
                        prev2_sum = sum(int(c) for c in prev2)
                        k2 = (prev2_sum, prev_sum)
                        if k2 not in sum_trans2:
                            sum_trans2[k2] = {}
                        sum_trans2[k2][cur_sum] = sum_trans2[k2].get(cur_sum, 0) + 1
                except ValueError:
                    pass

            self._trans1[prize_type] = trans1
            self._trans2[prize_type] = trans2
            self._sum_trans1[prize_type] = sum_trans1
            self._sum_trans2[prize_type] = sum_trans2
            self._last_values[prize_type] = history[-2:] if len(history) >= 2 else history[-1:]
            if history:
                self._last_sums[prize_type] = [
                    sum(int(c) for c in v) for v in self._last_values[prize_type]
                ]

        self._fitted = True

    def _transition_prob(self, prize_type: str, pos: int, context: tuple) -> list[float]:
        """Compute interpolated transition probabilities for one position."""
        laplace = self.laplace
        n_digits = 10

        ctx1 = (context[-1],)
        counts1 = self._trans1[prize_type][pos].get(ctx1, {})
        total1 = sum(counts1.values()) + laplace * n_digits
        probs1 = [(counts1.get(d, 0) + laplace) / total1 for d in range(n_digits)]

        if self.order < 2 or len(context) < 2:
            return probs1

        counts2 = self._trans2[prize_type][pos].get(context, {})
        total2 = sum(counts2.values()) + laplace * n_digits
        probs2 = [(counts2.get(d, 0) + laplace) / total2 for d in range(n_digits)]

        lam = self.lambda_2nd
        return [lam * p2 + (1 - lam) * p1 for p1, p2 in zip(probs1, probs2)]

    def predict_top_k(self, prize: PrizeType, k: int) -> list[Pick]:
        """Return k picks using Markov transition probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        n_pos = self._positions_for(prize)
        prize_str = str(prize)
        last = self._last_values.get(prize_str, [])

        if len(last) < 1:
            return [Pick(value="0" * n_pos, confidence=1e-10, rationale="No history")]

        if self.order >= 2 and len(last) >= 2:
            contexts = []
            prev2 = last[-2]
            prev1 = last[-1]
            for pos in range(n_pos):
                d2 = int(prev2[pos]) if prev2[pos].isdigit() else 0
                d1 = int(prev1[pos]) if prev1[pos].isdigit() else 0
                contexts.append((d2, d1))
        else:
            prev1 = last[-1]
            contexts = []
            for pos in range(n_pos):
                d1 = int(prev1[pos]) if prev1[pos].isdigit() else 0
                contexts.append((d1,))

        pos_probs = []
        for pos in range(n_pos):
            pos_probs.append(self._transition_prob(prize_str, pos, contexts[pos]))

        if prize == "first6":
            return self._top_k_beam(pos_probs, k, f"Markov order-{self.order} beam")

        candidates = []
        for digits in product(range(10), repeat=n_pos):
            score = 1.0
            for pos, d in enumerate(digits):
                score *= pos_probs[pos][d]
            candidates.append(("".join(str(d) for d in digits), score))

        candidates.sort(key=lambda x: -x[1])
        picks = []
        for value, score in candidates[:k]:
            picks.append(Pick(
                value=value,
                confidence=score,
                rationale=f"Markov order-{self.order} (Laplace={self.laplace})",
            ))
        return picks

    def _top_k_beam(self, pos_probs: list[list[float]], k: int, rationale: str) -> list[Pick]:
        """Beam top-k for 6-position."""
        import heapq
        top_digits_per_pos = [
            sorted(range(10), key=lambda d, pp=pp: -pp[d])[:5]
            for pp in pos_probs
        ]
        heap: list[tuple[float, str]] = []
        for digits in product(*top_digits_per_pos):
            score = 1.0
            for pos, d in enumerate(digits):
                score *= pos_probs[pos][d]
            value = "".join(str(d) for d in digits)
            if len(heap) < k:
                heapq.heappush(heap, (score, value))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, value))
        heap.sort(key=lambda x: -x[0])
        return [Pick(value=v, confidence=s, rationale=rationale) for s, v in heap]

    def score(self, draws: list) -> dict[str, float]:
        """Evaluate on holdout draws."""
        if not self._fitted:
            return {"brier": 1.0, "log_loss": 10.0, "hit_rate": 0.0}

        from fortuna.eval.metrics import brier_score, log_loss

        probs, labels = [], []
        for draw in draws:
            for prize_type in ("three_back", "two_back"):
                pt = cast(PrizeType, prize_type)
                picks = self.predict_top_k(pt, 1)
                if not picks:
                    continue
                actual_values = self._values_for(draw, pt)
                hit = int(picks[0].value in actual_values)
                probs.append(picks[0].confidence)
                labels.append(hit)
                vals = actual_values
                if vals:
                    hist = self._last_values.get(prize_type, [])
                    hist.append(vals[0])
                    self._last_values[prize_type] = hist[-2:]

        if not probs:
            return {"brier": 1.0, "log_loss": 10.0, "hit_rate": 0.0}
        return {
            "brier": brier_score(probs, labels),
            "log_loss": log_loss(probs, labels),
            "hit_rate": sum(labels) / len(labels),
        }

    def serialize(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        weights = {
            "trans1": self._trans1,
            "trans2": self._trans2,
            "sum_trans1": self._sum_trans1,
            "sum_trans2": self._sum_trans2,
            "last_values": self._last_values,
            "last_sums": self._last_sums,
            "fitted": self._fitted,
        }
        with open(path / "weights.pkl", "wb") as f:
            pickle.dump(weights, f)
        meta = {
            "model_id": self.model_id,
            "hyperparams": self.hyperparams,
            "feature_spec": self.feature_spec,
            "fingerprint": self.fingerprint(),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MarkovModel":
        with open(path / "meta.json") as f:
            meta = json.load(f)
        hp = meta["hyperparams"]
        model = cls(
            order=hp.get("order", 2),
            laplace=hp.get("laplace", 1.0),
            lambda_2nd=hp.get("lambda_2nd", 0.6),
        )
        with open(path / "weights.pkl", "rb") as f:
            weights = pickle.load(f)
        model._trans1 = weights["trans1"]
        model._trans2 = weights["trans2"]
        model._sum_trans1 = weights["sum_trans1"]
        model._sum_trans2 = weights["sum_trans2"]
        model._last_values = weights["last_values"]
        model._last_sums = weights["last_sums"]
        model._fitted = weights["fitted"]
        return model
