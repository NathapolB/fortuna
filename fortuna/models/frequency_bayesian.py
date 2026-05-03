"""FrequencyBayesian model — Beta-Bernoulli digit-frequency model. SPEC §4.

Per-position digit posteriors using Dirichlet prior.
Score combinations by product of marginals.
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


class FrequencyBayesian(BaseModel):
    """Beta-Bernoulli / Dirichlet-Multinomial per-position frequency model.

    For each digit position p and each digit value d in [0-9], we maintain
    a Dirichlet posterior:

        alpha[p][d] = prior_strength + count(d at position p)

    The posterior probability of digit d at position p is:
        P(d | pos=p) = alpha[p][d] / sum(alpha[p])

    Picks are scored by the product of marginals across positions.
    """

    model_id = "frequency-bayes-v1"
    feature_spec: list[str] = []
    hyperparams: dict = {"prior_strength": 1.0}  # Dirichlet concentration

    def __init__(self, prior_strength: float = 1.0) -> None:
        self.hyperparams = {"prior_strength": prior_strength}
        self.feature_spec = []
        self.prior_strength = prior_strength
        # posteriors[prize_type][position][digit] = alpha count
        self._posteriors: dict[str, list[list[float]]] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Position counts per prize type
    # ------------------------------------------------------------------

    @staticmethod
    def _positions_for(prize_type: PrizeType) -> int:
        return {"first6": 6, "three_back": 3, "two_back": 2}[prize_type]

    @staticmethod
    def _values_for(draw, prize_type: PrizeType) -> list[str]:
        """Extract relevant value(s) from a Draw object for a given prize type."""
        if prize_type == "first6":
            return [draw.first_prize]
        elif prize_type == "three_back":
            return draw.three_digit_back
        elif prize_type == "two_back":
            return [draw.two_digit_back]
        return []

    # ------------------------------------------------------------------
    # fit / predict / score
    # ------------------------------------------------------------------

    def fit(self, ctx: TrainContext) -> None:
        """Train on history in ctx.draws."""
        self._posteriors = {}
        for prize_type in ("first6", "three_back", "two_back"):
            pt = cast(PrizeType, prize_type)
            n_pos = self._positions_for(pt)
            # Initialize with Dirichlet prior (uniform α = prior_strength for each digit)
            alphas: list[list[float]] = [
                [self.prior_strength] * 10 for _ in range(n_pos)
            ]
            for draw in ctx.draws:
                for val in self._values_for(draw, pt):
                    if len(val) != n_pos:
                        continue
                    for pos, ch in enumerate(val):
                        if ch.isdigit():
                            alphas[pos][int(ch)] += 1.0
            self._posteriors[prize_type] = alphas
        self._fitted = True

    def _posterior_probs(self, prize_type: PrizeType) -> list[list[float]]:
        """Return normalized posterior probabilities per position."""
        alphas = self._posteriors[prize_type]
        result = []
        for pos_alphas in alphas:
            total = sum(pos_alphas)
            result.append([a / total for a in pos_alphas])
        return result

    def predict_top_k(self, prize: PrizeType, k: int) -> list[Pick]:
        """Return k picks scored by product of marginals, highest first."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        n_pos = self._positions_for(prize)
        probs = self._posterior_probs(prize)

        # For first6 (1M candidates), do top-k per position selection
        # rather than full enumeration
        if prize == "first6":
            return self._top_k_first6(probs, k)

        # For 3-digit (1000 values) and 2-digit (100 values), enumerate all
        candidates: list[tuple[str, float]] = []
        for digits in product(range(10), repeat=n_pos):
            score = 1.0
            for pos, d in enumerate(digits):
                score *= probs[pos][d]
            value = "".join(str(d) for d in digits)
            candidates.append((value, score))

        candidates.sort(key=lambda x: -x[1])
        picks = []
        for value, confidence in candidates[:k]:
            # Compute which digits are over-represented
            dominant_digits = []
            for pos, d in enumerate(value):
                d_int = int(d)
                if probs[pos][d_int] > 0.12:  # above uniform 0.10
                    dominant_digits.append(f"pos{pos}={d}")
            rationale = (
                f"Dirichlet posterior: {', '.join(dominant_digits) or 'uniform'}"
            )
            picks.append(Pick(value=value, confidence=confidence, rationale=rationale))
        return picks

    def _top_k_first6(self, probs: list[list[float]], k: int) -> list[Pick]:
        """Greedy top-k for first6 using per-position best digits.

        For 6-position lottery, use a beam/heap approach: take top-M digits
        per position and enumerate product of those.
        """
        import heapq

        # Take top-5 digits per position to keep candidate space manageable (5^6 = 15625)
        top_digits_per_pos = []
        for pos_probs in probs:
            ranked = sorted(range(10), key=lambda d: -pos_probs[d])[:5]
            top_digits_per_pos.append(ranked)

        # Use a min-heap (size k) to track top-k by score
        heap: list[tuple[float, str]] = []  # (neg_score, value)
        for digits in product(*top_digits_per_pos):
            score = 1.0
            for pos, d in enumerate(digits):
                score *= probs[pos][d]
            value = "".join(str(d) for d in digits)
            if len(heap) < k:
                heapq.heappush(heap, (score, value))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, value))

        heap.sort(key=lambda x: -x[0])
        picks = []
        for score, value in heap:
            rationale = f"Dirichlet product-of-marginals top-{k} from 5^6 beam"
            picks.append(Pick(value=value, confidence=score, rationale=rationale))
        return picks

    def score(self, draws: list) -> dict[str, float]:
        """Evaluate on holdout draws. Returns brier, log_loss, hit_rate."""
        if not self._fitted:
            return {"brier": 1.0, "log_loss": 10.0, "hit_rate": 0.0}

        from fortuna.eval.metrics import brier_score, log_loss, hit_rate
        import math

        results: dict[str, list] = {
            "probs": [], "labels": [], "hits": [], "tickets": []
        }

        for draw in draws:
            for prize_type in ("three_back", "two_back"):
                pt = cast(PrizeType, prize_type)
                space = {"three_back": 1000, "two_back": 100}[prize_type]
                picks = self.predict_top_k(pt, 1)
                if not picks:
                    continue
                top_pick = picks[0]
                actual_values = self._values_for(draw, pt)
                hit = int(top_pick.value in actual_values)
                results["probs"].append(top_pick.confidence)
                results["labels"].append(hit)
                results["hits"].append(hit)
                results["tickets"].append(1)

        if not results["probs"]:
            return {"brier": 1.0, "log_loss": 10.0, "hit_rate": 0.0}

        return {
            "brier": brier_score(results["probs"], results["labels"]),
            "log_loss": log_loss(results["probs"], results["labels"]),
            "hit_rate": sum(results["hits"]) / max(sum(results["tickets"]), 1),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self, path: Path) -> None:
        """Save model to path directory."""
        path.mkdir(parents=True, exist_ok=True)
        weights = {
            "posteriors": self._posteriors,
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
    def load(cls, path: Path) -> "FrequencyBayesian":
        """Load model from path directory."""
        with open(path / "meta.json") as f:
            meta = json.load(f)
        model = cls(prior_strength=meta["hyperparams"].get("prior_strength", 1.0))
        with open(path / "weights.pkl", "rb") as f:
            weights = pickle.load(f)
        model._posteriors = weights["posteriors"]
        model._fitted = weights["fitted"]
        return model
