"""Meta-learner stacker — logistic regression on base model confidence outputs. SPEC §5.

Takes the 4 base models' confidence outputs as features and produces calibrated
probabilities. Falls back to simple average if calibration is worse than uniform baseline.

Calibration check via sklearn.calibration.calibration_curve.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fortuna.models.base import BaseModel, Pick, PrizeType


class MetaStacker:
    """Logistic regression meta-learner that stacks base model outputs.

    For each (prize_type), builds a logistic regressor:
      X = [conf_bayes, conf_markov, conf_lstm, conf_rl]  (one feature per base model)
      y = 1 if any pick matched the actual, else 0

    Falls back to simple average if calibrated model has worse ECE than uniform.
    """

    def __init__(self, base_models: list[BaseModel]) -> None:
        self.base_models = base_models
        self._regressors: dict[str, LogisticRegression | None] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._use_average: dict[str, bool] = {}
        self._weights: dict[str, list[float]] = {}  # per-model weights if using avg
        self._fitted = False

    @staticmethod
    def _values_for(draw, prize_type: str) -> list[str]:
        if prize_type == "first6":
            return [draw.first_prize]
        elif prize_type == "three_back":
            return draw.three_digit_back
        elif prize_type == "two_back":
            return [draw.two_digit_back]
        return []

    def fit_on_walks(
        self,
        walk_results: dict[str, list[tuple[list[float], int]]],
    ) -> None:
        """Fit meta-learner from walk-forward CV results.

        walk_results[prize_type] = list of (model_confidences, hit_label) per draw.
        model_confidences = [conf_model0, conf_model1, ..., conf_modelN]
        """
        for prize_type in ("first6", "three_back", "two_back"):
            records = walk_results.get(prize_type, [])
            if len(records) < 10:
                self._use_average[prize_type] = True
                self._regressors[prize_type] = None
                self._weights[prize_type] = [1.0 / len(self.base_models)] * len(self.base_models)
                continue

            X = np.array([r[0] for r in records])
            y = np.array([r[1] for r in records])

            if y.sum() == 0:
                self._use_average[prize_type] = True
                self._regressors[prize_type] = None
                self._weights[prize_type] = [1.0 / len(self.base_models)] * len(self.base_models)
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._scalers[prize_type] = scaler

            lr = LogisticRegression(
                C=1.0,
                max_iter=500,
                random_state=42,
                class_weight="balanced",
            )
            lr.fit(X_scaled, y)

            y_pred_prob = lr.predict_proba(X_scaled)[:, 1]
            try:
                fraction_of_positives, mean_predicted = calibration_curve(
                    y, y_pred_prob, n_bins=5, strategy="quantile"
                )
                ece_model = np.mean(np.abs(fraction_of_positives - mean_predicted))
                base_rate = y.mean()
                ece_uniform = np.mean(np.abs(fraction_of_positives - base_rate))

                if ece_model > ece_uniform * 1.5:
                    self._use_average[prize_type] = True
                    self._regressors[prize_type] = None
                    self._weights[prize_type] = [
                        1.0 / len(self.base_models)
                    ] * len(self.base_models)
                else:
                    self._use_average[prize_type] = False
                    self._regressors[prize_type] = lr
                    self._weights[prize_type] = lr.coef_[0].tolist()
            except Exception:
                self._use_average[prize_type] = True
                self._regressors[prize_type] = None
                self._weights[prize_type] = [
                    1.0 / len(self.base_models)
                ] * len(self.base_models)

        self._fitted = True

    def predict_ensemble(
        self, prize_type: PrizeType, candidate_picks: list[list[Pick]], k: int
    ) -> list[Pick]:
        """Produce ensemble picks for a prize type.

        candidate_picks[i] = list of picks from base_model[i].
        Returns merged top-k picks with ensemble confidence scores.
        """
        prize_str = str(prize_type)

        if not self._fitted:
            return self._simple_average(prize_type, candidate_picks, k)

        use_avg = self._use_average.get(prize_str, True)
        if use_avg:
            return self._simple_average(prize_type, candidate_picks, k)

        value_to_confs: dict[str, list[float]] = {}
        n_models = len(self.base_models)

        for model_idx, picks in enumerate(candidate_picks):
            for pick in picks:
                if pick.value not in value_to_confs:
                    value_to_confs[pick.value] = [0.0] * n_models
                value_to_confs[pick.value][model_idx] = pick.confidence

        lr = self._regressors[prize_str]
        scaler = self._scalers.get(prize_str)

        if lr is None or scaler is None:
            return self._simple_average(prize_type, candidate_picks, k)

        scored: list[tuple[str, float]] = []
        for value, confs in value_to_confs.items():
            X = np.array([confs])
            X_scaled = scaler.transform(X)
            prob = lr.predict_proba(X_scaled)[0, 1]
            scored.append((value, prob))

        scored.sort(key=lambda x: -x[1])
        picks_out = []
        for value, confidence in scored[:k]:
            picks_out.append(Pick(
                value=value,
                confidence=confidence,
                rationale=f"MetaStacker (logistic) prize={prize_str}",
            ))
        return picks_out

    def _simple_average(
        self, prize_type: PrizeType, candidate_picks: list[list[Pick]], k: int
    ) -> list[Pick]:
        """Simple confidence average across models."""
        prize_str = str(prize_type)
        n = len(self.base_models) or 1
        weights = self._weights.get(prize_str, [1.0 / n] * n)

        value_scores: dict[str, float] = {}
        for model_idx, picks in enumerate(candidate_picks):
            w = weights[model_idx] if model_idx < len(weights) else 1.0 / n
            for pick in picks:
                value_scores[pick.value] = value_scores.get(pick.value, 0.0) + w * pick.confidence

        total = sum(value_scores.values()) or 1.0
        scored = sorted(value_scores.items(), key=lambda x: -x[1])

        picks_out = []
        for value, raw_score in scored[:k]:
            picks_out.append(Pick(
                value=value,
                confidence=raw_score / total,
                rationale=f"MetaStacker (avg) prize={prize_str}",
            ))
        return picks_out

    def get_calibration_info(self) -> dict[str, dict]:
        """Return calibration method used per prize type."""
        info = {}
        for prize_type in ("first6", "three_back", "two_back"):
            use_avg = self._use_average.get(prize_type, True)
            info[prize_type] = {
                "method": "simple_average" if use_avg else "logistic_stacker",
                "weights": self._weights.get(prize_type, []),
            }
        return info

    def serialize(self, path: Path) -> None:
        """Save meta-stacker state."""
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "regressors": self._regressors,
            "scalers": self._scalers,
            "use_average": self._use_average,
            "weights": self._weights,
            "fitted": self._fitted,
        }
        with open(path / "meta_stacker.pkl", "wb") as f:
            pickle.dump(state, f)
        with open(path / "meta_stacker_info.json", "w") as f:
            json.dump(self.get_calibration_info(), f, indent=2)

    @classmethod
    def load(cls, path: Path, base_models: list[BaseModel]) -> "MetaStacker":
        """Load meta-stacker from path."""
        stacker = cls(base_models=base_models)
        with open(path / "meta_stacker.pkl", "rb") as f:
            state = pickle.load(f)
        stacker._regressors = state["regressors"]
        stacker._scalers = state["scalers"]
        stacker._use_average = state["use_average"]
        stacker._weights = state["weights"]
        stacker._fitted = state["fitted"]
        return stacker
