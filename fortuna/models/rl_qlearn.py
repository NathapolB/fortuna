"""Q-learning RL agent for lottery prediction. SPEC §4.

State: recent draw history features (digit sums, gaps since last appearance).
Action: pick a number (per prize type).
Reward: +N if hit, 0 otherwise (per SPEC §4 reward shaping).
Train via experience replay on historical sequence.

One Q-table per prize type. Actions = all possible pick values.
"""

from __future__ import annotations

import json
import pickle
import random
from collections import deque
from pathlib import Path
from typing import cast

import numpy as np

from fortuna.models.base import BaseModel, Pick, PrizeType, TrainContext


class RLQLearner(BaseModel):
    """Tabular Q-learning agent for lottery prediction.

    State representation (per prize type):
      - Last 3 digit sums (normalized)
      - Gap (draws since last appearance) for the most recent winner, normalized
      - Position-wise digit frequency bucket (0=low, 1=mid, 2=high) for last 10 draws

    State is discretized into a tuple for tabular Q lookup.
    Action space: all possible values for the prize type.

    Reward shaping (SPEC §4):
      - first6 hit: +600 (scaled from 6M, /10000)
      - three_back hit: +40 (scaled from 4000, /100)
      - two_back hit: +20 (scaled from 2000, /100)
      - miss: -1 (small penalty to encourage meaningful picks)
    """

    model_id = "rl-qlearn-v1"
    feature_spec: list[str] = []
    hyperparams: dict = {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.3,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "replay_buffer_size": 500,
        "batch_size": 32,
        "episodes": 3,
        "seed": 42,
    }

    _REWARDS: dict[str, float] = {
        "first6": 600.0,
        "three_front": 40.0,
        "three_back": 40.0,
        "two_back": 20.0,
    }
    _MISS_PENALTY = -1.0

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 500,
        batch_size: int = 32,
        episodes: int = 3,
        seed: int = 42,
    ) -> None:
        self.hyperparams = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "replay_buffer_size": replay_buffer_size,
            "batch_size": batch_size,
            "episodes": episodes,
            "seed": seed,
        }
        self.feature_spec = []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.episodes = episodes
        self.seed = seed

        # Q-tables: prize_type -> {state_key: {action: q_value}}
        self._q_tables: dict[str, dict] = {}
        # Action spaces (list of all possible values for each prize type)
        self._action_spaces: dict[str, list[str]] = {}
        # Histories for state computation
        self._histories: dict[str, list[str]] = {}
        self._fitted = False

    @staticmethod
    def _positions_for(prize_type: str) -> int:
        return {"first6": 6, "three_front": 3, "three_back": 3, "two_back": 2}[prize_type]

    @staticmethod
    def _values_for(draw, prize_type: str) -> list[str]:
        if prize_type == "first6":
            return [draw.first_prize]
        elif prize_type == "three_front":
            return draw.three_digit_front
        elif prize_type == "three_back":
            return draw.three_digit_back
        elif prize_type == "two_back":
            return [draw.two_digit_back]
        return []

    def _build_action_space(self, prize_type: str) -> list[str]:
        """Build full action space for prize type."""
        n_pos = self._positions_for(prize_type)
        count = 10 ** n_pos
        return [str(i).zfill(n_pos) for i in range(count)]

    def _compute_state(
        self,
        history: list[str],
        prize_type: str,
        t: int,
        window: int = 10,
    ) -> tuple:
        """Compute discrete state tuple from history up to index t (exclusive).

        State features:
          1. Last 3 digit sums (bucketed into 0-2 based on tercile)
          2. Gap since last occurrence of any winner (capped at 10, then bucketed)
          3. Dominant digit per position in last window draws (most frequent digit)
        """
        n_pos = self._positions_for(prize_type)
        past = history[max(0, t - window):t]

        if not past:
            # Return zero state
            sums = (1, 1, 1)
            gaps = (5,)
            dom = tuple([0] * n_pos)
            return sums + gaps + dom

        # Digit sums for last 3 draws
        recent = history[max(0, t - 3):t]
        sum_buckets = []
        for val in (recent + ["0" * n_pos, "0" * n_pos, "0" * n_pos])[:3]:
            try:
                s = sum(int(c) for c in val[:n_pos])
                max_sum = 9 * n_pos
                # Bucket into 0/1/2
                bucket = min(2, int(s / (max_sum / 3)))
                sum_buckets.append(bucket)
            except ValueError:
                sum_buckets.append(1)

        # Gap since last value first appeared (simplified: use last draw index mod 5)
        gap = min(len(past), 10)
        gap_bucket = gap // 3  # 0-3

        # Dominant digit per position in window
        dominant = []
        for pos in range(min(n_pos, 3)):  # cap at 3 to limit state space
            counts = [0] * 10
            for val in past:
                if len(val) > pos and val[pos].isdigit():
                    counts[int(val[pos])] += 1
            dom_digit = counts.index(max(counts))
            dominant.append(dom_digit // 3)  # bucket 0-3

        return tuple(sum_buckets) + (gap_bucket,) + tuple(dominant)

    def _q_get(self, prize_type: str, state: tuple, action: str) -> float:
        """Get Q-value with default 0."""
        table = self._q_tables.get(prize_type, {})
        return table.get(state, {}).get(action, 0.0)

    def _q_set(self, prize_type: str, state: tuple, action: str, value: float) -> None:
        """Set Q-value."""
        if prize_type not in self._q_tables:
            self._q_tables[prize_type] = {}
        if state not in self._q_tables[prize_type]:
            self._q_tables[prize_type][state] = {}
        self._q_tables[prize_type][state][action] = value

    def _best_action(self, prize_type: str, state: tuple) -> str:
        """Greedy action with highest Q-value."""
        table = self._q_tables.get(prize_type, {})
        state_q = table.get(state, {})
        actions = self._action_spaces.get(prize_type, [])
        if not state_q or not actions:
            return random.choice(actions) if actions else "0"
        best = max(actions, key=lambda a: state_q.get(a, 0.0))
        return best

    def fit(self, ctx: TrainContext) -> None:
        """Train Q-learning agent via experience replay on historical sequence."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        self._q_tables = {}
        self._action_spaces = {}
        self._histories = {}

        for prize_type in ("first6", "three_front", "three_back", "two_back"):
            # Build action space (skip first6 — too large for tabular; use two/three digit)
            if prize_type == "first6":
                # For first6, keep only a sampled subset to keep table manageable
                n_pos = 6
                full_space = [str(i).zfill(6) for i in range(0, 1000000, 1000)]
                self._action_spaces[prize_type] = full_space[:200]
            else:
                self._action_spaces[prize_type] = self._build_action_space(prize_type)

            history: list[str] = []
            for draw in ctx.draws:
                vals = self._values_for(draw, prize_type)
                n_pos = self._positions_for(prize_type)
                if vals and len(vals[0]) == n_pos and vals[0].isdigit():
                    history.append(vals[0])

            self._histories[prize_type] = history
            self._q_tables[prize_type] = {}

            if len(history) < 5:
                continue

            # Experience replay buffer: list of (state, action, reward, next_state)
            replay: deque = deque(maxlen=self.replay_buffer_size)
            epsilon = self.epsilon

            # Multiple passes over history
            for episode in range(self.episodes):
                for t in range(1, len(history)):
                    state = self._compute_state(history, prize_type, t)
                    actual = history[t]

                    # Epsilon-greedy action
                    if random.random() < epsilon:
                        action = random.choice(self._action_spaces[prize_type])
                    else:
                        action = self._best_action(prize_type, state)

                    # Reward
                    reward = (
                        self._REWARDS[prize_type]
                        if action == actual
                        else self._MISS_PENALTY
                    )

                    next_state = self._compute_state(history, prize_type, min(t + 1, len(history)))
                    replay.append((state, action, reward, next_state))

                    # Sample mini-batch from replay
                    if len(replay) >= self.batch_size:
                        batch = random.sample(list(replay), self.batch_size)
                        for s, a, r, ns in batch:
                            # Best next Q
                            next_q = max(
                                self._q_get(prize_type, ns, na)
                                for na in self._action_spaces[prize_type]
                            )
                            current_q = self._q_get(prize_type, s, a)
                            new_q = current_q + self.alpha * (
                                r + self.gamma * next_q - current_q
                            )
                            self._q_set(prize_type, s, a, new_q)

                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        self._fitted = True

    def predict_top_k(self, prize: PrizeType, k: int) -> list[Pick]:
        """Return k picks ranked by Q-value."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        prize_str = str(prize)
        history = self._histories.get(prize_str, [])
        t = len(history)
        state = self._compute_state(history, prize_str, t)

        actions = self._action_spaces.get(prize_str, [])
        if not actions:
            n_pos = self._positions_for(prize)
            return [Pick(value="0" * n_pos, confidence=0.0, rationale="No action space")]

        # Score all actions by Q-value
        scored = [(a, self._q_get(prize_str, state, a)) for a in actions]
        scored.sort(key=lambda x: -x[1])

        # Convert Q-values to rough probabilities via softmax
        q_vals = [q for _, q in scored[:k]]
        if q_vals:
            max_q = max(q_vals)
            exp_q = [2.718 ** min(q - max_q, 10) for q in q_vals]
            total = sum(exp_q) + 1e-10
            confidences = [e / total for e in exp_q]
        else:
            confidences = [1.0 / k] * k

        picks = []
        for i, (action, _) in enumerate(scored[:k]):
            picks.append(Pick(
                value=action,
                confidence=confidences[i],
                rationale=f"Q-learning state={state} (eps_min={self.epsilon_min})",
            ))
        return picks

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
                actual_values = self._values_for(draw, prize_type)
                hit = int(picks[0].value in actual_values)
                probs.append(picks[0].confidence)
                labels.append(hit)
                vals = actual_values
                if vals:
                    hist = self._histories.get(prize_type, [])
                    hist.append(vals[0])
                    self._histories[prize_type] = hist

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
            "q_tables": dict(self._q_tables),
            "action_spaces": self._action_spaces,
            "histories": self._histories,
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
    def load(cls, path: Path) -> "RLQLearner":
        with open(path / "meta.json") as f:
            meta = json.load(f)
        hp = meta["hyperparams"]
        model = cls(**hp)
        with open(path / "weights.pkl", "rb") as f:
            weights = pickle.load(f)
        model._q_tables = weights["q_tables"]
        model._action_spaces = weights["action_spaces"]
        model._histories = weights["histories"]
        model._fitted = weights["fitted"]
        return model
