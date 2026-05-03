"""LSTM model — PyTorch sequence-to-sequence on digit history. SPEC §4.

Small model (~50K params), CPU-trainable in <5 min on 370 draws.
Architecture: embedding + 2 LSTM layers + linear head per digit position.
Walk-forward training approach.

Supports all three prize types: first6, three_back, two_back.
"""

from __future__ import annotations

import heapq
import json
import pickle
from itertools import product
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fortuna.models.base import BaseModel, Pick, PrizeType, TrainContext


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Small LSTM network for lottery digit prediction.

    Input: sequence of (n_pos * embed_dim) per timestep
    Output: n_pos * 10 logits (digit probabilities for each position)
    """

    def __init__(
        self,
        n_positions: int,
        embed_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.embed_dim = embed_dim

        # Per-digit embeddings (shared across positions)
        self.embedding = nn.Embedding(10, embed_dim)

        # LSTM
        input_size = n_positions * embed_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Separate linear head per position
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 10) for _ in range(n_positions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_positions) int indices.
        Returns: (batch, seq_len, n_positions, 10) logits.
        """
        batch, seq_len, n_pos = x.shape

        # Embed each position: (batch, seq_len, n_pos, embed_dim)
        embedded = self.embedding(x)
        # Flatten positions: (batch, seq_len, n_pos * embed_dim)
        embedded_flat = embedded.view(batch, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(embedded_flat)  # (batch, seq_len, hidden)

        # Apply heads
        outputs = []
        for head in self.heads:
            outputs.append(head(lstm_out))  # (batch, seq_len, 10)
        # Stack: (batch, seq_len, n_pos, 10)
        return torch.stack(outputs, dim=2)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LSTMModel(BaseModel):
    """LSTM-based digit sequence model.

    Trains on sequences of draw_history length, predicts next draw digits.
    """

    model_id = "lstm-v1"
    feature_spec: list[str] = []
    hyperparams: dict = {
        "seq_len": 10,
        "embed_dim": 8,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 16,
        "seed": 42,
    }

    def __init__(
        self,
        seq_len: int = 10,
        embed_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.hyperparams = {
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
        }
        self.feature_spec = []
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

        self._nets: dict[str, _LSTMNet] = {}
        self._histories: dict[str, list[str]] = {}
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

    def _build_sequences(
        self, history: list[str], n_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (X, y) tensors from history list.

        X: (n_samples, seq_len, n_pos) — input sequences
        y: (n_samples, n_pos) — target digits
        """
        X_list, y_list = [], []
        for i in range(len(history) - self.seq_len):
            seq = history[i : i + self.seq_len]
            target = history[i + self.seq_len]

            seq_digits = [[int(c) for c in draw_val[:n_pos]] for draw_val in seq]
            target_digits = [int(c) for c in target[:n_pos]]
            X_list.append(seq_digits)
            y_list.append(target_digits)

        if not X_list:
            return torch.empty(0), torch.empty(0)

        X = torch.tensor(X_list, dtype=torch.long)  # (n, seq_len, n_pos)
        y = torch.tensor(y_list, dtype=torch.long)  # (n, n_pos)
        return X, y

    def fit(self, ctx: TrainContext) -> None:
        """Train LSTM on draw history."""
        torch.manual_seed(self.seed)

        self._nets = {}
        self._histories = {}

        for prize_type in ("first6", "three_back", "two_back"):
            pt = cast(PrizeType, prize_type)
            n_pos = self._positions_for(pt)

            history: list[str] = []
            for draw in ctx.draws:
                vals = self._values_for(draw, pt)
                if vals and len(vals[0]) == n_pos and vals[0].isdigit():
                    history.append(vals[0])

            self._histories[prize_type] = history

            if len(history) < self.seq_len + 2:
                # Not enough data — skip this prize type
                continue

            X, y = self._build_sequences(history, n_pos)
            if X.shape[0] == 0:
                continue

            net = _LSTMNet(
                n_positions=n_pos,
                embed_dim=self.embed_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            dataset = TensorDataset(X, y)
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
            )

            net.train()
            for _epoch in range(self.epochs):
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    # x_batch: (B, seq_len, n_pos)
                    logits = net(x_batch)  # (B, seq_len, n_pos, 10)
                    last_logits = logits[:, -1, :, :]  # (B, n_pos, 10)

                    # Sum cross-entropy loss across positions
                    pos_losses = [
                        criterion(last_logits[:, pos, :], y_batch[:, pos])
                        for pos in range(n_pos)
                    ]
                    loss = sum(pos_losses) / n_pos  # type: ignore[arg-type]

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

            net.eval()
            self._nets[prize_type] = net

        self._fitted = True

    def _predict_probs(self, prize_type: str) -> list[list[float]]:
        """Return per-position softmax probabilities using last seq_len history."""
        n_pos = self._positions_for(cast(PrizeType, prize_type))
        history = self._histories.get(prize_type, [])
        net = self._nets.get(prize_type)

        if net is None or len(history) < self.seq_len:
            # Uniform fallback
            return [[1.0 / 10] * 10 for _ in range(n_pos)]

        seq = history[-self.seq_len :]
        seq_digits = [[int(c) for c in val[:n_pos]] for val in seq]
        x = torch.tensor([seq_digits], dtype=torch.long)  # (1, seq_len, n_pos)

        with torch.no_grad():
            logits = net(x)  # (1, seq_len, n_pos, 10)
            last = logits[0, -1, :, :]  # (n_pos, 10)
            probs = torch.softmax(last, dim=-1).numpy()  # (n_pos, 10)

        return [probs[pos].tolist() for pos in range(n_pos)]

    def predict_top_k(self, prize: PrizeType, k: int) -> list[Pick]:
        """Return k picks using LSTM-predicted probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

        prize_str = str(prize)
        n_pos = self._positions_for(prize)
        pos_probs = self._predict_probs(prize_str)

        if prize == "first6":
            return self._top_k_beam(pos_probs, k)

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
                rationale=f"LSTM seq_len={self.seq_len} hidden={self.hidden_size}",
            ))
        return picks

    def _top_k_beam(self, pos_probs: list[list[float]], k: int) -> list[Pick]:
        """Beam top-k for 6-position (5^6 = 15625 candidates)."""
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
        return [
            Pick(value=v, confidence=s, rationale=f"LSTM beam hidden={self.hidden_size}")
            for s, v in heap
        ]

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
                if actual_values:
                    hist = self._histories.get(prize_type, [])
                    hist.append(actual_values[0])
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
        nets_state = {pt: net.state_dict() for pt, net in self._nets.items()}
        net_configs = {
            pt: {
                "n_positions": net.n_positions,
                "embed_dim": net.embed_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
            for pt, net in self._nets.items()
        }
        torch.save({"state_dicts": nets_state, "configs": net_configs}, path / "weights.pt")
        with open(path / "histories.pkl", "wb") as f:
            pickle.dump(self._histories, f)
        meta = {
            "model_id": self.model_id,
            "hyperparams": self.hyperparams,
            "feature_spec": self.feature_spec,
            "fingerprint": self.fingerprint(),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        with open(path / "meta.json") as f:
            meta = json.load(f)
        hp = meta["hyperparams"]
        model = cls(**hp)
        checkpoint = torch.load(path / "weights.pt", map_location="cpu", weights_only=False)
        nets_state = checkpoint["state_dicts"]
        net_configs = checkpoint["configs"]
        for pt, state_dict in nets_state.items():
            cfg = net_configs[pt]
            net = _LSTMNet(
                n_positions=cfg["n_positions"],
                embed_dim=cfg["embed_dim"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            )
            net.load_state_dict(state_dict)
            net.eval()
            model._nets[pt] = net
        with open(path / "histories.pkl", "rb") as f:
            model._histories = pickle.load(f)
        model._fitted = True
        return model
