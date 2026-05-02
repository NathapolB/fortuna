"""BaseModel ABC — all models in the registry inherit from this. SPEC §4."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fortuna.schema import Draw

PrizeType = Literal["first6", "three_back", "two_back"]


@dataclass(frozen=True)
class Pick:
    """One prediction pick. SPEC §4."""
    value: str          # '123456' for first6, '789' for three_back, '34' for two_back
    confidence: float   # model's P(this exact value wins)
    rationale: str      # short human-readable explanation


@dataclass(frozen=True)
class TrainContext:
    """Context passed to model.fit(). SPEC §4."""
    draws: list  # list[Draw] — full history up to (excluding) target_draw
    features: dict[str, float]   # pre-computed feature vector for target
    target_draw_id: str
    git_sha: str


class BaseModel(ABC):
    """Every model in registry inherits this. SPEC §4."""

    model_id: str             # 'frequency-bayes-v2.1' etc.
    feature_spec: list[str]   # which features it needs
    hyperparams: dict         # used by fingerprint()

    @abstractmethod
    def fit(self, ctx: TrainContext) -> None:
        """Train on history. Must be deterministic given seed."""

    @abstractmethod
    def predict_top_k(self, prize: PrizeType, k: int) -> list[Pick]:
        """Return k picks sorted by confidence desc. Sum of confidences <= 1."""

    @abstractmethod
    def score(self, draws: list) -> dict[str, float]:
        """Self-evaluation on a holdout. Returns {brier, log_loss, hit_rate}."""

    @abstractmethod
    def serialize(self, path: Path) -> None:
        """Persist model weights and metadata to path."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from persisted state."""

    def fingerprint(self) -> str:
        """Stable hash of (class, hyperparams, feature_spec). SPEC §4."""
        payload = f"{type(self).__name__}|{self.hyperparams}|{self.feature_spec}"
        return sha256(payload.encode()).hexdigest()[:12]
