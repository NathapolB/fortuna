"""FeatureSpec abstract base. SPEC §1. Phase 2."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FeatureSpec(ABC):
    """Abstract base for all feature specifications."""

    name: str
    description: str

    @abstractmethod
    def compute(self, draws: list, target_draw_id: str) -> float | bytes:
        """Compute feature value. Must use only draws strictly before target_draw_id."""
