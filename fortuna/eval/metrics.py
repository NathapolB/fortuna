"""Evaluation metrics. SPEC §7.1. Phase 2 implementation."""

from __future__ import annotations

# Phase 2 stub — implement after models are built


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Brier score = mean((p - y)^2). SPEC §7.1."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have same length")
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def log_loss(probabilities: list[float], outcomes: list[int], eps: float = 1e-15) -> float:
    """Log loss = -mean(y log p + (1-y) log(1-p)). SPEC §7.1."""
    import math
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have same length")
    total = 0.0
    for p, y in zip(probabilities, outcomes):
        p = max(eps, min(1 - eps, p))
        total += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -total / len(probabilities)


def hit_rate(hits: int, tickets: int) -> float:
    """hit_rate = hits / tickets. SPEC §7.1."""
    if tickets == 0:
        return 0.0
    return hits / tickets


def lift(observed_hit_rate: float, random_hit_rate: float) -> float:
    """lift = hit_rate / random_hit_rate. SPEC §7.1."""
    if random_hit_rate == 0:
        return float("inf")
    return observed_hit_rate / random_hit_rate


def sharpe_pnl(per_draw_pnl: list[float], draws_per_year: int = 24) -> float:
    """Sharpe ratio of per-draw P&L. SPEC §7.1.

    sharpe = mean(pnl) / std(pnl) * sqrt(draws_per_year)
    24 draws/year = 2 draws/month * 12 months.
    """
    import math
    if len(per_draw_pnl) < 2:
        return 0.0
    n = len(per_draw_pnl)
    mean = sum(per_draw_pnl) / n
    variance = sum((x - mean) ** 2 for x in per_draw_pnl) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(draws_per_year)
