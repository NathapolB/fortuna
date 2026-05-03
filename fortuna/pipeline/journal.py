"""AI meta-cognition journal generator. SPEC §6.

Auto-generate data/reports/journal-YYYY-MM-DD.md after each draw with:
  - What models predicted
  - What hit/missed
  - Per-model performance vs random baseline
  - "What I'd try next" (rule-based)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fortuna.config import BKK, PRIZE_SPACE, REPORTS_DIR, check_not_icloud
from fortuna.store import get_or_init_db

logger = logging.getLogger(__name__)


def _random_hit_rate(prize_type: str) -> float:
    """Expected random hit rate for a single ticket."""
    space = PRIZE_SPACE[prize_type]
    winners = 2 if prize_type == "three_back" else 1
    return winners / space


def generate_journal(draw_id: str, settlement: dict | None = None) -> Path:
    """Generate AI meta-cognition journal for draw_id.

    Returns path to generated .md file.
    """
    check_not_icloud()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if settlement is None:
        settlement = {"draw_id": draw_id, "results": {}, "total_cost_thb": 0, "total_payout_thb": 0}

    now = datetime.now(BKK).isoformat()
    journal_path = REPORTS_DIR / f"journal-{draw_id}.md"

    lines = []
    lines.append(f"# AI Meta-Cognition Journal — {draw_id}")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Draw ID:** {draw_id}")
    lines.append("")

    # Summary section
    total_cost = settlement.get("total_cost_thb", 0)
    total_payout = settlement.get("total_payout_thb", 0)
    net_pnl = total_payout - total_cost

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total cost | {total_cost} THB |")
    lines.append(f"| Total payout | {total_payout} THB |")
    lines.append(f"| Net P&L | {net_pnl:+d} THB |")
    lines.append("")

    # Per prize type breakdown
    lines.append("## Per-Prize-Type Results")
    lines.append("")

    for prize_type in ("first6", "three_back", "two_back"):
        result = settlement.get("results", {}).get(prize_type, {})
        picks = result.get("picks", [])
        hits = result.get("hits", [])
        n_hits = result.get("n_hits", 0)
        n_tickets = result.get("n_tickets", 0)
        cost = result.get("cost_thb", 0)
        payout = result.get("payout_thb", 0)

        random_rate = _random_hit_rate(prize_type)
        model_rate = n_hits / max(n_tickets, 1)
        lift = model_rate / max(random_rate, 1e-10)

        lines.append(f"### {prize_type}")
        lines.append("")
        lines.append(f"- Picks: {[p['value'] for p in picks]}")
        lines.append(f"- Hits: {n_hits}/{n_tickets}")
        lines.append(f"- Hit rate: {model_rate:.4f} vs random {random_rate:.6f} (lift = {lift:.2f}x)")
        lines.append(f"- Cost: {cost} THB | Payout: {payout} THB")
        lines.append("")

    # Model performance analysis (from DB)
    lines.append("## Model Analysis")
    lines.append("")

    try:
        conn = get_or_init_db()
        model_rows = conn.execute(
            """
            SELECT m.model_id, m.fitness, m.hyperparams, m.train_end
            FROM model_runs m
            ORDER BY m.created_at DESC
            LIMIT 10
            """,
        ).fetchall()

        if model_rows:
            lines.append("| Model | Fitness | Train end |")
            lines.append("|-------|---------|-----------|")
            for row in model_rows:
                lines.append(
                    f"| {row['model_id']} | {row['fitness'] or 'N/A'} | {row['train_end'] or 'N/A'} |"
                )
            lines.append("")
        else:
            lines.append("_No model run records found._")
            lines.append("")
    except Exception as e:
        logger.warning("DB query for model analysis failed: %s", e)
        lines.append("_Model analysis unavailable._")
        lines.append("")

    # Honest assessment
    lines.append("## Honest Assessment")
    lines.append("")

    total_hits = sum(
        settlement.get("results", {}).get(pt, {}).get("n_hits", 0)
        for pt in ("first6", "three_back", "two_back")
    )
    total_tickets = sum(
        settlement.get("results", {}).get(pt, {}).get("n_tickets", 0)
        for pt in ("first6", "three_back", "two_back")
    )

    if total_hits == 0:
        lines.append(
            "All picks missed this draw. This is the expected baseline — "
            "the Thai Government Lottery draws are close to uniform random. "
            "A miss rate near 100% is exactly what we'd expect statistically."
        )
    else:
        lines.append(
            f"{total_hits}/{total_tickets} tickets hit. "
            "Note: even with AI predictions, positive results in individual draws "
            "are not statistically significant without 50+ draw sample."
        )
    lines.append("")

    # "What I'd try next" — rule-based suggestions
    lines.append("## What I Would Try Next")
    lines.append("")

    suggestions = _generate_suggestions(settlement)
    for suggestion in suggestions:
        lines.append(f"- {suggestion}")
    lines.append("")

    lines.append("---")
    lines.append(f"_Journal auto-generated by Fortuna pipeline at {now}_")

    content = "\n".join(lines)
    journal_path.write_text(content, encoding="utf-8")
    logger.info("Journal written to %s", journal_path)

    # Also insert into DB journal table
    try:
        conn = get_or_init_db()
        conn.execute(
            """
            INSERT OR REPLACE INTO journal
              (draw_id, summary_md, confidence, surprises, next_actions, written_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                draw_id,
                content,
                None,
                json.dumps({"total_hits": total_hits, "total_tickets": total_tickets}),
                json.dumps(suggestions),
                now,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning("DB journal insert failed: %s", e)

    return journal_path


def _generate_suggestions(settlement: dict) -> list[str]:
    """Generate rule-based improvement suggestions from settlement data."""
    suggestions = []

    for prize_type in ("first6", "three_back", "two_back"):
        result = settlement.get("results", {}).get(prize_type, {})
        n_hits = result.get("n_hits", 0)
        n_tickets = result.get("n_tickets", 0)

        if n_hits == 0 and n_tickets > 0:
            suggestions.append(
                f"{prize_type}: Zero hits this draw. Consider raising Dirichlet prior strength "
                "to reduce overconfidence in frequency model."
            )

    # Generic suggestions
    suggestions.append(
        "Walk-forward CV on last 50 draws is the true signal — "
        "accumulate 50+ draws before drawing any conclusions."
    )
    suggestions.append(
        "Consider extending LSTM sequence length from 10 to 20 draws "
        "once we have 200+ draws in history."
    )
    suggestions.append(
        "RL Q-learning action space for first6 is heavily compressed (200 actions) — "
        "consider switching to a neural policy for first6 predictions."
    )

    return suggestions
