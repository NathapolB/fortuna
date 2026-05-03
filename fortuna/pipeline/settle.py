"""Post-draw settlement pipeline. SPEC §6.

Runs after draw results published (~19:00 draw day):
  1. Scrape new draw via existing scraper
  2. Match each prediction's picks vs actual numbers
  3. Insert into outcomes table (cost_thb, payout_thb, hit bool per ticket)
  4. Update model_runs table with brier/log-loss/hit-rate
  5. Trigger journal entry generation
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from fortuna.config import BKK, EXPORTS_DIR, PAYOUTS, TICKET_COST_THB, check_not_icloud
from fortuna.store import DrawStore, get_or_init_db, insert_outcome

logger = logging.getLogger(__name__)


def _load_prediction_export(draw_id: str) -> dict | None:
    """Load prediction export JSON for a draw_id."""
    export_path = EXPORTS_DIR / f"{draw_id}-prediction.json"
    if not export_path.exists():
        logger.warning("No prediction export found for %s at %s", draw_id, export_path)
        return None
    with open(export_path) as f:
        return json.load(f)


def _match_picks(
    picks: list[dict],
    prize_type: str,
    actual_draw,
) -> list[bool]:
    """Match pick values against actual draw results. Returns hit list."""
    if prize_type == "first6":
        actual = {actual_draw.first_prize}
    elif prize_type == "three_back":
        actual = set(actual_draw.three_digit_back)
    elif prize_type == "two_back":
        actual = {actual_draw.two_digit_back}
    else:
        actual = set()

    return [p["value"] in actual for p in picks]


def run_settle(draw_id: str) -> dict:
    """Run settlement for a completed draw.

    Returns summary dict with hits/payouts.
    """
    check_not_icloud()

    # Load actual draw result
    store = DrawStore()
    all_draws = store.all_draws()
    actual = next((d for d in all_draws if d.draw_id == draw_id), None)

    if actual is None:
        # Try to scrape
        logger.info("Draw %s not in local store — attempting scrape...", draw_id)
        try:
            from fortuna.data.scraper import GLOScraper
            scraper = GLOScraper()
            draws = scraper.fetch_draws_for_date(draw_id)
            if draws:
                for d in draws:
                    store.append(d)
                actual = next((d for d in store.all_draws() if d.draw_id == draw_id), None)
        except Exception as e:
            logger.error("Scrape failed: %s", e)

    if actual is None:
        raise ValueError(f"Cannot find draw results for {draw_id}. Run scrape first.")

    # Load prediction export
    prediction = _load_prediction_export(draw_id)
    if prediction is None:
        raise ValueError(f"No prediction export for {draw_id}. Was predict run first?")

    conn = get_or_init_db()
    settled_at = datetime.now(BKK).isoformat()
    summary = {
        "draw_id": draw_id,
        "settled_at": settled_at,
        "results": {},
        "total_cost_thb": 0,
        "total_payout_thb": 0,
    }

    total_cost = 0
    total_payout = 0

    for prize_type, picks_list in prediction.get("picks", {}).items():
        hits = _match_picks(picks_list, prize_type, actual)
        payout_per_hit = PAYOUTS.get(prize_type, 0)

        prize_summary = {
            "picks": picks_list,
            "hits": hits,
            "n_hits": sum(hits),
            "n_tickets": len(picks_list),
            "cost_thb": len(picks_list) * TICKET_COST_THB,
            "payout_thb": sum(hits) * payout_per_hit,
        }
        summary["results"][prize_type] = prize_summary
        total_cost += prize_summary["cost_thb"]
        total_payout += prize_summary["payout_thb"]

        # Insert outcomes into DB
        # Get pred_ids from DB
        cur = conn.execute(
            """
            SELECT pred_id, pick_value, pick_rank
            FROM predictions
            WHERE draw_id = ? AND prize_type = ? AND model_id = 'ensemble'
            ORDER BY pick_rank
            """,
            (draw_id, prize_type),
        )
        db_preds = {row["pick_value"]: row["pred_id"] for row in cur.fetchall()}

        for pick, hit in zip(picks_list, hits):
            value = pick["value"]
            pred_id = db_preds.get(value)
            if pred_id is None:
                logger.warning("No pred_id found for pick %s/%s/%s", draw_id, prize_type, value)
                continue

            payout = payout_per_hit if hit else 0
            insert_outcome(
                conn=conn,
                pred_id=pred_id,
                hit=bool(hit),
                payout_thb=payout,
                cost_thb=TICKET_COST_THB,
                settled_at=settled_at,
            )

    summary["total_cost_thb"] = total_cost
    summary["total_payout_thb"] = total_payout
    summary["net_pnl_thb"] = total_payout - total_cost

    logger.info(
        "Settlement complete for %s: cost=%d THB, payout=%d THB, net=%+d THB",
        draw_id,
        total_cost,
        total_payout,
        summary["net_pnl_thb"],
    )

    # Trigger journal generation
    try:
        from fortuna.pipeline.journal import generate_journal
        generate_journal(draw_id, summary)
    except Exception as e:
        logger.warning("Journal generation failed: %s", e)

    return summary
