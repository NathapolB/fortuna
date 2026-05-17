"""Post-draw settlement pipeline. SPEC §6.

Runs after draw results published (~19:00 draw day):
  1. Scrape new draw via existing scraper
  2. Match each prediction's picks vs actual numbers
  3. Insert into outcomes table (cost_thb, payout_thb, hit bool per ticket)
  4. Update model_runs table with brier/log-loss/hit-rate
  5. Trigger journal entry generation
  6. Update Notion page status to "Settled" and append results (non-blocking)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from fortuna.config import BKK, EXPORTS_DIR, PAYOUTS, PAYOUTS_PAO_TANG, TICKET_COST_THB, check_not_icloud
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
    """Match pick values against actual draw results. Returns hit list.

    Legacy bucket-mode matcher — kept for backward compat with old prediction
    exports (pre-v2.4). New code path uses _score_ticket_multi_prize().
    """
    if prize_type == "first6":
        actual = {actual_draw.first_prize}
    elif prize_type == "three_back":
        actual = set(actual_draw.three_digit_back)
    elif prize_type == "two_back":
        actual = {actual_draw.two_digit_back}
    else:
        actual = set()

    return [p["value"] in actual for p in picks]


def _score_ticket_multi_prize(ticket_value: str, actual_draw) -> dict[str, int]:
    """Score one ticket against ALL Pao Tang prize types (v2.4).

    `ticket_value` may be 2, 3, or 6 digits. For shorter values (legacy 2-back /
    3-back exports), we check only the prize types that apply to that length.

    Returns dict of {prize_type: payout_thb} for each hit. Empty if no hit.
    """
    hits: dict[str, int] = {}

    first = actual_draw.first_prize                          # str, 6 digits
    near = set(actual_draw.first_prize_near or [])           # 2 numbers, 6 digits
    front3_set = set(actual_draw.three_digit_front or [])    # 2 numbers, 3 digits
    back3_set = set(actual_draw.three_digit_back or [])      # 2 numbers, 3 digits
    back2 = actual_draw.two_digit_back                       # str, 2 digits

    L = len(ticket_value)

    if L == 6:
        # Full 6-digit ticket — check every prize type
        if ticket_value == first:
            hits["first1"] = PAYOUTS_PAO_TANG["first1"]
        elif ticket_value in near:
            hits["first_near"] = PAYOUTS_PAO_TANG["first_near"]
        if ticket_value[:3] in front3_set:
            hits["front3"] = PAYOUTS_PAO_TANG["front3"]
        if ticket_value[-3:] in back3_set:
            hits["back3"] = PAYOUTS_PAO_TANG["back3"]
        if ticket_value[-2:] == back2:
            hits["back2"] = PAYOUTS_PAO_TANG["back2"]

    elif L == 3:
        # 3-digit pattern (legacy three_back pick) — Nash buys a 6-digit ticket
        # ending with this pattern. Check back3 exact + back2 partial.
        if ticket_value in back3_set:
            hits["back3"] = PAYOUTS_PAO_TANG["back3"]
        if ticket_value[-2:] == back2:
            hits["back2"] = PAYOUTS_PAO_TANG["back2"]

    elif L == 2:
        # 2-digit pattern (legacy two_back pick) — pure back2 lottery
        if ticket_value == back2:
            hits["back2"] = PAYOUTS_PAO_TANG["back2"]

    return hits


def _get_notion_page_id(conn: sqlite3.Connection, draw_id: str) -> str | None:
    """Fetch the Notion page ID for a given draw from the predictions table."""
    row = conn.execute(
        """
        SELECT notion_page_id FROM predictions
        WHERE draw_id = ? AND model_id = 'ensemble' AND notion_page_id IS NOT NULL
        LIMIT 1
        """,
        (draw_id,),
    ).fetchone()
    if row:
        return row["notion_page_id"]
    return None


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
        # Multi-prize scoring (v2.4): every ticket checks ALL Pao Tang prize
        # types. Legacy `prize_type` here is just the bucket the picker used to
        # generate the value — settlement is bucket-agnostic.
        per_ticket_hits: list[dict[str, int]] = []
        per_ticket_payout: list[int] = []
        any_hits: list[bool] = []

        for pick in picks_list:
            ticket_hits = _score_ticket_multi_prize(pick["value"], actual)
            payout = sum(ticket_hits.values())
            per_ticket_hits.append(ticket_hits)
            per_ticket_payout.append(payout)
            any_hits.append(bool(ticket_hits))

        prize_summary = {
            "picks": picks_list,
            "per_ticket_hits": per_ticket_hits,
            "per_ticket_payout_thb": per_ticket_payout,
            "n_tickets_with_any_hit": sum(any_hits),
            "n_tickets": len(picks_list),
            "cost_thb": len(picks_list) * TICKET_COST_THB,
            "payout_thb": sum(per_ticket_payout),
        }
        summary["results"][prize_type] = prize_summary
        total_cost += prize_summary["cost_thb"]
        total_payout += prize_summary["payout_thb"]

        # Insert outcomes into DB
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

        for pick, hit_dict, ticket_payout in zip(picks_list, per_ticket_hits, per_ticket_payout):
            value = pick["value"]
            pred_id = db_preds.get(value)
            if pred_id is None:
                logger.warning("No pred_id found for pick %s/%s/%s", draw_id, prize_type, value)
                continue

            insert_outcome(
                conn=conn,
                pred_id=pred_id,
                hit=bool(hit_dict),
                payout_thb=ticket_payout,
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

    # --- Enhancement-1: Update Notion page (non-blocking) ---
    try:
        notion_page_id = _get_notion_page_id(conn, draw_id)
        if notion_page_id:
            from fortuna.pipeline.notion_publisher import settle_prediction_page
            settle_prediction_page(notion_page_id, summary)
        else:
            logger.info(
                "No Notion page ID found for draw %s — skipping Notion settle update", draw_id
            )
    except Exception as e:
        logger.warning("Notion settle update failed (non-blocking): %s", e)

    return summary
