"""Notion publisher — create/update prediction pages in Notion. SPEC §Enhancement-1 (v2.3).

Uses the Notion REST API directly (not MCP, which is only available in agent contexts).

Target DB: Claude Scheduler DB (shared with other JARVIS scheduled outputs).
DB ID: 343d185571598079963ccfeb61f197c6

Real schema — ONLY 3 properties exist:
  Schedule Name  (title)   — pattern: "🎰 Fortuna — งวด YYYY-MM-DD"
  Run Date       (date)    — the draw date (ISO YYYY-MM-DD); used for calendar sorting
  Archive        (checkbox) — false by default; true to hide from active views

All prediction data (picks, model weights, tamper timestamp, honest framing) lives in
the page BODY as Notion blocks. Do NOT attempt to set any other properties — the API
will reject the request with a 400.

Environment variables required:
  NOTION_TOKEN          — integration token from https://www.notion.so/my-integrations
  NOTION_FORTUNA_DB_ID  — DB ID of the shared Claude Scheduler DB

If either env var is missing, all operations log a warning and return None.
Pipeline will never fail due to Notion errors (all calls wrapped in try/except).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# Thai month names for display
_THAI_MONTHS = [
    "", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]


def _draw_date_thai(iso_date: str) -> str:
    """Convert "2026-05-16" to "16 พ.ค. 2569" (Buddhist Era year)."""
    try:
        y, m, d = iso_date.split("-")
        be_year = int(y) + 543
        return f"{int(d)} {_THAI_MONTHS[int(m)]} {be_year}"
    except Exception:
        return iso_date


def _get_notion_headers() -> dict[str, str] | None:
    """Build Notion API headers from env. Returns None if token missing."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        logger.warning("NOTION_TOKEN not set — skipping Notion publish")
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _get_db_id() -> str | None:
    """Return NOTION_FORTUNA_DB_ID env var. Returns None if missing."""
    db_id = os.environ.get("NOTION_FORTUNA_DB_ID", "").strip()
    if not db_id:
        logger.warning("NOTION_FORTUNA_DB_ID not set — skipping Notion publish")
        return None
    return db_id


def _build_page_body(prediction: dict) -> list[dict]:
    """Build Notion block children for the prediction page body.

    All content lives here since the DB has only 3 properties.
    Sections: callout header, 3 prize sections, divider, model contributions,
    verifiable timestamp, honest framing callout.
    """
    draw_date = prediction.get("target_draw_id", "unknown")
    draw_date_th = _draw_date_thai(draw_date)
    picks_raw = prediction.get("picks", {})
    model_versions = prediction.get("model_versions", {})
    freeze_sha = prediction.get("freeze_commit_sha") or "N/A"
    frozen_at = prediction.get("frozen_at", "")

    short_sha = (
        freeze_sha[:7]
        if freeze_sha not in ("N/A", "dry-run", None)
        else freeze_sha
    )
    github_url = (
        f"https://github.com/bnash/fortuna/commit/{freeze_sha}"
        if freeze_sha not in ("N/A", "dry-run", None)
        else ""
    )

    blocks: list[dict] = []

    # ── Top callout ──────────────────────────────────────────────────────────
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"emoji": "🎰"},
            "color": "yellow_background",
            "rich_text": [{
                "type": "text",
                "text": {
                    "content": (
                        f"AI's Picks for งวด {draw_date_th}. "
                        "Buy via Pao Tang. Verifiable timestamp: GitHub commit"
                    )
                },
            }],
        },
    })

    # ── รางวัลที่ 1 — 2 picks ────────────────────────────────────────────────
    first6_picks = picks_raw.get("first6", [])
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🥇 รางวัลที่ 1 (2 ใบ × ฿80 = ฿160)"}}],
        },
    })
    for p in first6_picks:
        value = p.get("value", p) if isinstance(p, dict) else p
        conf = p.get("confidence") if isinstance(p, dict) else None
        conf_text = f" — confidence {conf:.4%}" if conf is not None else ""
        blocks.append({
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": str(value)},
                        "annotations": {"bold": True, "code": True},
                    },
                    {
                        "type": "text",
                        "text": {"content": conf_text},
                    },
                ],
            },
        })

    # ── เลขท้าย 3 ตัว — 3 picks ──────────────────────────────────────────────
    three_back_picks = picks_raw.get("three_back", [])
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🎯 เลขท้าย 3 ตัว (3 ใบ × ฿80 = ฿240)"}}],
        },
    })
    for p in three_back_picks:
        value = p.get("value", p) if isinstance(p, dict) else p
        conf = p.get("confidence") if isinstance(p, dict) else None
        conf_text = f" — confidence {conf:.4%}" if conf is not None else ""
        blocks.append({
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": str(value)},
                        "annotations": {"bold": True, "code": True},
                    },
                    {
                        "type": "text",
                        "text": {"content": conf_text},
                    },
                ],
            },
        })

    # ── เลขท้าย 2 ตัว — 5 picks as a paragraph with " · " separator ──────────
    two_back_picks = picks_raw.get("two_back", [])
    two_back_values = [
        (p.get("value", p) if isinstance(p, dict) else p)
        for p in two_back_picks
    ]
    blocks.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "🎲 เลขท้าย 2 ตัว (5 ใบ × ฿80 = ฿400)"}}],
        },
    })
    # Build inline rich_text: each value code-styled, separated by " · " plain text
    two_back_rich: list[dict] = []
    for i, val in enumerate(two_back_values):
        if i > 0:
            two_back_rich.append({
                "type": "text",
                "text": {"content": " · "},
            })
        two_back_rich.append({
            "type": "text",
            "text": {"content": str(val)},
            "annotations": {"code": True},
        })
    if two_back_rich:
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": two_back_rich},
        })

    # ── Divider ───────────────────────────────────────────────────────────────
    blocks.append({"object": "block", "type": "divider", "divider": {}})

    # ── Model contributions ───────────────────────────────────────────────────
    if model_versions:
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": "📊 Model Contributions"}}],
            },
        })
        total_models = len(model_versions)
        equal_weight = 1.0 / total_models if total_models else 0.0
        for model_name in model_versions:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": f"{model_name} (weight {equal_weight:.2f})"},
                    }],
                },
            })

    # ── Verifiable timestamp ──────────────────────────────────────────────────
    blocks.append({
        "object": "block",
        "type": "heading_3",
        "heading_3": {
            "rich_text": [{"type": "text", "text": {"content": "🔒 Verifiable Timestamp"}}],
        },
    })
    if github_url:
        ts_rich: list[dict] = [
            {"type": "text", "text": {"content": "GitHub commit: "}},
            {
                "type": "text",
                "text": {"content": short_sha, "link": {"url": github_url}},
                "annotations": {"code": True},
            },
            {"type": "text", "text": {"content": f" pushed at {frozen_at}"}},
        ]
    else:
        ts_rich = [
            {"type": "text", "text": {"content": f"Commit: {short_sha} — frozen at {frozen_at}"}},
        ]
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": ts_rich},
    })

    # ── Honest framing callout ────────────────────────────────────────────────
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"emoji": "⚠️"},
            "color": "gray_background",
            "rich_text": [{
                "type": "text",
                "text": {
                    "content": (
                        "Honest framing: AI may underperform random. "
                        "Lottery ≈ uniform random. "
                        "This is entertainment + ML learning, not financial advice."
                    )
                },
            }],
        },
    })

    return blocks


def publish_prediction(prediction: dict, notion_db_id: str | None = None) -> str | None:
    """Create a Notion page for a frozen prediction.

    Uses ONLY the 3 real properties of the Claude Scheduler DB:
      - Schedule Name (title)
      - Run Date (date)
      - Archive (checkbox)

    All pick data lives in the page body (children blocks).

    Args:
        prediction: payload dict from run_predict()
        notion_db_id: override DB ID (defaults to NOTION_FORTUNA_DB_ID env var)

    Returns:
        Notion page URL string, or None if publish was skipped/failed.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        logger.warning("requests library not installed — cannot publish to Notion")
        return None

    headers = _get_notion_headers()
    if headers is None:
        return None

    db_id = notion_db_id or _get_db_id()
    if db_id is None:
        return None

    draw_date = prediction.get("target_draw_id", "unknown")
    draw_date_th = _draw_date_thai(draw_date)

    # Only 3 real properties — no other properties exist in this DB
    page_payload: dict[str, Any] = {
        "parent": {"database_id": db_id},
        "properties": {
            "Schedule Name": {
                "title": [{"text": {"content": f"🎰 Fortuna — งวด {draw_date_th}"}}]
            },
            "Run Date": {
                "date": {"start": draw_date}  # ISO YYYY-MM-DD
            },
            "Archive": {
                "checkbox": False
            },
        },
        "children": _build_page_body(prediction),
    }

    try:
        response = requests.post(
            f"{NOTION_API_BASE}/pages",
            headers=headers,
            json=page_payload,
            timeout=30,
        )
        if response.status_code in (200, 201):
            data = response.json()
            page_id = data.get("id", "")
            page_url = data.get("url", f"https://www.notion.so/{page_id.replace('-', '')}")
            logger.info("Notion page created for draw %s: %s", draw_date, page_url)
            return page_url
        else:
            logger.warning(
                "Notion API returned %d for draw %s: %s",
                response.status_code,
                draw_date,
                response.text[:500],
            )
            return None
    except Exception as e:
        logger.warning("Notion publish failed for draw %s: %s", draw_date, e)
        return None


def settle_prediction_page(
    page_id: str,
    settlement_summary: dict,
) -> bool:
    """Append draw results to an existing Notion prediction page.

    Does NOT attempt to update any properties (the DB only has Schedule Name,
    Run Date, Archive — none of which reflect settlement status). All result
    data is appended to the page body via PATCH /v1/blocks/{page_id}/children.

    Args:
        page_id: Notion page ID stored in predictions.notion_page_id
        settlement_summary: dict from run_settle() with keys:
            draw_id, actual_results, tickets, net_pnl_thb, hit_count,
            brier_lift, settled_at

    Returns:
        True if the append succeeded, False otherwise.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        logger.warning("requests library not installed — cannot update Notion page")
        return False

    headers = _get_notion_headers()
    if headers is None:
        return False

    draw_id = settlement_summary.get("draw_id", "unknown")
    net_pnl = settlement_summary.get("net_pnl_thb", 0)
    hit_count = settlement_summary.get("hit_count", 0)
    brier_lift = settlement_summary.get("brier_lift", 0.0)
    settled_at = settlement_summary.get("settled_at", "")
    tickets = settlement_summary.get("tickets", [])  # list of {prize_type, pick, result, payout}
    actual = settlement_summary.get("actual_results", {})

    result_blocks: list[dict] = [
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "📈 ผลรางวัล"}}],
            },
        },
    ]

    # Actual winning numbers paragraph
    if actual:
        first_prize = actual.get("first_prize", "—")
        three_back = ", ".join(actual.get("three_back", []))
        two_back = actual.get("two_back", "—")
        result_blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "ผลออก: รางวัลที่ 1 "}},
                    {
                        "type": "text",
                        "text": {"content": str(first_prize)},
                        "annotations": {"bold": True},
                    },
                    {"type": "text", "text": {"content": f"  |  เลขท้าย 3 ตัว {three_back}"}},
                    {"type": "text", "text": {"content": f"  |  เลขท้าย 2 ตัว {two_back}"}},
                ],
            },
        })

    # Results table (header row + one row per ticket)
    prize_label_map = {
        "first6": "รางวัลที่ 1",
        "three_back": "เลขท้าย 3 ตัว",
        "two_back": "เลขท้าย 2 ตัว",
    }
    if tickets:
        table_rows: list[dict] = [
            {
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"type": "text", "text": {"content": "Prize"}}],
                        [{"type": "text", "text": {"content": "Pick"}}],
                        [{"type": "text", "text": {"content": "Result"}}],
                        [{"type": "text", "text": {"content": "Payout (฿)"}}],
                    ]
                },
            }
        ]
        for t in tickets:
            prize_label = prize_label_map.get(t.get("prize_type", ""), t.get("prize_type", ""))
            pick_val = str(t.get("pick", ""))
            result_str = "HIT ✓" if t.get("hit") else "miss"
            payout_str = f"{t.get('payout_thb', 0):,}"
            table_rows.append({
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"type": "text", "text": {"content": prize_label}}],
                        [{"type": "text", "text": {"content": pick_val}, "annotations": {"code": True}}],
                        [{"type": "text", "text": {"content": result_str}}],
                        [{"type": "text", "text": {"content": payout_str}}],
                    ]
                },
            })
        result_blocks.append({
            "object": "block",
            "type": "table",
            "table": {
                "table_width": 4,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_rows,
            },
        })

    # P&L summary callout
    pnl_positive = net_pnl > 0
    result_blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"emoji": "💰" if pnl_positive else "📉"},
            "color": "green_background" if pnl_positive else "red_background",
            "rich_text": [{
                "type": "text",
                "text": {
                    "content": (
                        f"Net P&L: ฿{net_pnl:+,d} | "
                        f"Hit {hit_count}/10 tickets | "
                        f"Brier vs random: {brier_lift:+.4f} | "
                        f"Settled: {settled_at}"
                    )
                },
            }],
        },
    })

    try:
        append_response = requests.patch(
            f"{NOTION_API_BASE}/blocks/{page_id}/children",
            headers=headers,
            json={"children": result_blocks},
            timeout=30,
        )
        if append_response.status_code in (200, 201):
            logger.info("Notion page updated with settlement results for draw %s", draw_id)
            return True
        else:
            logger.warning(
                "Notion settle append returned %d for draw %s: %s",
                append_response.status_code,
                draw_id,
                append_response.text[:500],
            )
            return False
    except Exception as e:
        logger.warning("Notion settle append failed for draw %s: %s", draw_id, e)
        return False
