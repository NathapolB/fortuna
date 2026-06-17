"""Align publisher — write/update prediction notes in the Align app. SPEC §Enhancement-2 (v2.5).

Replaces the Notion publish step (v2.3). Nash now reads Fortuna predictions inside
his own Align app (the "Lottery" notebook), not Notion.

Writes directly to Align's Supabase `notes` table via the PostgREST endpoint using
the service-role key (bypasses RLS — required for headless cron writes).

`notes.content` is a TipTap / ProseMirror "doc" JSON document. Align's editor uses
StarterKit, so heading / paragraph / orderedList / bulletList / listItem nodes and
bold + code marks all render natively.

Environment variables:
  ALIGN_SUPABASE_URL          — default https://dbxtnbknouplbrztwkvo.supabase.co
  ALIGN_SUPABASE_SERVICE_KEY  — service-role key (REQUIRED; from Align .env.local)
  ALIGN_LOTTERY_NOTEBOOK_ID   — default = the "Lottery" notebook UUID
  ALIGN_NOTE_USER_ID          — default = Nash's Align user UUID

If the service key is missing, every call logs a warning and returns None — the
prediction pipeline never fails because of an Align error (all calls try/except).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Baked-in (non-secret) defaults — discovered from Align Supabase, 16 มิ.ย. 2569.
_DEFAULT_SUPABASE_URL = "https://dbxtnbknouplbrztwkvo.supabase.co"
_DEFAULT_NOTEBOOK_ID = "3e9666f6-1a1e-499f-8a5e-afe315b7a33a"  # "Lottery" notebook
_DEFAULT_USER_ID = "8625443f-fef5-49ea-b400-e11cbc692275"      # Nash

_THAI_MONTHS = [
    "", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]

# Pao Tang prize labels (mirrors PAYOUTS_PAO_TANG in config).
_PRIZE_LABELS = {
    "first1": "รางวัลที่ 1",
    "first_near": "ข้างเคียงรางวัลที่ 1",
    "front3": "เลข 3 ตัวหน้า",
    "back3": "เลข 3 ตัวหลัง",
    "back2": "เลข 2 ตัวล่าง",
}


def _draw_date_thai(iso_date: str) -> str:
    """Convert "2026-07-01" to "1 ก.ค. 2569" (Buddhist Era)."""
    try:
        y, m, d = iso_date.split("-")
        return f"{int(d)} {_THAI_MONTHS[int(m)]} {int(y) + 543}"
    except Exception:
        return iso_date


# ---------------------------------------------------------------------------
# Env / HTTP helpers
# ---------------------------------------------------------------------------

def _cfg() -> dict[str, str] | None:
    """Resolve Align Supabase config from env. Returns None if service key absent."""
    key = os.environ.get("ALIGN_SUPABASE_SERVICE_KEY", "").strip()
    if not key:
        logger.warning("ALIGN_SUPABASE_SERVICE_KEY not set — skipping Align publish")
        return None
    return {
        "url": os.environ.get("ALIGN_SUPABASE_URL", _DEFAULT_SUPABASE_URL).rstrip("/"),
        "key": key,
        "notebook_id": os.environ.get("ALIGN_LOTTERY_NOTEBOOK_ID", _DEFAULT_NOTEBOOK_ID),
        "user_id": os.environ.get("ALIGN_NOTE_USER_ID", _DEFAULT_USER_ID),
    }


def _headers(key: str, *, write: bool = False) -> dict[str, str]:
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if write:
        h["Prefer"] = "return=representation"
    return h


# ---------------------------------------------------------------------------
# TipTap document builder
# ---------------------------------------------------------------------------

def _txt(content: str, *marks: str) -> dict:
    node: dict[str, Any] = {"type": "text", "text": content}
    if marks:
        node["marks"] = [{"type": m} for m in marks]
    return node


def _heading(level: int, text: str) -> dict:
    return {"type": "heading", "attrs": {"level": level}, "content": [_txt(text)]}


def _para(*nodes: dict) -> dict:
    p: dict[str, Any] = {"type": "paragraph"}
    if nodes:
        p["content"] = list(nodes)
    return p


def _ordered(items: list[list[dict]]) -> dict:
    return {
        "type": "orderedList",
        "attrs": {"start": 1},
        "content": [
            {"type": "listItem", "content": [_para(*item)]} for item in items
        ],
    }


def _bullet(items: list[list[dict]]) -> dict:
    return {
        "type": "bulletList",
        "content": [
            {"type": "listItem", "content": [_para(*item)]} for item in items
        ],
    }


def _pick_values(picks_raw: dict) -> list[str]:
    """Flatten all picks (v2.4 = 10 first6) into an ordered list of 6-digit strings."""
    values: list[str] = []
    for prize_type, picks_list in picks_raw.items():
        for p in picks_list:
            v = p.get("value", p) if isinstance(p, dict) else p
            values.append(str(v))
    return values


def _build_doc(prediction: dict) -> dict:
    """Build the TipTap doc JSON for a frozen prediction (v2.4 Pao Tang 10-ticket)."""
    draw_date = prediction.get("target_draw_id", "unknown")
    draw_th = _draw_date_thai(draw_date)
    picks_raw = prediction.get("picks", {})
    values = _pick_values(picks_raw)
    n = len(values)
    cost = prediction.get("total_cost_thb", n * 80)

    freeze_sha = prediction.get("freeze_commit_sha") or "N/A"
    short_sha = freeze_sha[:7] if freeze_sha not in ("N/A", "dry-run", None) else str(freeze_sha)
    frozen_at = prediction.get("frozen_at", "")

    content: list[dict] = []

    content.append(_para(
        _txt(f"🎰 AI Picks — งวด {draw_th}", "bold"),
    ))
    content.append(_para(
        _txt(f"{n} ใบ × ฿80 = ฿{cost:,} · ซื้อผ่านเป๋าตัง"),
    ))
    content.append(_para(
        _txt("ทุกใบเช็คทุกรางวัล: ที่ 1 · ข้างเคียง · 3 ตัวหน้า · 3 ตัวหลัง · 2 ตัวล่าง"),
    ))

    # Strategy 5/3/2 — render tickets grouped by prize target if present.
    plan = prediction.get("ticket_plan")
    if plan:
        groups = [
            ("front3_two_back", "🎯 5 ใบ — เลขหน้า 3 ตัว + เลขท้าย 2 ตัว"),
            ("two_back", "🎲 5 ใบ — เน้นเลขท้าย 2 ตัว"),
            ("front3_back3", "🎯 3 ใบ — เลขหน้า 3 ตัว + เลขท้าย 3 ตัว"),
            ("first1", "🥇 2 ใบ — รางวัลที่ 1"),
        ]
        for gkey, gtitle in groups:
            items = [t for t in plan if t.get("group") == gkey]
            if not items:
                continue
            content.append(_heading(3, gtitle))
            rows: list[list[dict]] = []
            for t in items:
                v = t["value"]
                if gkey == "front3_two_back":
                    rows.append([_txt(v[:3], "bold", "code"), _txt(v[3]),
                                 _txt(v[4:], "bold", "code"),
                                 _txt("  ← หน้า 3 / ท้าย 2")])
                elif gkey == "two_back":
                    rows.append([_txt(v[:4]), _txt(v[4:], "bold", "code"),
                                 _txt("  ← ท้าย 2 ตัว")])
                elif gkey == "front3_back3":
                    rows.append([_txt(v[:3], "bold", "code"), _txt(" · "),
                                 _txt(v[3:], "bold", "code")])
                else:
                    rows.append([_txt(v, "bold", "code")])
            content.append(_bullet(rows))
    else:
        content.append(_heading(3, f"🎟️ เลข {n} ใบ"))
        content.append(_ordered([[_txt(v, "bold", "code")] for v in values]))

    content.append(_heading(3, "🔒 Verifiable Timestamp"))
    content.append(_para(
        _txt("GitHub commit "),
        _txt(short_sha, "code"),
        _txt(f" · frozen {frozen_at}"),
    ))

    content.append(_heading(3, "⚠️ Honest framing"))
    content.append(_para(
        _txt(
            "AI อาจแพ้การสุ่ม — หวยใกล้เคียง uniform random. "
            "นี่คือ entertainment + ML learning ไม่ใช่คำแนะนำการลงทุน."
        ),
    ))

    return {"type": "doc", "content": content}


def _build_settle_nodes(summary: dict) -> list[dict]:
    """Build TipTap nodes to append after settlement (actual results)."""
    nodes: list[dict] = [{"type": "horizontalRule"}]
    net = summary.get("net_pnl_thb", 0)
    cost = summary.get("total_cost_thb", 0)
    payout = summary.get("total_payout_thb", 0)
    emoji = "✅" if net > 0 else "❌"
    nodes.append(_heading(3, f"{emoji} ผลออกแล้ว — สรุป"))
    nodes.append(_para(
        _txt(f"ลงทุน ฿{cost:,} · ได้คืน ฿{payout:,} · สุทธิ ", ),
        _txt(f"{net:+,} ฿", "bold"),
    ))

    # Per-ticket hits (flatten across prize buckets)
    hit_lines: list[list[dict]] = []
    for _bucket, res in summary.get("results", {}).items():
        picks = res.get("picks", [])
        per_hits = res.get("per_ticket_hits", [])
        per_payout = res.get("per_ticket_payout_thb", [])
        for pick, hits, pay in zip(picks, per_hits, per_payout):
            if not hits:
                continue
            val = pick.get("value", pick) if isinstance(pick, dict) else pick
            labels = " + ".join(_PRIZE_LABELS.get(k, k) for k in hits)
            hit_lines.append([
                _txt(str(val), "bold", "code"),
                _txt(f" → {labels} (+฿{pay:,})"),
            ])
    if hit_lines:
        nodes.append(_heading(3, "🎯 ใบที่ถูกรางวัล"))
        nodes.append(_bullet(hit_lines))
    else:
        nodes.append(_para(_txt("งวดนี้ไม่มีใบถูกรางวัล")))
    return nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_existing_note(cfg: dict, title: str) -> str | None:
    """Return the id of an existing note (same notebook + title), or None."""
    import requests  # type: ignore

    try:
        resp = requests.get(
            f"{cfg['url']}/rest/v1/notes",
            headers=_headers(cfg["key"]),
            params={
                "notebook_id": f"eq.{cfg['notebook_id']}",
                "title": f"eq.{title}",
                "select": "id",
                "limit": "1",
            },
            timeout=30,
        )
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]["id"]
    except Exception as e:
        logger.warning("Align note lookup failed: %s", e)
    return None


def publish_prediction_align(prediction: dict) -> str | None:
    """Create or update the Align note for a frozen prediction.

    Idempotent on (notebook_id, title): re-running a prediction updates the same note.

    Returns the Align note UUID, or None if skipped/failed.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        logger.warning("requests not installed — cannot publish to Align")
        return None

    cfg = _cfg()
    if cfg is None:
        return None

    draw_date = prediction.get("target_draw_id", "unknown")
    title = f"🎰 Fortuna — งวด {_draw_date_thai(draw_date)}"
    doc = _build_doc(prediction)

    note_id = _find_existing_note(cfg, title)

    try:
        if note_id:
            resp = requests.patch(
                f"{cfg['url']}/rest/v1/notes",
                headers=_headers(cfg["key"], write=True),
                params={"id": f"eq.{note_id}"},
                json={"content": doc, "updated_at": "now()"},
                timeout=30,
            )
        else:
            resp = requests.post(
                f"{cfg['url']}/rest/v1/notes",
                headers=_headers(cfg["key"], write=True),
                json={
                    "user_id": cfg["user_id"],
                    "notebook_id": cfg["notebook_id"],
                    "title": title,
                    "content": doc,
                    "space": "personal",
                    "tags": ["fortuna", "lottery"],
                },
                timeout=30,
            )
        if resp.status_code in (200, 201):
            data = resp.json()
            nid = data[0]["id"] if isinstance(data, list) and data else note_id
            logger.info("Align note %s for draw %s (%s)",
                        "updated" if note_id else "created", draw_date, nid)
            return nid
        logger.warning("Align API %d for draw %s: %s",
                       resp.status_code, draw_date, resp.text[:500])
        return None
    except Exception as e:
        logger.warning("Align publish failed for draw %s: %s", draw_date, e)
        return None


def settle_prediction_note(note_id: str, settlement_summary: dict) -> bool:
    """Append actual draw results to an existing Align prediction note.

    Fetches the note's current doc, appends settlement nodes, PATCHes it back.
    Idempotent-ish: re-running strips a prior settlement block before re-appending.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        return False

    cfg = _cfg()
    if cfg is None or not note_id:
        return False

    try:
        resp = requests.get(
            f"{cfg['url']}/rest/v1/notes",
            headers=_headers(cfg["key"]),
            params={"id": f"eq.{note_id}", "select": "content", "limit": "1"},
            timeout=30,
        )
        if resp.status_code != 200 or not resp.json():
            logger.warning("Align settle: note %s not found", note_id)
            return False
        doc = resp.json()[0].get("content") or {"type": "doc", "content": []}
        body = doc.get("content", [])

        # Drop any previous settlement block (everything from the first horizontalRule on)
        for i, node in enumerate(body):
            if node.get("type") == "horizontalRule":
                body = body[:i]
                break
        body.extend(_build_settle_nodes(settlement_summary))
        doc["content"] = body

        patch = requests.patch(
            f"{cfg['url']}/rest/v1/notes",
            headers=_headers(cfg["key"], write=True),
            params={"id": f"eq.{note_id}"},
            json={"content": doc, "updated_at": "now()"},
            timeout=30,
        )
        if patch.status_code in (200, 204):
            logger.info("Align note %s settled", note_id)
            return True
        logger.warning("Align settle API %d: %s", patch.status_code, patch.text[:500])
        return False
    except Exception as e:
        logger.warning("Align settle failed for note %s: %s", note_id, e)
        return False
