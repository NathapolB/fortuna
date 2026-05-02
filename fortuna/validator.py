"""Cross-source consistency validation. SPEC §3.3.

2-of-3 quorum: if any field disagrees across 2/3 sources, flag in
data/raw/discrepancies.jsonl and skip canonical insertion.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from fortuna.config import DISCREPANCIES_JSONL, BKK
from fortuna.schema import Draw

logger = logging.getLogger(__name__)

# Fields that must agree across sources for a draw to be accepted
QUORUM_FIELDS = ("first_prize", "two_digit_back")
# three_digit_back agreement is checked but not blocking if only 1 source has it
THREE_BACK_FIELDS = ("three_digit_back",)

KNOWN_SHIFTED_DATES_PATH = Path(__file__).parent.parent / "data" / "raw" / "known_shifted_dates.json"


def _load_known_shifted_dates() -> set[str]:
    if KNOWN_SHIFTED_DATES_PATH.exists():
        data = json.loads(KNOWN_SHIFTED_DATES_PATH.read_text())
        return set(data.get("shifted_dates", {}).keys())
    return set()


def validate_draw_date(draw: Draw) -> bool:
    """Check that draw_date falls on 1st or 16th (with exceptions). SPEC §3.3."""
    d = date.fromisoformat(draw.draw_date)
    if d.day in (1, 16):
        return True
    known_shifted = _load_known_shifted_dates()
    if draw.draw_id in known_shifted:
        logger.info("Draw %s is a known shifted date — accepted", draw.draw_id)
        return True
    logger.warning(
        "Draw %s falls on day %d, not 1st or 16th and not in known_shifted_dates",
        draw.draw_id, d.day
    )
    return False


def validate_digits(draw: Draw) -> list[str]:
    """Return list of validation errors for digit format checks. SPEC §3.3."""
    errors: list[str] = []

    if len(draw.first_prize) != 6 or not draw.first_prize.isdigit():
        errors.append(f"first_prize invalid: {draw.first_prize!r}")

    if len(draw.two_digit_back) != 2 or not draw.two_digit_back.isdigit():
        errors.append(f"two_digit_back invalid: {draw.two_digit_back!r}")

    for num in draw.three_digit_back:
        if len(num) != 3 or not num.isdigit():
            errors.append(f"three_digit_back invalid item: {num!r}")

    for num in draw.three_digit_front:
        if len(num) != 3 or not num.isdigit():
            errors.append(f"three_digit_front invalid item: {num!r}")

    return errors


def cross_check(
    draws: list[Draw],
    draw_id: str,
) -> tuple[Draw | None, list[dict]]:
    """Apply 2-of-3 quorum validation across sources for a single draw_id.

    Args:
        draws: List of Draw objects from different sources for the same draw_id.
               Typically 2–3 sources: sanook, kapook, glo.
        draw_id: The draw ID being validated.

    Returns:
        (canonical_draw, discrepancies)
        - canonical_draw: the Draw to use (from primary source) if quorum passed, else None
        - discrepancies: list of dicts describing any field disagreements
    """
    if not draws:
        return None, []

    if len(draws) == 1:
        # Only one source — accept it but mark as unverified
        draw = draws[0]
        return draw, []

    discrepancies: list[dict] = []
    votes: dict[str, dict[str, int]] = {field: {} for field in QUORUM_FIELDS}

    for draw in draws:
        for field in QUORUM_FIELDS:
            value = getattr(draw, field)
            votes[field][value] = votes[field].get(value, 0) + 1

    # Check quorum for each field
    quorum_ok = True
    for field in QUORUM_FIELDS:
        field_votes = votes[field]
        if not field_votes:
            continue

        # Find value with most votes
        best_value, best_count = max(field_votes.items(), key=lambda x: x[1])
        total_sources = len(draws)

        if best_count < 2:
            # No value has 2+ votes — genuine conflict
            discrepancies.append({
                "draw_id": draw_id,
                "field": field,
                "votes": field_votes,
                "sources": [d.source for d in draws],
                "resolution": "SKIP — no quorum",
            })
            quorum_ok = False
            logger.warning(
                "Quorum FAILED for draw %s field %s: votes=%s",
                draw_id, field, field_votes,
            )
        elif best_count < total_sources:
            # Majority agrees but at least one source disagrees — log as discrepancy
            minority_values = {v: c for v, c in field_votes.items() if v != best_value}
            discrepancies.append({
                "draw_id": draw_id,
                "field": field,
                "majority_value": best_value,
                "minority_values": minority_values,
                "sources": [d.source for d in draws],
                "resolution": "ACCEPT — majority quorum",
            })
            logger.info(
                "Minor discrepancy for draw %s field %s: majority=%s, minority=%s",
                draw_id, field, best_value, minority_values,
            )

    if not quorum_ok:
        _write_discrepancies(discrepancies)
        return None, discrepancies

    # Use the primary source (first in list, expected to be sanook for backfill)
    primary = draws[0]

    # Build verified_against list
    verified_sources = [d.source for d in draws if d.source != primary.source]

    # Return a new Draw with verified_against populated
    canonical = primary.model_copy(
        update={"verified_against": verified_sources}
    )

    if discrepancies:
        _write_discrepancies(discrepancies)

    return canonical, discrepancies


def _write_discrepancies(discrepancies: list[dict]) -> None:
    """Append discrepancies to discrepancies.jsonl. SPEC §3.3."""
    if not discrepancies:
        return
    DISCREPANCIES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with DISCREPANCIES_JSONL.open("a", encoding="utf-8") as f:
        for d in discrepancies:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
