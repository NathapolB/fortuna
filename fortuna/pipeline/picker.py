"""Picker — produce final 2/3/5 ticket selections from model outputs. SPEC §6.1.

Picks:
  - 2 first6 picks (Hamming distance >= 2 between picks)
  - 3 three_back picks (3 distinct values; Hamming distance >= 1)
  - 5 two_back picks (no diversity filter — only 100 possible values)

Total 10 tickets, 800 THB.
"""

from __future__ import annotations

import logging
import random
from typing import cast

from fortuna.config import PICK_SPLIT, PRIZE_SPACE
from fortuna.models.base import Pick, PrizeType

logger = logging.getLogger(__name__)


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _is_unhumanlike(value: str, prize_type: str) -> bool:
    """Reject picks that look obviously non-random to humans.

    Mathematically these are as likely as any other number, but historical
    evidence suggests Thai lottery outcomes rarely fall in these patterns,
    likely due to mechanical quirks of the draw machine. More importantly,
    Nash explicitly does not want them.

    Filters apply to first6 only — three_back/two_back are too short to
    meaningfully filter (e.g., "00" or "000" can and do appear).
    """
    if prize_type != "first6":
        return False

    # All same digit: 000000, 111111, ..., 999999
    if len(set(value)) == 1:
        return True

    # 4+ identical digits anywhere (e.g., 011000 has 4 zeros, 822888 has 4 eights)
    most_common_count = max(value.count(d) for d in set(value))
    if most_common_count >= 4:
        return True

    # 3+ leading zeros (000xxx) or trailing zeros (xxx000) — including 011000
    if value.startswith("000") or value.endswith("000"):
        return True

    # Strict ascending/descending sequence
    digits = [int(c) for c in value]
    diffs = [digits[i + 1] - digits[i] for i in range(len(digits) - 1)]
    if all(d == 1 for d in diffs) or all(d == -1 for d in diffs):
        return True

    return False


def _pad_value(value: str, prize_type: PrizeType) -> str:
    """Ensure value is correct length for prize type."""
    lengths = {"first6": 6, "three_front": 3, "three_back": 3, "two_back": 2}
    n = lengths[prize_type]
    return value.zfill(n)


def _generate_fallback_picks(
    prize_type: PrizeType,
    exclude: set[str],
    n: int,
    min_hamming: int,
    existing: list[str],
) -> list[str]:
    """Generate n fallback picks not in exclude, satisfying Hamming diversity.

    Builds the candidate list incrementally so that each new pick is checked
    against ALL previously accepted fallbacks (not just the initial existing
    list). This prevents consecutive sequential values (000000, 000001) from
    both being accepted when min_hamming >= 2.
    """
    length = {"first6": 6, "three_front": 3, "three_back": 3, "two_back": 2}[prize_type]
    total = 10 ** length

    # Local accumulator — starts with the already-selected picks so diversity
    # is checked against everything committed so far.
    accumulated: list[str] = list(existing)
    result: list[str] = []

    for i in range(total):
        if len(result) >= n:
            break
        val = str(i).zfill(length)
        if val in exclude:
            continue
        if _is_unhumanlike(val, prize_type):
            continue
        # Check Hamming diversity vs all accumulated picks (existing + already-generated fallbacks)
        if min_hamming > 0:
            ok = all(
                hamming_distance(val, e) >= min_hamming
                for e in accumulated
                if len(e) == length
            )
            if not ok:
                continue
        result.append(val)
        accumulated.append(val)
        exclude.add(val)

    return result


def select_picks(
    ensemble_picks: dict[str, list[Pick]],
    recent_winners: set[str] | None = None,
) -> dict[str, list[str]]:
    """Select final picks from ensemble output, enforcing diversity rules.

    ensemble_picks[prize_type] = list of Pick objects sorted by confidence desc.
    recent_winners: optional set of recent first_prize 6-digit values that
        should be rejected (recency-bias guard). Applied to first6 only —
        prevents top-1 prediction from echoing the most-recent draw, which is
        a known failure mode of LSTM/Markov when the lottery is actually iid
        (P(repeat) = 1/1M but recurrent models will weight it ~1).

    Returns {prize_type: [pick_value, ...]} with exactly PICK_SPLIT[prize_type] picks.
    Enforces:
      - first6: Hamming >= 2 between any two picks + not in recent_winners
      - three_back: Hamming >= 1 (all distinct)
      - two_back: no diversity filter (100 values only)
    """
    result: dict[str, list[str]] = {}
    recent_winners = recent_winners or set()

    for prize_type_str, target_count in PICK_SPLIT.items():
        prize_type = cast(PrizeType, prize_type_str)
        picks = ensemble_picks.get(prize_type_str, [])
        length = {"first6": 6, "three_back": 3, "two_back": 2}[prize_type_str]

        # Minimum Hamming distance between any two selected picks
        min_hamming: int = {
            "first6": 2,
            "three_back": 1,
            "two_back": 0,  # no filter
        }[prize_type_str]

        selected: list[str] = []
        # Seed excluded with recent winners for first6 only
        excluded: set[str] = set(recent_winners) if prize_type_str == "first6" else set()

        # Iterate through candidates in confidence order
        for pick in picks:
            if len(selected) >= target_count:
                break
            val = _pad_value(pick.value, prize_type)
            if not val.isdigit() or len(val) != length:
                continue
            if val in excluded:
                continue
            if _is_unhumanlike(val, prize_type_str):
                continue

            # Check Hamming diversity
            if min_hamming > 0:
                diverse = all(
                    hamming_distance(val, s) >= min_hamming for s in selected
                )
                if not diverse:
                    continue

            selected.append(val)
            excluded.add(val)

        # Fill remaining with fallback if needed
        if len(selected) < target_count:
            needed = target_count - len(selected)
            logger.warning(
                "Picker: only %d/%d picks from ensemble for %s — generating fallbacks",
                len(selected),
                target_count,
                prize_type_str,
            )
            fallbacks = _generate_fallback_picks(
                prize_type,
                excluded,
                needed,
                min_hamming,
                selected,
            )
            selected.extend(fallbacks[:needed])

        # Guarantee exact count (truncate if somehow over)
        result[prize_type_str] = selected[:target_count]

        # Validate
        final = result[prize_type_str]
        assert len(final) == target_count, (
            f"Expected {target_count} picks for {prize_type_str}, got {len(final)}"
        )
        assert len(set(final)) == len(final), (
            f"Duplicate picks in {prize_type_str}: {final}"
        )
        if min_hamming > 0 and len(final) > 1:
            for i in range(len(final)):
                for j in range(i + 1, len(final)):
                    hd = hamming_distance(final[i], final[j])
                    assert hd >= min_hamming, (
                        f"Hamming violation in {prize_type_str}: {final[i]} vs {final[j]} = {hd} < {min_hamming}"
                    )

    return result


# ---------------------------------------------------------------------------
# Strategy 5/3/2 — Nash's prize-targeted ticket construction (v2.5)
# ---------------------------------------------------------------------------
#
# All 10 Pao Tang tickets are 6-digit, but each is CONSTRUCTED to target a
# specific prize tier (Pao Tang auto-checks every tier, so a ticket can still
# win others incidentally):
#   • 5 tickets — เลขท้าย 2 ตัว  : last 2 digits = top-5 two_back picks
#                                   (leading 4 = best first6 prefixes, varied)
#   • 3 tickets — เลขหน้า 3 + ท้าย 3 : first 3 = top-3 three_front,
#                                       last 3 = top-3 three_back (glued)
#   • 2 tickets — รางวัลที่ 1     : top-2 first6 picks (recency-guarded)

_LEN = {"first6": 6, "three_front": 3, "three_back": 3, "two_back": 2}


def _top_values(
    ensemble_picks: dict[str, list[Pick]],
    prize_type: str,
    n: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Top-n distinct, valid, human-OK values for a prize type (with fallback)."""
    excl = set(exclude or set())
    length = _LEN[prize_type]
    out: list[str] = []
    for pick in ensemble_picks.get(prize_type, []):
        v = _pad_value(pick.value, cast(PrizeType, prize_type))
        if not v.isdigit() or len(v) != length:
            continue
        if v in out or v in excl:
            continue
        if _is_unhumanlike(v, prize_type):
            continue
        # Short tiers (2/3-digit) are ~uniform: skip degenerate all-same-digit
        # picks (000 / 00 / 111…) — they waste coverage even if model-top.
        if length <= 3 and len(set(v)) == 1:
            continue
        out.append(v)
        if len(out) >= n:
            break
    if len(out) < n:
        out.extend(
            _generate_fallback_picks(
                cast(PrizeType, prize_type), set(out) | excl, n - len(out), 0, out
            )
        )
    return out[:n]


def select_picks_532(
    ensemble_picks: dict[str, list[Pick]],
    recent_winners: set[str] | None = None,
) -> list[dict[str, str]]:
    """Build 10 prize-targeted 6-digit tickets. Returns ordered list of
    {"value", "group", "label"} — 5 two_back, 3 front3+back3, 2 first1.
    """
    recent = set(recent_winners or set())

    first6 = _top_values(ensemble_picks, "first6", 7, exclude=recent)
    two_back = _top_values(ensemble_picks, "two_back", 5)
    front3 = _top_values(ensemble_picks, "three_front", 3)
    back3 = _top_values(ensemble_picks, "three_back", 3)

    tickets: list[dict[str, str]] = []

    # 5 × เลขท้าย 2 ตัว
    for i in range(5):
        prefix = (first6[i] if i < len(first6) else first6[0])[:4]
        tickets.append(
            {"value": prefix + two_back[i], "group": "two_back", "label": "เลขท้าย 2 ตัว"}
        )

    # 3 × เลขหน้า 3 + ท้าย 3
    for i in range(3):
        tickets.append(
            {
                "value": front3[i] + back3[i],
                "group": "front3_back3",
                "label": "เลขหน้า 3 + ท้าย 3",
            }
        )

    # 2 × รางวัลที่ 1
    for i in range(2):
        idx = 5 + i if 5 + i < len(first6) else i
        tickets.append(
            {"value": first6[idx], "group": "first1", "label": "รางวัลที่ 1"}
        )

    # Guarantee 10 distinct 6-digit values — replace collisions with fallbacks.
    seen: set[str] = set()
    for t in tickets:
        if t["value"] in seen or len(t["value"]) != 6 or not t["value"].isdigit():
            fb = _generate_fallback_picks(
                cast(PrizeType, "first6"), set(seen), 1, 0, list(seen)
            )
            t["value"] = fb[0] if fb else t["value"]
        seen.add(t["value"])

    assert len(tickets) == 10, f"expected 10 tickets, got {len(tickets)}"
    assert len({t["value"] for t in tickets}) == 10, "duplicate tickets in 5/3/2 plan"
    return tickets
