"""Tests for fortuna/pipeline/notion_publisher.py — SPEC §Enhancement-1 v2.3.

Verifies:
  - Page payload uses ONLY the 3 real Claude Scheduler DB properties
    (Schedule Name, Run Date, Archive) — no legacy properties
  - Page body contains all 3 prize sections + verifiable timestamp + honest framing
  - settle_prediction_page() appends via PATCH without creating a new page
  - Graceful degradation when NOTION_TOKEN is missing (returns None, logs warning)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from fortuna.pipeline.notion_publisher import (
    _build_page_body,
    _draw_date_thai,
    publish_prediction,
    settle_prediction_page,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DRAW_DATE = "2026-05-16"
SAMPLE_SHA = "abc1234def5678901234567890"  # >7 chars


@pytest.fixture()
def sample_prediction() -> dict:
    return {
        "target_draw_id": SAMPLE_DRAW_DATE,
        "frozen_at": "2026-05-02T07:00:00+07:00",
        "freeze_commit_sha": SAMPLE_SHA,
        "model_versions": {
            "frequency-bayes-v1": "aaabbb1",
            "markov-v1": "ccc2222",
        },
        "picks": {
            "first6": [
                {"value": "309612", "rank": 1, "confidence": 0.0000045},
                {"value": "718234", "rank": 2, "confidence": 0.0000041},
            ],
            "three_back": [
                {"value": "612", "rank": 1},
                {"value": "234", "rank": 2},
                {"value": "891", "rank": 3},
            ],
            "two_back": [
                {"value": "12", "rank": 1},
                {"value": "34", "rank": 2},
                {"value": "56", "rank": 3},
                {"value": "78", "rank": 4},
                {"value": "90", "rank": 5},
            ],
        },
        "total_cost_thb": 800,
    }


@pytest.fixture()
def sample_settlement() -> dict:
    return {
        "draw_id": SAMPLE_DRAW_DATE,
        "net_pnl_thb": -800,
        "hit_count": 0,
        "brier_lift": -0.0012,
        "settled_at": "2026-05-16T17:00:00+07:00",
        "actual_results": {
            "first_prize": "123456",
            "three_back": ["789", "012"],
            "two_back": "34",
        },
        "tickets": [
            {"prize_type": "first6", "pick": "309612", "hit": False, "payout_thb": 0},
            {"prize_type": "first6", "pick": "718234", "hit": False, "payout_thb": 0},
            {"prize_type": "three_back", "pick": "612", "hit": False, "payout_thb": 0},
            {"prize_type": "three_back", "pick": "234", "hit": False, "payout_thb": 0},
            {"prize_type": "three_back", "pick": "891", "hit": False, "payout_thb": 0},
            {"prize_type": "two_back", "pick": "12", "hit": False, "payout_thb": 0},
            {"prize_type": "two_back", "pick": "34", "hit": True, "payout_thb": 2000},
            {"prize_type": "two_back", "pick": "56", "hit": False, "payout_thb": 0},
            {"prize_type": "two_back", "pick": "78", "hit": False, "payout_thb": 0},
            {"prize_type": "two_back", "pick": "90", "hit": False, "payout_thb": 0},
        ],
    }


def _mock_ok_response(page_id: str = "abc123") -> MagicMock:
    """Return a mock requests.Response that looks like a Notion 200."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "id": page_id,
        "url": f"https://www.notion.so/{page_id}",
    }
    r.text = "{}"
    return r


def _capture_payload(mock_call) -> dict:
    """Extract the json= keyword arg from a mock call."""
    if mock_call.call_args.kwargs.get("json") is not None:
        return mock_call.call_args.kwargs["json"]
    # Positional args: requests.post(url, **kwargs) — json is always a kwarg
    return mock_call.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# Thai date helper
# ---------------------------------------------------------------------------


def test_draw_date_thai_conversion():
    assert _draw_date_thai("2026-05-16") == "16 พ.ค. 2569"
    assert _draw_date_thai("2026-01-01") == "1 ม.ค. 2569"
    assert _draw_date_thai("2026-12-31") == "31 ธ.ค. 2569"


def test_draw_date_thai_invalid_graceful():
    # Bad input should not raise — returns original string
    result = _draw_date_thai("not-a-date")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Properties: only 3 allowed
# ---------------------------------------------------------------------------


def test_publish_prediction_uses_only_real_properties(sample_prediction):
    """Payload must contain ONLY Schedule Name, Run Date, Archive properties."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", return_value=_mock_ok_response()) as mock_post,
    ):
        result = publish_prediction(sample_prediction)

    assert result is not None
    payload = _capture_payload(mock_post)
    props = payload["properties"]

    # Must have exactly the 3 real properties
    assert set(props.keys()) == {"Schedule Name", "Run Date", "Archive"}


def test_publish_prediction_no_forbidden_properties(sample_prediction):
    """Payload must NOT contain legacy properties that were never in this DB."""
    forbidden = {"Draw Date", "Total Cost", "Status", "Freeze Commit", "Hit Count", "Net P&L", "Name"}
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", return_value=_mock_ok_response()) as mock_post,
    ):
        publish_prediction(sample_prediction)
        payload = _capture_payload(mock_post)
        props = set(payload["properties"].keys())

    assert props.isdisjoint(forbidden), f"Forbidden properties found: {props & forbidden}"


def test_publish_prediction_schedule_name_pattern(sample_prediction):
    """Schedule Name must match pattern '🎰 Fortuna — งวด DD MMM BE'."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", return_value=_mock_ok_response()) as mock_post,
    ):
        publish_prediction(sample_prediction)
        payload = _capture_payload(mock_post)

    title_content = payload["properties"]["Schedule Name"]["title"][0]["text"]["content"]
    assert title_content.startswith("🎰 Fortuna — งวด")
    assert "2569" in title_content  # BE year for 2026


def test_publish_prediction_run_date_iso(sample_prediction):
    """Run Date must be ISO YYYY-MM-DD."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", return_value=_mock_ok_response()) as mock_post,
    ):
        publish_prediction(sample_prediction)
        payload = _capture_payload(mock_post)

    assert payload["properties"]["Run Date"]["date"]["start"] == SAMPLE_DRAW_DATE


def test_publish_prediction_archive_default_false(sample_prediction):
    """Archive must default to False."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", return_value=_mock_ok_response()) as mock_post,
    ):
        publish_prediction(sample_prediction)
        payload = _capture_payload(mock_post)

    assert payload["properties"]["Archive"]["checkbox"] is False


# ---------------------------------------------------------------------------
# Page body — all sections present
# ---------------------------------------------------------------------------


def test_body_contains_three_prize_sections(sample_prediction):
    """Body must contain headings for all 3 prize types."""
    blocks = _build_page_body(sample_prediction)
    heading_texts = [
        b[b["type"]]["rich_text"][0]["text"]["content"]
        for b in blocks
        if b["type"] in ("heading_2", "heading_3")
    ]
    assert any("รางวัลที่ 1" in t for t in heading_texts), "Missing first6 heading"
    assert any("3 ตัว" in t for t in heading_texts), "Missing three_back heading"
    assert any("2 ตัว" in t for t in heading_texts), "Missing two_back heading"


def test_body_contains_top_callout(sample_prediction):
    """Body must start with a yellow_background callout (AI's Picks header)."""
    blocks = _build_page_body(sample_prediction)
    first_callout = next((b for b in blocks if b["type"] == "callout"), None)
    assert first_callout is not None
    assert first_callout["callout"]["color"] == "yellow_background"
    assert first_callout["callout"]["icon"]["emoji"] == "🎰"


def test_body_contains_honest_framing_callout(sample_prediction):
    """Body must end with a gray_background honest framing callout."""
    blocks = _build_page_body(sample_prediction)
    callouts = [b for b in blocks if b["type"] == "callout"]
    last_callout = callouts[-1]
    assert last_callout["callout"]["color"] == "gray_background"
    assert last_callout["callout"]["icon"]["emoji"] == "⚠️"
    text = last_callout["callout"]["rich_text"][0]["text"]["content"]
    assert "Honest framing" in text
    assert "not financial advice" in text


def test_body_contains_verifiable_timestamp_section(sample_prediction):
    """Body must have a Verifiable Timestamp heading and commit link."""
    blocks = _build_page_body(sample_prediction)
    heading_texts = [
        b[b["type"]]["rich_text"][0]["text"]["content"]
        for b in blocks
        if b["type"] == "heading_3"
    ]
    assert any("Verifiable Timestamp" in t for t in heading_texts)

    # Find the paragraph that follows the timestamp heading — should contain short SHA
    short_sha = SAMPLE_SHA[:7]
    found_sha = False
    for b in blocks:
        if b["type"] == "paragraph":
            for segment in b["paragraph"]["rich_text"]:
                if short_sha in segment["text"]["content"]:
                    found_sha = True
    assert found_sha, f"Short SHA {short_sha!r} not found in body paragraphs"


def test_body_contains_model_contributions_section(sample_prediction):
    """Body must have a Model Contributions heading with one bullet per model."""
    blocks = _build_page_body(sample_prediction)
    heading_texts = [
        b[b["type"]]["rich_text"][0]["text"]["content"]
        for b in blocks
        if b["type"] == "heading_3"
    ]
    assert any("Model Contributions" in t for t in heading_texts)

    bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
    model_names = list(sample_prediction["model_versions"].keys())
    for name in model_names:
        assert any(
            name in b["bulleted_list_item"]["rich_text"][0]["text"]["content"]
            for b in bullets
        ), f"Model {name!r} not found in bullets"


def test_body_first6_picks_are_code_styled(sample_prediction):
    """first6 numbered items should be code-styled with confidence text."""
    blocks = _build_page_body(sample_prediction)
    numbered = [b for b in blocks if b["type"] == "numbered_list_item"]
    assert len(numbered) >= 2
    # First item: value segment must be code-annotated with value "309612"
    value_seg = numbered[0]["numbered_list_item"]["rich_text"][0]
    assert value_seg["annotations"]["code"] is True
    assert value_seg["text"]["content"] == "309612"


def test_body_two_back_separator_style(sample_prediction):
    """two_back picks must be in a single paragraph with ' · ' separators."""
    blocks = _build_page_body(sample_prediction)
    separator_found = False
    for b in blocks:
        if b["type"] == "paragraph":
            for seg in b["paragraph"]["rich_text"]:
                if " · " in seg["text"]["content"]:
                    separator_found = True
    assert separator_found, "No ' · ' separator found in two_back paragraph"


def test_body_contains_divider(sample_prediction):
    """Body must contain at least one divider."""
    blocks = _build_page_body(sample_prediction)
    assert any(b["type"] == "divider" for b in blocks)


# ---------------------------------------------------------------------------
# settle_prediction_page — appends without creating new page
# ---------------------------------------------------------------------------


def test_settle_does_not_call_post(sample_settlement):
    """settle_prediction_page must never call POST /pages."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.post") as mock_post,
        patch("requests.patch", return_value=_mock_ok_response("page123")) as mock_patch,
    ):
        result = settle_prediction_page("page123", sample_settlement)

    assert result is True
    mock_post.assert_not_called()
    mock_patch.assert_called_once()


def test_settle_patches_blocks_endpoint(sample_settlement):
    """settle_prediction_page must call PATCH /v1/blocks/{page_id}/children."""
    page_id = "abc-def-123"
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.patch", return_value=_mock_ok_response()) as mock_patch,
    ):
        settle_prediction_page(page_id, sample_settlement)

    called_url = mock_patch.call_args.args[0] if mock_patch.call_args.args else mock_patch.call_args.kwargs["url"]
    assert f"/blocks/{page_id}/children" in called_url


def test_settle_appended_blocks_contain_results_heading(sample_settlement):
    """Appended blocks must contain a ผลรางวัล heading."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.patch", return_value=_mock_ok_response()) as mock_patch,
    ):
        settle_prediction_page("page123", sample_settlement)

    payload = _capture_payload(mock_patch)
    children = payload["children"]
    heading_texts = [
        b[b["type"]]["rich_text"][0]["text"]["content"]
        for b in children
        if b["type"] in ("heading_2", "heading_3")
    ]
    assert any("ผลรางวัล" in t for t in heading_texts)


def test_settle_appended_blocks_contain_pnl_callout(sample_settlement):
    """Appended blocks must contain a P&L summary callout."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.patch", return_value=_mock_ok_response()) as mock_patch,
    ):
        settle_prediction_page("page123", sample_settlement)

    payload = _capture_payload(mock_patch)
    children = payload["children"]
    callouts = [b for b in children if b["type"] == "callout"]
    assert callouts, "No callout found in settlement blocks"
    pnl_text = callouts[-1]["callout"]["rich_text"][0]["text"]["content"]
    assert "Net P&L" in pnl_text
    assert "Hit" in pnl_text
    assert "Brier" in pnl_text


def test_settle_pnl_positive_green_negative_red():
    """Callout colour must be green for profit, red for loss."""
    base = {
        "draw_id": "2026-05-16",
        "settled_at": "2026-05-16T17:00:00+07:00",
        "brier_lift": 0.0,
        "hit_count": 0,
        "actual_results": {},
        "tickets": [],
    }

    for pnl, expected_color in [(1000, "green_background"), (-800, "red_background")]:
        summary = {**base, "net_pnl_thb": pnl}
        with (
            patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
            patch("requests.patch", return_value=_mock_ok_response()) as mock_patch,
        ):
            settle_prediction_page("page123", summary)

        payload = _capture_payload(mock_patch)
        children = payload["children"]
        callouts = [b for b in children if b["type"] == "callout"]
        assert callouts[-1]["callout"]["color"] == expected_color, (
            f"Expected {expected_color} for pnl={pnl}"
        )


def test_settle_includes_table_for_tickets(sample_settlement):
    """Settlement blocks must include a table block when tickets are provided."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.patch", return_value=_mock_ok_response()) as mock_patch,
    ):
        settle_prediction_page("page123", sample_settlement)

    payload = _capture_payload(mock_patch)
    children = payload["children"]
    tables = [b for b in children if b["type"] == "table"]
    assert tables, "No table block found in settlement blocks"
    table = tables[0]
    assert table["table"]["has_column_header"] is True
    assert table["table"]["table_width"] == 4


# ---------------------------------------------------------------------------
# Graceful degradation — missing NOTION_TOKEN
# ---------------------------------------------------------------------------


def test_publish_returns_none_when_token_missing(sample_prediction, caplog):
    """publish_prediction must return None and log a warning when token is absent."""
    env = {"NOTION_TOKEN": "", "NOTION_FORTUNA_DB_ID": "testdb"}
    with (
        patch.dict("os.environ", env, clear=False),
        patch("requests.post") as mock_post,
        caplog.at_level(logging.WARNING, logger="fortuna.pipeline.notion_publisher"),
    ):
        result = publish_prediction(sample_prediction)

    assert result is None
    mock_post.assert_not_called()
    assert any("NOTION_TOKEN" in r.message for r in caplog.records)


def test_settle_returns_false_when_token_missing(sample_settlement, caplog):
    """settle_prediction_page must return False and log a warning when token absent."""
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": ""}, clear=False),
        patch("requests.patch") as mock_patch,
        caplog.at_level(logging.WARNING, logger="fortuna.pipeline.notion_publisher"),
    ):
        result = settle_prediction_page("page123", sample_settlement)

    assert result is False
    mock_patch.assert_not_called()
    assert any("NOTION_TOKEN" in r.message for r in caplog.records)


def test_publish_returns_none_when_db_id_missing(sample_prediction, caplog):
    """publish_prediction must return None when DB ID env var is absent."""
    env = {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": ""}
    with (
        patch.dict("os.environ", env, clear=False),
        patch("requests.post") as mock_post,
        caplog.at_level(logging.WARNING, logger="fortuna.pipeline.notion_publisher"),
    ):
        result = publish_prediction(sample_prediction)

    assert result is None
    mock_post.assert_not_called()
    assert any("NOTION_FORTUNA_DB_ID" in r.message for r in caplog.records)


def test_publish_no_exception_on_requests_error(sample_prediction):
    """publish_prediction must never raise even if requests raises."""
    import requests as req

    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test", "NOTION_FORTUNA_DB_ID": "testdb"}),
        patch("requests.post", side_effect=req.exceptions.ConnectionError("timeout")),
    ):
        result = publish_prediction(sample_prediction)

    assert result is None


def test_settle_no_exception_on_requests_error(sample_settlement):
    """settle_prediction_page must never raise even if requests raises."""
    import requests as req

    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "secret_test"}),
        patch("requests.patch", side_effect=req.exceptions.ConnectionError("timeout")),
    ):
        result = settle_prediction_page("page123", sample_settlement)

    assert result is False
