"""Pydantic models for draw data + SQLite DDL string.

SPEC §2.1 (draws.jsonl schema) and §2.2 (lab.db DDL).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Draw — canonical ground-truth record stored in draws.jsonl
# ---------------------------------------------------------------------------


class Draw(BaseModel):
    """One lottery draw. Append-only ground truth. SPEC §2.1."""

    draw_date: str = Field(
        description="ISO date string YYYY-MM-DD of the draw"
    )
    draw_id: str = Field(
        description="Unique key = draw_date (YYYY-MM-DD)"
    )
    first_prize: str = Field(
        description="6-digit winning number e.g. '123456'"
    )
    first_prize_near: list[str] = Field(
        default_factory=list,
        description="Adjacent near-miss numbers (typically draw_id ± 1)"
    )
    three_digit_front: list[str] = Field(
        default_factory=list,
        description="Three-digit front prize numbers (เลขหน้า 3 ตัว)"
    )
    three_digit_back: list[str] = Field(
        default_factory=list,
        description="Three-digit back prize numbers (เลขท้าย 3 ตัว)"
    )
    two_digit_back: str = Field(
        description="Two-digit back prize number (เลขท้าย 2 ตัว)"
    )
    bonus_prizes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Bonus/adjacent prizes keyed by tier: second, third, fourth, fifth"
    )
    source: str = Field(
        description="Primary source domain e.g. 'news.sanook.com'"
    )
    source_url: str = Field(
        description="URL of the page that yielded this record"
    )
    scraped_at: str = Field(
        description="ISO 8601 timestamp with TZ when this was fetched"
    )
    raw_html_sha256: str = Field(
        description="SHA-256 of raw HTML bytes used to parse this draw"
    )
    verified_against: list[str] = Field(
        default_factory=list,
        description="Sources that agreed on first_prize + three_digit_back + two_digit_back"
    )
    schema_version: int = Field(default=1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator("draw_date", "draw_id")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Expected YYYY-MM-DD, got: {v!r}") from exc
        return v

    @field_validator("first_prize")
    @classmethod
    def validate_first_prize(cls, v: str) -> str:
        if len(v) != 6 or not v.isdigit():
            raise ValueError(f"first_prize must be exactly 6 digits, got: {v!r}")
        return v

    @field_validator("two_digit_back")
    @classmethod
    def validate_two_digit_back(cls, v: str) -> str:
        if len(v) != 2 or not v.isdigit():
            raise ValueError(f"two_digit_back must be exactly 2 digits, got: {v!r}")
        return v

    @field_validator("three_digit_back", "three_digit_front", "first_prize_near")
    @classmethod
    def validate_digit_lists(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item.isdigit():
                raise ValueError(f"Expected digits-only string, got: {item!r}")
        return v

    def is_normal_draw_date(self) -> bool:
        """Return True if draw_date falls on 1st or 16th of month."""
        d = date.fromisoformat(self.draw_date)
        return d.day in (1, 16)


# ---------------------------------------------------------------------------
# DrawCorrection — append-only corrections referencing a draw_id
# ---------------------------------------------------------------------------


class DrawCorrection(BaseModel):
    """Correction record for draws_corrections.jsonl. SPEC §2.1 rules."""

    draw_id: str
    field_name: str
    old_value: str
    new_value: str
    reason: str
    corrected_at: str
    corrected_by: str = "manual"


# ---------------------------------------------------------------------------
# SQLite DDL (SPEC §2.2)
# ---------------------------------------------------------------------------

DDL = """
-- ทุก feature snapshot ก่อนแต่ละงวด (immutable per draw_id+feature_name)
CREATE TABLE IF NOT EXISTS features (
    draw_id        TEXT NOT NULL,
    feature_name   TEXT NOT NULL,
    feature_value  REAL,
    feature_blob   BLOB,              -- BLOB <= 4 KB enforced in store.py
    computed_at    TEXT NOT NULL,
    code_sha       TEXT NOT NULL,
    PRIMARY KEY (draw_id, feature_name)
);

-- ทุก prediction ที่ FROZEN ก่อน draw cutoff
CREATE TABLE IF NOT EXISTS predictions (
    pred_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_id            TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    prize_type         TEXT NOT NULL,
    pick_value         TEXT NOT NULL,
    pick_rank          INTEGER NOT NULL,
    confidence         REAL,
    purchased          INTEGER DEFAULT 0,
    frozen_at          TEXT NOT NULL,
    freeze_commit_sha  TEXT,
    notion_page_id     TEXT,          -- Notion page ID set after publish_prediction()
    UNIQUE (draw_id, model_id, prize_type, pick_rank),
    UNIQUE (draw_id, model_id, prize_type, pick_value)
);
-- Both UNIQUE constraints coexist intentionally. See SPEC §2.2 fix #1 note.
-- notion_page_id added in Enhancement-1 (v2.2). Migration: scripts/migrate_add_notion_page_id.py

-- post-draw settlement
CREATE TABLE IF NOT EXISTS outcomes (
    pred_id        INTEGER PRIMARY KEY,
    hit            INTEGER NOT NULL,
    payout_thb     INTEGER NOT NULL,
    cost_thb       INTEGER NOT NULL CHECK (cost_thb IN (0, 80)),
    settled_at     TEXT NOT NULL,
    FOREIGN KEY (pred_id) REFERENCES predictions(pred_id)
);

-- training/evolution runs
CREATE TABLE IF NOT EXISTS model_runs (
    run_id         TEXT PRIMARY KEY,
    model_id       TEXT NOT NULL,
    parent_ids     TEXT,
    train_start    TEXT,
    train_end      TEXT,
    hyperparams    TEXT,
    fitness        REAL,
    git_sha        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- feature lifecycle
CREATE TABLE IF NOT EXISTS feature_proposals (
    proposal_id    TEXT PRIMARY KEY,
    feature_code   TEXT NOT NULL,
    backtest_score REAL,
    status         TEXT NOT NULL,
    proposed_at    TEXT NOT NULL,
    decided_at     TEXT
);

-- AI meta-cognition (also mirrored to .md file)
CREATE TABLE IF NOT EXISTS journal (
    draw_id        TEXT PRIMARY KEY,
    summary_md     TEXT NOT NULL,
    confidence     REAL,
    surprises      TEXT,
    next_actions   TEXT,
    written_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_draw    ON predictions(draw_id);
CREATE INDEX IF NOT EXISTS idx_pred_model   ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_outcome_hit  ON outcomes(hit);
"""
