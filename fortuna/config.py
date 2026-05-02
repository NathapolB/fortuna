"""Central configuration — paths, draw schedule, payout constants.

SPEC §1 (paths), §2.5 (payouts), §9 (cron schedule / draw_cutoff).
"""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root: two levels up from this file (fortuna/config.py → fortuna/ → root)
REPO_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = RAW_DIR / "cache"
EXPORTS_DIR = DATA_DIR / "exports"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"

DRAWS_JSONL = RAW_DIR / "draws.jsonl"
CORRECTIONS_JSONL = RAW_DIR / "draws_corrections.jsonl"
DRAWS_CHECKSUM = RAW_DIR / "draws.checksum"
DISCREPANCIES_JSONL = RAW_DIR / "discrepancies.jsonl"
SCRAPE_LOG_JSONL = RAW_DIR / "scrape_log.jsonl"
KNOWN_SHIFTED_DATES = RAW_DIR / "known_shifted_dates.json"

LAB_DB = DATA_DIR / "lab.db"

NOSYNC_SENTINEL = DATA_DIR / ".nosync"

REGISTRY_JSON = MODELS_DIR / "registry.json"
GRAVEYARD_JSON = MODELS_DIR / "graveyard.json"
BREEDING_LOG_JSONL = MODELS_DIR / "breeding_log.jsonl"

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

BKK = ZoneInfo("Asia/Bangkok")

# Draw cutoff = 06:00 Asia/Bangkok on draw date (matches cron schedule, SPEC §9)
DRAW_CUTOFF_TIME = time(6, 0, 0)

# ---------------------------------------------------------------------------
# Payout constants (SPEC §2.5)
# ---------------------------------------------------------------------------

PAYOUTS: dict[str, int] = {
    "first6": 6_000_000,
    "three_back": 4_000,
    "two_back": 2_000,
}

TICKET_COST_THB = 80  # Pao Tang official wholesale price

# Break-even hit rates = TICKET_COST_THB / payout_thb
BREAK_EVEN: dict[str, float] = {
    pt: TICKET_COST_THB / payout for pt, payout in PAYOUTS.items()
}

# ---------------------------------------------------------------------------
# Pick split (locked 2/3/5 — SPEC §13 v2.1)
# ---------------------------------------------------------------------------

PICK_SPLIT: dict[str, int] = {
    "first6": 2,
    "three_back": 3,
    "two_back": 5,
}

TOTAL_TICKETS_PER_DRAW = sum(PICK_SPLIT.values())  # 10
COST_PER_DRAW_THB = TOTAL_TICKETS_PER_DRAW * TICKET_COST_THB  # 800
COST_PER_MONTH_THB = COST_PER_DRAW_THB * 2  # 1600 (1st + 16th)

# ---------------------------------------------------------------------------
# Prize space sizes (for lift calculation, SPEC §6.2)
# ---------------------------------------------------------------------------

PRIZE_SPACE: dict[str, int] = {
    "first6": 1_000_000,   # 000000–999999
    "three_back": 1_000,   # 000–999
    "two_back": 100,       # 00–99
}

# ---------------------------------------------------------------------------
# HTTP politeness (SPEC §3.2)
# ---------------------------------------------------------------------------

REQUEST_DELAY_SEC = 3.0       # 1 req / 3 sec
MAX_RETRIES = 3
BACKOFF_BASE = 2.0            # exponential backoff factor

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

# Override user agent via env var if set
if os.environ.get("GLO_USER_AGENT"):
    USER_AGENTS = [os.environ["GLO_USER_AGENT"]] + USER_AGENTS

# ---------------------------------------------------------------------------
# iCloud guard
# ---------------------------------------------------------------------------

ICLOUD_MARKER = "com~apple~CloudDocs"


def check_not_icloud() -> None:
    """Abort if repo appears to be inside iCloud Drive. SPEC §1 / §11."""
    repo_str = str(REPO_ROOT)
    if ICLOUD_MARKER in repo_str:
        raise RuntimeError(
            f"ABORT: repo path contains iCloud marker '{ICLOUD_MARKER}'.\n"
            f"  Path: {repo_str}\n"
            "  Move the repo to ~/projects/fortuna/ (outside iCloud Drive).\n"
            "  See SPEC §1 and §11."
        )
    if not NOSYNC_SENTINEL.exists():
        # Sentinel missing — create it defensively; warn but don't abort
        NOSYNC_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        NOSYNC_SENTINEL.touch()
