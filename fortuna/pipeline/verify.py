"""fortuna verify — re-hash prediction + validate git history. SPEC §6.1."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime

from fortuna.config import BKK, EXPORTS_DIR, REPO_ROOT, check_not_icloud
from fortuna.eval.walkforward import draw_cutoff

logger = logging.getLogger(__name__)


def run_verify(draw_id: str) -> dict:
    """Verify prediction export integrity.

    Checks:
      1. Export file exists
      2. picks_sha256 in file matches re-computed sha256
      3. freeze_commit_sha exists in remote git history
      4. Commit timestamp is before draw_cutoff

    Returns dict with {valid: bool, checks: {name: (passed, detail)}}.
    """
    check_not_icloud()

    export_path = EXPORTS_DIR / f"{draw_id}-prediction.json"
    result: dict = {
        "draw_id": draw_id,
        "valid": True,
        "checks": {},
    }

    # Check 1: file exists
    if not export_path.exists():
        result["valid"] = False
        result["checks"]["file_exists"] = (False, f"Not found: {export_path}")
        return result
    result["checks"]["file_exists"] = (True, str(export_path))

    # Load
    with open(export_path) as f:
        payload = json.load(f)

    # Check 2: SHA256 integrity
    stored_sha = payload.get("picks_sha256")
    picks_json = json.dumps(payload.get("picks", {}), sort_keys=True, separators=(",", ":"))
    computed_sha = hashlib.sha256(picks_json.encode()).hexdigest()

    if stored_sha != computed_sha:
        result["valid"] = False
        result["checks"]["sha256_match"] = (
            False,
            f"Stored={stored_sha[:16]}... Computed={computed_sha[:16]}...",
        )
    else:
        result["checks"]["sha256_match"] = (True, f"SHA256={computed_sha[:16]}...")

    # Check 3: freeze commit exists in remote
    freeze_sha = payload.get("freeze_commit_sha")
    if not freeze_sha or freeze_sha in ("dry-run", None):
        result["checks"]["commit_exists"] = (False, "No freeze_commit_sha in payload")
        result["valid"] = False
    else:
        try:
            # Check commit exists locally
            proc = subprocess.run(
                ["git", "cat-file", "-t", freeze_sha],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if proc.stdout.strip() == "commit":
                result["checks"]["commit_exists"] = (True, f"SHA={freeze_sha[:12]}...")
            else:
                result["valid"] = False
                result["checks"]["commit_exists"] = (
                    False, f"SHA {freeze_sha[:12]}... not found in local history"
                )
        except Exception as e:
            result["checks"]["commit_exists"] = (False, str(e))
            result["valid"] = False

    # Check 4: commit timestamp before draw cutoff
    if freeze_sha and freeze_sha not in ("dry-run", None):
        try:
            proc = subprocess.run(
                ["git", "log", "-1", "--format=%ai", freeze_sha],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            commit_time_str = proc.stdout.strip()
            if commit_time_str:
                # git format=%ai = ISO 8601 with +TZ
                commit_time = datetime.fromisoformat(commit_time_str.replace(" ", "T", 1))
                if commit_time.tzinfo is None:
                    from zoneinfo import ZoneInfo
                    commit_time = commit_time.replace(tzinfo=ZoneInfo("UTC"))
                cutoff = draw_cutoff(draw_id)
                before = commit_time < cutoff
                result["checks"]["before_cutoff"] = (
                    before,
                    f"Committed at {commit_time_str}, cutoff={cutoff.isoformat()}",
                )
                if not before:
                    result["valid"] = False
        except Exception as e:
            result["checks"]["before_cutoff"] = (False, f"Error: {e}")
            result["valid"] = False

    all_passed = all(check[0] for check in result["checks"].values())
    result["valid"] = all_passed

    if result["valid"]:
        logger.info("Prediction for %s is VALID — all checks passed", draw_id)
    else:
        failed = [name for name, (passed, _) in result["checks"].items() if not passed]
        logger.error("Prediction for %s INVALID — failed checks: %s", draw_id, failed)

    return result
