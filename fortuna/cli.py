"""CLI entrypoint — `python -m fortuna <cmd>`. SPEC §10.

Phase 1: scrape (backfill mode only).
Phase 2: features, predict, settle, journal, verify, train + stubs for evolve/tournament/etc.

New in v2.2 (Enhancement-2):
  - run-scheduled: runs on day 2 and 17 of each month (07:00 BKK). Determines
    previous draw to settle and next draw to predict automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("fortuna.cli")


def _cmd_scrape(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    if args.catch_up or args.target is None:
        logger.info("Running backfill scrape (catch-up mode)")
        from scripts.backfill import run_backfill
        return run_backfill()
    else:
        logger.info("Scraping target date: %s", args.target)
        logger.warning("Per-date scrape not yet implemented (Phase 2)")
        return 1


def _cmd_predict(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    date_str = args.date
    if date_str is None:
        logger.error("--date YYYY-MM-DD is required for predict")
        return 1

    dry_run = getattr(args, "dry_run", False)
    allow_leak = getattr(args, "allow_leak", False)
    logger.info(
        "Running prediction for date: %s (dry_run=%s, allow_leak=%s)",
        date_str, dry_run, allow_leak,
    )

    from fortuna.pipeline.predict import run_predict
    try:
        payload = run_predict(target_date=date_str, dry_run=dry_run, allow_leak=allow_leak)
        print(f"\nPrediction frozen: {date_str}")
        for prize_type, picks in payload.get("picks", {}).items():
            values = [p["value"] for p in picks]
            print(f"  {prize_type}: {values}")
        sha = payload.get("picks_sha256", "N/A")
        print(f"\nSHA256: {sha}")
        commit = payload.get("freeze_commit_sha", "N/A")
        print(f"Freeze commit: {commit}")
        notion_url = payload.get("notion_page_url")
        if notion_url:
            print(f"Notion page: {notion_url}")
        return 0
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        return 1


def _cmd_settle(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    date_str = args.date
    if date_str is None:
        logger.error("--date YYYY-MM-DD is required for settle")
        return 1

    logger.info("Running settlement for date: %s", date_str)
    from fortuna.pipeline.settle import run_settle
    try:
        summary = run_settle(draw_id=date_str)
        print(f"\nSettlement for {date_str}:")
        print(f"  Cost: {summary['total_cost_thb']} THB")
        print(f"  Payout: {summary['total_payout_thb']} THB")
        print(f"  Net P&L: {summary['net_pnl_thb']:+d} THB")
        return 0
    except Exception as e:
        logger.error("Settlement failed: %s", e, exc_info=True)
        return 1


def _cmd_verify(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    date_str = args.date
    if date_str is None:
        logger.error("--date YYYY-MM-DD is required for verify")
        return 1

    logger.info("Verifying prediction for date: %s", date_str)
    from fortuna.pipeline.verify import run_verify
    try:
        result = run_verify(draw_id=date_str)
        print(f"\nVerification for {date_str}: {'VALID' if result['valid'] else 'INVALID'}")
        for check_name, (passed, detail) in result.get("checks", {}).items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check_name}: {detail}")
        return 0 if result["valid"] else 1
    except Exception as e:
        logger.error("Verify failed: %s", e, exc_info=True)
        return 1


def _cmd_train(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    start = getattr(args, "start", None)
    end = getattr(args, "end", None)

    logger.info("Training models (start=%s, end=%s)", start, end)
    from fortuna.pipeline.train import run_train
    try:
        summary = run_train(start_date=start, end_date=end)
        print(f"\nTraining complete:")
        print(f"  Models trained: {summary['n_models_trained']}")
        print(f"  Training draws: {summary['n_draws']}")
        print(f"  Range: {summary['train_range']}")
        print("\nModel scores:")
        for m in summary["models"]:
            brier = m.get("holdout_brier")
            hr = m.get("holdout_hit_rate")
            brier_str = f"{brier:.4f}" if brier is not None else "N/A"
            hr_str = f"{hr:.4f}" if hr is not None else "N/A"
            print(f"  {m['model_id']}: brier={brier_str}, hit_rate={hr_str}")
        return 0
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1


def _resolve_scheduled_dates(today: date) -> tuple[str, str]:
    """Determine which draw to settle and which to predict for a given run date.

    Schedule (SPEC §Enhancement-2 / v2.2):
      - Day  2: settle draw from the 1st of this month, predict for the 16th of this month
      - Day 17: settle draw from the 16th of this month, predict for the 1st of next month

    Returns (settle_date, predict_date) as YYYY-MM-DD strings.

    Raises ValueError if called on a day that is not 2 or 17.
    """
    day = today.day
    year = today.year
    month = today.month

    if day == 2:
        # Settle the 1st of this month; predict for the 16th of this month
        settle = date(year, month, 1)
        predict = date(year, month, 16)
    elif day == 17:
        # Settle the 16th of this month; predict for the 1st of next month
        settle = date(year, month, 16)
        # Next month
        if month == 12:
            predict = date(year + 1, 1, 1)
        else:
            predict = date(year, month + 1, 1)
    else:
        raise ValueError(
            f"run-scheduled is designed for day 2 or 17 only. Today is day {day}. "
            "Use --date with predict/settle commands for manual runs."
        )

    return settle.isoformat(), predict.isoformat()


def _cmd_run_scheduled(args: argparse.Namespace) -> int:
    """Automated cron entrypoint — runs on day 2 and 17 of each month.

    Steps:
      1. Determine settle_date and predict_date from today's date
      2. Run scrape (catch-up)
      3. Run settle for previous draw
      4. Run train
      5. Run predict for upcoming draw

    SPEC §Enhancement-2 cron line:
      0 7 2,17 * * cd ~/projects/fortuna && .venv/bin/python -m fortuna run-scheduled >> logs/cron.log 2>&1
    """
    from fortuna.config import check_not_icloud
    check_not_icloud()

    from zoneinfo import ZoneInfo
    from datetime import datetime as dt

    bkk = ZoneInfo("Asia/Bangkok")
    today = dt.now(bkk).date()

    # Allow override via --date for testing/manual runs
    override_date = getattr(args, "date", None)
    if override_date:
        try:
            today = date.fromisoformat(override_date)
            logger.info("run-scheduled: using override date %s", today)
        except ValueError:
            logger.error("Invalid --date format: %s (expected YYYY-MM-DD)", override_date)
            return 1

    try:
        settle_date, predict_date = _resolve_scheduled_dates(today)
    except ValueError as e:
        logger.error("run-scheduled: %s", e)
        return 1

    logger.info(
        "run-scheduled: today=%s | settle=%s | predict=%s",
        today, settle_date, predict_date,
    )

    exit_code = 0

    # Step 1: Scrape catch-up
    logger.info("--- Step 1: Scrape catch-up ---")
    try:
        from scripts.backfill import run_backfill
        run_backfill()
    except Exception as e:
        logger.warning("Scrape failed (non-blocking): %s", e)

    # Step 2: Settle previous draw
    logger.info("--- Step 2: Settle draw %s ---", settle_date)
    try:
        from fortuna.pipeline.settle import run_settle
        summary = run_settle(draw_id=settle_date)
        print(f"\nSettlement for {settle_date}:")
        print(f"  Net P&L: {summary['net_pnl_thb']:+d} THB")
    except Exception as e:
        logger.error("Settlement failed for %s: %s", settle_date, e)
        exit_code = 1

    # Step 3: Train
    logger.info("--- Step 3: Train models ---")
    try:
        from fortuna.pipeline.train import run_train
        train_summary = run_train()
        logger.info("Train complete: %d models", train_summary.get("n_models_trained", 0))
    except Exception as e:
        logger.warning("Training failed (non-blocking, predict may still run): %s", e)

    # Step 4: Predict for upcoming draw
    logger.info("--- Step 4: Predict for draw %s ---", predict_date)
    try:
        from fortuna.pipeline.predict import run_predict
        payload = run_predict(target_date=predict_date, dry_run=False, allow_leak=False)
        print(f"\nPrediction frozen: {predict_date}")
        for prize_type, picks in payload.get("picks", {}).items():
            values = [p["value"] for p in picks]
            print(f"  {prize_type}: {values}")
        notion_url = payload.get("notion_page_url")
        if notion_url:
            print(f"Notion page: {notion_url}")
    except Exception as e:
        logger.error("Prediction failed for %s: %s", predict_date, e, exc_info=True)
        exit_code = 1

    return exit_code


def _cmd_stub(cmd_name: str) -> int:
    logger.warning("Command '%s' is not yet implemented", cmd_name)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fortuna", description="Project Fortuna CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Fetch lottery results")
    p_scrape.add_argument("--catch-up", action="store_true", help="Backfill missing draws")
    p_scrape.add_argument("--target", metavar="YYYY-MM-DD", help="Specific draw date")

    # predict
    p_predict = sub.add_parser("predict", help="Produce frozen prediction for a draw date")
    p_predict.add_argument("--date", metavar="YYYY-MM-DD", required=True)
    p_predict.add_argument("--dry-run", action="store_true", default=False,
                           help="Skip git push (for testing)")
    p_predict.add_argument("--allow-leak", action="store_true", default=False,
                           help="Downgrade leakage guard from ValueError to warning (testing only)")

    # settle
    p_settle = sub.add_parser("settle", help="Settle predictions after draw results")
    p_settle.add_argument("--date", metavar="YYYY-MM-DD", required=True)

    # verify
    p_verify = sub.add_parser("verify", help="Verify prediction integrity")
    p_verify.add_argument("--date", metavar="YYYY-MM-DD", required=True)

    # train
    p_train = sub.add_parser("train", help="Train all models and update registry")
    p_train.add_argument("--start", metavar="YYYY-MM-DD", default=None)
    p_train.add_argument("--end", metavar="YYYY-MM-DD", default=None)

    # run-scheduled (Enhancement-2): cron entrypoint for day 2 and 17
    p_scheduled = sub.add_parser(
        "run-scheduled",
        help="Cron entrypoint: settle previous draw + predict next draw (run on day 2 and 17)",
    )
    p_scheduled.add_argument(
        "--date", metavar="YYYY-MM-DD", default=None,
        help="Override today's date for testing (default: system date in Asia/Bangkok)",
    )

    # stubs for Phase 3+ commands
    for cmd in [
        "features", "journal", "evolve",
        "tournament", "breed", "propose-features", "status", "rollback",
    ]:
        p = sub.add_parser(cmd)
        p.add_argument("--target", metavar="YYYY-MM-DD", default=None)
        p.add_argument("--freeze", action="store_true", default=False)
        p.add_argument("--month", metavar="YYYY-MM", default=None)
        p.add_argument("--date", metavar="YYYY-MM-DD", default=None)
        p.add_argument("--post-draw", action="store_true", default=False)
        p.add_argument("--to-sha", metavar="SHA", default=None)

    args = parser.parse_args(argv)

    dispatch = {
        "scrape": _cmd_scrape,
        "predict": _cmd_predict,
        "settle": _cmd_settle,
        "verify": _cmd_verify,
        "train": _cmd_train,
        "run-scheduled": _cmd_run_scheduled,
    }

    if args.command in dispatch:
        return dispatch[args.command](args)
    else:
        return _cmd_stub(args.command)


if __name__ == "__main__":
    sys.exit(main())
