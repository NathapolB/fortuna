"""CLI entrypoint — `python -m fortuna <cmd>`. SPEC §10.

Phase 1: scrape (backfill mode only).
Phase 2: features, predict, settle, journal, verify, train + stubs for evolve/tournament/etc.
"""

from __future__ import annotations

import argparse
import logging
import sys

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

    date = args.date
    if date is None:
        logger.error("--date YYYY-MM-DD is required for predict")
        return 1

    dry_run = getattr(args, "dry_run", False)
    logger.info("Running prediction for date: %s (dry_run=%s)", date, dry_run)

    from fortuna.pipeline.predict import run_predict
    try:
        payload = run_predict(target_date=date, dry_run=dry_run)
        print(f"\nPrediction frozen: {date}")
        for prize_type, picks in payload.get("picks", {}).items():
            values = [p["value"] for p in picks]
            print(f"  {prize_type}: {values}")
        sha = payload.get("picks_sha256", "N/A")
        print(f"\nSHA256: {sha}")
        commit = payload.get("freeze_commit_sha", "N/A")
        print(f"Freeze commit: {commit}")
        return 0
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        return 1


def _cmd_settle(args: argparse.Namespace) -> int:
    from fortuna.config import check_not_icloud
    check_not_icloud()

    date = args.date
    if date is None:
        logger.error("--date YYYY-MM-DD is required for settle")
        return 1

    logger.info("Running settlement for date: %s", date)
    from fortuna.pipeline.settle import run_settle
    try:
        summary = run_settle(draw_id=date)
        print(f"\nSettlement for {date}:")
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

    date = args.date
    if date is None:
        logger.error("--date YYYY-MM-DD is required for verify")
        return 1

    logger.info("Verifying prediction for date: %s", date)
    from fortuna.pipeline.verify import run_verify
    try:
        result = run_verify(draw_id=date)
        print(f"\nVerification for {date}: {'VALID' if result['valid'] else 'INVALID'}")
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
            print(
                f"  {m['model_id']}: brier={brier:.4f if brier else 'N/A'}, "
                f"hit_rate={hr:.4f if hr else 'N/A'}"
            )
        return 0
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1


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
    }

    if args.command in dispatch:
        return dispatch[args.command](args)
    else:
        return _cmd_stub(args.command)


if __name__ == "__main__":
    sys.exit(main())
