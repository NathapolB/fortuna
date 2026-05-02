"""CLI entrypoint — `python -m fortuna <cmd>`. SPEC §10.

Phase 1 implements: scrape (backfill mode only).
Phase 2+ implements: features, predict, settle, journal, evolve, tournament,
    breed, propose-features, verify, status, rollback.
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
        # Delegate to backfill script logic
        from scripts.backfill import run_backfill
        return run_backfill()
    else:
        logger.info("Scraping target date: %s", args.target)
        # Phase 2
        logger.warning("Per-date scrape not yet implemented (Phase 2)")
        return 1


def _cmd_stub(cmd_name: str) -> int:
    logger.warning("Command '%s' is not yet implemented — Phase 2+", cmd_name)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fortuna", description="Project Fortuna CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Fetch lottery results")
    p_scrape.add_argument("--catch-up", action="store_true", help="Backfill missing draws")
    p_scrape.add_argument("--target", metavar="YYYY-MM-DD", help="Specific draw date")

    # stubs for Phase 2+ commands
    for cmd in [
        "features", "predict", "settle", "journal", "evolve",
        "tournament", "breed", "propose-features", "verify", "status", "rollback",
    ]:
        p = sub.add_parser(cmd)
        p.add_argument("--target", metavar="YYYY-MM-DD", default=None)
        p.add_argument("--freeze", action="store_true", default=False)
        p.add_argument("--month", metavar="YYYY-MM", default=None)
        p.add_argument("--date", metavar="YYYY-MM-DD", default=None)
        p.add_argument("--post-draw", action="store_true", default=False)
        p.add_argument("--to-sha", metavar="SHA", default=None)

    args = parser.parse_args(argv)

    if args.command == "scrape":
        return _cmd_scrape(args)
    else:
        return _cmd_stub(args.command)


if __name__ == "__main__":
    sys.exit(main())
