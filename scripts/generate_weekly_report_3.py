#!/usr/bin/env python3
"""Entry point to generate the Weekly Report 3 output."""

import argparse
import logging
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from error_log_monitor.weekly_report_3 import WeeklyReportGenerator3  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate weekly report 3")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today (UTC).")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown args: %s", " ".join(unknown))
    return args


def main():
    args = parse_args(sys.argv[1:])

    end_date = None
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)

    generator = WeeklyReportGenerator3()
    report = generator.generate_weekly_report(end_date=end_date)

    logger.info("Combined weekly report saved to %s", report.get("combined_excel_path"))
    for site, payload in report.get("site_reports", {}).items():
        logger.info(
            "Site %s: count=%s, excel=%s, html=%s",
            site,
            payload.get("count"),
            payload.get("excel_path"),
            payload.get("html_path"),
        )


if __name__ == "__main__":
    main()
