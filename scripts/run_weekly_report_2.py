#!/usr/bin/env python3
"""Utility script to generate weekly reports using WeeklyReportGenerator2."""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from error_log_monitor.config import load_config  # noqa: E402
from error_log_monitor.weekly_report_2 import WeeklyReportGenerator2  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly report using WeeklyReportGenerator2")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 7 days before end date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown arguments: %s", " ".join(unknown))
    return args


def parse_date(date_str: str, default: datetime) -> datetime:
    if not date_str:
        return default
    return datetime.strptime(date_str, "%Y-%m-%d")


def main() -> None:
    args = parse_args(sys.argv[1:])

    end_date_default = datetime.utcnow()
    end_date = parse_date(args.end_date, end_date_default)
    start_date_default = end_date - timedelta(days=7)
    start_date = parse_date(args.start_date, start_date_default)

    if start_date > end_date:
        raise ValueError("start date must not be after end date")

    logger.info("Loading configuration")
    config = load_config()

    generator = WeeklyReportGenerator2(config)
    logger.info("Generating weekly report 2 from %s to %s", start_date.date(), end_date.date())
    report = generator.generate_weekly_report(start_date, end_date)

    combined_path = report.get("combined_excel_path")
    logger.info("Weekly report generated. Combined Excel path: %s", combined_path)
    for site, data in report.get("site_reports", {}).items():
        logger.info("Site %s Excel path: %s", site, data.get("excel_path"))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        logger.error("Missing dependency: %s. Ensure required packages are installed.", exc)
        raise
