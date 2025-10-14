#!/usr/bin/env python3
"""
Daily Report Runner Script

This script can be used to run daily reports manually or as a scheduled task.
"""

import sys
import logging
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from error_log_monitor.config import load_config
from error_log_monitor.daily_report import DailyReportGenerator
from error_log_monitor.email_service import EmailService
from error_log_monitor.email_templates import generate_daily_report_html_email

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Configure logging

logger = logging.getLogger(__name__)


def run_daily_report(send_email: bool = True, report_type: str = "daily") -> bool:
    """
    Run daily or weekly report generation and optionally send email.

    Args:
        send_email: Whether to send email notification
        report_type: Type of report to generate ("daily" or "weekly")

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting {report_type} report generation...")

        # Load configuration
        config = load_config()

        # Initialize services
        report_generator = DailyReportGenerator()
        email_service = EmailService(config.email) if send_email else None

        # Generate report based on type
        if report_type == "weekly":
            logger.info("Generating weekly report...")
            report_data = report_generator.generate_weekly_report()
        else:
            logger.info("Generating daily report...")
            report_data = report_generator.generate_daily_report()

        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) for site_data in report_data.get("site_reports", {}).values()
        )

        # Print summary
        report_title = "Weekly" if report_type == "weekly" else "Daily"
        print(f"\n📊 {report_title} Report Summary:")
        print(
            f"  Period: {report_data['start_date'].strftime('%Y-%m-%d %H:%M')} to {report_data['end_date'].strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"  Total Issues: {total_issues}")

        for site_name, site_data in report_data['site_reports'].items():
            issues_count = len(site_data.get('issues', []))
            print(f"  {site_name.title()} Site: {issues_count} issues")
            if site_data.get('excel_path'):
                print(f"    Excel Report: {site_data['excel_path']}")
            if site_data.get('html_path'):
                print(f"    HTML Report: {site_data['html_path']}")

        if report_data.get('combined_excel_path'):
            print(f"  Combined Excel Report: {report_data['combined_excel_path']}")

        # Send email if requested
        if send_email and email_service:
            logger.info("Generating and sending email notification...")

            # Generate HTML email content
            html_content = generate_daily_report_html_email(
                site_reports=report_data.get("site_reports", {}),
                start_date=report_data.get("start_date"),
                end_date=report_data.get("end_date"),
                total_issues=total_issues,
            )

            # Send email
            email_sent = email_service.send_daily_report_email(html_content)

            if email_sent:
                print("  ✅ Email notification sent successfully")
            else:
                print("  ❌ Failed to send email notification")
                return False

        print(f"\n✅ {report_title} report completed successfully!")
        return True

    except Exception as e:
        logger.error(f"{report_type.title()} report failed: {str(e)}", exc_info=True)
        print(f"\n❌ {report_type.title()} report failed: {str(e)}")
        return False


def main():
    """Main function to run daily or weekly report."""
    import argparse

    parser = argparse.ArgumentParser(description="Run daily or weekly error report")
    parser.add_argument("--no-email", action="store_true", help="Generate report without sending email")
    parser.add_argument(
        "--type", choices=["daily", "weekly"], default="daily", help="Type of report to generate (default: daily)"
    )

    args = parser.parse_args()
    # Run report
    success = run_daily_report(
        send_email=not args.no_email,
        report_type=args.type,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
