"""
Generate a sample HTML using generate_daily_report_html_email for manual verification.

Usage:
  python scripts/generate_daily_email_html.py

Output:
  ./daily_report_sample.html
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from error_log_monitor.daily_report import DailyReportRow
from error_log_monitor.email_templates import generate_daily_report_html_email


def main() -> None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=24)
    end = now

    # Minimal dummy data for one site
    issues = [
        DailyReportRow(
            key="JIRA-1234",
            site="prod",
            count=7,
            error_message="Sample error message to preview dark theme email layout",
            status="Open",
            log_group="/aws/lambda/sample-group",
            latest_update=end,
        ),
        DailyReportRow(
            key="JIRA-5678",
            site="prod",
            count=3,
            error_message="Database connection timeout when querying user profile",
            status="In Progress",
            log_group="/aws/ecs/sample-service",
            latest_update=end - timedelta(hours=6),
        ),
    ]

    site_reports = {
        "prod": {
            "issues": issues,
        }
    }

    total_issues = sum(len(site_data.get("issues", [])) for site_data in site_reports.values())

    html = generate_daily_report_html_email(
        site_reports=site_reports,
        start_date=start,
        end_date=end,
        total_issues=total_issues,
    )

    output_path = Path.cwd() / "daily_report_sample.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote sample HTML: {output_path}")


if __name__ == "__main__":
    main()
