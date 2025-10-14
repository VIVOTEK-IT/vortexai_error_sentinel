"""Common report utilities for generating Excel and HTML reports."""

import os
from datetime import datetime
from typing import Any, Dict, List, Protocol

import pandas as pd


_ILLEGAL_CHAR_LIMIT = 32767


def sanitize_for_excel(value: Any) -> str:
    """Sanitize text for Excel compatibility."""
    text = value if isinstance(value, str) else str(value)
    sanitized = "".join(ch for ch in text if 32 <= ord(ch) <= 126 or ord(ch) in (9, 10, 13))
    if len(sanitized) > _ILLEGAL_CHAR_LIMIT:
        sanitized = sanitized[: _ILLEGAL_CHAR_LIMIT - 3] + "..."
    return sanitized


def get_reports_dir() -> str:
    """Get the reports directory path and create it if it doesn't exist."""
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


class ReportRow(Protocol):
    """Protocol for report row data."""

    key: str
    site: str
    count: int
    error_message: str
    status: str
    log_group: str
    latest_update: datetime
    note: str


def generate_excel_report(
    site: str, rows: List[ReportRow], start: datetime, end: datetime, report_type: str = "report"
) -> str:
    """Generate Excel report for a specific site."""
    reports_dir = get_reports_dir()
    filename = f"{report_type}_{site}_{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}.xlsx"
    filepath = os.path.join(reports_dir, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        if rows:
            # Generate report with data
            data = [
                {
                    "Key": sanitize_for_excel(row.key),
                    "Site": sanitize_for_excel(row.site),
                    "Count": row.count,
                    "Error_Message": sanitize_for_excel(row.error_message),
                    "Status": sanitize_for_excel(row.status),
                    "Log Group": sanitize_for_excel(row.log_group),
                    "Latest Update": row.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                    "Note": sanitize_for_excel(row.note),
                }
                for row in rows
            ]
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f"{site.title()} Report", index=False)
        else:
            # Generate empty report with summary
            summary_data = {
                "Message": [f"No issues found for {site} site"],
                "Start Date": [start.strftime("%Y-%m-%d %H:%M:%S")],
                "End Date": [end.strftime("%Y-%m-%d %H:%M:%S")],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name=f"{site.title()} Report", index=False)

    return filepath


def generate_html_report(
    site: str,
    rows: List[ReportRow],
    start: datetime,
    end: datetime,
    report_type: str = "report",
    store_to_file: bool = False,
) -> str:
    """Generate HTML report for a specific site."""
    filepath = None
    if store_to_file:
        reports_dir = get_reports_dir()
        filename = f"{report_type}_{site}_{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}.html"
        filepath = os.path.join(reports_dir, filename)
    html_rows = "".join(
        f"<tr><td>{sanitize_for_excel(row.key)}</td><td>{sanitize_for_excel(row.site)}</td><td>{row.count}</td>"
        f"<td>{sanitize_for_excel(row.error_message)}</td><td>{sanitize_for_excel(row.status)}</td>"
        f"<td>{sanitize_for_excel(row.log_group)}</td><td>{row.latest_update.strftime('%Y-%m-%d %H:%M:%S')}</td>"
        f"<td>{sanitize_for_excel(row.note)}</td></tr>"
        for row in rows
    )
    html = f"""
    <html>
    <head><title>{report_type.title()} - {site}</title></head>
    <body>
    <h1>{report_type.title()} - {site.title()}</h1>
    <p>Period: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}</p>
    <table border="1" cellspacing="0" cellpadding="4">
        <thead>
            <tr>
                <th>Key</th><th>Site</th><th>Count</th><th>Error Message</th>
                <th>Status</th><th>Log Group</th><th>Latest Update</th><th>Note</th>
            </tr>
        </thead>
        <tbody>
            {html_rows}
        </tbody>
    </table>
    </body>
    </html>
    """
    if store_to_file:
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(html)

    return filepath, html


def generate_combined_excel_report(
    site_reports: Dict[str, Dict[str, Any]], start: datetime, end: datetime, report_type: str = "report"
) -> str:
    """Generate combined Excel report for all sites."""
    reports_dir = get_reports_dir()
    filename = f"{report_type}_combined_{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%Y%m%d_%H%M')}.xlsx"
    filepath = os.path.join(reports_dir, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        sheets_created = False

        for site, payload in site_reports.items():
            rows = payload.get("issues", [])
            if not rows:  # Skip empty sites
                continue

            df = pd.DataFrame(
                [
                    {
                        "Key": sanitize_for_excel(row.key),
                        "Site": sanitize_for_excel(row.site),
                        "Count": row.count,
                        "Error_Message": sanitize_for_excel(row.error_message),
                        "Status": sanitize_for_excel(row.status),
                        "Log Group": sanitize_for_excel(row.log_group),
                        "Latest Update": row.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                        "Note": sanitize_for_excel(row.note),
                    }
                    for row in rows
                ]
            )
            df.to_excel(writer, sheet_name=site.title(), index=False)
            sheets_created = True

        # If no sheets were created (all sites empty), create a summary sheet
        if not sheets_created:
            summary_data = {
                "Message": ["No issues found for the specified time period"],
                "Start Date": [start.strftime("%Y-%m-%d %H:%M:%S")],
                "End Date": [end.strftime("%Y-%m-%d %H:%M:%S")],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    return filepath
