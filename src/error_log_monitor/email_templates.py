"""HTML email templates for daily reports."""

from datetime import datetime
from typing import Dict, List, Any

from error_log_monitor.daily_report import DailyReportRow


def generate_daily_report_html_email(
    site_reports: Dict[str, Dict[str, Any]], start_date: datetime, end_date: datetime, total_issues: int = 0
) -> str:
    """
    Generate HTML email content for daily report.

    Args:
        site_reports: Dictionary containing site-specific report data
        start_date: Report start date
        end_date: Report end date
        total_issues: Total number of issues across all sites

    Returns:
        HTML email content
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Generate site-specific sections
    site_sections = []
    for site, report_data in site_reports.items():
        issues = report_data.get("issues", [])
        if not issues:
            continue

        site_section = _generate_site_section(site, issues, start_date, end_date)
        site_sections.append(site_section)

    # Calculate summary statistics
    total_stage_issues = len(site_reports.get("stage", {}).get("issues", []))
    total_prod_issues = len(site_reports.get("prod", {}).get("issues", []))

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily Error Report - {today}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .summary {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary h2 {{
                color: #667eea;
                margin-top: 0;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .site-section {{
                background: white;
                margin-bottom: 30px;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .site-header {{
                background: #667eea;
                color: white;
                padding: 20px;
                margin: 0;
            }}
            .site-header h2 {{
                margin: 0;
                font-size: 1.5em;
            }}
            .issues-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 0;
            }}
            .issues-table th {{
                background: #f8f9fa;
                color: #333;
                padding: 15px 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #dee2e6;
            }}
            .issues-table td {{
                padding: 12px;
                border-bottom: 1px solid #dee2e6;
                vertical-align: top;
            }}
            .issues-table tr:hover {{
                background-color: #f8f9fa;
            }}
            .status-badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }}
            .status-open {{
                background-color: #d4edda;
                color: #155724;
            }}
            .status-in-progress {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .status-resolved {{
                background-color: #d1ecf1;
                color: #0c5460;
            }}
            .status-closed {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .error-message {{
                max-width: 300px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .count-badge {{
                background: #667eea;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .no-issues {{
                text-align: center;
                padding: 40px;
                color: #666;
                font-style: italic;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
            .footer a {{
                color: #667eea;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Daily Error Report</h1>
            <p>{today} | VortexAI Error Monitoring System</p>
        </div>

        <div class="summary">
            <h2>📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_issues}</div>
                    <div class="stat-label">Total Issues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_stage_issues}</div>
                    <div class="stat-label">Stage Issues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_prod_issues}</div>
                    <div class="stat-label">Production Issues</div>
                </div>
            </div>
            <p><strong>Report Period:</strong> {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')} UTC</p>
        </div>

        {''.join(site_sections) if site_sections else '<div class="no-issues"><h3>No issues found for the reporting period</h3></div>'}

        <div class="footer">
            <p>Generated by VortexAI Error Monitoring System</p>
            <p>For support, contact: <a href="mailto:vortexai.dashboard@vortex.vivotek.com">vortexai.dashboard@vortex.vivotek.com</a></p>
        </div>
    </body>
    </html>
    """

    return html_content


def _generate_site_section(site: str, issues: List[DailyReportRow], start_date: datetime, end_date: datetime) -> str:
    """Generate HTML section for a specific site."""

    if not issues:
        return f"""
        <div class="site-section">
            <div class="site-header">
                <h2>{site.title()} Site</h2>
            </div>
            <div class="no-issues">
                <h3>No issues found for {site.title()} site</h3>
            </div>
        </div>
        """

    # Generate table rows
    table_rows = []
    for issue in issues:
        status_class = _get_status_class(issue.status)
        table_rows.append(
            f"""
            <tr>
                <td><strong>{issue.key}</strong></td>
                <td><span class="count-badge">{issue.count}</span></td>
                <td class="error-message" title="{issue.error_message}">{issue.error_message[:100]}{'...' if len(issue.error_message) > 100 else ''}</td>
                <td><span class="status-badge {status_class}">{issue.status}</span></td>
                <td>{issue.log_group}</td>
                <td>{issue.latest_update.strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        """
        )

    return f"""
    <div class="site-section">
        <div class="site-header">
            <h2>{site.title()} Site ({len(issues)} issues)</h2>
        </div>
        <table class="issues-table">
            <thead>
                <tr>
                    <th>Jira Key</th>
                    <th>Count</th>
                    <th>Error Message</th>
                    <th>Status</th>
                    <th>Log Group</th>
                    <th>Latest Update</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """


def _get_status_class(status: str) -> str:
    """Get CSS class for status badge based on status value."""
    status_lower = status.lower()
    if 'open' in status_lower:
        return 'status-open'
    elif 'progress' in status_lower or 'in progress' in status_lower:
        return 'status-in-progress'
    elif 'resolved' in status_lower or 'fixed' in status_lower:
        return 'status-resolved'
    elif 'closed' in status_lower or 'done' in status_lower:
        return 'status-closed'
    else:
        return 'status-open'  # Default
