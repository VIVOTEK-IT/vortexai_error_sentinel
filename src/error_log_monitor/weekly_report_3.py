"""Weekly Report 3 generator based on Jira, embedding DB, and recent error logs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from error_log_monitor.config import load_config
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB
from error_log_monitor.jira_cloud_client import JiraCloudClient, JiraIssueDetails
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.report_shared import (
    JiraIssueSnapshot,
    build_log_embedding,
    fetch_embedding_docs,
    fetch_error_logs,
    fetch_jira_snapshots,
    filter_occurrence_timestamps,
    merge_orphan_embedding_docs,
    parse_iso_datetime,
    sync_embedding_statuses,
    update_embedding_with_error_logs,
)

logger = logging.getLogger(__name__)

_ILLEGAL_CHAR_LIMIT = 32767


def _sanitize(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    sanitized = "".join(ch for ch in text if 32 <= ord(ch) <= 126 or ord(ch) in (9, 10, 13))
    if len(sanitized) > _ILLEGAL_CHAR_LIMIT:
        sanitized = sanitized[: _ILLEGAL_CHAR_LIMIT - 3] + "..."
    return sanitized


@dataclass
class WeeklyReportRow:
    key: str
    site: str
    count: int
    error_message: str
    status: str
    log_group: str
    latest_update: datetime
    note: str = ""


@dataclass
class JiraIssueSnapshot:
    key: str
    status: str
    site: Optional[str]
    log_group: Optional[str]
    summary: str
    updated: Optional[datetime]


class WeeklyReportGenerator3:
    """Generate weekly reports using Jira issues, embedding DB, and error logs."""

    def __init__(self):
        self.config = load_config()
        self.embedding_service = EmbeddingService(model_name=self.config.vector_db.embedding_model)
        self.jira_embedding_db = JiraIssueEmbeddingDB(embedding_service=self.embedding_service, config=self.config)
        self.jira_client = JiraCloudClient(self.config.jira)
        self.opensearch_client = OpenSearchClient(self.config.opensearch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_weekly_report(self, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        end_date = (end_date or datetime.now(timezone.utc)).astimezone(timezone.utc)
        a_week_ago = (end_date - timedelta(days=7)).astimezone(timezone.utc)
        six_months_ago = (end_date - timedelta(days=180)).astimezone(timezone.utc)

        logger.info("Fetching Jira issues for the past six months")
        jira_snapshots = self._fetch_recent_jira_issues(since=six_months_ago)
        jira_by_key = {issue.key: issue for issue in jira_snapshots if issue.key}

        logger.info("Fetching embedding issues for the past six months")
        weekly_embedding_docs = self._fetch_embedding_issues(a_week_ago, end_date)

        logger.info("Synchronizing statuses and merging orphan embedding entries")
        self._sync_embedding_statuses(weekly_embedding_docs, jira_by_key)

        logger.info("Fetching error logs for the past seven days")
        error_logs_7_days = self._fetch_recent_error_logs(a_week_ago, end_date)

        logger.info(f"Updating {len(error_logs_7_days)} embedding occurrences based on error logs")
        self._update_embedding_with_error_logs(error_logs_7_days)

        logger.info("Refreshing embedding issues after updates")
        weekly_embedding_docs = self._fetch_embedding_issues(a_week_ago, end_date)
        self._merge_orphan_embedding_docs(weekly_embedding_docs)
        weekly_embedding_docs = self._fetch_embedding_issues(a_week_ago, end_date)
        site_reports = self._build_site_reports(weekly_embedding_docs, a_week_ago, end_date)
        combined_path = self._generate_combined_excel(site_reports, a_week_ago, end_date)

        return {
            "start_date": a_week_ago,
            "end_date": end_date,
            "site_reports": site_reports,
            "combined_excel_path": combined_path,
        }

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def _fetch_recent_jira_issues(self, since: Optional[datetime] = None) -> List[JiraIssueSnapshot]:
        return fetch_jira_snapshots(
            self.jira_client,
            project_key=self.config.jira.project_key,
            since=since,
            duration_in_days=180,
        )

    def _fetch_embedding_issues(self, since: datetime, until: datetime) -> List[Dict[str, Any]]:
        return fetch_embedding_docs(self.jira_embedding_db, since, until)

    def _fetch_recent_error_logs(self, start: datetime, end: datetime) -> List[ErrorLog]:

        return fetch_error_logs(self.opensearch_client, start, end)

    # ------------------------------------------------------------------
    # Cleanup and synchronization
    # ------------------------------------------------------------------
    def _sync_embedding_statuses(
        self, embedding_docs: List[Dict[str, Any]], jira_by_key: Dict[str, JiraIssueDetails]
    ) -> None:
        sync_embedding_statuses(self.jira_embedding_db, embedding_docs, jira_by_key)

    def _merge_orphan_embedding_docs(self, embedding_docs: List[Dict[str, Any]]) -> None:
        merge_orphan_embedding_docs(self.jira_embedding_db, embedding_docs)

    # ------------------------------------------------------------------
    # Error log processing
    # ------------------------------------------------------------------
    def _update_embedding_with_error_logs(self, logs: List[ErrorLog]) -> None:
        update_embedding_with_error_logs(
            self.jira_embedding_db, self.embedding_service, logs, similarity_threshold=0.85
        )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _build_site_reports(
        self, embedding_docs: List[Dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        reports: Dict[str, Dict[str, Any]] = {}
        for doc in embedding_docs:
            key = doc.get("key")
            if not key:
                continue
            site = doc.get("site", "unknown")
            occurrences = doc.get("occurrence_list", []) or []
            recent_timestamps = filter_occurrence_timestamps(occurrences, start_date, end_date)
            if not recent_timestamps:
                continue

            latest_update = max(recent_timestamps)
            row = WeeklyReportRow(
                key=key,
                site=site,
                count=len(recent_timestamps),
                error_message=doc.get("error_message", ""),
                status=doc.get("status", "Unknown"),
                log_group=doc.get("log_group", "Unknown"),
                latest_update=latest_update,
            )
            reports.setdefault(site, {"issues": []})
            reports[site]["issues"].append(row)

        for site, payload in reports.items():
            rows: List[WeeklyReportRow] = payload["issues"]
            rows.sort(key=lambda r: r.latest_update, reverse=True)
            payload["excel_path"] = self._generate_excel(site, rows, start_date, end_date)
            payload["html_path"] = self._generate_html(site, rows, start_date, end_date)
            payload["count"] = len(rows)
        return reports

    def _generate_excel(self, site: str, rows: List[WeeklyReportRow], start: datetime, end: datetime) -> str:
        data = [
            {
                "Key": _sanitize(row.key),
                "Site": _sanitize(row.site),
                "Count": row.count,
                "Error_Message": _sanitize(row.error_message),
                "Status": _sanitize(row.status),
                "Log Group": _sanitize(row.log_group),
                "Latest Update": row.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                "Note": _sanitize(row.note),
            }
            for row in rows
        ]
        df = pd.DataFrame(data)
        reports_dir = self._reports_dir()
        filename = f"weekly_report_3_{site}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.xlsx"
        filepath = os.path.join(reports_dir, filename)
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=f"{site.title()} Report", index=False)
        return filepath

    def _generate_html(self, site: str, rows: List[WeeklyReportRow], start: datetime, end: datetime) -> str:
        reports_dir = self._reports_dir()
        filename = f"weekly_report_3_{site}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.html"
        filepath = os.path.join(reports_dir, filename)
        html_rows = "".join(
            f"<tr><td>{_sanitize(row.key)}</td><td>{_sanitize(row.site)}</td><td>{row.count}</td>"
            f"<td>{_sanitize(row.error_message)}</td><td>{_sanitize(row.status)}</td>"
            f"<td>{_sanitize(row.log_group)}</td><td>{row.latest_update.strftime('%Y-%m-%d %H:%M:%S')}</td>"
            f"<td>{_sanitize(row.note)}</td></tr>"
            for row in rows
        )
        html = f"""
        <html>
        <head><title>Weekly Report 3 - {site}</title></head>
        <body>
        <h1>Weekly Report 3 - {site.title()}</h1>
        <p>Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}</p>
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
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(html)
        return filepath

    def _generate_combined_excel(self, site_reports: Dict[str, Dict[str, Any]], start: datetime, end: datetime) -> str:
        reports_dir = self._reports_dir()
        filename = f"weekly_report_3_combined_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.xlsx"
        filepath = os.path.join(reports_dir, filename)
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for site, payload in site_reports.items():
                rows = payload.get("issues", [])
                df = pd.DataFrame(
                    [
                        {
                            "Key": _sanitize(row.key),
                            "Site": _sanitize(row.site),
                            "Count": row.count,
                            "Error_Message": _sanitize(row.error_message),
                            "Status": _sanitize(row.status),
                            "Log Group": _sanitize(row.log_group),
                            "Latest Update": row.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                            "Note": _sanitize(row.note),
                        }
                        for row in rows
                    ]
                )
                df.to_excel(writer, sheet_name=site.title(), index=False)
        return filepath

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _reports_dir(self) -> str:
        reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        return reports_dir

    # Utility retained for compatibility if direct parsing required elsewhere
    _parse_date = staticmethod(parse_iso_datetime)
