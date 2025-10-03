"""Daily Report generator based on Jira, embedding DB, and recent error logs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from error_log_monitor.config import load_config
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB
from error_log_monitor.jira_cloud_client import JiraCloudClient
from error_log_monitor.opensearch_client import OpenSearchClient, ErrorLog
from error_log_monitor.report_shared import (
    JiraIssueSnapshot,
    fetch_embedding_docs,
    fetch_error_logs,
    fetch_jira_snapshots,
    filter_occurrence_timestamps,
    merge_orphan_embedding_docs,
    sync_embedding_statuses,
    update_embedding_with_error_logs,
)
from error_log_monitor.report_utils import (
    generate_excel_report,
    generate_html_report,
    generate_combined_excel_report,
)

logger = logging.getLogger(__name__)


@dataclass
class DailyReportRow:
    key: str
    site: str
    count: int
    error_message: str
    status: str
    log_group: str
    latest_update: datetime
    note: str = ""

    def __post_init__(self):
        # Ensure this class implements the ReportRow protocol
        pass


@dataclass
class JiraIssueSnapshot:
    key: str
    status: str
    site: Optional[str]
    log_group: Optional[str]
    summary: str
    updated: Optional[datetime]


class DailyReportGenerator:
    """Generate daily reports using Jira issues, embedding DB, and error logs."""

    def __init__(self):
        self.config = load_config()
        self.embedding_service = EmbeddingService(model_name=self.config.vector_db.embedding_model)
        self.jira_embedding_db = JiraIssueEmbeddingDB(embedding_service=self.embedding_service, config=self.config)
        self.jira_client = JiraCloudClient(self.config.jira)
        self.opensearch_client = OpenSearchClient(self.config.opensearch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily report covering the most recent 24-hour window."""
        end_date = datetime.now(timezone.utc).astimezone(timezone.utc)
        a_day_ago = (end_date - timedelta(hours=24)).astimezone(timezone.utc)

        logger.info("Fetching Jira issues for the past 1 day")
        jira_snapshots = self._fetch_recent_jira_issues(duration_in_days=1)
        jira_by_key = {issue.key: issue for issue in jira_snapshots if issue.key}

        logger.info("Fetching embedding issues for the past 24 hours")
        daily_embedding_docs = self._fetch_embedding_issues(a_day_ago, end_date)
        logger.info(f"Found {len(daily_embedding_docs)} embedding issues")
        logger.info("Synchronizing statuses and merging orphan embedding entries")
        self._sync_embedding_statuses(daily_embedding_docs, jira_by_key)

        logger.info("Fetching error logs for the past 24 hours")
        error_logs_24_hours = self._fetch_recent_error_logs(a_day_ago, end_date)

        logger.info(f"Updating {len(error_logs_24_hours)} embedding occurrences based on error logs")
        self._update_embedding_with_error_logs(error_logs_24_hours)

        logger.info("Refreshing embedding issues after updates")
        daily_embedding_docs = self._fetch_embedding_issues(a_day_ago, end_date)
        self._merge_orphan_embedding_docs(daily_embedding_docs)
        daily_embedding_docs = self._fetch_embedding_issues(a_day_ago, end_date)
        site_reports = self._build_site_reports(daily_embedding_docs, a_day_ago, end_date)
        combined_path = generate_combined_excel_report(site_reports, a_day_ago, end_date, "daily_report")

        return {
            "start_date": a_day_ago,
            "end_date": end_date,
            "site_reports": site_reports,
            "combined_excel_path": combined_path,
        }

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def _fetch_recent_jira_issues(self, duration_in_days: int = 30) -> List[JiraIssueSnapshot]:
        return fetch_jira_snapshots(
            self.jira_client,
            project_key=self.config.jira.project_key,
            duration_in_days=duration_in_days,
        )

    def _fetch_embedding_issues(self, since: datetime, until: datetime) -> List[Dict[str, Any]]:
        return fetch_embedding_docs(self.jira_embedding_db, since, until)

    def _fetch_recent_error_logs(self, start: datetime, end: datetime) -> List[ErrorLog]:
        return fetch_error_logs(self.opensearch_client, start, end)

    # ------------------------------------------------------------------
    # Cleanup and synchronization
    # ------------------------------------------------------------------
    def _sync_embedding_statuses(
        self, embedding_docs: List[Dict[str, Any]], jira_by_key: Dict[str, JiraIssueSnapshot]
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
            row = DailyReportRow(
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
            rows: List[DailyReportRow] = payload["issues"]
            rows.sort(key=lambda r: r.latest_update, reverse=True)
            payload["excel_path"] = generate_excel_report(site, rows, start_date, end_date, "daily_report")
            payload["html_path"] = generate_html_report(site, rows, start_date, end_date, "daily_report")
            payload["count"] = len(rows)
        return reports

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    # Retained for backward compatibility but no additional utilities required
