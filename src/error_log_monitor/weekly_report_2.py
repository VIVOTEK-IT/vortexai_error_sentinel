"""Simplified Weekly Report generator that reads Jira issues directly from JiraIssueEmbeddingDB."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Iterable, Optional

import pandas as pd
import re

from error_log_monitor.config import SystemConfig
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB

logger = logging.getLogger(__name__)


_ILLEGAL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_for_excel(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    sanitized = _ILLEGAL_CHAR_PATTERN.sub("", text)
    if len(sanitized) > 32767:
        sanitized = sanitized[:32764] + "..."
    return sanitized


@dataclass
class WeeklyReportIssue:
    key: str
    site: str
    count: int
    summary: str
    error_message: str
    status: str
    log_group: str
    latest_update: datetime
    parent_issue_key: Optional[str] = None
    updated_raw: str = ""
    note: str = ""


class WeeklyReportGenerator2:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.jira_embedding_db = JiraIssueEmbeddingDB(embedding_service=None, config=config)

    def generate_weekly_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)

        if start_date > end_date:
            raise ValueError("start_date must be before end_date")

        logger.info("Generating weekly report 2 using Jira embedding database")
        site_reports: Dict[str, Dict[str, Any]] = {}
        all_issues = self._fetch_issues(start_date, end_date)

        for site, issues in all_issues.items():
            weekly_issues = [self._convert_issue(issue) for issue in issues]
            weekly_issues.sort(key=lambda issue: issue.latest_update, reverse=True)
            excel_path = self._generate_excel_report(weekly_issues, start_date, end_date, site)

            site_reports[site] = {
                "site": site,
                "start_date": start_date,
                "end_date": end_date,
                "weekly_issues": len(weekly_issues),
                "excel_path": excel_path,
                "issues": weekly_issues,
            }

        combined_path = self._generate_combined_excel_report(site_reports, start_date, end_date)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "site_reports": site_reports,
            "combined_excel_path": combined_path,
        }

    def _fetch_issues(self, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        client = self.jira_embedding_db.opensearch_connect
        index = self.jira_embedding_db.get_current_index_name()

        query = {
            "size": 500,
            "query": {"match_all": {}},
            "_source": [
                "key",
                "summary",
                "status",
                "error_message",
                "site",
                "log_group",
                "parent_issue_key",
                "updated",
                "occurrence_list",
            ],
        }

        response = client.search(index=index, body=query, scroll="2m")
        scroll_id = response.get("_scroll_id")

        issues_by_site: Dict[str, List[Dict[str, Any]]] = {}

        try:
            while True:
                hits = response.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for issue in self._filter_hits(hits, start_date, end_date):
                    site = issue.get("site", "unknown")
                    issues_by_site.setdefault(site, []).append(issue)

                if not scroll_id:
                    break

                response = client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response.get("_scroll_id")
        finally:
            if scroll_id:
                try:
                    client.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    logger.warning("Failed to clear OpenSearch scroll context", exc_info=True)

        for site in list(issues_by_site.keys()):
            issues_by_site[site] = self._merge_child_issues(issues_by_site[site])

        return issues_by_site

    def _filter_hits(
        self, hits: Iterable[Dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> Iterable[Dict[str, Any]]:
        for hit in hits:
            source = hit.get("_source", {})
            occurrences = source.get("occurrence_list", []) or []
            updated_value = source.get("updated")
            if updated_value:
                updated_dt = self._parse_iso_datetime(updated_value)
                occurrences.append({"timestamp": updated_dt})
            timestamps = self._filter_occurrence_timestamps(occurrences, start_date, end_date)
            if not timestamps:
                continue

            latest_update = max(timestamps).isoformat()
            log_group = source.get("log_group")
            if not log_group:
                log_group = source.get("log_group.keyword") or "Unknown"
            yield {
                "key": source.get("key", "unknown"),
                "summary": source.get("summary", ""),
                "status": source.get("status", "Unknown"),
                "error_message": source.get("error_message", ""),
                "site": source.get("site", "unknown"),
                "log_group": log_group,
                "parent_issue_key": source.get("parent_issue_key"),
                "updated": latest_update,
                "count": len(timestamps),
            }

    def _filter_occurrence_timestamps(
        self, occurrences: Iterable[Dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        timestamps: List[datetime] = []

        for occurrence in occurrences:
            ts = occurrence.get("timestamp")
            if not ts:
                continue
            parsed = self._parse_iso_datetime(ts)
            if start_date <= parsed <= end_date:
                timestamps.append(parsed)
        return timestamps

    def _merge_child_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues_by_key = {issue.get("key"): issue for issue in issues if issue.get("key")}
        removed_keys: set[str] = set()

        for issue in issues:
            status = (issue.get("status") or "").upper()
            parent_key = issue.get("parent_issue_key")
            if status == "SUB ISSUE" and parent_key:
                parent = issues_by_key.get(parent_key)
                if not parent:
                    continue

                parent["count"] = parent.get("count", 0) + issue.get("count", 0)

                parent_updated = parent.get("updated")
                child_updated = issue.get("updated")
                if child_updated:
                    if parent_updated:
                        parent_dt = self._parse_iso_datetime(parent_updated)
                        child_dt = self._parse_iso_datetime(child_updated)
                        parent["updated"] = max(parent_dt, child_dt).isoformat()
                    else:
                        parent["updated"] = child_updated

                removed_keys.add(issue.get("key"))

        return [issue for issue in issues if issue.get("key") not in removed_keys]

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _convert_issue(self, issue: Dict[str, Any]) -> WeeklyReportIssue:
        latest_update = issue.get("updated")
        latest_update_dt = self._parse_iso_datetime(latest_update)

        return WeeklyReportIssue(
            key=issue.get("key", "unknown"),
            site=issue.get("site", "unknown"),
            count=int(issue.get("count", 0) or 0),
            summary=issue.get("summary", ""),
            error_message=issue.get("error_message", ""),
            status=issue.get("status", "Unknown"),
            log_group=issue.get("log_group", "Unknown"),
            latest_update=latest_update_dt,
            parent_issue_key=issue.get("parent_issue_key"),
            updated_raw=issue.get("updated", ""),
        )

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        return datetime.utcnow()

    def _generate_excel_report(
        self, weekly_issues: List[WeeklyReportIssue], start_date: datetime, end_date: datetime, site: str
    ) -> str:
        data = [
            {
                "Key": sanitize_for_excel(issue.key),
                "Site": sanitize_for_excel(issue.site),
                "Count": issue.count,
                "Summary": sanitize_for_excel(issue.summary),
                "Error_Message": sanitize_for_excel(issue.error_message),
                "Status": sanitize_for_excel(issue.status),
                "Log Group": sanitize_for_excel(issue.log_group),
                "Latest Update": issue.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for issue in weekly_issues
        ]

        df = pd.DataFrame(data)
        filename = f"weekly_report_2_{site}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
        reports_dir = os.path.abspath(reports_dir)
        os.makedirs(reports_dir, exist_ok=True)
        filepath = os.path.join(reports_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=f"{site.title()} Report", index=False)
        logger.info("Generated weekly report 2 excel: %s", filepath)
        return filepath

    def _generate_combined_excel_report(
        self, site_reports: Dict[str, Dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> str:
        combined_filename = (
            f"weekly_report_2_combined_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
        )
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
        reports_dir = os.path.abspath(reports_dir)
        os.makedirs(reports_dir, exist_ok=True)
        combined_path = os.path.join(reports_dir, combined_filename)

        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            for site, data in site_reports.items():
                issues = data.get("issues", [])
                df = pd.DataFrame(
                    [
                        {
                            "Key": sanitize_for_excel(issue.key),
                            "Site": sanitize_for_excel(issue.site),
                            "Count": issue.count,
                            "Summary": sanitize_for_excel(issue.summary),
                            "Error_Message": sanitize_for_excel(issue.error_message),
                            "Status": sanitize_for_excel(issue.status),
                            "Log Group": sanitize_for_excel(issue.log_group),
                            "Latest Update": issue.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        for issue in issues
                    ]
                )
                df.to_excel(writer, sheet_name=site.title(), index=False)
        logger.info("Generated combined weekly report 2 excel: %s", combined_path)
        return combined_path
