"""Shared utilities for report generation modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from error_log_monitor.embedding_service import EmbeddingService, cosine_similarity
from error_log_monitor.jira_cloud_client import JiraCloudClient, JiraIssueDetails
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB
from error_log_monitor.opensearch_client import ErrorLog, OpenSearchClient


logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_FIELDS: Sequence[str] = (
    "summary",
    "status",
    "error_message",
    "site",
    "log_group",
    "parent_issue_key",
    "occurrence_list",
    "key",
    "updated",
)


@dataclass
class JiraIssueSnapshot:
    key: str
    status: str
    site: Optional[str]
    log_group: Optional[str]
    summary: str
    updated: Optional[datetime]


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def fetch_jira_snapshots(
    jira_client: JiraCloudClient,
    project_key: Optional[str],
    since: Optional[datetime] = None,
    duration_in_days: Optional[int] = None,
) -> List[JiraIssueDetails]:
    raw_issues: List[JiraIssueDetails] = jira_client.get_all_issues(
        project_key=project_key, duration_in_days=duration_in_days
    )
    results: List[JiraIssueDetails] = []
    for issue in raw_issues:
        key = getattr(issue, "key", None)
        if not key:
            continue
        updated_dt = parse_iso_datetime(getattr(issue, "updated", None))
        created_dt = parse_iso_datetime(getattr(issue, "created", None))
        reference = updated_dt or created_dt
        if since and reference and reference < since:
            continue
        # Normalize updated/created back to ISO strings on the object if needed
        if updated_dt:
            issue.updated = updated_dt.isoformat()
        if created_dt:
            issue.created = created_dt.isoformat()
        results.append(issue)
    return results


def fetch_embedding_docs(
    jira_embedding_db: JiraIssueEmbeddingDB,
    since: datetime,
    until: datetime,
    source_fields: Optional[Sequence[str]] = None,
    batch_size: int = 500,
    scroll_ttl: str = "2m",
) -> List[Dict[str, Any]]:
    client = jira_embedding_db.opensearch_connect
    index_name = jira_embedding_db.get_current_index_name()

    query: Dict[str, Any] = {
        "size": batch_size,
        "query": {
            "range": {
                "updated": {
                    "gte": since.isoformat(),
                    "lte": until.isoformat(),
                }
            }
        },
    }

    query["_source"] = list(source_fields or _DEFAULT_SOURCE_FIELDS)

    response = client.search(index=index_name, body=query, scroll=scroll_ttl)
    scroll_id = response.get("_scroll_id")
    results: List[Dict[str, Any]] = []

    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                source = hit.get("_source", {})
                source["doc_id"] = hit.get("_id")
                results.append(source)

            if not scroll_id:
                break
            response = client.scroll(scroll_id=scroll_id, scroll=scroll_ttl)
            scroll_id = response.get("_scroll_id")
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                logger.warning("Failed to clear OpenSearch scroll context", exc_info=True)

    return results


def fetch_error_logs(
    opensearch_client: OpenSearchClient,
    start: datetime,
    end: datetime,
    sites: Optional[Sequence[str]] = None,
    limit: int = 5000,
) -> List[ErrorLog]:
    site_set: Set[str] = set(filter(None, sites or []))
    if not site_set:
        site_set = {"prod", "stage"}

    logs: List[ErrorLog] = []
    for site in site_set:
        try:
            site_logs = opensearch_client.get_error_logs(site, start, end, limit=limit)
            logs.extend(site_logs)
        except Exception:
            logger.warning("Failed to fetch logs for site %s", site, exc_info=True)
    return logs


def sync_embedding_statuses(
    jira_embedding_db: JiraIssueEmbeddingDB,
    embedding_docs: List[Dict[str, Any]],
    jira_by_key: Dict[str, JiraIssueSnapshot],
) -> None:

    index_name = jira_embedding_db.get_current_index_name()
    for doc in embedding_docs:
        key = doc.get("key")
        if not key or key not in jira_by_key:
            continue
        snapshot = jira_by_key[key]
        jira_status = snapshot.status or "Unknown"
        current_status = doc.get("status") or "Unknown"
        payload = {"status": jira_status}
        if snapshot.log_group:
            payload["log_group"] = snapshot.log_group
        if jira_status != current_status or (snapshot.log_group and doc.get("log_group") != snapshot.log_group):
            try:
                jira_embedding_db.opensearch_connect.update(
                    index=index_name, id=doc.get("doc_id"), body={"doc": {**payload}}
                )
                doc.update(payload)
            except Exception:
                logger.warning("Failed to update status/log_group for %s", key, exc_info=True)

    # # Update Jira issues that are not in the Jira
    # for embedding_doc in embedding_docs:
    #     issue = jira_embedding_db.find_jira_issue_by_key(embedding_doc.get("key"))
    #     if not issue:
    #         jira_data = JiraIssueDetails(
    #             key=embedding_doc.get("key"),
    #             summary=embedding_doc.get("summary", ""),
    #             status="Pending",
    #             site=embedding_doc.get("site", "unknown"),
    #             log_group=embedding_doc.get("log_group", None),
    #             error_message=embedding_doc.get("error_message", ""),
    #             error_type=embedding_doc.get("error_type", ""),
    #             traceback=embedding_doc.get("traceback", ""),
    #             request_id=embedding_doc.get("request_id", ""),
    #             count=embedding_doc.get("count", None),
    #             created=embedding_doc.get("created", None),
    #             updated=embedding_doc.get("updated", None),
    #             description=embedding_doc.get("description", ""),
    #             is_parent=True,
    #             not_commit_to_jira=False,
    #         )
    #         jira_embedding_db.add_jira_issue(jira_data)


def merge_orphan_embedding_docs(
    jira_embedding_db: JiraIssueEmbeddingDB,
    embedding_docs: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> None:
    for doc in embedding_docs:
        if doc.get("key"):
            continue
        _merge_single_orphan(jira_embedding_db, doc, now=now)


def _merge_single_orphan(
    jira_embedding_db: JiraIssueEmbeddingDB,
    orphan: Dict[str, Any],
    now: Optional[datetime] = None,
) -> None:
    summary = orphan.get("summary", "")
    error_message = orphan.get("error_message", "")
    status = orphan.get("status", "Unknown")
    site = orphan.get("site") or "unknown"
    timestamp = (now or datetime.now(timezone.utc)).isoformat()

    placeholder = JiraIssueDetails(
        key="",
        summary=summary,
        status=status,
        parent_issue_key=None,
        child_issue_keys=[],
        error_message=error_message,
        error_type=None,
        traceback=None,
        site=site,
        request_id=None,
        log_group=None,
        count=None,
        created=timestamp,
        updated=timestamp,
        description="",
    )

    embedding = jira_embedding_db._generate_embedding_from_data(placeholder)
    if not embedding:
        return

    match = jira_embedding_db.find_similar_jira_issue(embedding, site)
    if not match or not match.get("key"):
        return

    target_doc_id = match["doc_id"]
    occurrences = orphan.get("occurrence_list", []) or []
    for occ in occurrences:
        try:
            ts = occ.get("timestamp")
            if ts:
                jira_embedding_db.add_occurrence(
                    source_doc_id=target_doc_id,
                    doc_id=occ.get("doc_id", f"merged-{orphan.get('doc_id')}-{ts}"),
                    timestamp=ts,
                )
        except Exception:
            logger.warning("Failed to merge occurrence into %s", target_doc_id, exc_info=True)

    try:
        jira_embedding_db.delete_issue(orphan.get("doc_id"))
    except Exception:
        logger.warning("Failed to delete orphan embedding doc %s", orphan.get("doc_id"), exc_info=True)


def update_embedding_with_error_logs(
    jira_embedding_db: JiraIssueEmbeddingDB,
    embedding_service: EmbeddingService,
    logs: Sequence[ErrorLog],
    similarity_threshold: float = 0.85,
) -> None:
    # Step 1: Precompute all embeddings
    enriched: List[Dict[str, Any]] = []
    for log in logs:
        emb = build_log_embedding(embedding_service, log)
        if emb is None:
            continue
        enriched.append({"log": log, "embedding": emb})

    if not enriched:
        return

    # Step 2: Local merge by cosine similarity (greedy clustering), grouped by site

    clusters: List[Dict[str, Any]] = []
    by_site: Dict[str, List[Dict[str, Any]]] = {}
    for item in enriched:
        by_site.setdefault(item["log"].site or "unknown", []).append(item)

    local_threshold = max(0.88, similarity_threshold)

    for site, items in by_site.items():
        used: set = set()
        for i, item in enumerate(items):
            if i in used:
                continue
            base = item
            cluster = {"site": site, "embedding": base["embedding"], "members": [base["log"]]}
            used.add(i)
            for j in range(i + 1, len(items)):
                if j in used:
                    continue
                other = items[j]
                sim = cosine_similarity(base["embedding"], other["embedding"])
                if sim >= local_threshold:
                    cluster["members"].append(other["log"])
                    used.add(j)
            clusters.append(cluster)

    if not clusters:
        return

    # Step 3: For each cluster, try to update existing issue; if none, create a new one from representative
    for cluster in clusters:
        site = cluster["site"]
        # Representative is the first member
        rep_log: ErrorLog = cluster["members"][0]
        rep_emb: List[float] = cluster["embedding"]

        match = jira_embedding_db.find_similar_jira_issue(rep_emb, site, similarity_threshold=similarity_threshold)
        if match and match.get("key"):
            for log in cluster["members"]:
                try:
                    jira_embedding_db.add_occurrence(
                        source_doc_id=match["doc_id"],
                        doc_id=log.message_id,
                        timestamp=log.timestamp.isoformat(),
                    )
                except Exception:
                    logger.warning("Failed to add occurrence for %s", match.get("key"), exc_info=True)
            try:
                jira_embedding_db.remove_duplicate_occurrences(match["doc_id"])
            except Exception:
                logger.warning("Failed to dedupe occurrences for %s", match.get("key"), exc_info=True)
        else:
            # Create new embedding issue from representative log
            try:
                new_issue = jira_embedding_db._create_jira_issue_from_error_log(rep_log, site)
                if new_issue:
                    result = jira_embedding_db.add_jira_issue_from_jira_issue_detail(new_issue)
                    source_doc_id = None
                    similar_issue_key = None
                    if result.get('result') == 'added':
                        source_doc_id = result.get("id", None)
                    elif result.get('result') == 'skipped':
                        similar_issue = result.get("similar_issue", None)
                        similar_issue_key = similar_issue.get("key", None)
                        if not similar_issue_key:
                            logger.warning("Similar issue key is None for doc id %s", result.get("doc_id", None))
                        if similar_issue:
                            source_doc_id = similar_issue.get("doc_id", None)
                    # Add occurrences for all logs in the cluster
                    for log in cluster["members"]:
                        try:
                            jira_embedding_db.add_occurrence(
                                source_doc_id=source_doc_id,
                                doc_id=log.message_id,
                                timestamp=log.timestamp.isoformat(),
                            )
                        except Exception:
                            logger.warning("Failed to add occurrence for new issue %s", new_issue.key, exc_info=True)
                    source_doc = jira_embedding_db.opensearch_connect.get(
                        index=jira_embedding_db.get_current_index_name(), id=source_doc_id
                    )
                    if '_source' in source_doc:
                        source_doc = source_doc['_source']
                    if not source_doc.get("key", None):
                        jira_cloud_client = JiraCloudClient(jira_embedding_db.config.jira)
                        jira_issue_key = jira_cloud_client.create_jira_issue(source_doc)
                        source_doc["key"] = jira_issue_key
                        jira_embedding_db.opensearch_connect.update(
                            index=jira_embedding_db.get_current_index_name(), id=source_doc_id, body={"doc": source_doc}
                        )

            except Exception:
                logger.warning(
                    "Failed to create new issue from representative log %s",
                    getattr(rep_log, "message_id", ""),
                    exc_info=True,
                )


def build_log_embedding(embedding_service: EmbeddingService, log: ErrorLog) -> Optional[List[float]]:
    text = " ".join(filter(None, [log.error_message, log.error_type, log.traceback]))
    if not text.strip():
        return None
    try:
        return embedding_service.generate_embedding(text)
    except Exception:
        logger.warning("Failed to generate embedding for log %s", log.message_id, exc_info=True)
        return None


def filter_occurrence_timestamps(
    occurrences: Iterable[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
) -> List[datetime]:
    timestamps: List[datetime] = []
    for occurrence in occurrences:
        ts = occurrence.get("timestamp")
        if not ts:
            continue
        parsed = parse_iso_datetime(ts) if isinstance(ts, str) else ts
        if not isinstance(parsed, datetime):
            continue
        parsed = parsed.astimezone(timezone.utc)
        if start_date <= parsed <= end_date:
            timestamps.append(parsed)
    return timestamps
