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
    parent_issue_key: Optional[str]
    is_parent: Optional[bool]


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
    page_size: int = 1000,
) -> List[ErrorLog]:
    site_set: Set[str] = set(filter(None, sites or []))
    if not site_set:
        site_set = {"prod", "stage"}

    logs: List[ErrorLog] = []
    for site in site_set:
        try:
            site_logs = opensearch_client.get_error_logs(site, start, end, page_size=page_size)
            logs.extend(site_logs)
        except Exception:
            logger.warning("Failed to fetch logs for site %s", site, exc_info=True)
    return logs


def sync_embedding_statuses(
    jira_embedding_db: JiraIssueEmbeddingDB,
    embedding_docs: List[Dict[str, Any]],
    jira_by_key: Dict[str, JiraIssueDetails],
) -> None:

    index_name = jira_embedding_db.get_current_index_name()
    # Update Jira issues that are not in the Jira embedding db
    for key in jira_by_key.keys():
        jira_issue: JiraIssueDetails = jira_by_key[key]
        result = jira_embedding_db.find_jira_issue_by_key(key)
        # Compare result and jira_issue, if not match, update the jira_embedding_db
        try:
            if not result:
                try:
                    jira_embedding_db.add_jira_issue_from_jira_issue_detail(jira_issue)
                except Exception:
                    logger.warning("Failed to add Jira issue %s into embedding DB", key, exc_info=True)
                continue

            # Prepare payload of fields to sync from Jira into embedding DB
            payload: Dict[str, Any] = {}

            desired_status = jira_issue.status or "Unknown"
            current_status = result.get("status") or "Unknown"
            if desired_status != current_status:
                payload["status"] = desired_status

            if jira_issue.log_group and result.get("log_group") != jira_issue.log_group:
                payload["log_group"] = jira_issue.log_group

            if jira_issue.parent_issue_key and result.get("parent_issue_key", None) != jira_issue.parent_issue_key:
                payload["parent_issue_key"] = jira_issue.parent_issue_key

            desired_is_parent = True if not jira_issue.parent_issue_key else False
            current_is_parent = True if not result.get("parent_issue_key") else False
            if desired_is_parent != current_is_parent:
                payload["is_parent"] = desired_is_parent

            # Keep summary aligned if Jira summary changed
            if getattr(jira_issue, "summary", None) and result.get("summary") != jira_issue.summary:
                payload["summary"] = jira_issue.summary

            # Update document if there are changes
            if payload:
                doc_id = result.get("doc_id") or result.get("key")
                if doc_id:
                    try:
                        jira_embedding_db.opensearch_connect.update(index=index_name, id=doc_id, body={"doc": payload})
                    except Exception:
                        logger.warning("Failed to update embedding doc for %s", key, exc_info=True)
        except Exception:
            logger.warning("Failed to sync Jira issue %s into embedding DB", key, exc_info=True)

    for doc in embedding_docs:
        payload = {}
        key = doc.get("key")
        if not key or key not in jira_by_key:
            continue
        jira_issue_detail = jira_by_key[key]
        jira_status = jira_issue_detail.status or "Unknown"
        current_status = doc.get("status") or "Unknown"

        if jira_status != current_status:
            payload["status"] = jira_status
        if jira_issue_detail.log_group and doc.get("log_group") != jira_issue_detail.log_group:
            payload["log_group"] = jira_issue_detail.log_group
        if (
            jira_issue_detail.parent_issue_key
            and doc.get("parent_issue_key", None) != jira_issue_detail.parent_issue_key
        ):
            payload["parent_issue_key"] = jira_issue_detail.parent_issue_key
        is_parent = True if not jira_issue_detail.parent_issue_key else False
        is_parent_doc = True if not doc.get("parent_issue_key") else False
        if is_parent != is_parent_doc:
            payload["is_parent"] = is_parent

        if len(payload.keys()) > 0:
            try:
                doc_id = doc.get("doc_id")
                jira_embedding_db.opensearch_connect.update(index=index_name, id=doc_id, body={"doc": {**payload}})
                doc.update(payload)
            except Exception:
                logger.warning("Failed to update status/log_group for %s", key, exc_info=True)


def merge_orphan_embedding_docs(
    error_log_db: OpenSearchClient,
    jira_embedding_db: JiraIssueEmbeddingDB,
    embedding_docs: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> None:
    for doc in embedding_docs:
        if doc.get("key"):
            continue
        _merge_single_orphan(error_log_db, jira_embedding_db, doc, now=now)


def _merge_single_orphan(
    error_log_db: OpenSearchClient,
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
    update_error_log_with_jira_reference(error_log_db, site, orphan.get("doc_id"), match["doc_id"])
    # target_doc_id = match["doc_id"]
    # occurrences = orphan.get("occurrence_list", []) or []
    # for occ in occurrences:
    #     try:
    #         ts = occ.get("timestamp")
    #         if ts:
    #             jira_embedding_db.add_occurrence(
    #                 source_doc_id=target_doc_id,
    #                 doc_id=occ.get("doc_id", f"merged-{orphan.get('doc_id')}-{ts}"),
    #                 timestamp=ts,
    #             )
    #     except Exception:
    #         logger.warning("Failed to merge occurrence into %s", target_doc_id, exc_info=True)

    try:
        jira_embedding_db.delete_issue(orphan.get("doc_id"))
    except Exception:
        logger.warning("Failed to delete orphan embedding doc %s", orphan.get("doc_id"), exc_info=True)


def update_error_log_with_jira_reference(
    error_log_db: OpenSearchClient,
    site: str,
    old_jira_reference: str,
    new_jira_reference: str,
) -> int:
    """
    Find all error logs with old_jira_reference and update them with new_jira_reference.

    This function searches across all error log indices (prod and stage) for error logs
    that have the specified old Jira reference and updates them to use the new reference.
    This is useful when Jira issues are merged, renamed, or when correcting incorrect
    references in the error log database.

    Args:
        error_log_db: OpenSearch client for error logs
        old_jira_reference: The old Jira reference to find and replace
        new_jira_reference: The new Jira reference to set

    Returns:
        Number of error logs updated

    Example:
        >>> from error_log_monitor.opensearch_client import OpenSearchClient
        >>> from error_log_monitor.config import load_config
        >>>
        >>> config = load_config()
        >>> error_log_db = OpenSearchClient(config.opensearch)
        >>>
        >>> # Update all error logs from old reference to new reference
        >>> updated_count = update_error_log_with_jira_reference(
        ...     error_log_db=error_log_db,
        ...     old_jira_reference="old-jira-key-123",
        ...     new_jira_reference="new-jira-key-456"
        ... )
        >>> print(f"Updated {updated_count} error logs")
    """
    if not old_jira_reference or not new_jira_reference:
        logger.warning("Both old_jira_reference and new_jira_reference must be provided")
        return 0

    updated_count = 0
    try:
        # Build query to find error logs with the old Jira reference
        query = {
            "query": {"term": {"jira_reference": old_jira_reference}},
            "size": 1000,  # Process in batches
            "_source": ["message_id", "jira_reference"],  # Only fetch necessary fields
        }

        # Search for error logs with the old reference
        response = error_log_db.client.search(
            index=f"error-logs-{site}*", body=query, scroll="2m"  # Search all error log indices for this site
        )

        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])

        # Process all hits
        while hits:
            # Update each error log found
            for hit in hits:
                doc_id = hit["_id"]
                index_name = hit["_index"]

                try:
                    # Update the error log with the new Jira reference
                    update_response = error_log_db.client.update(
                        index=index_name, id=doc_id, body={"doc": {"jira_reference": new_jira_reference}}
                    )

                    if update_response.get("result") in ["updated"]:
                        updated_count += 1
                        logger.debug(
                            "Updated error log %s from %s to %s", doc_id, old_jira_reference, new_jira_reference
                        )
                    else:
                        logger.warning("Unexpected update result for %s: %s", doc_id, update_response.get("result"))

                except Exception as e:
                    logger.error("Failed to update error log %s: %s", doc_id, str(e), exc_info=True)

            # Get next batch if there's a scroll ID
            if not scroll_id:
                break

            response = error_log_db.client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])

        # Clear scroll context
        if scroll_id:
            try:
                error_log_db.client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                logger.warning("Failed to clear scroll context for site %s", site)

    except Exception as e:
        logger.error("Failed to search/update error logs for site %s: %s", site, str(e), exc_info=True)

    logger.info(
        "Updated %d error logs from Jira reference %s to %s", updated_count, old_jira_reference, new_jira_reference
    )
    return updated_count


def find_error_logs_by_jira_reference(
    error_log_db: OpenSearchClient,
    jira_reference: str,
    site: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """
    Find error logs that have a specific Jira reference.

    This function searches across error log indices for logs that have the specified
    Jira reference. Useful for debugging, verification, or analysis of error logs
    associated with specific Jira issues.

    Args:
        error_log_db: OpenSearch client for error logs
        jira_reference: The Jira reference to search for
        site: Site to search in (defaults to ["prod", "stage"])
        limit: Maximum number of results to return

    Returns:
        List of error log documents with the specified Jira reference.
        Each document includes the original fields plus '_id' and '_index'.

    Example:
        >>> from error_log_monitor.opensearch_client import OpenSearchClient
        >>> from error_log_monitor.config import load_config
        >>>
        >>> config = load_config()
        >>> error_log_db = OpenSearchClient(config.opensearch)
        >>>
        >>> # Find error logs with a specific Jira reference
        >>> logs = find_error_logs_by_jira_reference(
        ...     error_log_db=error_log_db,
        ...     jira_reference="JIRA-123",
        ...     limit=50
        ... )
        >>> print(f"Found {len(logs)} error logs for JIRA-123")
    """
    if not jira_reference:
        logger.warning("jira_reference must be provided")
        return []

    site_set: Set[str] = set(filter(None, site or []))
    if not site_set:
        site_set = {"prod", "stage"}

    results: List[Dict] = []
    sites_to_search: List[str] = [site] if site else ["prod", "stage"]

    for s in sites_to_search:
        try:
            # Build query to find error logs with the Jira reference
            query = {
                "query": {"term": {"jira_reference": jira_reference}},
                "size": limit,
                "_source": True,  # Return all fields
            }

            # Search for error logs with the reference
            response = error_log_db.client.search(index=f"error-logs-{s}*", body=query)

            hits = response.get("hits", {}).get("hits", [])
            for hit in hits:
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                doc["_index"] = hit["_index"]
                results.append(doc)

        except Exception as e:
            logger.error("Failed to search error logs for site %s: %s", s, str(e), exc_info=True)

    return results


def update_embedding_with_error_logs(
    jira_embedding_db: JiraIssueEmbeddingDB,
    error_log_db: OpenSearchClient,
    embedding_service: EmbeddingService,
    logs: Sequence[ErrorLog],
    similarity_threshold: float = 0.85,
) -> None:
    # Step 1: Precompute all embeddings
    enriched: List[Dict[str, Any]] = []
    for log in logs:
        if log.jira_reference:
            # skip processed logs
            continue

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
            source_doc_id = match["doc_id"]
            for log in cluster["members"]:
                index = log.index_name
                try:
                    # Skip update occurrence
                    # jira_embedding_db.add_occurrence(
                    #     source_doc_id=source_doc_id,
                    #     doc_id=log.message_id,
                    #     timestamp=log.timestamp.isoformat(),
                    # )
                    resoponse = error_log_db.client.update(
                        index=index, id=log.message_id, body={"doc": {"jira_reference": source_doc_id}}
                    )
                except Exception:
                    logger.warning("Failed to add occurrence for %s", match.get("key"), exc_info=True)
            # try:
            #     jira_embedding_db.remove_duplicate_occurrences(match["doc_id"])
            # except Exception:
            #     logger.warning("Failed to dedupe occurrences for %s", match.get("key"), exc_info=True)
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
                            # jira_embedding_db.add_occurrence(
                            #     source_doc_id=source_doc_id,
                            #     doc_id=log.message_id,
                            #     timestamp=log.timestamp.isoformat(),
                            # )
                            response = error_log_db.client.update(
                                index=log.index_name, id=log.message_id, body={"doc": {"jira_reference": source_doc_id}}
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
