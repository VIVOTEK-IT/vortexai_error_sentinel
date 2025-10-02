#!/usr/bin/env python3
"""Synchronize log_group field in Jira Issue Embedding DB with current Jira data."""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from error_log_monitor.config import load_config  # noqa: E402
from error_log_monitor.jira_cloud_client import JiraCloudClient  # noqa: E402
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB  # noqa: E402
from scripts.init_jira_db import get_all_jira_issues  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize log_group field for Jira embeddings")
    parser.add_argument(
        "--project-key",
        type=str,
        help="Override Jira project key. Defaults to config.jira.project_key.",
    )
    parser.add_argument(
        "--default-log-group",
        type=str,
        default="",
        help="Value to use when Jira issue has no log_group (default: empty string)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not perform updates, only report changes",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown arguments: %s", " ".join(unknown))
    return args


def fetch_jira_log_groups(config, project_key: str | None) -> Dict[str, str]:
    jira_client = JiraCloudClient(config.jira)
    jira_issues = get_all_jira_issues(
        jira_cloud_client=jira_client,
        project_key=project_key or config.jira.project_key,
    )

    mapping: Dict[str, str] = {}
    for issue in jira_issues:
        if issue.key:
            mapping[issue.key] = issue.log_group or ""
    logger.info("Fetched %d Jira issues for log_group synchronization", len(mapping))
    return mapping


def iterate_embedding_docs(db: JiraIssueEmbeddingDB, batch_size: int = 500) -> Iterable[Tuple[str, Dict[str, str]]]:
    client = db.opensearch_connect
    index_name = db.get_current_index_name()

    query = {
        "size": batch_size,
        "query": {"match_all": {}},
        "_source": ["key", "log_group"],
    }

    response = client.search(index=index_name, body=query, scroll="2m")
    scroll_id = response.get("_scroll_id")

    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                yield hit.get("_id"), hit.get("_source", {})

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


def main() -> None:
    args = parse_args(sys.argv[1:])
    config = load_config()

    jira_log_groups = fetch_jira_log_groups(config, args.project_key)
    db = JiraIssueEmbeddingDB(embedding_service=None, config=config)

    updates = 0
    total = 0
    missing_in_jira = 0

    logger.info("Scanning embedding database for log_group synchronization")

    for doc_id, source in iterate_embedding_docs(db):
        total += 1
        key = source.get("key")
        if not key:
            continue

        jira_value = jira_log_groups.get(key)
        if jira_value is None:
            missing_in_jira += 1
            continue

        current_value = source.get("log_group") or ""
        desired_value = jira_value or args.default_log_group

        if current_value == desired_value:
            continue

        updates += 1
        logger.info("Updating %s log_group: '%s' -> '%s'", key, current_value, desired_value)

        if args.dry_run:
            continue

        db.opensearch_connect.update(
            index=db.get_current_index_name(),
            id=doc_id,
            body={
                "doc": {
                    "log_group": desired_value,
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            },
        )

    logger.info("Processed %d embedding documents", total)
    logger.info("Documents missing Jira data: %d", missing_in_jira)
    if args.dry_run:
        logger.info("Dry run complete. %d documents would be updated.", updates)
    else:
        logger.info("Updated %d documents.", updates)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        logger.error("Missing dependency: %s", exc)
        raise
