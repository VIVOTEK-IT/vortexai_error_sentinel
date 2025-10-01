#!/usr/bin/env python3
"""
Simple Jira Issue Embedding Database Initialization Script

This script provides a simple interface to initialize the Jira Issue Embedding Database.
It fetches Jira issues and error logs, then processes them through the initialization workflow.

Usage:
    python scripts/init_jira_db.py [--dry-run] [--sites prod,stage] [--months 6] \\
        [--project-key PROJECT] [--max-issues 1000] [--page-size 100]

    Examples:
    # Dry run with Jira API (default)
    python scripts/init_jira_db.py --dry-run

    # Test with specific project
    python scripts/init_jira_db.py --project-key VEL --dry-run

    # Fetch specific project with limit and custom page size
    python scripts/init_jira_db.py --project-key VEL --max-issues 500 --page-size 50 --dry-run

    # Docker usage
    docker-compose exec error-monitor python /app/scripts/init_jira_db.py --dry-run
"""

import sys
import os
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient, ErrorLog
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB, JiraIssueData
from error_log_monitor.jira_cloud_client import JiraCloudClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_jira_issues(
    jira_cloud_client: JiraCloudClient, project_key: str = None, page_size: int = 100
) -> List[JiraIssueData]:
    """
    Get all Jira issues from Jira Cloud API and format them for the embedding database.

    Args:
        jira_cloud_client: JiraCloudClient instance
        project_key: Specific project key to search (if None, searches all accessible projects)
        page_size: Number of issues to fetch per page (default: 100)

    Returns:
        List of JiraIssueData objects
    """
    try:
        logger.info(f"Fetching Jira issues from API (project: {project_key or 'all'})")

        # Get issues from Jira Cloud API
        jira_issue_details = jira_cloud_client.get_all_issues(project_key=project_key, page_size=page_size)

        if not jira_issue_details:
            logger.warning("No Jira issues found")
            return []

        # Format issues for embedding database
        formatted_issues = []
        for issue_detail in jira_issue_details:
            # Use site from Jira issue or determine from issue key
            site = issue_detail.site or _determine_site_from_issue_key(issue_detail.issue_key)

            # Use error information directly from Jira issue or extract from description
            error_message = issue_detail.error_message
            error_type = issue_detail.error_type
            traceback = issue_detail.traceback

            # If no error info from Jira fields, try to extract from description
            if not error_message or not error_type:
                extracted_error_message, extracted_error_type, extracted_traceback = _extract_error_info(issue_detail)
                error_message = error_message or extracted_error_message
                error_type = error_type or extracted_error_type
                traceback = traceback or extracted_traceback
            is_parent = True
            if (
                issue_detail.parent_issue_key and len(issue_detail.parent_issue_key) > 3
            ) or issue_detail.status == "SUB ISSUES":
                is_parent = False
            formatted_issue = JiraIssueData(
                key=issue_detail.issue_key,
                summary=issue_detail.summary or "",
                description=issue_detail.description or "",
                status=issue_detail.status or "Unknown",
                created=issue_detail.created or "",
                updated=issue_detail.updated or "",
                site=site,
                parent_issue_key=issue_detail.parent_issue_key or "",
                error_message=error_message or "",
                error_type=error_type or "",
                traceback=traceback or "",
                request_id=issue_detail.request_id or f"jira-{issue_detail.issue_key}",
                log_group=issue_detail.log_group or "",
                count=issue_detail.count or 1,
                is_parent=is_parent,
                not_commit_to_jira=False,  # Default to False for existing Jira issues
            )
            formatted_issues.append(formatted_issue)

        logger.info(f"Successfully formatted {len(formatted_issues)} Jira issues")
        return formatted_issues

    except Exception as e:
        logger.error(f"Failed to fetch Jira issues: {e}")
        return []


def get_all_error_logs(
    opensearch_client: OpenSearchClient, sites: List[str], months: int = 6
) -> Dict[str, List[ErrorLog]]:
    """
    Get all error logs from OpenSearch organized by site using scroll API for large datasets.

    Args:
        opensearch_client: OpenSearch client
        sites: List of sites to fetch logs for
        months: Number of months of logs to fetch

    Returns:
        Dictionary mapping site names to lists of error log data
    """
    logger.info("📊 Fetching error logs from OpenSearch using scroll API...")

    try:
        error_logs_by_site = {}

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months * 30)

        logger.info(f"   - Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   - Sites: {', '.join(sites)}")

        for site in sites:
            logger.info(f"   - Fetching logs for site: {site}")

            try:
                # Search for error logs in the specified date range for this site
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"site": site}},
                                {"range": {"timestamp": {"gte": start_date.isoformat(), "lte": end_date.isoformat()}}},
                            ]
                        }
                    },
                    "size": 1000,  # Batch size for scroll
                    "sort": [{"timestamp": {"order": "asc"}}],
                }

                # Initialize scroll search
                response = opensearch_client.client.search(
                    index="error_log_*", body=query, scroll="5m"  # Keep scroll context alive for 5 minutes
                )

                # Extract scroll ID and total hits
                scroll_id = response.get('_scroll_id')
                total_hits = response['hits']['total']['value']
                logger.info(f"   - Found {total_hits} total error logs for {site}")

                # Extract error logs from initial response
                error_logs = []
                for hit in response['hits']['hits']:
                    source = hit['_source']

                    # Parse timestamp
                    timestamp_str = source.get('timestamp', '')
                    try:
                        if timestamp_str.endswith('Z'):
                            timestamp = datetime.fromisoformat(timestamp_str[:-1]).replace(tzinfo=timezone.utc)
                        else:
                            timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)

                    # Calculate hashes
                    error_message = source.get('error_message', '')
                    error_message_hash = (
                        int(hashlib.md5(error_message.encode('utf-8')).hexdigest()[:8], 16) if error_message else 0
                    )

                    traceback = source.get('traceback', '')
                    traceback_hash = (
                        int(hashlib.md5(traceback.encode('utf-8')).hexdigest()[:8], 16) if traceback else None
                    )

                    error_type = source.get('error_type', '')
                    error_type_hash = (
                        int(hashlib.md5(error_type.encode('utf-8')).hexdigest()[:8], 16) if error_type else None
                    )

                    # Compose service name from log_group and module_name
                    log_group = source.get('log_group', '')
                    module_name = source.get('module_name', '')

                    if log_group and module_name:
                        service = f"{log_group}.{module_name}"
                    elif log_group:
                        service = log_group
                    elif module_name:
                        service = module_name
                    else:
                        service = 'unknown'

                    error_log = ErrorLog(
                        message_id=hit['_id'],
                        timestamp=timestamp,
                        error_message=error_message,
                        error_message_hash=error_message_hash,
                        traceback=traceback,
                        traceback_hash=traceback_hash,
                        error_type=error_type,
                        error_type_hash=error_type_hash,
                        site=source.get('site', site),
                        service=service,
                        index_name=hit.get('_index'),
                        topic=source.get('topic'),
                        count=source.get('count', 1),
                        request_id=source.get('request_id'),
                        category=source.get('category'),
                        log_group=log_group,
                        module_name=module_name,
                        version=source.get('version'),
                    )
                    error_logs.append(error_log)

                # Continue scrolling through remaining results
                while scroll_id and len(response['hits']['hits']) > 0:
                    try:
                        response = opensearch_client.client.scroll(scroll_id=scroll_id, scroll="5m")
                        # Extract error logs from current batch
                        for hit in response['hits']['hits']:
                            source = hit['_source']

                            # Parse timestamp
                            timestamp_str = source.get('timestamp', '')
                            try:
                                if timestamp_str.endswith('Z'):
                                    timestamp = datetime.fromisoformat(timestamp_str[:-1]).replace(tzinfo=timezone.utc)
                                else:
                                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                            except ValueError:
                                timestamp = datetime.now(timezone.utc)

                            # Calculate hashes
                            error_message = source.get('error_message', '')
                            error_message_hash = (
                                int(hashlib.md5(error_message.encode('utf-8')).hexdigest()[:8], 16)
                                if error_message
                                else 0
                            )

                            traceback = source.get('traceback', '')
                            traceback_hash = (
                                int(hashlib.md5(traceback.encode('utf-8')).hexdigest()[:8], 16) if traceback else None
                            )

                            error_type = source.get('error_type', '')
                            error_type_hash = (
                                int(hashlib.md5(error_type.encode('utf-8')).hexdigest()[:8], 16) if error_type else None
                            )

                            # Compose service name from log_group and module_name
                            log_group = source.get('log_group', '')
                            module_name = source.get('module_name', '')

                            if log_group and module_name:
                                service = f"{log_group}.{module_name}"
                            elif log_group:
                                service = log_group
                            elif module_name:
                                service = module_name
                            else:
                                service = 'unknown'

                            error_log = ErrorLog(
                                message_id=hit['_id'],
                                timestamp=timestamp,
                                error_message=error_message,
                                error_message_hash=error_message_hash,
                                traceback=traceback,
                                traceback_hash=traceback_hash,
                                error_type=error_type,
                                error_type_hash=error_type_hash,
                                site=source.get('site', site),
                                service=service,
                                index_name=hit.get('_index'),
                                topic=source.get('topic'),
                                count=source.get('count', 1),
                                request_id=source.get('request_id'),
                                category=source.get('category'),
                                log_group=log_group,
                                module_name=module_name,
                                version=source.get('version'),
                            )
                            error_logs.append(error_log)

                        # Update scroll ID for next iteration
                        scroll_id = response.get('_scroll_id')
                        # Log progress every 1000 records
                        if len(error_logs) % 1000 == 0:
                            logger.info(f"   - Processed {len(error_logs)}/{total_hits} error logs for {site}")

                    except Exception as scroll_error:
                        logger.warning(f"⚠️  Scroll error for site {site}: {scroll_error}")
                        break

                # Clear scroll context
                if scroll_id:
                    try:
                        opensearch_client.client.clear_scroll(scroll_id=scroll_id)
                    except Exception as clear_error:
                        logger.warning(f"⚠️  Failed to clear scroll context: {clear_error}")

                error_logs_by_site[site] = error_logs
                logger.info(f"   - Fetched {len(error_logs)} error logs for {site}")

            except Exception as e:
                logger.warning(f"⚠️  Failed to fetch logs for site {site}: {e}")
                error_logs_by_site[site] = []

        total_logs = sum(len(logs) for logs in error_logs_by_site.values())
        logger.info(f"✅ Fetched {total_logs} total error logs across {len(sites)} sites")

        return error_logs_by_site

    except Exception as e:
        logger.error(f"❌ Failed to fetch error logs: {e}")
        raise


def _determine_site_from_issue_key(issue_key: str) -> str:
    """
    Determine site from Jira issue key.

    Args:
        issue_key: Jira issue key (e.g., "VEL-123", "PROD-456")

    Returns:
        Site name (stage/prod)
    """
    # Simple logic - you may need to customize this based on your Jira setup
    if 'STAGE' in issue_key.upper() or 'TEST' in issue_key.upper() or 'DEV' in issue_key.upper():
        return 'stage'
    else:
        return 'prod'


def _extract_error_info(issue_detail) -> tuple[str, str, str]:
    """
    Extract error information from Jira issue details.

    Args:
        issue_detail: JiraIssueDetails object

    Returns:
        Tuple of (error_message, error_type, traceback)
    """
    error_message = ""
    error_type = ""
    traceback = ""

    # Try to extract error information from description
    description = issue_detail.description or ""
    summary = issue_detail.summary or ""

    # Look for common error patterns in description
    if "error" in description.lower() or "exception" in description.lower():
        # Extract error message (first line that looks like an error)
        lines = description.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'timeout']):
                error_message = line.strip()
                break

        # Look for traceback
        if "traceback" in description.lower():
            traceback_start = description.lower().find("traceback")
            if traceback_start != -1:
                traceback = description[traceback_start:].strip()

    # If no error info in description, try summary
    if not error_message and any(keyword in summary.lower() for keyword in ['error', 'exception', 'failed']):
        error_message = summary

    # Determine error type from common patterns
    if error_message:
        if "timeout" in error_message.lower():
            error_type = "TimeoutError"
        elif "connection" in error_message.lower():
            error_type = "ConnectionError"
        elif "permission" in error_message.lower() or "access" in error_message.lower():
            error_type = "PermissionError"
        elif "not found" in error_message.lower():
            error_type = "NotFoundError"
        else:
            error_type = "Error"

    return error_message, error_type, traceback


def initialize_database(
    dry_run: bool = False,
    sites: List[str] = None,
    months: int = 6,
    project_key: str = None,
    page_size: int = 100,
    skip_jira: bool = False,
    skip_error_logs: bool = False,
    skip_error_logs_count: dict = None,  # {site: count}
    max_process_error_logs: dict = None,  # {site: count}
):
    """
    Initialize the Jira Issue Embedding Database.

    Args:
        dry_run: If True, run in preview mode without making changes
        sites: List of sites to process (default: ['prod', 'stage'])
        months: Number of months of error logs to process (default: 6)
        project_key: Specific Jira project key to fetch (default: from config)
        page_size: Number of issues to fetch per page (default: 100)
        skip_jira: If True, skip Jira issues (default: False)
        skip_error_logs: If True, skip error logs (default: False)
        skip_error_logs_count: Dictionary of site counts to skip (default: None)
        max_process_error_logs: Dictionary of site counts to max process (default: None)
    """
    config = load_config()
    embedding_service = EmbeddingService(model_name=config.vector_db.embedding_model)
    jira_embedding_db = JiraIssueEmbeddingDB(embedding_service=embedding_service, config=config)
    if not skip_jira:
        try:
            logger.info("🚀 Starting Jira Issue Embedding Database initialization")

            # Load configuration
            logger.info("📋 Loading configuration...")

            # Initialize clients
            logger.info("🔌 Initializing clients...")

            jira_cloud_client = JiraCloudClient(config.jira)

            # Fetch Jira issues
            logger.info("📝 Fetching Jira issues from API...")
            try:
                # Test Jira connection first
                if not jira_cloud_client.test_connection():
                    logger.error("❌ Jira Cloud API connection failed")
                    raise ConnectionError(
                        "Failed to connect to Jira Cloud API. Please check your credentials and network connection."
                    )

                logger.info("✅ Jira Cloud API connection successful")
                jira_issues = get_all_jira_issues(
                    jira_cloud_client=jira_cloud_client,
                    project_key=project_key or config.jira.project_key,
                    page_size=page_size,
                )

                if not jira_issues:
                    logger.error("❌ No Jira issues found")
                    raise ValueError("No Jira issues found. Please check your project key and permissions.")

            except Exception as e:
                logger.error(f"❌ Failed to fetch Jira issues: {e}")
                raise
            if dry_run:
                logger.info("🔍 DRY RUN: Would process the following data:")
                logger.info(f"   - {len(jira_issues)} Jira issues")
                logger.info(f"   - Time period: {months} months")
            else:
                logger.info("🔄 Initializing embedding database...")
                results = jira_embedding_db.initialize_database(jira_issues, None)
        except Exception as e:
            logger.error(f"❌ Failed to fetch Jira issues: {e}")
            raise

        logger.info(f"   Jira Issues Processed: {results['jira_issues_processed']}")
        logger.info(f"   Jira Issues Added: {results.get('jira_issues_added', 0)}")
        logger.info(f"   Jira Issues Skipped: {results.get('jira_issues_skipped', 0)}")

    if not skip_error_logs:
        opensearch_client = OpenSearchClient(config.opensearch)
        if sites is None:
            sites = ['prod', 'stage']
        if not skip_error_logs_count:
            skip_error_logs_count = {site: 0 for site in sites}
        if not max_process_error_logs:
            max_process_error_logs = {site: 1000000 for site in sites}
        # Fetch error logs from OpenSearch
        logger.info("📊 Fetching error logs from OpenSearch...")
        try:
            error_logs_by_site = get_all_error_logs(opensearch_client=opensearch_client, sites=sites, months=months)

        except Exception as e:
            logger.error(f"❌ Failed to fetch error logs: {e}")
            raise

        total_error_logs = sum(len(logs) for logs in error_logs_by_site.values())
        logger.info(f"   - {total_error_logs} error logs across {len(error_logs_by_site)} sites")

        if dry_run:
            logger.info("🔍 DRY RUN: Would process the following data:")
            logger.info(f"   - {total_error_logs} error logs across {len(error_logs_by_site)} sites")
            logger.info("   - Sites: " + ", ".join(sites))
            logger.info(f"   - Time period: {months} months")
        else:
            # Initialize database
            logger.info("🔄 Initializing embedding database...")
            total_static = {
                "error_logs_processed": 0,
                "similar_issues_found": 0,
                "new_issues_created": 0,
                "occurrences_added": 0,
            }
            for site in sites:
                if skip_error_logs_count[site] > 0:
                    max_size = min(max_process_error_logs[site], len(error_logs_by_site[site]))
                    current_error_logs = error_logs_by_site[site][skip_error_logs_count[site] : max_size]
                    skip_error_logs_count[site] = 0
                else:
                    current_error_logs = error_logs_by_site[site]
                for error_log_page in range(0, min(max_process_error_logs[site], len(current_error_logs)), page_size):
                    logger.warning(
                        f"   - Processing error logs for {site} page {error_log_page // page_size + 1} of {min(max_process_error_logs[site], len(current_error_logs)) // page_size + 1}"
                    )
                    current_page = current_error_logs[error_log_page : error_log_page + page_size]
                    results = jira_embedding_db.initialize_database(
                        None,
                        current_page,
                        skip_error_logs_count[site] + error_log_page,
                        min(max_process_error_logs[site], len(current_error_logs)),
                    )
                    logger.info("\n📊 Initialization Results:")
                    logger.info(f"   Status: {results['status']}")
                    logger.info(f"   - {results['error_logs_processed']} error logs for {site}")
                    total_static["error_logs_processed"] += results["error_logs_processed"]
                    logger.info(f"   - {results['similar_issues_found']} similar issues found for {site}")
                    total_static["similar_issues_found"] += results["similar_issues_found"]
                    logger.info(f"   - {results['new_issues_created']} new issues created for {site}")
                    total_static["new_issues_created"] += results["new_issues_created"]
                    logger.info(f"   - {results['occurrences_added']} occurrences added for {site}")
                    total_static["occurrences_added"] += results["occurrences_added"]

            logger.info(f"   Total error logs processed: {total_static['error_logs_processed']}")
            logger.info(f"   Total similar issues found: {total_static['similar_issues_found']}")
            logger.info(f"   Total new issues created: {total_static['new_issues_created']}")
            logger.info(f"   Total occurrences added: {total_static['occurrences_added']}")

            # Display results
            if results.get('errors'):
                logger.warning(f"   Errors: {len(results['errors'])}")
                for error in results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"     - {error}")
                if len(results['errors']) > 5:
                    logger.warning(f"     ... and {len(results['errors']) - 5} more errors")

    logger.info("✅ Database initialization completed successfully!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Jira Issue Embedding Database")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    parser.add_argument("--sites", type=str, default="prod,stage", help="Comma-separated list of sites")
    parser.add_argument("--months", type=int, default=12, help="Number of months of error logs to process")
    parser.add_argument("--project-key", type=str, help="Specific Jira project key to fetch (default: from config)")
    # parser.add_argument("--skip-jira", action="store_true", default=True, help="Skip Jira issues (default: True)")
    # parser.add_argument("--skip-error-logs", action="store_true", default=True, help="Skip error logs (default: True)")
    # parser.add_argument("--skip-error-logs-count", type=dict, help="Dictionary of site counts to skip (default: None)")
    parser.add_argument("--page-size", type=int, default=100, help="Number of issues to fetch per page (default: 100)")

    args = parser.parse_args()
    sites = ["stage", "prod"]
    skip_error_logs_count = {"stage": 0, "prod": 0}
    max_process_error_logs = {"stage": 1000000, "prod": 1000000}

    try:
        initialize_database(
            dry_run=args.dry_run,
            sites=sites,
            months=48,
            project_key=args.project_key,
            page_size=args.page_size,
            skip_jira=False,
            skip_error_logs=True,
            skip_error_logs_count=skip_error_logs_count,
            max_process_error_logs=max_process_error_logs,
        )
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


# script example
# python scripts/init_jira_db.py --dry-run --sites prod,stage --months 12 --project-key VEL --page-size 100 --skip-jira False --skip-error-logs True --skip-error-logs-count {"prod": 0, "stage": 0}
# python scripts/init_jira_db.py --dry-run --sites prod,stage --months 6 --project-key VEL --page-size 100 --skip-jira False --skip-error-logs False --skip-error-logs-count {"prod": 0, "stage": 0}
# python scripts/init_jira_db.py --dry-run --sites prod,stage --months 6 --project-key VEL --page-size 100 --skip-jira False --skip-error-logs False --skip-error-logs-count {"prod": 100, "stage": 100}
