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
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB
from error_log_monitor.jira_helper import JiraHelper
from error_log_monitor.jira_cloud_client import JiraCloudClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_jira_issues(
    jira_cloud_client: JiraCloudClient, project_key: str = None, max_results: int = 1000, page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Get all Jira issues from Jira Cloud API and format them for the embedding database.

    Args:
        jira_cloud_client: JiraCloudClient instance
        project_key: Specific project key to search (if None, searches all accessible projects)
        max_results: Maximum number of issues to return
        page_size: Number of issues to fetch per page (default: 100)

    Returns:
        List of formatted Jira issues
    """
    try:
        logger.info(f"Fetching Jira issues from API (project: {project_key or 'all'}, max: {max_results})")

        # Get issues from Jira Cloud API
        jira_issue_details = jira_cloud_client.get_all_issues(
            project_key=project_key, max_results=max_results, page_size=page_size
        )

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

            formatted_issue = {
                "key": issue_detail.issue_key,
                "summary": issue_detail.summary or "",
                "description": issue_detail.description or "",
                "status": issue_detail.status or "Unknown",
                "created": issue_detail.created or "",
                "updated": issue_detail.updated or "",
                "site": site,
                "parent_issue_key": issue_detail.parent_issue_key or "",
                "error_message": error_message or "",
                "error_type": error_type or "",
                "traceback": traceback or "",
                "request_id": issue_detail.request_id or f"jira-{issue_detail.issue_key}",
            }
            formatted_issues.append(formatted_issue)

        logger.info(f"Successfully formatted {len(formatted_issues)} Jira issues")
        return formatted_issues

    except Exception as e:
        logger.error(f"Failed to fetch Jira issues: {e}")
        return []


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
    max_issues: int = 1000,
    page_size: int = 100,
):
    """
    Initialize the Jira Issue Embedding Database.

    Args:
        dry_run: If True, run in preview mode without making changes
        sites: List of sites to process (default: ['prod', 'stage'])
        months: Number of months of error logs to process (default: 6)
        project_key: Specific Jira project key to fetch (default: from config)
        max_issues: Maximum number of Jira issues to fetch (default: 1000)
        page_size: Number of issues to fetch per page (default: 100)
    """
    if sites is None:
        sites = ['prod', 'stage']

    try:
        logger.info("🚀 Starting Jira Issue Embedding Database initialization")

        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config()

        # Initialize clients
        logger.info("🔌 Initializing clients...")
        embedding_service = EmbeddingService(model_name=config.vector_db.embedding_model)
        jira_cloud_client = JiraCloudClient(config.jira)

        # Initialize embedding database
        jira_embedding_db = JiraIssueEmbeddingDB(embedding_service=embedding_service, config=config)

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
                max_results=max_issues,
                page_size=page_size,
            )

            if not jira_issues:
                logger.error("❌ No Jira issues found")
                raise ValueError("No Jira issues found. Please check your project key and permissions.")

            # Create empty error logs for now (could be enhanced to fetch from OpenSearch)
            error_logs_by_site = {"prod": [], "stage": []}

        except Exception as e:
            logger.error(f"❌ Failed to fetch Jira issues: {e}")
            raise

        logger.info(f"   - {len(jira_issues)} Jira issues")
        total_error_logs = sum(len(logs) for logs in error_logs_by_site.values())
        logger.info(f"   - {total_error_logs} error logs across {len(error_logs_by_site)} sites")

        if dry_run:
            logger.info("🔍 DRY RUN: Would process the following data:")
            logger.info(f"   - {len(jira_issues)} Jira issues")
            logger.info(f"   - {total_error_logs} error logs across {len(error_logs_by_site)} sites")
            logger.info("   - Sites: " + ", ".join(sites))
            logger.info(f"   - Time period: {months} months")
            return

        # Initialize database
        logger.info("🔄 Initializing embedding database...")
        results = jira_embedding_db.initialize_database(jira_issues, error_logs_by_site)

        # Display results
        logger.info("\n📊 Initialization Results:")
        logger.info(f"   Status: {results['status']}")
        logger.info(f"   Jira Issues Processed: {results['jira_issues_processed']}")
        logger.info(f"   Error Logs Processed: {results['error_logs_processed']}")
        logger.info(f"   Similar Issues Found: {results['similar_issues_found']}")
        logger.info(f"   New Issues Created: {results['new_issues_created']}")
        logger.info(f"   Occurrences Added: {results['occurrences_added']}")

        if results.get('errors'):
            logger.warning(f"   Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                logger.warning(f"     - {error}")
            if len(results['errors']) > 5:
                logger.warning(f"     ... and {len(results['errors']) - 5} more errors")

        logger.info("✅ Database initialization completed successfully!")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Jira Issue Embedding Database")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    parser.add_argument("--sites", type=str, default="prod,stage", help="Comma-separated list of sites")
    parser.add_argument("--months", type=int, default=12, help="Number of months of error logs to process")
    parser.add_argument("--project-key", type=str, help="Specific Jira project key to fetch (default: from config)")
    parser.add_argument(
        "--max-issues", type=int, default=1000, help="Maximum number of Jira issues to fetch (default: 1000)"
    )
    parser.add_argument("--page-size", type=int, default=100, help="Number of issues to fetch per page (default: 100)")

    args = parser.parse_args()

    sites = [site.strip() for site in args.sites.split(",")]

    try:
        initialize_database(
            dry_run=args.dry_run,
            sites=sites,
            months=args.months,
            project_key=args.project_key,
            max_issues=args.max_issues,
            page_size=args.page_size,
        )
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
