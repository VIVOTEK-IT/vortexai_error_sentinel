#!/usr/bin/env python3
"""
Jira Issue Embedding Database Initialization Script

This script initializes the Jira Issue Embedding Database by:
1. Fetching Jira issues from the API
2. Retrieving error logs from OpenSearch (past 6 months)
3. Processing both datasets through the initialization workflow
4. Setting up the embedding database for correlation

Usage:
    python scripts/initialize_jira_embedding_db.py [options]

Options:
    --dry-run          Run without making changes (preview only)
    --sites SITES      Comma-separated list of sites to process (default: all)
    --months MONTHS    Number of months of error logs to process (default: 6)
    --batch-size SIZE  Batch size for processing (default: 100)
    --verbose          Enable verbose logging
    --help             Show this help message
"""

import argparse
import logging
import sys
import os
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

logger = logging.getLogger(__name__)


class JiraEmbeddingDBInitializer:
    """Initializes the Jira Issue Embedding Database with real data."""

    def __init__(self, config=None, dry_run: bool = False):
        """
        Initialize the database initializer.

        Args:
            config: System configuration (loads from environment if None)
            dry_run: If True, run in preview mode without making changes
        """
        self.config = config or load_config()
        self.dry_run = dry_run

        # Initialize clients
        self.opensearch_client = OpenSearchClient(self.config.opensearch)
        self.embedding_service = EmbeddingService(model_name=self.config.vector_db.embedding_model)
        self.jira_helper = JiraHelper(self.config.opensearch)  # JiraHelper needs OpenSearchConfig, not JiraConfig
        self.jira_cloud_client = JiraCloudClient(self.config.jira)

        # Initialize embedding database
        self.jira_embedding_db = JiraIssueEmbeddingDB(
            opensearch_client=self.opensearch_client, embedding_service=self.embedding_service, config=self.config
        )

        logger.info(f"Initialized JiraEmbeddingDBInitializer (dry_run={dry_run})")

    def fetch_jira_issues(self, sites: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch Jira issues from the API for specified sites.

        Args:
            sites: List of sites to fetch issues for

        Returns:
            List of Jira issues with required fields
        """
        logger.info(f"Fetching Jira issues for sites: {sites}")

        try:
            # Get all Jira issues from the helper
            all_issues = self.jira_helper.get_all_issues()

            # Filter issues by site and format for embedding database
            formatted_issues = []
            for issue in all_issues:
                # Determine site from issue data (you may need to adjust this logic)
                site = self._determine_site_from_issue(issue)

                if site in sites or not sites:  # Include if site matches or no site filter
                    formatted_issue = {
                        "key": issue.issue_key,
                        "summary": issue.summary or "",
                        "description": issue.description or "",
                        "status": issue.status or "Unknown",
                        "created": issue.created.isoformat() if issue.created else "",
                        "updated": issue.updated.isoformat() if issue.updated else "",
                        "site": site,
                        "parent_issue_key": "",  # Will be determined based on issue structure
                        "error_message": issue.error_message or "",
                        "error_type": issue.error_type or "",
                        "traceback": issue.traceback or "",
                        "request_id": issue.request_id or "",
                    }
                    formatted_issues.append(formatted_issue)

            logger.info(f"Fetched {len(formatted_issues)} Jira issues")
            return formatted_issues

        except Exception as e:
            logger.error(f"Failed to fetch Jira issues: {e}")
            raise

    def _determine_site_from_issue(self, issue) -> str:
        """
        Determine site from Jira issue data.

        Args:
            issue: Jira issue object

        Returns:
            Site name (stage/prod)
        """
        # This is a simplified implementation - you may need to adjust based on your data
        if hasattr(issue, 'site') and issue.site:
            return issue.site

        # Default logic - you may need to customize this
        if 'stage' in issue.issue_key.lower() or 'test' in issue.issue_key.lower():
            return 'stage'
        else:
            return 'prod'

    def fetch_error_logs(self, sites: List[str], months: int = 6) -> Dict[str, List[Any]]:
        """
        Fetch error logs from OpenSearch for specified sites and time period.

        Args:
            sites: List of sites to fetch logs for
            months: Number of months of logs to fetch

        Returns:
            Dictionary mapping site names to lists of ErrorLog objects
        """
        logger.info(f"Fetching error logs for sites: {sites} (last {months} months)")

        try:
            error_logs_by_site = {}
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=months * 30)

            for site in sites:
                logger.info(f"Fetching error logs for site: {site}")

                # Query OpenSearch for error logs
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"site": site}},
                                {"range": {"timestamp": {"gte": start_date.isoformat(), "lte": end_date.isoformat()}}},
                            ]
                        }
                    },
                    "size": 10000,  # Adjust based on your data volume
                    "sort": [{"timestamp": {"order": "desc"}}],
                }

                # Search across all error log indices
                indices = self._get_error_log_indices()
                response = self.opensearch_client.client.search(index=",".join(indices), body=query)

                error_logs = []
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    error_log = self._create_error_log_from_source(source)
                    if error_log:
                        error_logs.append(error_log)

                error_logs_by_site[site] = error_logs
                logger.info(f"Fetched {len(error_logs)} error logs for site: {site}")

            total_logs = sum(len(logs) for logs in error_logs_by_site.values())
            logger.info(f"Total error logs fetched: {total_logs}")
            return error_logs_by_site

        except Exception as e:
            logger.error(f"Failed to fetch error logs: {e}")
            raise

    def _get_error_log_indices(self) -> List[str]:
        """
        Get list of error log indices to search.

        Returns:
            List of index names
        """
        try:
            # Get all indices that match the error log pattern
            indices = self.opensearch_client.client.cat.indices(index="vortex_ecs_execution_log_*", format="json")

            index_names = [idx["index"] for idx in indices]
            logger.info(f"Found {len(index_names)} error log indices")
            return index_names

        except Exception as e:
            logger.warning(f"Failed to get error log indices: {e}")
            # Fallback to common index names
            return ["vortex_ecs_execution_log_prod_*", "vortex_ecs_execution_log_stage_*"]

    def _create_error_log_from_source(self, source: Dict[str, Any]) -> Optional[Any]:
        """
        Create ErrorLog object from OpenSearch source data.

        Args:
            source: OpenSearch document source

        Returns:
            ErrorLog object or None if creation fails
        """
        try:
            from error_log_monitor.opensearch_client import ErrorLog

            # Parse timestamp
            timestamp_str = source.get("timestamp", "")
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
            else:
                timestamp = datetime.now(timezone.utc)

            return ErrorLog(
                message_id=source.get("message_id", ""),
                timestamp=timestamp,
                error_message=source.get("error_message", ""),
                error_type=source.get("error_type", ""),
                traceback=source.get("traceback", ""),
                site=source.get("site", ""),
                service=source.get("service", ""),
                request_id=source.get("request_id", ""),
                category=source.get("category", ""),
            )

        except Exception as e:
            logger.warning(f"Failed to create ErrorLog from source: {e}")
            return None

    def initialize_database(self, sites: List[str], months: int = 6) -> Dict[str, Any]:
        """
        Initialize the Jira Issue Embedding Database.

        Args:
            sites: List of sites to process
            months: Number of months of error logs to process

        Returns:
            Initialization results
        """
        logger.info("Starting database initialization")

        try:
            # Fetch data
            logger.info("Step 1: Fetching Jira issues...")
            jira_issues = self.fetch_jira_issues(sites)

            logger.info("Step 2: Fetching error logs...")
            error_logs_by_site = self.fetch_error_logs(sites, months)

            if self.dry_run:
                logger.info("DRY RUN: Would process the following data:")
                logger.info(f"  - {len(jira_issues)} Jira issues")
                total_logs = sum(len(logs) for logs in error_logs_by_site.values())
                logger.info(f"  - {total_logs} error logs across {len(error_logs_by_site)} sites")

                return {
                    "status": "dry_run",
                    "jira_issues_count": len(jira_issues),
                    "error_logs_count": total_logs,
                    "sites": sites,
                    "months": months,
                }

            # Initialize database
            logger.info("Step 3: Initializing embedding database...")
            results = self.jira_embedding_db.initialize_database(jira_issues, error_logs_by_site)

            logger.info("Database initialization completed")
            return results

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def print_results(self, results: Dict[str, Any]):
        """
        Print initialization results in a formatted way.

        Args:
            results: Initialization results
        """
        print("\n" + "=" * 60)
        print("JIRA ISSUE EMBEDDING DATABASE INITIALIZATION RESULTS")
        print("=" * 60)

        if results.get("status") == "dry_run":
            print(f"Mode: DRY RUN (no changes made)")
            print(f"Jira Issues: {results['jira_issues_count']}")
            print(f"Error Logs: {results['error_logs_count']}")
            print(f"Sites: {', '.join(results['sites'])}")
            print(f"Time Period: {results['months']} months")
            return

        print(f"Status: {results['status']}")
        print(f"Timestamp: {results['timestamp']}")
        print()

        print("📊 Processing Statistics:")
        print(f"  Jira Issues Processed: {results['jira_issues_processed']}")
        print(f"  Error Logs Processed: {results['error_logs_processed']}")
        print(f"  Similar Issues Found: {results['similar_issues_found']}")
        print(f"  New Issues Created: {results['new_issues_created']}")
        print(f"  Occurrences Added: {results['occurrences_added']}")

        if results.get('errors'):
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")

        print("\n📈 Site Statistics:")
        for site, stats in results.get('site_stats', {}).items():
            print(f"  {site.upper()}:")
            print(f"    Error Logs: {stats['error_logs_processed']}")
            print(f"    Similar Issues: {stats['similar_issues_found']}")
            print(f"    New Issues: {stats['new_issues_created']}")
            print(f"    Occurrences: {stats['occurrences_added']}")
            if stats.get('errors'):
                print(f"    Errors: {len(stats['errors'])}")

        print("\n" + "=" * 60)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point for the initialization script."""
    parser = argparse.ArgumentParser(
        description="Initialize Jira Issue Embedding Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--dry-run", action="store_true", help="Run without making changes (preview only)")

    parser.add_argument("--sites", type=str, default="", help="Comma-separated list of sites to process (default: all)")

    parser.add_argument("--months", type=int, default=6, help="Number of months of error logs to process (default: 6)")

    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    try:
        # Parse sites
        sites = [site.strip() for site in args.sites.split(",")] if args.sites else ["prod", "stage"]

        # Initialize the database initializer
        initializer = JiraEmbeddingDBInitializer(dry_run=args.dry_run)

        # Run initialization
        results = initializer.initialize_database(sites, args.months)

        # Print results
        initializer.print_results(results)

        # Exit with appropriate code
        if results.get("status") == "failed":
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
