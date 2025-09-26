#!/usr/bin/env python3
"""
Migration utilities for Jira Issue Embedding Database.

This script provides utilities for migrating data between year-based indices,
archiving old data, and managing the lifecycle of embedding databases.
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB

logger = logging.getLogger(__name__)


class JiraEmbeddingMigrator:
    """Utility class for migrating Jira embedding data between years."""

    def __init__(self, config=None):
        """Initialize the migrator with configuration."""
        self.config = config or load_config()
        self.opensearch_client = OpenSearchClient(self.config.opensearch)

        # Initialize embedding service only if API key is available
        self.embedding_service = None
        try:
            self.embedding_service = EmbeddingService(model_name=self.config.vector_db.embedding_model)
        except ValueError as e:
            if "OpenAI API key is required" in str(e):
                print(f"Warning: OpenAI API key not available. Some operations may be limited: {e}")
            else:
                raise

        self.jira_embedding_db = JiraIssueEmbeddingDB(
            opensearch_client=self.opensearch_client, embedding_service=self.embedding_service, config=self.config
        )

    def migrate_year_to_year(self, from_year: int, to_year: int, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate all issues from one year to another.

        Args:
            from_year: Source year
            to_year: Target year
            dry_run: If True, only simulate the migration without making changes

        Returns:
            Migration results
        """
        logger.info(f"Starting migration from {from_year} to {to_year} (dry_run={dry_run})")

        try:
            # Check if source index exists
            from_index = self.jira_embedding_db.get_index_name_for_year(from_year)
            if not self.opensearch_client.client.indices.exists(index=from_index):
                logger.error(f"Source index {from_index} does not exist")
                return {"status": "failed", "error": f"Source index {from_index} does not exist"}

            # Check if target index exists
            to_index = self.jira_embedding_db.get_index_name_for_year(to_year)
            if self.opensearch_client.client.indices.exists(index=to_index):
                if not dry_run:
                    logger.warning(f"Target index {to_index} already exists")
                    return {"status": "failed", "error": f"Target index {to_index} already exists"}
                else:
                    logger.info(f"Target index {to_index} already exists (dry run)")

            # Get source index stats
            source_stats = self.opensearch_client.client.indices.stats(index=from_index)
            doc_count = source_stats["indices"][from_index]["total"]["docs"]["count"]

            logger.info(f"Source index {from_index} contains {doc_count} documents")

            if dry_run:
                logger.info("DRY RUN: Would migrate {} documents from {} to {}".format(doc_count, from_index, to_index))
                return {
                    "status": "dry_run",
                    "from_year": from_year,
                    "to_year": to_year,
                    "total_documents": doc_count,
                    "from_index": from_index,
                    "to_index": to_index,
                }

            # Create target index if it doesn't exist
            if not self.opensearch_client.client.indices.exists(index=to_index):
                logger.info(f"Creating target index {to_index}")
                self.jira_embedding_db.create_index(to_year)

            # Perform migration
            result = self.jira_embedding_db.migrate_old_issues(from_year, to_year)

            logger.info(f"Migration completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {"status": "failed", "error": str(e)}

    def archive_year(self, year: int, archive_location: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Archive a year's data by creating a snapshot.

        Args:
            year: Year to archive
            archive_location: Location to store archive (optional)
            dry_run: If True, only simulate the archiving

        Returns:
            Archive results
        """
        logger.info(f"Archiving year {year} (dry_run={dry_run})")

        try:
            index_name = self.jira_embedding_db.get_index_name_for_year(year)

            if not self.opensearch_client.client.indices.exists(index=index_name):
                logger.error(f"Index {index_name} does not exist")
                return {"status": "failed", "error": f"Index {index_name} does not exist"}

            # Get index stats
            stats = self.opensearch_client.client.indices.stats(index=index_name)
            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
            size_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

            logger.info(f"Index {index_name} contains {doc_count} documents ({size_bytes} bytes)")

            if dry_run:
                logger.info("DRY RUN: Would archive index {} with {} documents".format(index_name, doc_count))
                return {
                    "status": "dry_run",
                    "year": year,
                    "index_name": index_name,
                    "total_documents": doc_count,
                    "size_bytes": size_bytes,
                }

            # Create snapshot repository if it doesn't exist
            repo_name = f"jira_embeddings_archive_{year}"
            if not self.opensearch_client.client.snapshot.get_repository(repository=repo_name, ignore=[404]):
                logger.info(f"Creating snapshot repository {repo_name}")
                # This would require proper repository configuration
                logger.warning("Snapshot repository creation requires additional configuration")
                return {"status": "failed", "error": "Snapshot repository not configured"}

            # Create snapshot
            snapshot_name = f"jira_embeddings_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            snapshot_body = {"indices": index_name, "ignore_unavailable": True, "include_global_state": False}

            snapshot_result = self.opensearch_client.client.snapshot.create(
                repository=repo_name, snapshot=snapshot_name, body=snapshot_body
            )

            logger.info(f"Created snapshot {snapshot_name}: {snapshot_result}")

            return {
                "status": "success",
                "year": year,
                "index_name": index_name,
                "snapshot_name": snapshot_name,
                "total_documents": doc_count,
                "size_bytes": size_bytes,
            }

        except Exception as e:
            logger.error(f"Archiving failed: {e}")
            return {"status": "failed", "error": str(e)}

    def delete_year(self, year: int, dry_run: bool = False) -> Dict[str, Any]:
        """
        Delete a year's data (use with caution).

        Args:
            year: Year to delete
            dry_run: If True, only simulate the deletion

        Returns:
            Deletion results
        """
        logger.info(f"Deleting year {year} (dry_run={dry_run})")

        try:
            index_name = self.jira_embedding_db.get_index_name_for_year(year)

            if not self.opensearch_client.client.indices.exists(index=index_name):
                logger.error(f"Index {index_name} does not exist")
                return {"status": "failed", "error": f"Index {index_name} does not exist"}

            # Get index stats
            stats = self.opensearch_client.client.indices.stats(index=index_name)
            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
            size_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

            logger.info(f"Index {index_name} contains {doc_count} documents ({size_bytes} bytes)")

            if dry_run:
                logger.info("DRY RUN: Would delete index {} with {} documents".format(index_name, doc_count))
                return {
                    "status": "dry_run",
                    "year": year,
                    "index_name": index_name,
                    "total_documents": doc_count,
                    "size_bytes": size_bytes,
                }

            # Delete index
            delete_result = self.opensearch_client.client.indices.delete(index=index_name)

            logger.info(f"Deleted index {index_name}: {delete_result}")

            return {
                "status": "success",
                "year": year,
                "index_name": index_name,
                "total_documents": doc_count,
                "size_bytes": size_bytes,
            }

        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            return {"status": "failed", "error": str(e)}

    def list_available_years(self) -> List[int]:
        """List all available years with data."""
        try:
            years = self.jira_embedding_db.get_available_years()
            logger.info(f"Available years: {years}")
            return years
        except Exception as e:
            logger.error(f"Failed to list available years: {e}")
            return []

    def get_year_stats(self, year: int) -> Dict[str, Any]:
        """Get statistics for a specific year."""
        try:
            stats = self.jira_embedding_db.get_embedding_stats(year)
            logger.info(f"Stats for year {year}: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for year {year}: {e}")
            return {}

    def cleanup_old_occurrences(self, year: int, days_threshold: int = 90, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up old occurrences from a year's data.

        Args:
            year: Year to clean up
            days_threshold: Remove occurrences older than this many days
            dry_run: If True, only simulate the cleanup

        Returns:
            Cleanup results
        """
        logger.info(f"Cleaning up old occurrences for year {year} (days_threshold={days_threshold}, dry_run={dry_run})")

        try:
            index_name = self.jira_embedding_db.get_index_name_for_year(year)

            if not self.opensearch_client.client.indices.exists(index=index_name):
                logger.error(f"Index {index_name} does not exist")
                return {"status": "failed", "error": f"Index {index_name} does not exist"}

            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_threshold)
            cutoff_timestamp = cutoff_date.isoformat()

            if dry_run:
                logger.info(f"DRY RUN: Would remove occurrences older than {cutoff_timestamp}")
                return {
                    "status": "dry_run",
                    "year": year,
                    "cutoff_timestamp": cutoff_timestamp,
                    "days_threshold": days_threshold,
                }

            # This would require implementing the cleanup logic in JiraIssueEmbeddingDB
            logger.warning("Occurrence cleanup requires implementation in JiraIssueEmbeddingDB")
            return {"status": "not_implemented", "error": "Occurrence cleanup not yet implemented"}

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"status": "failed", "error": str(e)}


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(description="Jira Embedding Database Migration Utilities")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Simulate operations without making changes")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate data between years")
    migrate_parser.add_argument("from_year", type=int, help="Source year")
    migrate_parser.add_argument("to_year", type=int, help="Target year")

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a year's data")
    archive_parser.add_argument("year", type=int, help="Year to archive")
    archive_parser.add_argument("--location", help="Archive location")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a year's data")
    delete_parser.add_argument("year", type=int, help="Year to delete")

    # List command
    subparsers.add_parser("list", help="List available years")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics for a year")
    stats_parser.add_argument("year", type=int, help="Year to get stats for")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old occurrences")
    cleanup_parser.add_argument("year", type=int, help="Year to clean up")
    cleanup_parser.add_argument("--days", type=int, default=90, help="Days threshold for cleanup")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.command:
        parser.print_help()
        return

    # Load configuration
    config = load_config()

    # Create migrator
    migrator = JiraEmbeddingMigrator(config)

    # Execute command
    try:
        if args.command == "migrate":
            result = migrator.migrate_year_to_year(args.from_year, args.to_year, args.dry_run)
        elif args.command == "archive":
            result = migrator.archive_year(args.year, args.location, args.dry_run)
        elif args.command == "delete":
            result = migrator.delete_year(args.year, args.dry_run)
        elif args.command == "list":
            years = migrator.list_available_years()
            print(f"Available years: {years}")
            return
        elif args.command == "stats":
            result = migrator.get_year_stats(args.year)
        elif args.command == "cleanup":
            result = migrator.cleanup_old_occurrences(args.year, args.days, args.dry_run)
        else:
            print(f"Unknown command: {args.command}")
            return

        # Print result
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
