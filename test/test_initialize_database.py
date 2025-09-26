#!/usr/bin/env python3
"""
Test script for Jira Issue Embedding Database initialization workflow.

This script demonstrates the initialize_database method with sample data.
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient, ErrorLog
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB


def create_sample_jira_issues() -> List[Dict[str, Any]]:
    """Create sample Jira issues for testing."""
    return [
        {
            "key": "VEL-123",
            "summary": "Database connection timeout error",
            "description": "Users experiencing timeout when connecting to the database",
            "status": "Open",
            "created": "2024-01-15T10:30:00Z",
            "updated": "2024-01-20T14:45:00Z",
            "site": "prod",
            "parent_issue_key": "",
            "error_message": "Connection timeout after 30 seconds",
            "error_type": "DatabaseError",
            "traceback": "Traceback (most recent call last):\n  File 'db.py', line 45, in connect\n    conn = psycopg2.connect(...)\npsycopg2.OperationalError: connection timeout",
            "request_id": "req-001",
        },
        {
            "key": "VEL-124",
            "summary": "API rate limit exceeded",
            "description": "API calls are being rate limited too aggressively",
            "status": "In Progress",
            "created": "2024-01-16T09:15:00Z",
            "updated": "2024-01-22T11:20:00Z",
            "site": "stage",
            "parent_issue_key": "",
            "error_message": "Rate limit exceeded: 100 requests per minute",
            "error_type": "RateLimitError",
            "traceback": "Traceback (most recent call last):\n  File 'api.py', line 23, in make_request\n    raise RateLimitError('Too many requests')\nRateLimitError: Too many requests",
            "request_id": "req-002",
        },
        {
            "key": "VEL-125",
            "summary": "Memory leak in background process",
            "description": "Background process consuming increasing memory over time",
            "status": "Closed",
            "created": "2024-01-10T08:00:00Z",
            "updated": "2024-01-25T16:30:00Z",
            "site": "prod",
            "parent_issue_key": "",
            "error_message": "Memory usage exceeded 2GB",
            "error_type": "MemoryError",
            "traceback": "Traceback (most recent call last):\n  File 'worker.py', line 67, in process\n    data = load_large_dataset()\nMemoryError: Unable to allocate memory",
            "request_id": "req-003",
        },
    ]


def create_sample_error_logs() -> Dict[str, List[ErrorLog]]:
    """Create sample error logs for testing."""
    now = datetime.now(timezone.utc)

    # Create error logs for different sites
    error_logs_by_site = {
        "prod": [
            ErrorLog(
                message_id="log-001",
                timestamp=now - timedelta(hours=2),
                error_message="Database connection timeout after 30 seconds",
                error_type="DatabaseError",
                traceback="Traceback (most recent call last):\n  File 'db.py', line 45, in connect\n    conn = psycopg2.connect(...)\npsycopg2.OperationalError: connection timeout",
                site="prod",
                service="database",
                request_id="req-004",
            ),
            ErrorLog(
                message_id="log-002",
                timestamp=now - timedelta(hours=1),
                error_message="Memory usage exceeded 2GB in worker process",
                error_type="MemoryError",
                traceback="Traceback (most recent call last):\n  File 'worker.py', line 67, in process\n    data = load_large_dataset()\nMemoryError: Unable to allocate memory",
                site="prod",
                service="worker",
                request_id="req-005",
            ),
            ErrorLog(
                message_id="log-003",
                timestamp=now - timedelta(minutes=30),
                error_message="New error: File not found in upload directory",
                error_type="FileNotFoundError",
                traceback="Traceback (most recent call last):\n  File 'upload.py', line 12, in process_file\n    with open(file_path, 'r') as f:\nFileNotFoundError: [Errno 2] No such file or directory",
                site="prod",
                service="upload",
                request_id="req-006",
            ),
        ],
        "stage": [
            ErrorLog(
                message_id="log-004",
                timestamp=now - timedelta(hours=3),
                error_message="Rate limit exceeded: 100 requests per minute",
                error_type="RateLimitError",
                traceback="Traceback (most recent call last):\n  File 'api.py', line 23, in make_request\n    raise RateLimitError('Too many requests')\nRateLimitError: Too many requests",
                site="stage",
                service="api",
                request_id="req-007",
            ),
            ErrorLog(
                message_id="log-005",
                timestamp=now - timedelta(minutes=45),
                error_message="Authentication failed for user admin",
                error_type="AuthenticationError",
                traceback="Traceback (most recent call last):\n  File 'auth.py', line 8, in authenticate\n    raise AuthenticationError('Invalid credentials')\nAuthenticationError: Invalid credentials",
                site="stage",
                service="auth",
                request_id="req-008",
            ),
        ],
    }

    return error_logs_by_site


def test_initialize_database():
    """Test the initialize_database method."""
    try:
        print("🚀 Starting Jira Issue Embedding Database initialization test")

        # Load configuration
        print("📋 Loading configuration...")
        config = load_config()

        # Initialize clients
        print("🔌 Initializing OpenSearch client...")
        opensearch_client = OpenSearchClient(config.opensearch)

        print("🤖 Initializing embedding service...")
        embedding_service = EmbeddingService(model_name=config.vector_db.embedding_model)

        print("🗄️ Initializing Jira Issue Embedding Database...")
        jira_embedding_db = JiraIssueEmbeddingDB(
            opensearch_client=opensearch_client, embedding_service=embedding_service, config=config
        )

        # Create sample data
        print("📝 Creating sample data...")
        jira_issues = create_sample_jira_issues()
        error_logs_by_site = create_sample_error_logs()

        print(f"   - {len(jira_issues)} Jira issues")
        total_error_logs = sum(len(logs) for logs in error_logs_by_site.values())
        print(f"   - {total_error_logs} error logs across {len(error_logs_by_site)} sites")

        # Initialize database
        print("🔄 Initializing database...")
        results = jira_embedding_db.initialize_database(jira_issues, error_logs_by_site)

        # Display results
        print("\n📊 Initialization Results:")
        print(f"   Status: {results['status']}")
        print(f"   Jira Issues Processed: {results['jira_issues_processed']}")
        print(f"   Error Logs Processed: {results['error_logs_processed']}")
        print(f"   Similar Issues Found: {results['similar_issues_found']}")
        print(f"   New Issues Created: {results['new_issues_created']}")
        print(f"   Occurrences Added: {results['occurrences_added']}")

        if results['errors']:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"   - {error}")

        print("\n📈 Site Statistics:")
        for site, stats in results['site_stats'].items():
            print(f"   {site.upper()}:")
            print(f"     - Error Logs Processed: {stats['error_logs_processed']}")
            print(f"     - Similar Issues Found: {stats['similar_issues_found']}")
            print(f"     - New Issues Created: {stats['new_issues_created']}")
            print(f"     - Occurrences Added: {stats['occurrences_added']}")
            if stats['errors']:
                print(f"     - Errors: {len(stats['errors'])}")

        # Test search functionality
        print("\n🔍 Testing search functionality...")
        test_query = "database connection timeout"
        search_results = jira_embedding_db.search_similar_issues(
            query_text=test_query, top_k=3, similarity_threshold=0.7
        )

        print(f"   Search query: '{test_query}'")
        print(f"   Results found: {len(search_results)}")
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result['key']} (score: {result['score']:.3f}) - {result['summary']}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_initialize_database()
