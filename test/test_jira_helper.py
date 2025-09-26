"""
Test cases for Jira Helper module.
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

# Add src to path
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
except NameError:
    # Handle case when __file__ is not defined (e.g., when exec'd)
    sys.path.insert(0, '/app/src')

from error_log_monitor.config import load_config, OpenSearchConfig
from error_log_monitor.jira_helper import JiraHelper, JiraIssue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config() -> OpenSearchConfig:
    """Create test configuration for Jira helper."""
    try:
        config = load_config()
        return config.opensearch
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        # Create minimal config for testing
        return OpenSearchConfig(
            host=os.getenv("OPENSEARCH_HOST", "43.207.106.51"),
            port=int(os.getenv("OPENSEARCH_PORT", "443")),
            username=os.getenv("OPENSEARCH_USERNAME"),
            password=os.getenv("OPENSEARCH_PASSWORD"),
        )


def test_jira_helper_connection():
    """Test Jira helper connection."""
    print("=== Testing Jira Helper Connection ===")

    try:
        config = create_test_config()
        jira_helper = JiraHelper(config)

        if jira_helper.client:
            print("✅ Jira helper connection successful")
            return True
        else:
            print("❌ Jira helper connection failed")
            return False
    except Exception as e:
        print(f"❌ Error testing Jira helper connection: {e}")
        return False


def test_get_recent_issues():
    """Test retrieving recent Jira issues."""
    print("\n=== Testing Get Recent Issues ===")

    try:
        config = create_test_config()
        jira_helper = JiraHelper(config)

        if not jira_helper.client:
            print("❌ Jira helper not connected")
            return False

        # Test getting issues from last 7 days
        issues = jira_helper.get_recent_issues(days=7)

        print(f"📊 Retrieved {len(issues)} Jira issues from last 7 days")

        if issues:
            print("\n🔍 Sample Issues:")
            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                print(f"  {i+1}. {issue.issue_key} - {issue.error_message[:50]}...")
                print(f"     Time: {issue.issue_time}")
                print(f"     Site: {issue.site}")
                if issue.jira_summary:
                    print(f"     Summary: {issue.jira_summary[:50]}...")
                print()

        print("✅ Get recent issues test completed")
        return True

    except Exception as e:
        print(f"❌ Error testing get recent issues: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_issues_by_date_range():
    """Test retrieving Jira issues by date range."""
    print("\n=== Testing Get Issues by Date Range ===")

    try:
        config = create_test_config()
        jira_helper = JiraHelper(config)

        if not jira_helper.client:
            print("❌ Jira helper not connected")
            return False

        # Test getting issues from last 3 days
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=3)

        issues = jira_helper.get_issues_by_date_range(start_date, end_date)

        print(f"📊 Retrieved {len(issues)} Jira issues from {start_date.date()} to {end_date.date()}")

        if issues:
            print("\n🔍 Sample Issues:")
            for i, issue in enumerate(issues[:2]):  # Show first 2 issues
                print(f"  {i+1}. {issue.issue_key} - {issue.error_message[:50]}...")
                print(f"     Time: {issue.issue_time}")
                print()

        print("✅ Get issues by date range test completed")
        return True

    except Exception as e:
        print(f"❌ Error testing get issues by date range: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_issues_by_error_message():
    """Test retrieving Jira issues by error message pattern."""
    print("\n=== Testing Get Issues by Error Message ===")

    try:
        config = create_test_config()
        jira_helper = JiraHelper(config)

        if not jira_helper.client:
            print("❌ Jira helper not connected")
            return False

        # Test searching for timeout-related issues
        issues = jira_helper.get_issues_by_error_message("timeout", days=30)

        print(f"📊 Retrieved {len(issues)} Jira issues containing 'timeout' in last 30 days")

        if issues:
            print("\n🔍 Sample Timeout Issues:")
            for i, issue in enumerate(issues[:2]):  # Show first 2 issues
                print(f"  {i+1}. {issue.issue_key} - {issue.error_message[:50]}...")
                print(f"     Time: {issue.issue_time}")
                print()

        print("✅ Get issues by error message test completed")
        return True

    except Exception as e:
        print(f"❌ Error testing get issues by error message: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_issue_statistics():
    """Test getting Jira issue statistics."""
    print("\n=== Testing Get Issue Statistics ===")

    try:
        config = create_test_config()
        jira_helper = JiraHelper(config)

        if not jira_helper.client:
            print("❌ Jira helper not connected")
            return False

        # Test getting statistics for last 7 days
        stats = jira_helper.get_issue_statistics(days=7)

        print("📊 Jira Issue Statistics (Last 7 days):")
        print(f"  Total Issues: {stats['total_issues']}")
        print(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        if stats['sites']:
            print("  Sites:")
            for site, count in stats['sites'].items():
                print(f"    {site}: {count} issues")

        if stats['error_types']:
            print("  Error Types:")
            for error_type, count in list(stats['error_types'].items())[:5]:  # Show top 5
                print(f"    {error_type}: {count} issues")

        print("✅ Get issue statistics test completed")
        return True

    except Exception as e:
        print(f"❌ Error testing get issue statistics: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all Jira helper tests."""
    print("Running Jira Helper Tests...")
    print("=" * 50)

    tests = [
        test_jira_helper_connection,
        test_get_recent_issues,
        test_get_issues_by_date_range,
        test_get_issues_by_error_message,
        test_get_issue_statistics,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
