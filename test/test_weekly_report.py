"""
Test cases for Weekly Report module.
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

from error_log_monitor.config import load_config
from error_log_monitor.weekly_report import WeeklyReportGenerator, WeeklyReportIssue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create test configuration for weekly report."""
    try:
        config = load_config()
        return config
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return None


def test_weekly_report_generator_initialization():
    """Test weekly report generator initialization."""
    print("=== Testing Weekly Report Generator Initialization ===")

    try:
        config = create_test_config()
        if not config:
            print("❌ Could not create test configuration")
            return False

        report_generator = WeeklyReportGenerator(config)

        # Check if all components are initialized
        assert hasattr(report_generator, 'opensearch_client'), 'Missing opensearch_client'
        assert hasattr(report_generator, 'vector_db_client'), 'Missing vector_db_client'
        assert hasattr(report_generator, 'rag_engine'), 'Missing rag_engine'
        assert hasattr(report_generator, 'jira_helper'), 'Missing jira_helper'
        assert hasattr(report_generator, 'error_analyzer'), 'Missing error_analyzer'

        print("✅ Weekly report generator initialization successful")
        return True

    except Exception as e:
        print(f"❌ Error testing weekly report generator initialization: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_weekly_report_issue_creation():
    """Test WeeklyReportIssue dataclass creation."""
    print("\n=== Testing Weekly Report Issue Creation ===")

    try:
        # Create a test weekly report issue
        issue = WeeklyReportIssue(
            key="https://jira.example.com/PROJ-123",
            site="prod",
            count=5,
            summary="Test error issue",
            status="Open",
            log_group="test-service",
            latest_update=datetime.now(timezone.utc),
            note="Test root cause analysis",
            child_issues=["PROJ-124", "PROJ-125"],
        )

        # Verify all fields are set correctly
        assert issue.key == "https://jira.example.com/PROJ-123"
        assert issue.site == "prod"
        assert issue.count == 5
        assert issue.summary == "Test error issue"
        assert issue.status == "Open"
        assert issue.log_group == "test-service"
        assert issue.note == "Test root cause analysis"
        assert len(issue.child_issues) == 2
        assert "PROJ-124" in issue.child_issues
        assert "PROJ-125" in issue.child_issues

        print("✅ Weekly report issue creation successful")
        return True

    except Exception as e:
        print(f"❌ Error testing weekly report issue creation: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_weekly_report_generation():
    """Test weekly report generation (without actual data)."""
    print("\n=== Testing Weekly Report Generation ===")

    try:
        config = create_test_config()
        if not config:
            print("❌ Could not create test configuration")
            return False

        report_generator = WeeklyReportGenerator(config)

        # Test with a small date range (likely no data)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)

        print(f"Testing report generation from {start_date.date()} to {end_date.date()}")

        # This might fail due to no data, but should not crash
        try:
            report_data = report_generator.generate_weekly_report(start_date, end_date)

            # Verify report structure
            required_keys = [
                'start_date',
                'end_date',
                'total_error_logs',
                'stage_logs',
                'prod_logs',
                'merged_issues',
                'weekly_issues',
                'issues',
            ]

            for key in required_keys:
                assert key in report_data, f"Missing key: {key}"

            print(f"✅ Weekly report generation successful")
            print(f"  Total Error Logs: {report_data['total_error_logs']}")
            print(f"  Merged Issues: {report_data['merged_issues']}")
            print(f"  Weekly Issues: {report_data['weekly_issues']}")

            return True

        except Exception as e:
            if "No error logs found" in str(e) or "No data" in str(e):
                print("✅ Weekly report generation handled empty data correctly")
                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ Error testing weekly report generation: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_excel_report_generation():
    """Test Excel report generation."""
    print("\n=== Testing Excel Report Generation ===")

    try:
        # Create test data
        test_issues = [
            WeeklyReportIssue(
                key="https://jira.example.com/PROJ-123",
                site="prod",
                count=3,
                summary="Test error 1",
                status="Open",
                log_group="service1",
                latest_update=datetime.now(timezone.utc),
                note="Test root cause 1",
                child_issues=["PROJ-124"],
            ),
            WeeklyReportIssue(
                key="https://jira.example.com/PROJ-125",
                site="stage",
                count=1,
                summary="Test error 2",
                status="Closed",
                log_group="service2",
                latest_update=datetime.now(timezone.utc),
                note="Test root cause 2",
                child_issues=[],
            ),
        ]

        config = create_test_config()
        if not config:
            print("❌ Could not create test configuration")
            return False

        report_generator = WeeklyReportGenerator(config)

        # Test Excel generation
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        excel_path = report_generator._generate_excel_report(test_issues, start_date, end_date)

        if excel_path and os.path.exists(excel_path):
            print(f"✅ Excel report generated successfully: {excel_path}")
            return True
        else:
            print(f"❌ Excel report generation failed or file not found")
            return False

    except Exception as e:
        print(f"❌ Error testing Excel report generation: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_html_report_generation():
    """Test HTML report generation."""
    print("\n=== Testing HTML Report Generation ===")

    try:
        # Create test data
        test_issues = [
            WeeklyReportIssue(
                key="https://jira.example.com/PROJ-123",
                site="prod",
                count=3,
                summary="Test error 1",
                status="Open",
                log_group="service1",
                latest_update=datetime.now(timezone.utc),
                note="Test root cause 1",
                child_issues=["PROJ-124"],
            )
        ]

        config = create_test_config()
        if not config:
            print("❌ Could not create test configuration")
            return False

        report_generator = WeeklyReportGenerator(config)

        # Test HTML generation
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        html_path = report_generator._generate_html_report(test_issues, start_date, end_date)

        if html_path and os.path.exists(html_path):
            print(f"✅ HTML report generated successfully: {html_path}")
            return True
        else:
            print(f"❌ HTML report generation failed or file not found")
            return False

    except Exception as e:
        print(f"❌ Error testing HTML report generation: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all weekly report tests."""
    print("Running Weekly Report Tests...")
    print("=" * 50)

    tests = [
        test_weekly_report_generator_initialization,
        test_weekly_report_issue_creation,
        test_weekly_report_generation,
        test_excel_report_generation,
        test_html_report_generation,
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
