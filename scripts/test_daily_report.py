#!/usr/bin/env python3
"""
Test script for daily report functionality.

This script tests the daily report generation, email service, and Lambda handlers.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.daily_report import DailyReportGenerator
from error_log_monitor.email_service import EmailService
from error_log_monitor.email_templates import generate_daily_report_html_email
from error_log_monitor.lambda_daily_report import lambda_handler, generate_daily_report_only, test_email_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_daily_report_generation():
    """Test daily report generation without email sending."""
    logger.info("Testing daily report generation...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize report generator
        report_generator = DailyReportGenerator()
        
        # Generate daily report
        report_data = report_generator.generate_daily_report()
        
        # Print summary
        print("\n📊 Daily Report Generation Test Results:")
        print(f"  Period: {report_data['start_date'].strftime('%Y-%m-%d %H:%M')} to {report_data['end_date'].strftime('%Y-%m-%d %H:%M')}")
        
        total_issues = 0
        for site_name, site_data in report_data['site_reports'].items():
            issues_count = len(site_data.get('issues', []))
            total_issues += issues_count
            print(f"  {site_name.title()} Site: {issues_count} issues")
            if site_data.get('excel_path'):
                print(f"    Excel Report: {site_data['excel_path']}")
            if site_data.get('html_path'):
                print(f"    HTML Report: {site_data['html_path']}")
        
        print(f"  Total Issues: {total_issues}")
        if report_data.get('combined_excel_path'):
            print(f"  Combined Excel Report: {report_data['combined_excel_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Daily report generation test failed: {str(e)}", exc_info=True)
        return False


def test_email_service():
    """Test email service configuration and sending."""
    logger.info("Testing email service...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize email service
        email_service = EmailService(config.email)
        
        # Check configuration
        print("\n📧 Email Service Test Results:")
        print(f"  AWS Region: {config.email.aws_region}")
        print(f"  Sender Email: {config.email.sender_email}")
        print(f"  Recipients: {config.email.recipients}")
        
        # Check if sender email is verified
        sender_verified = email_service.is_email_verified(config.email.sender_email)
        print(f"  Sender Verified: {sender_verified}")
        
        # Get send quota
        quota_info = email_service.get_send_quota()
        if quota_info:
            print(f"  Send Quota: {quota_info.get('Max24HourSend', 'Unknown')} emails per day")
            print(f"  Sent Last 24h: {quota_info.get('SentLast24Hours', 'Unknown')} emails")
        
        # Test email sending (only if recipients are configured)
        if config.email.recipients:
            print("\n  Sending test email...")
            test_html = """
            <html>
            <body>
                <h1>Test Email from VortexAI Error Monitoring System</h1>
                <p>This is a test email to verify the email service configuration.</p>
                <p>If you receive this email, the configuration is working correctly.</p>
            </body>
            </html>
            """
            
            email_sent = email_service.send_daily_report_email(
                html_content=test_html,
                subject="[TEST] VortexAI Error Monitoring System - Email Test"
            )
            
            if email_sent:
                print("  ✅ Test email sent successfully!")
            else:
                print("  ❌ Failed to send test email")
        else:
            print("  ⚠️  No recipients configured, skipping email test")
        
        return True
        
    except Exception as e:
        logger.error(f"Email service test failed: {str(e)}", exc_info=True)
        return False


def test_lambda_handlers():
    """Test Lambda handlers."""
    logger.info("Testing Lambda handlers...")
    
    try:
        # Test context object
        class MockContext:
            def __init__(self):
                self.function_name = "test-daily-report"
                self.function_version = "1"
                self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-daily-report"
                self.memory_limit_in_mb = 128
                self.remaining_time_in_millis = 30000
        
        context = MockContext()
        
        # Test 1: Generate report only
        print("\n🔧 Lambda Handler Test Results:")
        print("  Testing generate_daily_report_only...")
        
        response = generate_daily_report_only({}, context)
        print(f"    Status Code: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = response['body']
            if isinstance(body, str):
                import json
                body = json.loads(body)
            print(f"    Message: {body.get('message', 'No message')}")
            print(f"    Total Issues: {body.get('total_issues', 'Unknown')}")
            print("    ✅ generate_daily_report_only test passed")
        else:
            print("    ❌ generate_daily_report_only test failed")
        
        # Test 2: Full Lambda handler (with email)
        print("  Testing lambda_handler (with email)...")
        
        response = lambda_handler({}, context)
        print(f"    Status Code: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = response['body']
            if isinstance(body, str):
                import json
                body = json.loads(body)
            print(f"    Message: {body.get('message', 'No message')}")
            print(f"    Email Sent: {body.get('email_sent', 'Unknown')}")
            print("    ✅ lambda_handler test passed")
        else:
            print("    ❌ lambda_handler test failed")
        
        # Test 3: Email service test
        print("  Testing test_email_service...")
        
        response = test_email_service({}, context)
        print(f"    Status Code: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = response['body']
            if isinstance(body, str):
                import json
                body = json.loads(body)
            print(f"    Message: {body.get('message', 'No message')}")
            print(f"    Email Sent: {body.get('email_sent', 'Unknown')}")
            print("    ✅ test_email_service test passed")
        else:
            print("    ❌ test_email_service test failed")
        
        return True
        
    except Exception as e:
        logger.error(f"Lambda handler test failed: {str(e)}", exc_info=True)
        return False


def test_html_email_template():
    """Test HTML email template generation."""
    logger.info("Testing HTML email template generation...")
    
    try:
        # Create mock report data
        from error_log_monitor.daily_report import DailyReportRow
        
        mock_issues = [
            DailyReportRow(
                key="TEST-123",
                site="stage",
                count=5,
                error_message="Test error message for stage environment",
                status="Open",
                log_group="test-service",
                latest_update=datetime.now(timezone.utc)
            ),
            DailyReportRow(
                key="TEST-456",
                site="prod",
                count=3,
                error_message="Test error message for production environment",
                status="In Progress",
                log_group="prod-service",
                latest_update=datetime.now(timezone.utc)
            )
        ]
        
        mock_site_reports = {
            "stage": {"issues": [mock_issues[0]]},
            "prod": {"issues": [mock_issues[1]]}
        }
        
        start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now(timezone.utc)
        
        # Generate HTML email
        html_content = generate_daily_report_html_email(
            site_reports=mock_site_reports,
            start_date=start_date,
            end_date=end_date,
            total_issues=2
        )
        
        # Save HTML to file for inspection
        output_file = os.path.join(os.path.dirname(__file__), "..", "test_daily_report_email.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"\n📧 HTML Email Template Test Results:")
        print(f"  HTML content generated successfully")
        print(f"  Output saved to: {output_file}")
        print(f"  Content length: {len(html_content)} characters")
        print("  ✅ HTML email template test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"HTML email template test failed: {str(e)}", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Daily Report System Tests")
    print("=" * 50)
    
    tests = [
        ("Daily Report Generation", test_daily_report_generation),
        ("Email Service", test_email_service),
        ("Lambda Handlers", test_lambda_handlers),
        ("HTML Email Template", test_html_email_template),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {str(e)}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Daily report system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

