"""AWS Lambda handler for daily report generation and email sending."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from error_log_monitor.config import load_config
from error_log_monitor.daily_report import DailyReportGenerator
from error_log_monitor.email_service import EmailService
from error_log_monitor.email_templates import generate_daily_report_html_email

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for daily report generation and email sending.

    Args:
        event: Lambda event data
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    try:
        logger.info("Starting daily report generation Lambda function")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        report_generator = DailyReportGenerator()
        email_service = EmailService(config.email)
        
        # Generate daily report
        logger.info("Generating daily report...")
        report_data = report_generator.generate_daily_report()
        
        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) 
            for site_data in report_data.get("site_reports", {}).values()
        )
        
        # Generate HTML email content
        logger.info("Generating HTML email content...")
        html_content = generate_daily_report_html_email(
            site_reports=report_data.get("site_reports", {}),
            start_date=report_data.get("start_date"),
            end_date=report_data.get("end_date"),
            total_issues=total_issues
        )
        
        # Send email
        logger.info("Sending daily report email...")
        email_sent = email_service.send_daily_report_email(html_content)
        
        if not email_sent:
            logger.error("Failed to send daily report email")
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": "Failed to send daily report email",
                    "report_generated": True,
                    "total_issues": total_issues
                })
            }
        
        # Prepare response
        response_data = {
            "message": "Daily report generated and sent successfully",
            "report_period": {
                "start_date": report_data.get("start_date").isoformat(),
                "end_date": report_data.get("end_date").isoformat()
            },
            "total_issues": total_issues,
            "site_reports": {
                site: {
                    "count": len(site_data.get("issues", [])),
                    "excel_path": site_data.get("excel_path"),
                    "html_path": site_data.get("html_path")
                }
                for site, site_data in report_data.get("site_reports", {}).items()
            },
            "combined_excel_path": report_data.get("combined_excel_path"),
            "email_sent": True
        }
        
        logger.info(f"Daily report completed successfully. Total issues: {total_issues}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error in daily report Lambda function: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Internal server error: {str(e)}",
                "report_generated": False
            })
        }


def generate_daily_report_only(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for generating daily report without sending email.
    Useful for testing or when email sending is not required.

    Args:
        event: Lambda event data
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    try:
        logger.info("Starting daily report generation (no email)")
        
        # Load configuration
        config = load_config()
        
        # Initialize report generator
        report_generator = DailyReportGenerator()
        
        # Generate daily report
        logger.info("Generating daily report...")
        report_data = report_generator.generate_daily_report()
        
        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) 
            for site_data in report_data.get("site_reports", {}).values()
        )
        
        # Prepare response
        response_data = {
            "message": "Daily report generated successfully",
            "report_period": {
                "start_date": report_data.get("start_date").isoformat(),
                "end_date": report_data.get("end_date").isoformat()
            },
            "total_issues": total_issues,
            "site_reports": {
                site: {
                    "count": len(site_data.get("issues", [])),
                    "excel_path": site_data.get("excel_path"),
                    "html_path": site_data.get("html_path")
                }
                for site, site_data in report_data.get("site_reports", {}).items()
            },
            "combined_excel_path": report_data.get("combined_excel_path"),
            "email_sent": False
        }
        
        logger.info(f"Daily report generated successfully. Total issues: {total_issues}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error in daily report generation: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Internal server error: {str(e)}",
                "report_generated": False
            })
        }


def test_email_service(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for testing email service configuration.
    Sends a test email to verify SES setup.

    Args:
        event: Lambda event data
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    try:
        logger.info("Testing email service configuration")
        
        # Load configuration
        config = load_config()
        
        # Initialize email service
        email_service = EmailService(config.email)
        
        # Check if sender email is verified
        sender_verified = email_service.is_email_verified(config.email.sender_email)
        
        # Get send quota
        quota_info = email_service.get_send_quota()
        
        # Send test email
        test_html = """
        <html>
        <body>
            <h1>Test Email</h1>
            <p>This is a test email from the VortexAI Error Monitoring System.</p>
            <p>If you receive this email, the SES configuration is working correctly.</p>
        </body>
        </html>
        """
        
        email_sent = email_service.send_daily_report_email(
            html_content=test_html,
            subject="[TEST] VortexAI Error Monitoring System - Email Test",
            recipients=config.email.recipients
        )
        
        response_data = {
            "message": "Email service test completed",
            "sender_email": config.email.sender_email,
            "sender_verified": sender_verified,
            "recipients": config.email.recipients,
            "email_sent": email_sent,
            "quota_info": quota_info
        }
        
        return {
            "statusCode": 200,
            "body": json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error in email service test: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Email service test failed: {str(e)}"
            })
        }

