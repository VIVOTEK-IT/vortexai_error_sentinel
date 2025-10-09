"""
AWS Lambda handlers for daily and weekly report generation and email sending.

Available Lambda Handlers:
==========================

1. lambda_handler(event, context)
   - Main handler supporting both daily and weekly reports
   - Event parameter: {"report_type": "daily" | "weekly"} (optional, defaults to "daily")
   - Generates report and sends email

2. generate_daily_report(event, context)
   - Convenience handler specifically for daily reports
   - Always generates daily report and sends email

3. generate_weekly_report(event, context)
   - Convenience handler specifically for weekly reports
   - Always generates weekly report and sends email

4. generate_report_only(event, context)
   - Generates report without sending email
   - Event parameter: {"report_type": "daily" | "weekly"} (optional, defaults to "daily")
   - Useful for testing or when email is not required

5. test_email_service(event, context)
   - Tests email service configuration
   - Sends a test email to verify SES setup
   - Returns email service status and quota information

Usage Examples:
==============

Daily Report (with email):
{
  "report_type": "daily"
}

Weekly Report (with email):
{
  "report_type": "weekly"
}

Report only (no email):
{
  "report_type": "daily"
}

Email test:
{}
"""

import json
import logging
from typing import Dict, Any

from error_log_monitor.config import load_config
from error_log_monitor.daily_report import DailyReportGenerator
from error_log_monitor.email_service import EmailService
from error_log_monitor.email_templates import generate_daily_report_html_email

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for daily and weekly report generation and email sending.

    Args:
        event: Lambda event data with optional 'report_type' field ("daily" or "weekly")
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    try:
        # Extract report type from event, default to "daily"
        report_type = event.get("report_type", "daily").lower()
        if report_type not in ["daily", "weekly"]:
            logger.warning(f"Invalid report_type '{report_type}', defaulting to 'daily'")
            report_type = "daily"

        logger.info(f"Starting {report_type} report generation Lambda function")

        # Load configuration
        config = load_config()

        # Initialize services (disable Excel generation for Lambda)
        report_generator = DailyReportGenerator(generate_excel=False)
        email_service = EmailService(config.email)

        # Generate report based on type
        if report_type == "weekly":
            logger.info("Generating weekly report...")
            report_data = report_generator.generate_weekly_report()
        else:
            logger.info("Generating daily report...")
            report_data = report_generator.generate_daily_report()

        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) for site_data in report_data.get("site_reports", {}).values()
        )

        # Generate HTML email content
        logger.info("Generating HTML email content...")
        html_content = generate_daily_report_html_email(
            site_reports=report_data.get("site_reports", {}),
            start_date=report_data.get("start_date"),
            end_date=report_data.get("end_date"),
            total_issues=total_issues,
        )

        # Send email
        logger.info(f"Sending {report_type} report email...")
        email_sent = email_service.send_daily_report_email(html_content)

        if not email_sent:
            logger.error(f"Failed to send {report_type} report email")
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "error": f"Failed to send {report_type} report email",
                        "report_generated": True,
                        "total_issues": total_issues,
                        "report_type": report_type,
                    }
                ),
            }

        # Prepare response
        response_data = {
            "message": f"{report_type.title()} report generated and sent successfully",
            "report_type": report_type,
            "report_period": {
                "start_date": report_data.get("start_date").isoformat(),
                "end_date": report_data.get("end_date").isoformat(),
            },
            "total_issues": total_issues,
            "site_reports": {
                site: {
                    "count": len(site_data.get("issues", [])),
                    "excel_path": site_data.get("excel_path"),
                    "html_path": site_data.get("html_path"),
                }
                for site, site_data in report_data.get("site_reports", {}).items()
            },
            "combined_excel_path": report_data.get("combined_excel_path"),
            "email_sent": True,
        }

        logger.info(f"{report_type.title()} report completed successfully. Total issues: {total_issues}")

        return {"statusCode": 200, "body": json.dumps(response_data, default=str)}

    except Exception as e:
        logger.error(f"Error in {report_type} report Lambda function: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}", "report_generated": False}),
        }


def generate_report_only(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for generating daily or weekly report without sending email.
    Useful for testing or when email sending is not required.

    Args:
        event: Lambda event data with optional 'report_type' field ("daily" or "weekly")
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    try:
        # Extract report type from event, default to "daily"
        report_type = event.get("report_type", "daily").lower()
        if report_type not in ["daily", "weekly"]:
            logger.warning(f"Invalid report_type '{report_type}', defaulting to 'daily'")
            report_type = "daily"

        logger.info(f"Starting {report_type} report generation (no email)")

        # Load configuration
        config = load_config()

        # Initialize report generator (disable Excel generation for Lambda)
        report_generator = DailyReportGenerator(generate_excel=False)

        # Generate report based on type
        if report_type == "weekly":
            logger.info("Generating weekly report...")
            report_data = report_generator.generate_weekly_report()
        else:
            logger.info("Generating daily report...")
            report_data = report_generator.generate_daily_report()

        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) for site_data in report_data.get("site_reports", {}).values()
        )

        # Prepare response
        response_data = {
            "message": f"{report_type.title()} report generated successfully",
            "report_type": report_type,
            "report_period": {
                "start_date": report_data.get("start_date").isoformat(),
                "end_date": report_data.get("end_date").isoformat(),
            },
            "total_issues": total_issues,
            "site_reports": {
                site: {
                    "count": len(site_data.get("issues", [])),
                    "excel_path": site_data.get("excel_path"),
                    "html_path": site_data.get("html_path"),
                }
                for site, site_data in report_data.get("site_reports", {}).items()
            },
            "combined_excel_path": report_data.get("combined_excel_path"),
            "email_sent": False,
        }

        logger.info(f"{report_type.title()} report generated successfully. Total issues: {total_issues}")

        return {"statusCode": 200, "body": json.dumps(response_data, default=str)}

    except Exception as e:
        logger.error(f"Error in {report_type} report generation: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}", "report_generated": False}),
        }


def generate_weekly_report(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler specifically for weekly report generation and email sending.
    Convenience function for weekly reports.

    Args:
        event: Lambda event data
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    # Force weekly report type
    event["report_type"] = "weekly"
    return lambda_handler(event, context)


def generate_daily_report(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler specifically for daily report generation and email sending.
    Convenience function for daily reports.

    Args:
        event: Lambda event data
        context: Lambda context

    Returns:
        Lambda response dictionary
    """
    # Force daily report type
    event["report_type"] = "daily"
    return lambda_handler(event, context)


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
            recipients=config.email.recipients,
        )

        response_data = {
            "message": "Email service test completed",
            "sender_email": config.email.sender_email,
            "sender_verified": sender_verified,
            "recipients": config.email.recipients,
            "email_sent": email_sent,
            "quota_info": quota_info,
        }

        return {"statusCode": 200, "body": json.dumps(response_data, default=str)}

    except Exception as e:
        logger.error(f"Error in email service test: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": f"Email service test failed: {str(e)}"})}
