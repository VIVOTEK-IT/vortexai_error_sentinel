#!/usr/bin/env python3
"""
Daily Report Runner Script

This script can be used to run daily reports manually or as a scheduled task.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.daily_report import DailyReportGenerator
from error_log_monitor.email_service import EmailService
from error_log_monitor.email_templates import generate_daily_report_html_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_daily_report(send_email: bool = True, end_date: Optional[datetime] = None) -> bool:
    """
    Run daily report generation and optionally send email.

    Args:
        send_email: Whether to send email notification
        end_date: End date for the report (defaults to now)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting daily report generation...")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        report_generator = DailyReportGenerator()
        email_service = EmailService(config.email) if send_email else None
        
        # Generate daily report
        logger.info("Generating daily report...")
        report_data = report_generator.generate_daily_report(end_date)
        
        # Calculate total issues across all sites
        total_issues = sum(
            len(site_data.get("issues", [])) 
            for site_data in report_data.get("site_reports", {}).values()
        )
        
        # Print summary
        print("\n📊 Daily Report Summary:")
        print(f"  Period: {report_data['start_date'].strftime('%Y-%m-%d %H:%M')} to {report_data['end_date'].strftime('%Y-%m-%d %H:%M')}")
        print(f"  Total Issues: {total_issues}")
        
        for site_name, site_data in report_data['site_reports'].items():
            issues_count = len(site_data.get('issues', []))
            print(f"  {site_name.title()} Site: {issues_count} issues")
            if site_data.get('excel_path'):
                print(f"    Excel Report: {site_data['excel_path']}")
            if site_data.get('html_path'):
                print(f"    HTML Report: {site_data['html_path']}")
        
        if report_data.get('combined_excel_path'):
            print(f"  Combined Excel Report: {report_data['combined_excel_path']}")
        
        # Send email if requested
        if send_email and email_service:
            logger.info("Generating and sending email notification...")
            
            # Generate HTML email content
            html_content = generate_daily_report_html_email(
                site_reports=report_data.get("site_reports", {}),
                start_date=report_data.get("start_date"),
                end_date=report_data.get("end_date"),
                total_issues=total_issues
            )
            
            # Send email
            email_sent = email_service.send_daily_report_email(html_content)
            
            if email_sent:
                print("  ✅ Email notification sent successfully")
            else:
                print("  ❌ Failed to send email notification")
                return False
        
        print("\n✅ Daily report completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Daily report failed: {str(e)}", exc_info=True)
        print(f"\n❌ Daily report failed: {str(e)}")
        return False


def main():
    """Main function to run daily report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run daily error report")
    parser.add_argument(
        "--no-email", 
        action="store_true", 
        help="Generate report without sending email"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for the report (YYYY-MM-DD format)"
    )
    
    args = parser.parse_args()
    
    # Parse end date if provided
    end_date = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            return 1
    
    # Run daily report
    success = run_daily_report(
        send_email=not args.no_email,
        end_date=end_date
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

