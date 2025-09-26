"""
Main application for Error Log Monitoring System.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import List

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.vector_db_client import VectorDBClient
from error_log_monitor.rag_engine import RAGEngine
from error_log_monitor.error_analyzer import ErrorAnalyzer
from error_log_monitor.weekly_report import WeeklyReportGenerator

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('error_monitor.log')],
)

    # Suppress all ChromaDB telemetry errors
    for logger_name in [
        'chromadb.telemetry',
        'chromadb.telemetry.posthog',
        'chromadb.telemetry.product',
        'chromadb.telemetry.product.posthog',
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def test_connections(config):
    """Test all database connections."""
    logger.info("Testing database connections...")

    # Test OpenSearch connection
    try:
        opensearch_client = OpenSearchClient(config.opensearch)
        if opensearch_client.test_connection():
            logger.info("✅ OpenSearch connection successful")
        else:
            logger.error("❌ OpenSearch connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ OpenSearch connection error: {e}")
        return False

    # Test Vector DB connection
    try:
        vector_db_client = VectorDBClient(config.vector_db)
        if vector_db_client.test_connection():
            logger.info("✅ Vector database connection successful")
        else:
            logger.error("❌ Vector database connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Vector database connection error: {e}")
        return False

    # Test RDS connection (optional)
    try:
        from error_log_monitor.rds_client import create_rds_client

        rds_client = create_rds_client(config.rds)
        if rds_client:
            logger.info("✅ RDS connection successful")
        else:
            logger.warning("⚠️ RDS connection not configured")
    except Exception as e:
        logger.warning(f"⚠️ RDS connection error: {e}")

    logger.info("All connection tests completed")
    return True


def run_analysis(site: str, start_date: datetime, end_date: datetime):
    """Run error analysis for specified site and date range."""
    logger.info(f"Starting error analysis for site: {site}")
    logger.info(f"Date range: {start_date} to {end_date}")

    try:
        # Load configuration
        config = load_config()

        # Initialize clients
        opensearch_client = OpenSearchClient(config.opensearch)
        vector_db_client = VectorDBClient(config.vector_db)
        rag_engine = RAGEngine(config, vector_db_client)
        error_analyzer = ErrorAnalyzer(config, rag_engine)

        # Retrieve error logs
        logger.info("Retrieving error logs from OpenSearch...")
        error_logs = opensearch_client.get_error_logs(site, start_date, end_date)

        if not error_logs:
            logger.warning(f"No error logs found for site {site} in the specified date range")
            return

        logger.info(f"Retrieved {len(error_logs)} error logs")

        # Merge similar issues using RAG
        logger.info("Merging similar issues using RAG...")
        merged_issues = rag_engine.merge_similar_issues(error_logs)
        logger.info(f"Merged into {len(merged_issues)} similar issue groups")

        # Analyze merged issues
        logger.info("Analyzing merged issues...")
        analyses = error_analyzer.analyze_merged_issues(merged_issues)
        logger.info(f"Completed analysis of {len(analyses)} issues")

        # Generate report
        generate_report(analyses, site, start_date, end_date)

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


def generate_report(analyses: List, site: str, start_date: datetime, end_date: datetime):
    """Generate analysis report."""
    logger.info("Generating analysis report...")

    # Create report data
    report_data = {
        "site": site,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "summary": {
            "total_issues": len(analyses),
            "level_1_issues": len([a for a in analyses if a.severity.value == "LEVEL_1"]),
            "level_2_issues": len([a for a in analyses if a.severity.value == "LEVEL_2"]),
            "level_3_issues": len([a for a in analyses if a.severity.value == "LEVEL_3"]),
            "human_action_needed": len([a for a in analyses if a.remediation_plan.human_action_needed]),
            "total_cost": sum(a.estimated_cost for a in analyses),
        },
        "issues": [],
    }

    # Add issue details
    for analysis in analyses:
        issue_data = {
            "error_id": analysis.error_id,
            "error_message": analysis.error_message,
            "severity": analysis.severity.value,
            "service": analysis.service,
            "confidence_score": analysis.confidence_score,
            "estimated_cost": analysis.estimated_cost,
            "analysis_time": analysis.analysis_time,
            "scope": {
                "affected_services": analysis.scope.affected_services,
                "technical_impact": analysis.scope.technical_impact,
                "estimated_downtime": analysis.scope.estimated_downtime,
            },
            "remediation_plan": {
                "human_action_needed": analysis.remediation_plan.human_action_needed,
                "action_guidelines": analysis.remediation_plan.action_guidelines,
                "damaged_modules": analysis.remediation_plan.damaged_modules,
                "root_cause": analysis.remediation_plan.root_cause,
                "urgency": analysis.remediation_plan.urgency,
            },
            "analysis_model": analysis.analysis_model,
        }
        report_data["issues"].append(issue_data)

    # Save JSON report
    import json

    report_file = f"reports/analysis_report_{site}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    # Also save as latest.json for web interface
    latest_file = "reports/latest.json"
    with open(latest_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    logger.info(f"Report saved to: {report_file}")
    logger.info(f"Latest report saved to: {latest_file}")

    # Generate HTML report
    generate_html_report(report_data, site)


def generate_html_report(report_data: dict, site: str):
    """Generate HTML report."""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Analysis Report - {site}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .issue {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .severity-level-3 {{ border-left: 5px solid #d32f2f; background-color: #ffebee; }}
        .severity-level-2 {{ border-left: 5px solid #f57c00; background-color: #fff3e0; }}
        .severity-level-1 {{ border-left: 5px solid #1976d2; background-color: #e3f2fd; }}
        .action-needed {{ background-color: #ffcdd2; padding: 10px; border-radius: 3px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Error Analysis Report - {site}</h1>
        <p>Analysis Date: {report_data['analysis_date']}</p>
        <p>Date Range: {report_data['date_range']['start']} to {report_data['date_range']['end']}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Issues</h3>
            <p style="font-size: 24px; font-weight: bold;">{report_data['summary']['total_issues']}</p>
        </div>
        <div class="summary-card">
            <h3>Level 1 Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #1976d2;">{report_data['summary']['level_1_issues']}</p>
        </div>
        <div class="summary-card">
            <h3>Level 2 Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #f57c00;">{report_data['summary']['level_2_issues']}</p>
        </div>
        <div class="summary-card">
            <h3>Level 3 Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #d32f2f;">{report_data['summary']['level_3_issues']}</p>
        </div>
        <div class="summary-card">
            <h3>Human Action Needed</h3>
            <p style="font-size: 24px; font-weight: bold; color: #d32f2f;">{report_data['summary']['human_action_needed']}</p>
        </div>
        <div class="summary-card">
            <h3>Total Cost</h3>
            <p style="font-size: 24px; font-weight: bold;">${report_data['summary']['total_cost']:.4f}</p>
        </div>
    </div>

    <h2>Issue Details</h2>
"""

    for issue in report_data['issues']:
        action_class = "action-needed" if issue['remediation_plan']['human_action_needed'] else ""
        html_content += f"""
    <div class="issue severity-{issue['severity'].lower()} {action_class}">
        <h3>Issue: {issue['error_id']}</h3>
        <p><strong>Error Message:</strong> {issue['error_message']}</p>
        <p><strong>Severity:</strong> {issue['severity']}</p>
        <p><strong>Service:</strong> {issue['service']}</p>
        <p><strong>Confidence Score:</strong> {issue['confidence_score']:.2f}</p>
        <p><strong>Technical Impact:</strong> {issue['scope']['technical_impact']}</p>
        <p><strong>Affected Services:</strong> {', '.join(issue['scope']['affected_services'])}</p>
        <p><strong>Human Action Needed:</strong> {'Yes' if issue['remediation_plan']['human_action_needed'] else 'No'}</p>
        <p><strong>Urgency:</strong> {issue['remediation_plan']['urgency']}</p>
        <p><strong>Root Cause:</strong> {issue['remediation_plan']['root_cause'] or 'Not identified'}</p>
        <p><strong>Analysis Model:</strong> {issue['analysis_model']}</p>
        <p><strong>Estimated Cost:</strong> ${issue['estimated_cost']:.4f}</p>
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Save HTML report
    html_file = f"reports/analysis_report_{site}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_file, 'w') as f:
        f.write(html_content)

    logger.info(f"HTML report saved to: {html_file}")


def run_weekly_report(start_date: datetime, end_date: datetime):
    """Run weekly report generation."""
    logger.info(f"Starting weekly report generation from {start_date.date()} to {end_date.date()}")
    
    try:
        config = load_config()
        report_generator = WeeklyReportGenerator(config)
        
        # Generate the weekly report
        report_data = report_generator.generate_weekly_report(start_date, end_date)
        
        # Print summary
        print("\n📊 Weekly Report Summary:")
        print(f"  Period: {report_data['start_date'].date()} to {report_data['end_date'].date()}")
        print(f"  Total Error Logs: {report_data['total_error_logs']}")
        print(f"  Stage Logs: {report_data['stage_logs']}")
        print(f"  Production Logs: {report_data['prod_logs']}")
        
        # Print site-specific summaries
        for site_name, site_data in report_data['site_reports'].items():
            print(f"\n  {site_name.title()} Site:")
            print(f"    Error Logs: {site_data['total_error_logs']}")
            print(f"    Merged Issues: {site_data['merged_issues']}")
            print(f"    Weekly Issues: {site_data['weekly_issues']}")
            if site_data['excel_path']:
                print(f"    Excel Report: {site_data['excel_path']}")
            if site_data['html_path']:
                print(f"    HTML Report: {site_data['html_path']}")
        
        # Print combined report paths
        if report_data['combined_excel_path']:
            print(f"\n  Combined Excel Report: {report_data['combined_excel_path']}")
        if report_data['combined_html_path']:
            print(f"  Combined HTML Report: {report_data['combined_html_path']}")
        
        # Show sample issues from each site
        for site_name, site_data in report_data['site_reports'].items():
            if site_data['issues']:
                print(f"\n🔍 Sample Issues from {site_name.title()} Site:")
                for i, issue in enumerate(site_data['issues'][:3], 1):
                    print(f"  {i}. {issue.key} - {issue.summary[:50]}...")
                    print(f"     Site: {issue.site}, Count: {issue.count}, Status: {issue.status}")
                    if issue.note:
                        print(f"     Note: {issue.note[:100]}...")
                    print()
        
        logger.info("Weekly report generation completed successfully")

    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        raise


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Error Log Monitoring System")
    parser.add_argument("command", choices=["test", "analyze", "weekly-report"], help="Command to execute")
    parser.add_argument("--site", required=True, help="Site name (dev, stage, prod)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        if args.command == "test":
        config = load_config()
        if test_connections(config):
                logger.info("All tests passed successfully")
            sys.exit(0)
        else:
                logger.error("Some tests failed")
                sys.exit(1)

        elif args.command == "analyze":
            if not args.start_date or not args.end_date:
                logger.error("Start date and end date are required for analysis")
            sys.exit(1)

            # Parse dates
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Run analysis
            run_analysis(args.site, start_date, end_date)
            logger.info("Analysis completed successfully")

        elif args.command == "weekly-report":
            if not args.start_date or not args.end_date:
                logger.error("Start date and end date are required for weekly report")
                sys.exit(1)

            # Parse dates
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # Generate weekly report
            config = load_config()
            if not test_connections(config):
                logger.error("Connection tests failed")
                sys.exit(1)

            run_weekly_report(start_date, end_date)
            logger.info("Weekly report completed successfully")

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
