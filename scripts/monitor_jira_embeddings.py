#!/usr/bin/env python3
"""
Monitoring dashboard for Jira Issue Embedding Database.

This script provides real-time monitoring, health checks, and performance metrics
for the Jira Issue Embedding Database system.
"""

import argparse
import json
import logging
import sys
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.config import load_config
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.embedding_service import EmbeddingService
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB

logger = logging.getLogger(__name__)


class JiraEmbeddingMonitor:
    """Monitoring dashboard for Jira Issue Embedding Database."""

    def __init__(self, config=None):
        """Initialize the monitor with configuration."""
        self.config = config or load_config()
        self.opensearch_client = OpenSearchClient(self.config.opensearch)
        self.embedding_service = EmbeddingService(model_name=self.config.vector_db.embedding_model)
        self.jira_embedding_db = JiraIssueEmbeddingDB(
            opensearch_client=self.opensearch_client, embedding_service=self.embedding_service, config=self.config
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the embedding database."""
        try:
            health_status = self.jira_embedding_db.health_check()
            return health_status
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "overall_status": "unhealthy", "error": str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            metrics = self.jira_embedding_db.performance_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}

    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary."""
        try:
            summary = self.jira_embedding_db.get_database_summary()
            return summary
        except Exception as e:
            logger.error(f"Failed to get database summary: {e}")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}

    def get_error_monitoring(self, hours: int = 24) -> Dict[str, Any]:
        """Get error monitoring data."""
        try:
            error_data = self.jira_embedding_db.error_rate_monitoring(hours)
            return error_data
        except Exception as e:
            logger.error(f"Failed to get error monitoring: {e}")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}

    def get_available_years(self) -> List[int]:
        """Get list of available years with data."""
        try:
            years = self.jira_embedding_db.get_available_years()
            return years
        except Exception as e:
            logger.error(f"Failed to get available years: {e}")
            return []

    def get_year_stats(self, year: int) -> Dict[str, Any]:
        """Get statistics for a specific year."""
        try:
            stats = self.jira_embedding_db.get_embedding_stats(year)
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for year {year}: {e}")
            return {"error": str(e)}

    def monitor_continuously(self, interval: int = 60, duration: int = None):
        """
        Monitor the database continuously.

        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds (None for infinite)
        """
        logger.info(f"Starting continuous monitoring (interval={interval}s, duration={duration}s)")

        start_time = time.time()
        iteration = 0

        try:
            while True:
                iteration += 1
                current_time = time.time()

                # Check if duration limit reached
                if duration and (current_time - start_time) >= duration:
                    logger.info("Monitoring duration reached, stopping")
                    break

                logger.info(f"Monitoring iteration {iteration}")

                # Get all monitoring data
                health = self.get_health_status()
                performance = self.get_performance_metrics()
                error_monitoring = self.get_error_monitoring()

                # Print status
                print(f"\n=== Monitoring Iteration {iteration} - {datetime.now().isoformat()} ===")
                print(f"Health Status: {health.get('overall_status', 'unknown')}")

                if health.get('overall_status') == 'unhealthy':
                    print("ERRORS:")
                    for error in health.get('errors', []):
                        print(f"  - {error}")

                # Print performance metrics
                if 'search_performance' in performance:
                    search_latency = performance['search_performance'].get('average_search_latency_ms', 0)
                    print(f"Search Latency: {search_latency:.2f}ms")

                if 'embedding_performance' in performance:
                    embedding_latency = performance['embedding_performance'].get('total_embedding_latency_ms', 0)
                    print(f"Embedding Latency: {embedding_latency:.2f}ms")

                # Print error rates
                if 'error_rate_percentage' in error_monitoring:
                    error_rate = error_monitoring['error_rate_percentage']
                    print(f"Error Rate: {error_rate:.2f}%")

                # Wait for next iteration
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")

    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive monitoring report.

        Args:
            output_file: Optional file to save the report

        Returns:
            Generated report
        """
        logger.info("Generating comprehensive monitoring report")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_info": {
                "index_template": self.jira_embedding_db.index_name_template,
                "current_index": self.jira_embedding_db.get_current_index_name(),
                "available_years": self.get_available_years(),
            },
            "health_status": self.get_health_status(),
            "performance_metrics": self.get_performance_metrics(),
            "error_monitoring": self.get_error_monitoring(),
            "year_statistics": {},
        }

        # Get stats for each available year
        for year in report["database_info"]["available_years"]:
            year_stats = self.get_year_stats(year)
            report["year_statistics"][str(year)] = year_stats

        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_file}: {e}")

        return report

    def check_alerts(self, thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.

        Args:
            thresholds: Alert thresholds (search_latency_ms, error_rate_percentage, etc.)

        Returns:
            List of active alerts
        """
        if thresholds is None:
            thresholds = {
                "search_latency_ms": 1000.0,  # 1 second
                "error_rate_percentage": 5.0,  # 5%
                "embedding_latency_ms": 2000.0,  # 2 seconds
            }

        alerts = []

        try:
            # Get current metrics
            health = self.get_health_status()
            performance = self.get_performance_metrics()
            error_monitoring = self.get_error_monitoring()

            # Check health status
            if health.get('overall_status') != 'healthy':
                alerts.append(
                    {
                        "type": "health",
                        "severity": "critical",
                        "message": f"Database health status: {health.get('overall_status')}",
                        "details": health.get('errors', []),
                    }
                )

            # Check search latency
            search_latency = performance.get('search_performance', {}).get('average_search_latency_ms', 0)
            if search_latency > thresholds['search_latency_ms']:
                alerts.append(
                    {
                        "type": "performance",
                        "severity": "warning",
                        "message": f"Search latency {search_latency:.2f}ms exceeds threshold {thresholds['search_latency_ms']}ms",
                    }
                )

            # Check embedding latency
            embedding_latency = performance.get('embedding_performance', {}).get('total_embedding_latency_ms', 0)
            if embedding_latency > thresholds['embedding_latency_ms']:
                alerts.append(
                    {
                        "type": "performance",
                        "severity": "warning",
                        "message": f"Embedding latency {embedding_latency:.2f}ms exceeds threshold {thresholds['embedding_latency_ms']}ms",
                    }
                )

            # Check error rate
            error_rate = error_monitoring.get('error_rate_percentage', 0)
            if error_rate > thresholds['error_rate_percentage']:
                alerts.append(
                    {
                        "type": "error_rate",
                        "severity": "critical",
                        "message": f"Error rate {error_rate:.2f}% exceeds threshold {thresholds['error_rate_percentage']}%",
                    }
                )

        except Exception as e:
            alerts.append({"type": "monitoring", "severity": "critical", "message": f"Failed to check alerts: {e}"})

        return alerts


def main():
    """Main entry point for the monitoring script."""
    parser = argparse.ArgumentParser(description="Jira Embedding Database Monitoring Dashboard")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", help="Output file for reports")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    health_parser = subparsers.add_parser("health", help="Check database health")

    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Get performance metrics")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get database summary")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start continuous monitoring")
    monitor_parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    monitor_parser.add_argument("--duration", type=int, help="Monitoring duration in seconds")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate monitoring report")
    report_parser.add_argument("--output", help="Output file for report")

    # Alerts command
    alerts_parser = subparsers.add_parser("alerts", help="Check for alerts")
    alerts_parser.add_argument(
        "--search-latency-threshold", type=float, default=1000.0, help="Search latency threshold in ms"
    )
    alerts_parser.add_argument("--error-rate-threshold", type=float, default=5.0, help="Error rate threshold in %")
    alerts_parser.add_argument(
        "--embedding-latency-threshold", type=float, default=2000.0, help="Embedding latency threshold in ms"
    )

    # Years command
    years_parser = subparsers.add_parser("years", help="List available years")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics for a year")
    stats_parser.add_argument("year", type=int, help="Year to get stats for")

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

    # Create monitor
    monitor = JiraEmbeddingMonitor(config)

    # Execute command
    try:
        if args.command == "health":
            result = monitor.get_health_status()
        elif args.command == "performance":
            result = monitor.get_performance_metrics()
        elif args.command == "summary":
            result = monitor.get_database_summary()
        elif args.command == "monitor":
            monitor.monitor_continuously(args.interval, args.duration)
            return
        elif args.command == "report":
            result = monitor.generate_report(args.output)
        elif args.command == "alerts":
            thresholds = {
                "search_latency_ms": args.search_latency_threshold,
                "error_rate_percentage": args.error_rate_threshold,
                "embedding_latency_ms": args.embedding_latency_threshold,
            }
            result = monitor.check_alerts(thresholds)
        elif args.command == "years":
            years = monitor.get_available_years()
            print(f"Available years: {years}")
            return
        elif args.command == "stats":
            result = monitor.get_year_stats(args.year)
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
