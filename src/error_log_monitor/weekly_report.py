"""
Weekly Report module for generating comprehensive error analysis reports.
"""

import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from jinja2 import Template

from error_log_monitor.config import SystemConfig
from error_log_monitor.opensearch_client import OpenSearchClient
from error_log_monitor.vector_db_client import VectorDBClient
from error_log_monitor.rag_engine import RAGEngine, MergedIssue
from error_log_monitor.jira_helper import JiraHelper, JiraIssue
from error_log_monitor.jira_cloud_client import JiraCloudClient
from error_log_monitor.error_analyzer import ErrorAnalyzer
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB
from error_log_monitor.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def sanitize_for_excel(text: str) -> str:
    """Sanitize text for Excel compatibility by removing illegal characters."""
    if not isinstance(text, str):
        return str(text)

    # Remove control characters (ASCII 0-31 except tab, newline, carriage return)
    # Keep only printable characters and common whitespace
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Replace any remaining problematic characters with safe alternatives
    sanitized = sanitized.replace('\x00', '')  # Null character
    sanitized = sanitized.replace('\x01', '')  # Start of heading
    sanitized = sanitized.replace('\x02', '')  # Start of text
    sanitized = sanitized.replace('\x03', '')  # End of text
    sanitized = sanitized.replace('\x04', '')  # End of transmission
    sanitized = sanitized.replace('\x05', '')  # Enquiry
    sanitized = sanitized.replace('\x06', '')  # Acknowledge
    sanitized = sanitized.replace('\x07', '')  # Bell
    sanitized = sanitized.replace('\x08', '')  # Backspace
    sanitized = sanitized.replace('\x0b', '')  # Vertical tab
    sanitized = sanitized.replace('\x0c', '')  # Form feed
    sanitized = sanitized.replace('\x0e', '')  # Shift out
    sanitized = sanitized.replace('\x0f', '')  # Shift in
    sanitized = sanitized.replace('\x10', '')  # Data link escape
    sanitized = sanitized.replace('\x11', '')  # Device control 1
    sanitized = sanitized.replace('\x12', '')  # Device control 2
    sanitized = sanitized.replace('\x13', '')  # Device control 3
    sanitized = sanitized.replace('\x14', '')  # Device control 4
    sanitized = sanitized.replace('\x15', '')  # Negative acknowledge
    sanitized = sanitized.replace('\x16', '')  # Synchronous idle
    sanitized = sanitized.replace('\x17', '')  # End of transmission block
    sanitized = sanitized.replace('\x18', '')  # Cancel
    sanitized = sanitized.replace('\x19', '')  # End of medium
    sanitized = sanitized.replace('\x1a', '')  # Substitute
    sanitized = sanitized.replace('\x1b', '')  # Escape
    sanitized = sanitized.replace('\x1c', '')  # File separator
    sanitized = sanitized.replace('\x1d', '')  # Group separator
    sanitized = sanitized.replace('\x1e', '')  # Record separator
    sanitized = sanitized.replace('\x1f', '')  # Unit separator
    sanitized = sanitized.replace('\x7f', '')  # Delete

    # Limit length to prevent Excel issues
    if len(sanitized) > 32767:  # Excel cell character limit
        sanitized = sanitized[:32764] + "..."

    return sanitized


@dataclass
class WeeklyReportIssue:
    """Weekly report issue combining merged error logs with Jira issues."""

    key: str  # Jira issue URL (parent issue if available)
    site: str
    count: int  # Occurrence count within the week
    summary: str  # Jira issue summary
    error_message: str  # Error message preview (first 100 characters)
    status: str  # Jira issue status
    log_group: str
    latest_update: datetime  # Latest error timestamp
    note: str  # Root cause description from LLM
    child_issues: List[str]  # Child Jira issue keys
    primary_jira_issue: Optional[JiraIssue] = None
    merged_issue: Optional[MergedIssue] = None


class WeeklyReportGenerator:
    """Generator for weekly error analysis reports."""

    def __init__(self, config: SystemConfig):
        """Initialize weekly report generator."""
        self.config = config
        self.opensearch_client = OpenSearchClient(config.opensearch)
        self.vector_db_client = VectorDBClient(config.vector_db)
        self.rag_engine = RAGEngine(config, self.vector_db_client)
        self.jira_helper = JiraHelper(config.opensearch)
        self.jira_cloud_client = JiraCloudClient(config.jira)
        self.error_analyzer = ErrorAnalyzer(config, self.rag_engine)

        # Initialize Jira Issue Embedding Database
        self.embedding_service = EmbeddingService(model_name=config.vector_db.embedding_model)
        self.jira_embedding_db = JiraIssueEmbeddingDB(
            embedding_service=self.embedding_service, config=config
        )

    def generate_weekly_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate weekly report for the specified date range, separated by site.

        Args:
            start_date: Start date for the report
            end_date: End date for the report

        Returns:
            Dictionary containing report data and file paths for each site
        """
        try:
            logger.info(f"Generating weekly report from {start_date.date()} to {end_date.date()}")

            # Step 1: Grab error logs from OpenSearch for stage and prod sites
            logger.info("Step 1: Retrieving error logs from OpenSearch...")
            stage_logs = self.opensearch_client.get_error_logs("stage", start_date, end_date)
            prod_logs = self.opensearch_client.get_error_logs("prod", start_date, end_date)

            all_logs = stage_logs + prod_logs
            logger.info(f"Retrieved {len(all_logs)} error logs ({len(stage_logs)} stage, {len(prod_logs)} prod)")

            if not all_logs:
                logger.warning("No error logs found for the specified period")
                return self._create_empty_report(start_date, end_date)

            # Process each site separately
            site_reports = {}
            sites = [("stage", stage_logs), ("prod", prod_logs)]

            for site_name, site_logs in sites:
                if not site_logs:
                    logger.info(f"No logs found for {site_name} site")
                    site_reports[site_name] = self._create_empty_site_report(site_name, start_date, end_date)
                    continue

                logger.info(f"Processing {site_name} site with {len(site_logs)} logs...")

                # Step 2: Merge similar issues using RAG engine for this site
                logger.info(f"Step 2: Merging similar issues for {site_name} site...")
                merged_issues = self.rag_engine.merge_similar_issues(site_logs)
                logger.info(
                    f"Created {len(merged_issues)} merged issues for {site_name} from {len(site_logs)} error logs"
                )

                # Step 3: Correlate with Jira issues for this site
                logger.info(f"Step 3: Correlating merged issues with Jira issues for {site_name}...")
                weekly_issues = self._correlate_with_jira_issues(merged_issues, start_date, end_date)
                logger.info(f"Correlated {len(weekly_issues)} issues with Jira for {site_name}")

                # Step 4: Generate LLM analysis for root cause for this site
                logger.info(f"Step 4: Generating LLM analysis for root cause for {site_name}...")
                weekly_issues = self._generate_root_cause_analysis(weekly_issues)

                # Step 5: Generate reports for this site
                logger.info(f"Step 5: Generating Excel and HTML reports for {site_name}...")
                excel_path = self._generate_excel_report(weekly_issues, start_date, end_date, site_name)
                html_path = self._generate_html_report(weekly_issues, start_date, end_date, site_name)

                site_reports[site_name] = {
                    "site": site_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_error_logs": len(site_logs),
                    "merged_issues": len(merged_issues),
                    "weekly_issues": len(weekly_issues),
                    "excel_path": excel_path,
                    "html_path": html_path,
                    "issues": weekly_issues,
                }

            # Create combined report data
            combined_report = {
                "start_date": start_date,
                "end_date": end_date,
                "total_error_logs": len(all_logs),
                "stage_logs": len(stage_logs),
                "prod_logs": len(prod_logs),
                "site_reports": site_reports,
                "combined_excel_path": self._generate_combined_excel_report(site_reports, start_date, end_date),
                "combined_html_path": self._generate_combined_html_report(site_reports, start_date, end_date),
            }

            logger.info("Weekly report generated successfully for all sites")
            return combined_report

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}", exc_info=True)
            raise

    def _correlate_with_jira_issues(
        self, merged_issues: List[MergedIssue], start_date: datetime, end_date: datetime
    ) -> List[WeeklyReportIssue]:
        """Correlate merged issues with Jira issues using multiple search strategies."""
        weekly_issues = []
        used_jira_issues = set()  # Track used Jira issues globally

        for merged_issue in merged_issues:
            try:
                # Try multiple correlation strategies
                jira_issues = self._find_correlated_jira_issues(merged_issue, start_date, end_date)

                if jira_issues:
                    # Filter out already used Jira issues
                    available_issues = [issue for issue in jira_issues if issue.issue_key not in used_jira_issues]

                    if available_issues:
                        # Sort by issue_time to get the latest as primary
                        available_issues.sort(key=lambda x: x.issue_time, reverse=True)
                        primary_issue = available_issues[0]
                        # Use child_issue_keys from Jira Cloud API if available, otherwise use other correlated issues
                        child_issues = primary_issue.child_issue_keys or [
                            issue.issue_key for issue in available_issues[1:]
                        ]

                        # Mark this Jira issue as used
                        used_jira_issues.add(primary_issue.issue_key)
                    else:
                        # No available Jira issues, create placeholder
                        primary_issue = None
                        child_issues = []

                    # Calculate occurrence count within the week
                    count = self._calculate_occurrence_count(merged_issue, start_date, end_date)

                    # Get latest error timestamp
                    latest_update = max(
                        log.timestamp for log in merged_issue.similar_logs + [merged_issue.representative_log]
                    )

                    if primary_issue:
                        # Create issue with Jira correlation
                        # Use parent_issue_key if available, otherwise use primary issue key
                        issue_key = primary_issue.parent_issue_key or primary_issue.issue_key
                        issue_url = primary_issue.jira_url or f"JIRA-{issue_key}"

                        # Get error message preview
                        error_message_preview = (
                            merged_issue.representative_log.error_message[:100]
                            if merged_issue.representative_log.error_message
                            else "No error message available"
                        )

                        # Determine status with fallback
                        issue_status = "Unknown"
                        if primary_issue.jira_status and primary_issue.jira_status != "Unknown":
                            issue_status = primary_issue.jira_status
                        elif hasattr(primary_issue, 'status') and primary_issue.status:
                            issue_status = primary_issue.status
                        else:
                            # Use a more descriptive default status based on Jira Cloud availability
                            if self.jira_cloud_client.client:
                                issue_status = "Status Not Available"
                            else:
                                issue_status = "Jira Cloud Not Connected"

                        logger.debug(
                            f"Jira issue {primary_issue.issue_key} status: {issue_status} "
                            f"(jira_status: {primary_issue.jira_status})"
                        )

                        weekly_issue = WeeklyReportIssue(
                            key=issue_url,
                            site=merged_issue.representative_log.site,
                            count=count,
                            summary=(primary_issue.jira_summary or "No summary available"),
                            error_message=error_message_preview,
                            status=issue_status,
                            log_group=(merged_issue.representative_log.log_group or "Unknown"),
                            latest_update=latest_update,
                            note="",  # Will be filled by LLM analysis
                            child_issues=child_issues,
                            primary_jira_issue=primary_issue,
                            merged_issue=merged_issue,
                        )
                    else:
                        # Create placeholder issue (no Jira correlation available)
                        # Get error message preview
                        error_message_preview = (
                            merged_issue.representative_log.error_message[:100]
                            if merged_issue.representative_log.error_message
                            else "No error message available"
                        )

                        weekly_issue = WeeklyReportIssue(
                            key="No Jira Issue",
                            site=merged_issue.representative_log.site,
                            count=count,
                            summary="No Jira issue found",
                            error_message=error_message_preview,
                            status="Not Tracked",
                            log_group=(merged_issue.representative_log.log_group or "Unknown"),
                            latest_update=latest_update,
                            note="",  # Will be filled by LLM analysis
                            child_issues=[],
                            primary_jira_issue=None,
                            merged_issue=merged_issue,
                        )

                    weekly_issues.append(weekly_issue)

            except Exception as e:
                logger.error(f"Error correlating merged issue {merged_issue.issue_id} with Jira: {e}")
                continue

        return weekly_issues

    def _find_correlated_jira_issues(
        self, merged_issue: MergedIssue, start_date: datetime, end_date: datetime
    ) -> List[JiraIssue]:
        """Find Jira issues using embedding-based similarity correlation with JiraIssueEmbeddingDB."""
        error_message = merged_issue.representative_log.error_message
        error_type = merged_issue.representative_log.error_type
        traceback = merged_issue.representative_log.traceback
        site = merged_issue.representative_log.site

        logger.info(f"Finding correlated Jira issues for site: {site}")
        logger.info(f"Error message: {error_message[:100]}...")

        try:
            # Ensure the embedding database index exists
            if not self.jira_embedding_db.client.client.indices.exists(
                index=self.jira_embedding_db.get_current_index_name()
            ):
                logger.info("Creating Jira embedding database index")
                self.jira_embedding_db.create_index()

            # Calculate embedding for the current error log (automatically normalized)
            current_error_text = self._combine_error_fields(error_message, error_type, traceback)

            # Check if text input is empty or only whitespace
            if not current_error_text or not current_error_text.strip():
                logger.warning("Empty text input for error log correlation, skipping embedding generation")
                return []

            try:
                current_embedding = self.rag_engine.embedding_service.generate_embedding(current_error_text)
            except (ValueError, RuntimeError) as e:
                logger.error(f"Failed to generate embedding for error log correlation: {e}")
                return []

            # Use JiraIssueEmbeddingDB to find similar issues
            logger.info("Searching for similar Jira issues using embedding database")
            similar_issue = self.jira_embedding_db.find_similar_jira_issue(
                error_log_embedding=current_embedding, site=site, similarity_threshold=0.85
            )

            if not similar_issue:
                logger.info("No similar Jira issues found in embedding database")
                return []

            # Convert the result to JiraIssue format
            correlated_issues = []
            jira_issue = JiraIssue(
                issue_key=similar_issue["key"],
                site=similar_issue["site"],
                error_message=similar_issue["error_message"],
                error_type=similar_issue["error_type"],
                traceback=similar_issue["traceback"],
                jira_summary=similar_issue["summary"],
                jira_status=similar_issue["status"],
                parent_issue_key=similar_issue["parent_issue_key"],
                child_issue_keys=[],  # Will be populated from Jira Cloud if available
                assignee="",  # Will be populated from Jira Cloud if available
                priority="",  # Will be populated from Jira Cloud if available
                issue_type="",  # Will be populated from Jira Cloud if available
                created=similar_issue["created"],
                updated=similar_issue["updated"],
                description=similar_issue["description"],
            )

            correlated_issues.append(jira_issue)
            logger.info(
                f"Found similar Jira issue: {similar_issue['key']} with score: {similar_issue.get('score', 'N/A')}"
            )

            # Fetch additional details from Jira Cloud API
            if correlated_issues and self.jira_cloud_client.client:
                logger.info("Fetching additional details from Jira Cloud API")
                issue_keys = [issue.issue_key for issue in correlated_issues]
                jira_details = self.jira_cloud_client.get_multiple_issue_details(issue_keys)

                logger.info(f"Retrieved details for {len(jira_details)} out of {len(issue_keys)} issues")

                # Update JiraIssue objects with additional details
                for issue in correlated_issues:
                    if issue.issue_key in jira_details:
                        details = jira_details[issue.issue_key]
                        logger.debug(f"Updating issue {issue.issue_key} with status: {details.status}")
                        # Update the issue with additional information
                        issue.jira_summary = details.summary
                        issue.jira_status = details.status
                        issue.parent_issue_key = details.parent_issue_key
                        issue.child_issue_keys = details.child_issue_keys
                        issue.assignee = details.assignee
                        issue.priority = details.priority
                        issue.issue_type = details.issue_type
                        issue.created = details.created
                        issue.updated = details.updated
                        issue.description = details.description
                    else:
                        logger.warning(f"No Jira Cloud details found for issue {issue.issue_key}")
            else:
                if correlated_issues:
                    logger.warning("Jira Cloud client not available, using embedding database data only")
                    for issue in correlated_issues:
                        logger.debug(f"Issue {issue.issue_key} jira_status: {issue.jira_status}")

            logger.info(f"Found {len(correlated_issues)} similar Jira issues using embedding database")
            return correlated_issues

        except Exception as e:
            logger.error(f"Error in embedding-based correlation: {e}")
            # Fallback to original method if embedding database fails
            logger.info("Falling back to original correlation method")
            return self._find_correlated_jira_issues_fallback(merged_issue, start_date, end_date)

    def _find_correlated_jira_issues_fallback(
        self, merged_issue: MergedIssue, start_date: datetime, end_date: datetime
    ) -> List[JiraIssue]:
        """Fallback method for finding Jira issues using original correlation logic."""
        error_message = merged_issue.representative_log.error_message
        error_type = merged_issue.representative_log.error_type
        traceback = merged_issue.representative_log.traceback
        site = merged_issue.representative_log.site

        logger.info(f"Using fallback correlation method for site: {site}")

        # Step 1: Grab all issues from Jira (past 0.5 year)
        logger.info("Step 1: Retrieving all Jira issues from past 0.5 year")
        all_jira_issues = self.jira_helper.get_recent_issues(days=180, site=site)  # 0.5 year = ~180 days
        logger.info(f"Retrieved {len(all_jira_issues)} Jira issues from past 0.5 year")

        if not all_jira_issues:
            logger.warning("No Jira issues found for correlation")
            return []

        # Step 2: Calculate embedding of each Jira issue using error_message, error_type and traceback
        logger.info("Step 2: Calculating embeddings for Jira issues")
        jira_embeddings = self._calculate_jira_issue_embeddings(all_jira_issues)

        # Calculate embedding for the current error log
        current_error_text = self._combine_error_fields(error_message, error_type, traceback)
        current_embedding = self.rag_engine.embedding_service.generate_embedding(current_error_text)

        # Step 3: Find similar issues that exceed similarity threshold (0.85) and same site
        logger.info("Step 3: Finding similar issues with 0.85 similarity threshold")
        from error_log_monitor.embedding_service import cosine_similarity

        similar_issues = []
        similarity_threshold = 0.85

        for i, jira_issue in enumerate(all_jira_issues):
            if jira_issue.site != site:
                continue  # Skip issues from different sites

            jira_embedding = jira_embeddings[i]
            similarity = cosine_similarity(current_embedding, jira_embedding)

            if similarity >= similarity_threshold:
                similar_issues.append((jira_issue, similarity))
                logger.debug(f"Jira issue {jira_issue.issue_key} similarity: {similarity:.3f}")

        # Sort by similarity (highest first)
        similar_issues.sort(key=lambda x: x[1], reverse=True)

        # Return only the JiraIssue objects (without similarity scores)
        correlated_issues = [issue for issue, similarity in similar_issues]

        # Fetch additional details from Jira Cloud API
        if correlated_issues and self.jira_cloud_client.client:
            logger.info("Fetching additional details from Jira Cloud API")
            issue_keys = [issue.issue_key for issue in correlated_issues]
            jira_details = self.jira_cloud_client.get_multiple_issue_details(issue_keys)

            logger.info(f"Retrieved details for {len(jira_details)} out of {len(issue_keys)} issues")

            # Update JiraIssue objects with additional details
            for issue in correlated_issues:
                if issue.issue_key in jira_details:
                    details = jira_details[issue.issue_key]
                    logger.debug(f"Updating issue {issue.issue_key} with status: {details.status}")
                    # Update the issue with additional information
                    issue.jira_summary = details.summary
                    issue.jira_status = details.status
                    issue.parent_issue_key = details.parent_issue_key
                    issue.child_issue_keys = details.child_issue_keys
                    issue.assignee = details.assignee
                    issue.priority = details.priority
                    issue.issue_type = details.issue_type
                    issue.created = details.created
                    issue.updated = details.updated
                    issue.description = details.description
                else:
                    logger.warning(f"No Jira Cloud details found for issue {issue.issue_key}")
        else:
            if correlated_issues:
                logger.warning("Jira Cloud client not available, using OpenSearch data only")
                for issue in correlated_issues:
                    logger.debug(f"Issue {issue.issue_key} jira_status: {issue.jira_status}")

        logger.info(f"Found {len(correlated_issues)} similar Jira issues with similarity >= {similarity_threshold}")
        return correlated_issues

    def _calculate_jira_issue_embeddings(self, jira_issues: List[JiraIssue]) -> List[List[float]]:
        """Calculate embeddings for a list of Jira issues."""
        try:
            # Combine error fields for each Jira issue
            jira_texts = []
            for jira_issue in jira_issues:
                combined_text = self._combine_error_fields(
                    jira_issue.error_message or "", jira_issue.error_type or "", jira_issue.traceback or ""
                )
                jira_texts.append(combined_text)

            # Generate embeddings for all Jira issues at once
            embeddings = self.rag_engine.embedding_service.generate_embeddings(jira_texts)
            logger.info(f"Generated embeddings for {len(embeddings)} Jira issues")
            return embeddings

        except Exception as e:
            logger.error(f"Error calculating Jira issue embeddings: {e}")
            return []

    def _combine_error_fields(self, error_message: str, error_type: str, traceback: str) -> str:
        """Combine error fields into a single text for embedding calculation."""
        # Filter out None values and empty strings
        fields = [field for field in [error_message, error_type, traceback] if field and field.strip()]

        # Join with newlines for better separation
        combined_text = "\n".join(fields)

        # Limit length to prevent excessive token usage
        if len(combined_text) > 8000:  # Reasonable limit for embedding
            combined_text = combined_text[:8000] + "..."

        return combined_text

    def _extract_keywords_with_gpt5_nano(self, error_message: str) -> List[str]:
        """Extract keywords from error message using GPT-5-nano."""
        try:
            import openai

            # Use GPT-5-nano for keyword extraction
            client = openai.OpenAI(api_key=self.config.openai_api_key)

            prompt = f"""
Extract the most important keywords from this error message that would help find related Jira issues.
Return only the keywords as a comma-separated list, no explanations.

Error message: {error_message}

Keywords:"""

            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=100,
                temperature=0.1,
            )

            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

            logger.info(f"GPT-5-nano extracted keywords: {keywords}")
            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords with GPT-5-nano: {e}")
            return []

    def _find_issues_by_keyword_matching(
        self, keywords: List[str], days: int, site: str, threshold: float = 0.9
    ) -> List[JiraIssue]:
        """Find Jira issues that match a threshold percentage of keywords."""
        try:
            if not keywords:
                return []

            # Get all recent Jira issues for the site
            all_issues = self.jira_helper.get_recent_issues(days=days, site=site)

            matching_issues = []

            for issue in all_issues:
                # Check keyword match percentage
                match_percentage = self._calculate_keyword_match_percentage(keywords, issue.error_message)

                if match_percentage >= threshold:
                    matching_issues.append(issue)
                    logger.debug(f"Issue {issue.issue_key} matched {match_percentage:.2%} of keywords")

            # Sort by match percentage (highest first)
            matching_issues.sort(
                key=lambda x: self._calculate_keyword_match_percentage(keywords, x.error_message), reverse=True
            )

            logger.info(f"Found {len(matching_issues)} issues matching {threshold:.0%} of keywords")
            return matching_issues

        except Exception as e:
            logger.error(f"Error in keyword matching: {e}")
            return []

    def _calculate_keyword_match_percentage(self, keywords: List[str], error_message: str) -> float:
        """Calculate the percentage of keywords that match in the error message."""
        if not keywords or not error_message:
            return 0.0

        error_lower = error_message.lower()
        matched_keywords = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in error_lower:
                matched_keywords += 1

        return matched_keywords / len(keywords)

    def _calculate_occurrence_count(self, merged_issue: MergedIssue, start_date: datetime, end_date: datetime) -> int:
        """Calculate occurrence count within the specified date range."""
        count = 0
        for log in merged_issue.similar_logs + [merged_issue.representative_log]:
            if start_date <= log.timestamp <= end_date:
                count += log.count
        return count

    def _generate_root_cause_analysis(self, weekly_issues: List[WeeklyReportIssue]) -> List[WeeklyReportIssue]:
        """Generate LLM analysis for root cause of each issue."""
        for weekly_issue in weekly_issues:
            try:
                if weekly_issue.merged_issue:
                    # Use error analyzer to get root cause analysis
                    analysis_result = self.error_analyzer.analyze_issue(weekly_issue.merged_issue)

                    if analysis_result and analysis_result.remediation_plan:
                        weekly_issue.note = analysis_result.remediation_plan.root_cause or "Root cause not identified"
                    else:
                        weekly_issue.note = "Analysis failed or incomplete"
                else:
                    weekly_issue.note = "No merged issue available for analysis"

            except Exception as e:
                logger.error(f"Error generating root cause analysis for {weekly_issue.key}: {e}")
                weekly_issue.note = f"Analysis error: {str(e)[:100]}"

        return weekly_issues

    def _generate_excel_report(
        self, weekly_issues: List[WeeklyReportIssue], start_date: datetime, end_date: datetime, site: str = "combined"
    ) -> str:
        """Generate Excel report."""
        try:
            # Prepare data for Excel
            data = []
            for issue in weekly_issues:
                data.append(
                    {
                        "Key": sanitize_for_excel(issue.key),
                        "Site": sanitize_for_excel(issue.site),
                        "Count": issue.count,
                        "Summary": sanitize_for_excel(issue.summary),
                        "Error_Message": sanitize_for_excel(issue.error_message),
                        "Status": sanitize_for_excel(issue.status),
                        "Log Group": sanitize_for_excel(issue.log_group),
                        "Latest Update": issue.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                        "Note": sanitize_for_excel(issue.note),
                        "Child Issues": sanitize_for_excel(
                            ", ".join(issue.child_issues) if issue.child_issues else "None"
                        ),
                    }
                )

            # Create DataFrame
            df = pd.DataFrame(data)

            # Generate filename
            filename = f"weekly_report_{site}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
            filepath = os.path.join("reports", filename)

            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)

            # Write to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'{site.title()} Report', index=False)

                # Add summary sheet
                summary_data = {
                    "Metric": [
                        "Total Issues",
                        "Issues with Jira",
                        "Issues without Jira",
                    ],
                    "Count": [
                        len(weekly_issues),
                        len([i for i in weekly_issues if i.primary_jira_issue is not None]),
                        len([i for i in weekly_issues if i.primary_jira_issue is None]),
                    ],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            logger.info(f"Excel report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            raise

    def _generate_html_report(
        self, weekly_issues: List[WeeklyReportIssue], start_date: datetime, end_date: datetime, site: str = "combined"
    ) -> str:
        """Generate HTML report."""
        try:
            # Generate filename
            filename = f"weekly_report_{site}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
            filepath = os.path.join("reports", filename)

            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)

            # Prepare data for template
            report_data = {
                "site": site,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "total_issues": len(weekly_issues),
                "jira_issues": len([i for i in weekly_issues if i.primary_jira_issue is not None]),
                "issues": weekly_issues,
            }

            # Generate HTML content
            html_content = self._create_html_template(report_data)

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise

    def _create_html_template(self, report_data: Dict[str, Any]) -> str:
        """Create HTML template for weekly report."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Error Report - {{ start_date }} to {{ end_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px;
                  border-radius: 5px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 20px; margin: 20px 0; }
        .summary-card { background-color: #e8f4f8; padding: 15px; border-radius: 5px;
                        text-align: center; }
        .issues-table { background-color: white; border-radius: 5px; overflow: hidden;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .jira-link { color: #3498db; text-decoration: none; }
        .jira-link:hover { text-decoration: underline; }
        .status-tracked { color: #27ae60; font-weight: bold; }
        .status-untracked { color: #e74c3c; font-weight: bold; }
        .note { max-width: 300px; word-wrap: break-word; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Weekly Error Report - {{ site.title() }} Site</h1>
        <p>Period: {{ start_date }} to {{ end_date }}</p>
        <p>Generated on: {{ current_time }}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #2c3e50;">{{ total_issues }}</p>
        </div>
        <div class="summary-card">
            <h3>Jira Tracked</h3>
            <p style="font-size: 24px; font-weight: bold; color: #27ae60;">{{ jira_issues }}</p>
        </div>
        <div class="summary-card">
            <h3>Not Tracked</h3>
            <p style="font-size: 24px; font-weight: bold; color: #e74c3c;">{{ total_issues - jira_issues }}</p>
        </div>
    </div>

    <div class="issues-table">
        <table>
            <thead>
                <tr>
                    <th>Key</th>
                    <th>Site</th>
                    <th>Count</th>
                    <th>Summary</th>
                    <th>Error_Message</th>
                    <th>Status</th>
                    <th>Log Group</th>
                    <th>Latest Update</th>
                    <th>Note</th>
                    <th>Child Issues</th>
                </tr>
            </thead>
            <tbody>
                {% for issue in issues %}
                <tr>
                    <td>
                        {% if issue.key != "No Jira Issue" %}
                            <a href="{{ issue.key }}" class="jira-link" target="_blank">
                                {{ issue.key.split('/')[-1] if '/' in issue.key else issue.key }}
                            </a>
                        {% else %}
                            <span class="status-untracked">{{ issue.key }}</span>
                        {% endif %}
                    </td>
                    <td>{{ issue.site }}</td>
                    <td>{{ issue.count }}</td>
                    <td>{{ issue.summary[:100] }}{% if issue.summary|length > 100 %}...{% endif %}</td>
                    <td>{{ issue.error_message[:100] }}{% if issue.error_message|length > 100 %}...{% endif %}</td>
                    <td>
                        {% if issue.status == "Unknown" %}
                            <span class="status-tracked">{{ issue.status }}</span>
                        {% else %}
                            <span class="status-untracked">{{ issue.status }}</span>
                        {% endif %}
                    </td>
                    <td>{{ issue.log_group }}</td>
                    <td>{{ issue.latest_update.strftime('%Y-%m-%d %H:%M') }}</td>
                    <td class="note">{{ issue.note[:200] }}{% if issue.note|length > 200 %}...{% endif %}</td>
                    <td>{{ issue.child_issues|join(', ') if issue.child_issues else 'None' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        template = Template(template_str)
        return template.render(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **report_data)

    def _create_empty_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Create empty report when no data is found."""
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_error_logs": 0,
            "stage_logs": 0,
            "prod_logs": 0,
            "site_reports": {},
            "combined_excel_path": None,
            "combined_html_path": None,
        }

    def _create_empty_site_report(self, site: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Create empty report for a specific site when no data is found."""
        return {
            "site": site,
            "start_date": start_date,
            "end_date": end_date,
            "total_error_logs": 0,
            "merged_issues": 0,
            "weekly_issues": 0,
            "excel_path": None,
            "html_path": None,
            "issues": [],
        }

    def _generate_combined_excel_report(
        self, site_reports: Dict[str, Any], start_date: datetime, end_date: datetime
    ) -> str:
        """Generate combined Excel report with all sites."""
        try:
            # Generate filename
            filename = f"weekly_report_combined_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
            filepath = os.path.join("reports", filename)

            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)

            # Write to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Create combined data
                all_issues = []
                for site_name, site_data in site_reports.items():
                    for issue in site_data.get("issues", []):
                        all_issues.append(issue)

                if all_issues:
                    # Prepare data for Excel
                    data = []
                    for issue in all_issues:
                        data.append(
                            {
                                "Key": sanitize_for_excel(issue.key),
                                "Site": sanitize_for_excel(issue.site),
                                "Count": issue.count,
                                "Summary": sanitize_for_excel(issue.summary),
                                "Error_Message": sanitize_for_excel(issue.error_message),
                                "Status": sanitize_for_excel(issue.status),
                                "Log Group": sanitize_for_excel(issue.log_group),
                                "Latest Update": issue.latest_update.strftime("%Y-%m-%d %H:%M:%S"),
                                "Note": sanitize_for_excel(issue.note),
                                "Child Issues": sanitize_for_excel(
                                    ", ".join(issue.child_issues) if issue.child_issues else "None"
                                ),
                            }
                        )

                    # Create DataFrame
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name='Combined Report', index=False)

                # Add summary sheet
                summary_data = {
                    "Metric": [
                        "Total Issues",
                        "Stage Issues",
                        "Prod Issues",
                        "Issues with Jira",
                        "Issues without Jira",
                    ],
                    "Count": [
                        len(all_issues),
                        len([i for i in all_issues if i.site == "stage"]),
                        len([i for i in all_issues if i.site == "prod"]),
                        len([i for i in all_issues if i.primary_jira_issue is not None]),
                        len([i for i in all_issues if i.primary_jira_issue is None]),
                    ],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            logger.info(f"Combined Excel report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating combined Excel report: {e}")
            raise

    def _generate_combined_html_report(
        self, site_reports: Dict[str, Any], start_date: datetime, end_date: datetime
    ) -> str:
        """Generate combined HTML report with all sites."""
        try:
            # Generate filename
            filename = f"weekly_report_combined_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.html"
            filepath = os.path.join("reports", filename)

            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)

            # Prepare data for template
            all_issues = []
            for site_name, site_data in site_reports.items():
                for issue in site_data.get("issues", []):
                    all_issues.append(issue)

            report_data = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "total_issues": len(all_issues),
                "stage_issues": len([i for i in all_issues if i.site == "stage"]),
                "prod_issues": len([i for i in all_issues if i.site == "prod"]),
                "jira_issues": len([i for i in all_issues if i.primary_jira_issue is not None]),
                "issues": all_issues,
                "site_reports": site_reports,
            }

            # Generate HTML content
            html_content = self._create_combined_html_template(report_data)

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Combined HTML report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating combined HTML report: {e}")
            raise

    def _create_combined_html_template(self, report_data: Dict[str, Any]) -> str:
        """Create HTML template for combined weekly report."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Error Report - Combined - {{ start_date }} to {{ end_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px;
                  border-radius: 5px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 20px; margin: 20px 0; }
        .summary-card { background-color: #e8f4f8; padding: 15px; border-radius: 5px;
                        text-align: center; }
        .site-section { background-color: white; border-radius: 5px; margin: 20px 0;
                        padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .site-header { background-color: #34495e; color: white; padding: 10px 20px;
                       margin: -20px -20px 20px -20px; border-radius: 5px 5px 0 0; }
        .issues-table { background-color: white; border-radius: 5px; overflow: hidden;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .jira-link { color: #3498db; text-decoration: none; }
        .jira-link:hover { text-decoration: underline; }
        .status-tracked { color: #27ae60; font-weight: bold; }
        .status-untracked { color: #e74c3c; font-weight: bold; }
        .note { max-width: 300px; word-wrap: break-word; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Weekly Error Report - Combined</h1>
        <p>Period: {{ start_date }} to {{ end_date }}</p>
        <p>Generated on: {{ current_time }}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #2c3e50;">{{ total_issues }}</p>
        </div>
        <div class="summary-card">
            <h3>Stage Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #f39c12;">{{ stage_issues }}</p>
        </div>
        <div class="summary-card">
            <h3>Production Issues</h3>
            <p style="font-size: 24px; font-weight: bold; color: #e74c3c;">{{ prod_issues }}</p>
        </div>

        <div class="summary-card">
            <h3>Jira Tracked</h3>
            <p style="font-size: 24px; font-weight: bold; color: #27ae60;">{{ jira_issues }}</p>
        </div>
    </div>

    {% for site_name, site_data in site_reports.items() %}
    <div class="site-section">
        <div class="site-header">
            <h2>{{ site_name.title() }} Site - {{ site_data.weekly_issues }} Issues</h2>
        </div>

        {% if site_data.issues %}
        <div class="issues-table">
            <table>
                <thead>
                    <tr>
                        <th>Key</th>
                        <th>Count</th>
                        <th>Summary</th>
                        <th>Error_Message</th>
                        <th>Status</th>
                        <th>Log Group</th>
                        <th>Latest Update</th>
                        <th>Note</th>
                        <th>Child Issues</th>
                    </tr>
                </thead>
                <tbody>
                    {% for issue in site_data.issues %}
                    <tr>
                        <td>
                            {% if issue.key != "No Jira Issue" %}
                                <a href="{{ issue.key }}" class="jira-link" target="_blank">
                                    {{ issue.key.split('/')[-1] if '/' in issue.key else issue.key }}
                                </a>
                            {% else %}
                                <span class="status-untracked">{{ issue.key }}</span>
                            {% endif %}
                        </td>
                        <td>{{ issue.count }}</td>
                        <td>{{ issue.summary[:100] }}{% if issue.summary|length > 100 %}...{% endif %}</td>
                        <td>{{ issue.error_message[:100] }}{% if issue.error_message|length > 100 %}...{% endif %}</td>
                        <td>
                            {% if issue.status == "Unknown" %}
                                <span class="status-tracked">{{ issue.status }}</span>
                            {% else %}
                                <span class="status-untracked">{{ issue.status }}</span>
                            {% endif %}
                        </td>
                        <td>{{ issue.log_group }}</td>
                        <td>{{ issue.latest_update.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td class="note">{{ issue.note[:200] }}{% if issue.note|length > 200 %}...{% endif %}</td>
                        <td>{{ issue.child_issues|join(', ') if issue.child_issues else 'None' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <p>No issues found for {{ site_name }} site.</p>
        {% endif %}
    </div>
    {% endfor %}
</body>
</html>
        """

        template = Template(template_str)
        return template.render(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **report_data)
