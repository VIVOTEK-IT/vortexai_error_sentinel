"""
Jira Cloud API client for fetching detailed issue information.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from jira import JIRA

    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False
    JIRA = None

from error_log_monitor.config import JiraConfig

logger = logging.getLogger(__name__)


@dataclass
class JiraIssueDetails:
    """Detailed Jira issue information from Jira Cloud API."""

    issue_key: str
    summary: str
    status: str
    parent_issue_key: Optional[str] = None
    child_issue_keys: List[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    site: Optional[str] = None
    request_id: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if self.child_issue_keys is None:
            self.child_issue_keys = []


class JiraCloudClient:
    """Client for interacting with Jira Cloud API."""

    def __init__(self, config: JiraConfig):
        """Initialize Jira Cloud client."""
        self.config = config
        self.client = None
        self._connect()

    def _connect(self):
        """Establish connection to Jira Cloud."""
        try:
            if not JIRA_AVAILABLE:
                logger.error("Jira Python library not available. Install with: pip install jira")
                return

            if not all([self.config.server_url, self.config.username, self.config.api_token]):
                logger.warning("Jira configuration incomplete, skipping Jira Cloud integration")
                return

            self.client = JIRA(server=self.config.server_url, basic_auth=(self.config.username, self.config.api_token))
            logger.info(f"Connected to Jira Cloud at {self.config.server_url}")

        except Exception as e:
            logger.error(f"Error connecting to Jira Cloud: {e}")
            self.client = None

    def get_issue_details(self, issue_key: str) -> Optional[JiraIssueDetails]:
        """Get detailed information for a specific Jira issue."""
        try:
            if not self.client:
                logger.warning("Jira client not initialized")
                return None

            # Fetch the issue with all necessary fields
            issue = self.client.issue(
                issue_key,
                fields=[
                    'summary',
                    'status',
                    'parent',
                    'subtasks',
                    'error_message',
                    'error_type',
                    'traceback',
                    'site',
                    'request_id',
                    'created',
                    'updated',
                    'description',
                ],
            )

            # Extract parent issue key
            parent_issue_key = None
            if hasattr(issue.fields, 'parent') and issue.fields.parent:
                parent_issue_key = issue.fields.parent.key

            # Extract child issue keys (subtasks)
            child_issue_keys = []
            if hasattr(issue.fields, 'subtasks') and issue.fields.subtasks:
                child_issue_keys = [subtask.key for subtask in issue.fields.subtasks]

            # Extract other fields
            status = issue.fields.status.name if issue.fields.status else "Unknown"
            summary = issue.fields.summary if issue.fields.summary else "No summary"
            error_message = getattr(issue.fields, 'error_message', None)
            error_type = getattr(issue.fields, 'error_type', None)
            traceback = getattr(issue.fields, 'traceback', None)
            site = getattr(issue.fields, 'site', None)
            request_id = getattr(issue.fields, 'request_id', None)
            created = str(issue.fields.created) if issue.fields.created else None
            updated = str(issue.fields.updated) if issue.fields.updated else None
            description = issue.fields.description if issue.fields.description else None

            return JiraIssueDetails(
                issue_key=issue_key,
                summary=summary,
                status=status,
                parent_issue_key=parent_issue_key,
                child_issue_keys=child_issue_keys,
                error_message=error_message,
                error_type=error_type,
                traceback=traceback,
                site=site,
                request_id=request_id,
                created=created,
                updated=updated,
                description=description,
            )

        except Exception as e:
            logger.error(f"Error fetching Jira issue {issue_key}: {e}")
            return None

    def get_multiple_issue_details(self, issue_keys: List[str]) -> Dict[str, JiraIssueDetails]:
        """Get detailed information for multiple Jira issues."""
        details = {}

        for issue_key in issue_keys:
            try:
                issue_details = self.get_issue_details(issue_key)
                if issue_details:
                    details[issue_key] = issue_details
            except Exception as e:
                logger.error(f"Error fetching details for issue {issue_key}: {e}")
                continue

        logger.info(f"Fetched details for {len(details)} out of {len(issue_keys)} issues")
        return details

    def get_all_issues(
        self, project_key: str = None, max_results: int = 1000, page_size: int = 100
    ) -> List[JiraIssueDetails]:
        """
        Get all Jira issues for a project or all accessible projects using pagination.

        Args:
            project_key: Specific project key to search (if None, searches all accessible projects)
            max_results: Maximum number of issues to return (default: 1000)
            page_size: Number of issues to fetch per page (default: 100)

        Returns:
            List of JiraIssueDetails objects
        """
        try:
            if not self.client:
                logger.warning("Jira client not initialized")
                return []

            # Build JQL query
            if project_key:
                jql = f"project = {project_key}"
            else:
                jql = "ORDER BY created DESC"

            logger.info(f"Searching for Jira issues with JQL: {jql} (max: {max_results}, page_size: {page_size})")

            issue_details = []
            start_at = 0
            total_fetched = 0

            while total_fetched < max_results:
                # Calculate how many issues to fetch in this batch
                remaining = max_results - total_fetched
                current_page_size = min(page_size, remaining)

                logger.info(f"Fetching page starting at {start_at} with {current_page_size} issues...")

                # Search for issues with pagination
                issues = self.client.enhanced_search_issues(
                    jql,
                    fields=[
                        'key',
                        'summary',
                        'status',
                        'parent',
                        'subtasks',
                        'error_message',
                        'error_type',
                        'traceback',
                        'site',
                        'request_id',
                        'created',
                        'updated',
                        'description',
                    ],
                    startAt=start_at,
                    maxResults=current_page_size,
                    expand=['changelog'],
                )

                # Check if we got any issues
                if not issues or len(issues) == 0:
                    logger.info("No more issues found, stopping pagination")
                    break

                # Process issues in this batch
                batch_count = 0
                for issue in issues:
                    try:
                        # Extract parent issue key
                        parent_issue_key = None
                        if hasattr(issue.fields, 'parent') and issue.fields.parent:
                            parent_issue_key = issue.fields.parent.key

                        # Extract child issue keys (subtasks)
                        child_issue_keys = []
                        if hasattr(issue.fields, 'subtasks') and issue.fields.subtasks:
                            child_issue_keys = [subtask.key for subtask in issue.fields.subtasks]

                        # Extract other fields
                        status = issue.fields.status.name if issue.fields.status else "Unknown"
                        summary = issue.fields.summary if issue.fields.summary else "No summary"
                        error_message = getattr(issue.fields, 'error_message', None)
                        error_type = getattr(issue.fields, 'error_type', None)
                        traceback = getattr(issue.fields, 'traceback', None)
                        site = getattr(issue.fields, 'site', None)
                        request_id = getattr(issue.fields, 'request_id', None)
                        created = str(issue.fields.created) if issue.fields.created else None
                        updated = str(issue.fields.updated) if issue.fields.updated else None
                        description = issue.fields.description if issue.fields.description else None

                        issue_detail = JiraIssueDetails(
                            issue_key=issue.key,
                            summary=summary,
                            status=status,
                            parent_issue_key=parent_issue_key,
                            child_issue_keys=child_issue_keys,
                            error_message=error_message,
                            error_type=error_type,
                            traceback=traceback,
                            site=site,
                            request_id=request_id,
                            created=created,
                            updated=updated,
                            description=description,
                        )
                        issue_details.append(issue_detail)
                        batch_count += 1

                    except Exception as e:
                        logger.warning(f"Error processing issue {issue.key}: {e}")
                        continue

                total_fetched += batch_count
                start_at += current_page_size

                logger.info(f"Fetched {batch_count} issues in this batch (total: {total_fetched})")

                # If we got fewer issues than requested, we've reached the end
                if batch_count < current_page_size:
                    logger.info("Reached end of available issues")
                    break

                # Add a small delay to avoid overwhelming the API
                import time

                time.sleep(0.1)

            logger.info(f"Successfully fetched {len(issue_details)} Jira issues using pagination")
            return issue_details

        except Exception as e:
            logger.error(f"Error fetching all Jira issues: {e}")
            return []

    def test_connection(self) -> bool:
        """Test Jira Cloud connection."""
        try:
            if not self.client:
                return False

            # Try to fetch current user info
            current_user = self.client.current_user()
            logger.info(f"Jira Cloud connection test successful. User: {current_user}")
            return True

        except Exception as e:
            logger.error(f"Jira Cloud connection test failed: {e}")
            return False
