"""
Jira Cloud API client for fetching detailed issue information.
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from jira import JIRA

    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False
    JIRA = None

from error_log_monitor.config import JiraConfig, load_config

logger = logging.getLogger(__name__)


def get_jira_field_index(jira=None):
    config = load_config()
    field_id_index = {}
    """Get Jira field index."""
    field_id_index = {}
    reverse_field_id_index = {}
    remove_dummy_issue = False
    if not jira:
        jira = JIRA(
            server=config.jira.server_url,
            basic_auth=(config.jira.username, config.jira.api_token),
        )
    jql = f'project = {config.jira.project_key} AND issuetype = Task ORDER BY created DESC'
    logger.info(f"📋 JQL: {jql}")
    issues = jira.search_issues(jql, maxResults=1)
    try:
        issue = issues[0]
    except Exception:
        remove_dummy_issue = True
        issue = jira.create_issue(
            fields={
                'project': {'key': config.jira.project_key},
                'summary': 'Dummy Issue',
                'issuetype': {'name': 'Task'},
            }
        )
    fields = jira.fields()
    all_fields = {}
    for field in fields:
        if field['id'].startswith('customfield_'):
            all_fields[field['id']] = field['name']

    for field_id, val in issue.raw['fields'].items():
        if field_id.startswith('customfield_') and field_id in all_fields.keys():
            field_name = all_fields[field_id]
            field_id_index[field_name] = field_id
            reverse_field_id_index[field_id] = field_name
            logging.info(f"✅ {field_name}:{field_name} added to field_id_index")
    if remove_dummy_issue:
        jira._session.delete(f"{jira._get_url('issue')}/{issue.key}")
        logging.info("✅ dummy issue removed")
    return field_id_index, reverse_field_id_index


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
    log_group: Optional[str] = None
    count: Optional[int] = None
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

    def get_issue_details(
        self, issue_key: str, custom_field_mapping: Optional[Dict[str, str]] = None
    ) -> Optional[JiraIssueDetails]:
        """Get detailed information for a specific Jira issue."""
        try:
            if not self.client:
                logger.warning("Jira client not initialized")
                return None

            # Get custom field mapping if not provided
            if custom_field_mapping is None:
                field_id_index, _ = get_jira_field_index(self.client)
                custom_field_mapping = field_id_index

            # Fetch the issue with all necessary fields
            issue = self.client.issue(issue_key)

            # Extract parent issue key
            parent_issue_key = None
            if hasattr(issue.fields, 'parent') and issue.fields.parent:
                parent_issue_key = issue.fields.parent.key

            # Extract child issue keys (subtasks)
            child_issue_keys = []
            if hasattr(issue.fields, 'subtasks') and issue.fields.subtasks:
                child_issue_keys = [subtask.key for subtask in issue.fields.subtasks]

            # Extract standard fields
            status = issue.fields.status.name if issue.fields.status else "Unknown"
            summary = issue.fields.summary if issue.fields.summary else "No summary"
            created = str(issue.fields.created) if issue.fields.created else None
            updated = str(issue.fields.updated) if issue.fields.updated else None
            description = issue.fields.description if issue.fields.description else None

            # Extract custom fields using dynamic mapping
            error_message = None
            error_type = None
            traceback = None
            site = None
            request_id = None
            log_group = None
            count = None

            if custom_field_mapping:
                # Map custom field IDs to field names
                reverse_mapping = {v: k for k, v in custom_field_mapping.items()}

                for field_id, field_name in reverse_mapping.items():
                    field_value = getattr(issue.fields, field_id, None)
                    if field_value is not None:
                        # Handle different field value types
                        if isinstance(field_value, dict):
                            if 'value' in field_value:
                                field_value = field_value['value']
                            elif 'name' in field_value:
                                field_value = field_value['name']
                            elif 'key' in field_value:
                                field_value = field_value['key']
                            else:
                                field_value = str(field_value)

                        # Map to the appropriate variable
                        if field_name == 'error_message':
                            error_message = field_value
                        elif field_name == 'error_type':
                            error_type = field_value
                        elif field_name == 'traceback':
                            traceback = field_value
                        elif field_name == 'site':
                            site = field_value
                        elif field_name == 'request_id':
                            request_id = field_value
                        elif field_name == 'log_group':
                            log_group = field_value
                        elif field_name == 'count':
                            count = int(field_value) if field_value else None
                        elif field_name == 'parent issue':
                            parent_issue_key = field_value

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
                log_group=log_group,
                count=count,
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

    def clean_traceback(self, traceback: str) -> str:
        """Remove {code:python} from head and remove  {code} from tail"""
        if traceback.startswith("{code:python}"):
            traceback = traceback[len("{code:python}") :].strip()
        if traceback.endswith("{code}"):
            traceback = traceback[: -len("{code}")].strip()
        if traceback.startswith("\n"):
            traceback = traceback[1:].strip()
        if traceback.endswith("\n"):
            traceback = traceback[:-1].strip()
        return traceback

    def get_all_issues(
        self, project_key: str = None, max_results: int = 1000, page_size: int = 100
    ) -> List[JiraIssueDetails]:
        """
        Get all Jira issues for a project or all accessible projects using search_issues with pagination.

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
                jql = f"project = {project_key} AND issuetype = Task ORDER BY created ASC"
            else:
                jql = "issuetype = Task ORDER BY created DESC"

            logger.info(f"Searching for Jira issues with JQL: {jql} (max: {max_results}, page_size: {page_size})")

            # Get custom field mapping
            field_id_index, _ = get_jira_field_index(self.client)
            issue_details = []
            nextPageToken = None
            total_fetched = 0

            while total_fetched < max_results:
                # Calculate how many issues to fetch in this batch
                remaining = max_results - total_fetched
                current_page_size = min(page_size, remaining)

                # Search for issues using search_issues with pagination
                search_result = self.client.enhanced_search_issues(
                    jql,
                    nextPageToken=nextPageToken,
                    maxResults=current_page_size,
                    expand=['changelog'],
                )

                # Extract issues from the search result
                # search_issues returns a list of issues directly
                if hasattr(search_result, 'issues'):
                    issues = search_result.issues
                elif hasattr(search_result, '__iter__'):
                    issues = list(search_result)
                else:
                    issues = []

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

                        # Extract standard fields
                        status = issue.fields.status.name if issue.fields.status else "Unknown"
                        summary = issue.fields.summary if issue.fields.summary else "No summary"
                        created = str(issue.fields.created) if issue.fields.created else None
                        updated = str(issue.fields.updated) if issue.fields.updated else None
                        description = issue.fields.description if issue.fields.description else None

                        # Extract custom fields using dynamic mapping
                        error_message = None
                        error_type = None
                        traceback = None
                        site = None
                        request_id = None
                        log_group = None
                        count = None

                        if field_id_index:
                            # Map custom field IDs to field names
                            reverse_mapping = {v: k for k, v in field_id_index.items()}

                            for field_id, field_name in reverse_mapping.items():
                                field_value = getattr(issue.fields, field_id, None)
                                if field_value is not None:
                                    # Handle different field value types
                                    if isinstance(field_value, dict):
                                        if 'value' in field_value:
                                            field_value = field_value['value']
                                        elif 'name' in field_value:
                                            field_value = field_value['name']
                                        elif 'key' in field_value:
                                            field_value = field_value['key']
                                        else:
                                            field_value = str(field_value)

                                    # Map to the appropriate variable
                                    if field_name == 'error_message':
                                        error_message = field_value
                                    elif field_name == 'error_type':
                                        error_type = field_value
                                    elif field_name == 'traceback':
                                        traceback = field_value
                                    elif field_name == 'site':
                                        site = field_value
                                    elif field_name == 'request_id':
                                        request_id = field_value
                                    elif field_name == 'log_group':
                                        log_group = field_value
                                    elif field_name == 'count':
                                        count = int(field_value) if field_value else None
                                    elif field_name == 'parent issue':
                                        parent_issue_key = field_value

                        issue_detail = JiraIssueDetails(
                            issue_key=issue.key,
                            summary=summary,
                            status=status,
                            parent_issue_key=parent_issue_key,
                            child_issue_keys=child_issue_keys,
                            error_message=error_message,
                            error_type=error_type,
                            traceback=self.clean_traceback(traceback),
                            site=site,
                            request_id=request_id,
                            log_group=log_group,
                            count=count,
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
                nextPageToken = search_result.nextPageToken

                logger.info(f"Fetched {batch_count} issues in this batch (total: {total_fetched})")

                # If we got fewer issues than requested, we've reached the end of available issues
                if batch_count < current_page_size and total_fetched > 400:
                    logger.info(
                        f"Reached end of available issues (got {batch_count} out of {current_page_size} requested)"
                    )
                    break

                # Add a small delay to avoid overwhelming the API
                import time

                time.sleep(0.1)

            logger.info(f"Successfully fetched {len(issue_details)} Jira issues using pagination")
            return issue_details

        except Exception as e:
            logger.error(f"Error fetching all Jira issues: {e}", exc_info=True)
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
