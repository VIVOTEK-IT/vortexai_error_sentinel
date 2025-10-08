"""
Jira Cloud API client for fetching detailed issue information.
"""

import datetime
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from jira import JIRA

    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False
    JIRA = None

from error_log_monitor.config import JiraConfig, SystemConfig

logger = logging.getLogger(__name__)


@dataclass
class JiraIssueDetails:
    """Detailed Jira issue information from Jira Cloud API."""

    key: str
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
    is_parent: bool = True
    not_commit_to_jira: bool = False

    def __post_init__(self):
        if self.child_issue_keys is None:
            self.child_issue_keys = []


def format_traceback_for_jira_rtf(raw_text: str) -> str:
    """
    將 raw traceback（以分號分隔）格式化為 Jira RTF 欄位可用的 Markdown 文字，
    使用三個大於號（{code}）包起來，適合貼入 Jira 支援 RTF 的段落欄位。
    """
    lines = raw_text.strip().split(";")
    formatted = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("File "):
            formatted.append("")  # 空行斷段
            formatted.append(line)
        else:
            formatted.append("    " + line)

    formatted_text = "\n".join(formatted)

    # 包裝成 Jira Wiki RTF code block 格式
    return "{code:python}\n" + formatted_text + "\n{code}"


def add_custom_fields(config: JiraConfig, jira_issue_dict: dict, issue_dict: dict):

    field_id_index = config.field_id_index
    for key_name in jira_issue_dict.keys():
        if key_name in field_id_index.keys():
            field_id = field_id_index[key_name]
            if jira_issue_dict[key_name] is not None:
                issue_dict[field_id] = jira_issue_dict[key_name]
                # logging.info(f"✅ {field_id}:{key_name}:{jira_issue_dict[key_name]} added to issue_dict")
                if key_name == "traceback":
                    issue_dict[field_id] = format_traceback_for_jira_rtf(jira_issue_dict[key_name])


def get_region_by_site(site: str) -> str:
    if site == 'dev':
        return 'ap-northeast-1'
    elif site == 'stage':
        return 'us-west-2'
    elif site == 'prod':
        return 'us-west-2'
    return 'ap-northeast-1'


def clean_summary_for_creation(summary: str) -> str:
    # 移除控制字符
    cleaned = summary.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 限制长度
    if len(cleaned) > 255:
        cleaned = cleaned[:252] + "..."

    return cleaned.strip()


def clean_summary_for_creation(summary: str) -> str:
    # 移除控制字符
    cleaned = summary.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 限制长度
    if len(cleaned) > 255:
        cleaned = cleaned[:252] + "..."

    return cleaned.strip()


def extract_summary_from_issue(issue: dict) -> str:
    err_msg = issue.get('error_message', None)
    summary = err_msg[:100] if err_msg else "Unknown Error"
    issue_time = issue.get('created', datetime.datetime.now(datetime.timezone.utc))
    if isinstance(issue_time, str):
        issue_time = datetime.datetime.fromisoformat(issue_time)
    issue_time = issue_time.strftime("%Y%m%d")
    site = issue.get('site', 'unknown')
    summary = f"[{issue_time}][{site}] - {summary}"
    return clean_summary_for_creation(summary)


class JiraCloudClient:
    """Client for interacting with Jira Cloud API."""

    def __init__(self, config: JiraConfig):
        """Initialize Jira Cloud client."""
        self.config: JiraConfig = config
        if not self.config:
            self.config = config.load_config().jira

        self.client = None
        self._connect()

    def _connect(self):
        if isinstance(self.config, SystemConfig):
            self.config = self.config.jira
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
            logger.error(f"Error connecting to Jira Cloud: {e}", exc_info=True)
            self.client = None

    def get_issue_details(self, issue_key: str) -> Optional[JiraIssueDetails]:
        """Get detailed information for a specific Jira issue."""
        try:
            if not self.client:
                logger.warning("Jira client not initialized")
                return None

            # Get custom field mapping if not provided
            field_id_index = self.config.field_id_index
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

            return JiraIssueDetails(
                key=issue_key,
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
        if not traceback:
            return traceback
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
        self, project_key: str = None, max_results: int = 1000, page_size: int = 100, duration_in_days: int = 180
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
                jql = f"project = {project_key} AND issuetype = Task AND created >= -{duration_in_days}d ORDER BY created ASC"
            else:
                jql = f"issuetype = Task AND created >= -{duration_in_days}d ORDER BY created DESC"

            logger.info(f"Searching for Jira issues with JQL: {jql} (max: {max_results}, page_size: {page_size})")

            # Get custom field mapping

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

                        if self.config.field_id_index:
                            # Map custom field IDs to field names
                            reverse_mapping = {v: k for k, v in self.config.field_id_index.items()}

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
                            key=issue.key,
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
                        logger.warning(f"Error processing issue {issue.key}: {e}", exc_info=True)
                        continue

                total_fetched += batch_count
                nextPageToken = search_result.nextPageToken

                logger.info(f"Fetched {batch_count} issues in this batch (total: {total_fetched})")

                # If we got fewer issues than requested, we've reached the end of available issues
                if batch_count < current_page_size:
                    try:
                        created_str = issue_details[-1].created
                        last_created = datetime.datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                        twelve_hours_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=12)
                        if last_created > twelve_hours_ago:
                            logger.info(
                                "Reached end of available issues (got %s out of %s requested)",
                                batch_count,
                                current_page_size,
                            )
                            break
                    except Exception:
                        logger.info(
                            "Reached end of available issues (got %s out of %s requested)",
                            batch_count,
                            current_page_size,
                        )

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

    def create_jira_issue(self, jira_issue: dict, title: str | None = None):
        """Create a new Jira issue in the configured project from JiraIssueDetails.

        Returns the created issue key (e.g., "PROJ-123").
        """
        try:
            if not self.client:
                raise RuntimeError("Jira client not initialized")

            project_key = self.config.project_key

            if jira_issue.get("created", None) is None:
                logger.error("issue_time is None, cannot create Jira issue", exc_info=True)
                raise ValueError("issue_time cannot be None")

            if jira_issue.get('request_id', None):
                issue_time = jira_issue.get('created', None)
                if isinstance(issue_time, str):
                    issue_time = datetime.datetime.fromisoformat(issue_time)
                start = (
                    (issue_time - datetime.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000Z").replace(":", "*3a")
                )
                end = (issue_time + datetime.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000Z").replace(":", "*3a")
                site = jira_issue.get('site', None)
                region = get_region_by_site(site) if site else "unknown"
                timezone = "UTC"
                request_id = jira_issue.get('request_id', None)
                log_group = jira_issue.get('log_group', '')
                if log_group:
                    log_group = log_group.replace("/", "*2f")

                jira_issue['full_log_url'] = (
                    f"""[https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:logs-insights$3FqueryDetail$3D~(end~'{end}~start~'{start}~timeType~'ABSOLUTE~tz~'{timezone}~editorString~'fields*20*40timestamp*2c*20*40message*2c*20*40logStream*2c*20*40log*2c*20strcontains*28*40message*2c*20*27{request_id}*27*29*20as*20unf*0a*7c*20filter*20unf*20*3d*201*0a*7c*20sort*20*40timestamp*20asc*0a*7c*20limit*209999~source~(~'{log_group})~lang~'CWLI\)]"""
                )
            # Jira summary has length limits; trim conservatively
            summary = extract_summary_from_issue(jira_issue)
            if len(summary) > 255:
                summary = summary[:252] + "..."

            # Description: prefer description, else compose from available fields
            description = jira_issue.get('description', '')
            if not description:
                parts = []
                if jira_issue.get('error_message', None):
                    parts.append(f"Error: {jira_issue.get('error_message', None)}")
                if jira_issue.get('error_type', None):
                    parts.append(f"Type: {jira_issue.get('error_type', None)}")
                if jira_issue.get('traceback', None):
                    tb = jira_issue.get('traceback', None)
                    if len(tb) > 4000:
                        tb = tb[:3997] + "..."
                    parts.append(f"Traceback:\n{tb}")
                description = "\n\n".join(parts) or "Created by automation"

            labels = []
            if jira_issue.get('site', None):
                labels.append(str(jira_issue.get('site', None)))
            if jira_issue.get('request_id', None):
                labels.append(str(jira_issue.get('request_id', None)))
            if jira_issue.get('log_group', None):
                labels.append(str(jira_issue.get('log_group', None)))

            issue_dict = {
                "project": {"key": project_key},
                "summary": summary,
                "issuetype": {"name": "Task"},
                "description": description,
            }
            if labels:
                issue_dict["labels"] = labels

            add_custom_fields(self.config, jira_issue, issue_dict)

            new_issue = self.client.create_issue(fields=issue_dict)
            logger.info(
                f"✅ issue created! issue key: {new_issue.key} ; issue url: {self.config.server_url}/browse/{new_issue.key}"
            )
            # Reflect created key/status back into the details object
            jira_issue['key'] = new_issue.key

            return new_issue.key

        except Exception as e:
            logger.error(f"Failed to create Jira issue: {e}", exc_info=True)
            raise
