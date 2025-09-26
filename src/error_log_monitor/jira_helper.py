"""
Jira Helper module for retrieving Jira issues from OpenSearch.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

from error_log_monitor.config import OpenSearchConfig

logger = logging.getLogger(__name__)


@dataclass
class JiraIssue:
    """Jira issue entry from OpenSearch."""

    issue_key: str
    issue_time: datetime
    error_message: str
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    traceback_hash: Optional[int] = None
    modified_error_message: Optional[str] = None
    modified_error_message_hash: Optional[int] = None
    modified_traceback: Optional[str] = None
    jira_summary: Optional[str] = None
    jira_url: Optional[str] = None
    full_log_url: Optional[str] = None
    site: str = "unknown"
    log_group: Optional[str] = None
    request_id: Optional[str] = None
    related_issue_keys: Optional[List[str]] = None
    related_traceback_hashes: Optional[List[int]] = None
    count: int = 1
    # Additional fields from Jira Cloud API
    jira_status: Optional[str] = None
    parent_issue_key: Optional[str] = None
    child_issue_keys: Optional[List[str]] = None
    assignee: Optional[str] = None
    priority: Optional[str] = None
    issue_type: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    description: Optional[str] = None


class JiraHelper:
    """Helper class for retrieving Jira issues from OpenSearch."""

    def __init__(self, config: OpenSearchConfig):
        """Initialize Jira helper with OpenSearch configuration."""
        self.config = config
        self.client = None
        self._connect()

    def _connect(self):
        """Establish connection to OpenSearch."""
        try:
            self.client = OpenSearch(
                hosts=[
                    {
                        'host': self.config.host,
                        'port': self.config.port,
                        'use_ssl': self.config.use_ssl,
                        'verify_certs': self.config.verify_certs,
                    }
                ],
                http_auth=(self.config.username, self.config.password) if self.config.username else None,
                use_ssl=self.config.use_ssl,
                verify_certs=self.config.verify_certs,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
            )
            logger.info(f"Connected to OpenSearch at {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Error connecting to OpenSearch: {e}")
            self.client = None

    def get_recent_issues(self, days: int = 7, site: Optional[str] = None) -> List[JiraIssue]:
        """
        Retrieve Jira issues from recent days.

        Args:
            days: Number of recent days to retrieve (default: 7)
            site: Optional site filter

        Returns:
            List of JiraIssue objects
        """
        try:
            if not self.client:
                logger.error("OpenSearch client not available")
                return []

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Get current year for index name
            current_year = datetime.now().year
            index_name = f"jira_issue_{current_year}"

            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return []

            # Build query
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "issue_time": {
                                        "gte": start_date.isoformat(),
                                        "lte": end_date.isoformat(),
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"issue_time": {"order": "desc"}}],
                "size": 1000,  # Adjust based on needs
            }

            # Add site filter if specified
            if site:
                query["query"]["bool"]["must"].append({"term": {"site.keyword": site}})

            logger.info(f"Searching for Jira issues in {index_name} from {start_date} to {end_date}")
            if site:
                logger.info(f"Filtering by site: {site}")

            # Execute search
            response = self.client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            # Parse results
            issues = []
            for hit in hits:
                source = hit.get("_source", {})
                issue = self._parse_jira_issue(source)
                if issue:
                    issues.append(issue)

            logger.info(f"Retrieved {len(issues)} Jira issues from recent {days} days")
            return issues

        except OpenSearchException as e:
            logger.error(f"OpenSearch error retrieving Jira issues: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving Jira issues: {e}")
            return []

    def get_issues_by_date_range(
        self, start_date: datetime, end_date: datetime, site: Optional[str] = None
    ) -> List[JiraIssue]:
        """
        Retrieve Jira issues within a specific date range.

        Args:
            start_date: Start date for the range
            end_date: End date for the range
            site: Optional site filter

        Returns:
            List of JiraIssue objects
        """
        try:
            if not self.client:
                logger.error("OpenSearch client not available")
                return []

            # Get current year for index name
            current_year = datetime.now().year
            index_name = f"jira_issue_{current_year}"

            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return []

            # Build query
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "issue_time": {
                                        "gte": start_date.isoformat(),
                                        "lte": end_date.isoformat(),
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [{"issue_time": {"order": "desc"}}],
                "size": 1000,
            }

            # Add site filter if specified
            if site:
                query["query"]["bool"]["must"].append({"term": {"site.keyword": site}})

            logger.info(f"Searching for Jira issues in {index_name} from {start_date} to {end_date}")
            if site:
                logger.info(f"Filtering by site: {site}")

            # Execute search
            response = self.client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            # Parse results
            issues = []
            for hit in hits:
                source = hit.get("_source", {})
                issue = self._parse_jira_issue(source)
                if issue:
                    issues.append(issue)

            logger.info(f"Retrieved {len(issues)} Jira issues in date range")
            return issues

        except OpenSearchException as e:
            logger.error(f"OpenSearch error retrieving Jira issues: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving Jira issues: {e}")
            return []

    def get_issues_by_error_message(
        self, error_message: str, days: int = 7, site: Optional[str] = None
    ) -> List[JiraIssue]:
        """
        Retrieve Jira issues by error message pattern.

        Args:
            error_message: Error message to search for
            days: Number of recent days to search (default: 7)
            site: Optional site filter

        Returns:
            List of JiraIssue objects
        """
        try:
            if not self.client:
                logger.error("OpenSearch client not available")
                return []

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Get current year for index name
            current_year = datetime.now().year
            index_name = f"jira_issue_{current_year}"

            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return []

            # Build query
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "issue_time": {
                                        "gte": start_date.isoformat(),
                                        "lte": end_date.isoformat(),
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            },
                            {
                                "match": {
                                    "error_message": {"query": error_message, "operator": "or", "fuzziness": "AUTO"}
                                }
                            },
                        ]
                    }
                },
                "sort": [{"issue_time": {"order": "desc"}}],
                "size": 1000,
            }

            # Add site filter if specified
            if site:
                query["query"]["bool"]["must"].append({"term": {"site.keyword": site}})

            logger.info(f"Searching for Jira issues with error message pattern: {error_message}")
            logger.info(f"Date range: {start_date} to {end_date}")
            if site:
                logger.info(f"Filtering by site: {site}")

            # Execute search
            response = self.client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            # Parse results
            issues = []
            for hit in hits:
                source = hit.get("_source", {})
                issue = self._parse_jira_issue(source)
                if issue:
                    issues.append(issue)

            logger.info(f"Retrieved {len(issues)} Jira issues matching error message pattern")
            return issues

        except OpenSearchException as e:
            logger.error(f"OpenSearch error retrieving Jira issues: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving Jira issues: {e}")
            return []

    def search_issues_flexible(
        self, search_terms: List[str], days: int = 7, site: Optional[str] = None
    ) -> List[JiraIssue]:
        """
        Search for Jira issues using flexible matching with multiple terms.

        Args:
            search_terms: List of search terms to match against
            days: Number of recent days to search
            site: Optional site filter

        Returns:
            List of JiraIssue objects
        """
        try:
            if not self.client:
                logger.error("OpenSearch client not available")
                return []

            if not search_terms:
                return []

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Get current year for index name
            current_year = datetime.now().year
            index_name = f"jira_issue_{current_year}"

            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return []

            # Build flexible query with multiple search strategies
            should_queries = []

            for term in search_terms:
                if len(term) > 3:  # Only search meaningful terms
                    # Exact phrase match (highest priority)
                    should_queries.append({"match_phrase": {"error_message": {"query": term, "boost": 3}}})

                    # Fuzzy match
                    should_queries.append(
                        {"match": {"error_message": {"query": term, "fuzziness": "AUTO", "boost": 2}}}
                    )

                    # Partial match
                    should_queries.append({"wildcard": {"error_message": {"value": f"*{term}*", "boost": 1.5}}})

            # Build main query
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "issue_time": {
                                        "gte": start_date.isoformat(),
                                        "lte": end_date.isoformat(),
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            }
                        ],
                        "should": should_queries,
                        "minimum_should_match": 1,
                    }
                },
                "sort": [{"issue_time": {"order": "desc"}}],
                "size": 1000,
            }

            # Add site filter if specified
            if site:
                query["query"]["bool"]["must"].append({"term": {"site.keyword": site}})

            logger.info(f"Flexible search for terms: {search_terms}")
            logger.info(f"Date range: {start_date} to {end_date}")
            if site:
                logger.info(f"Filtering by site: {site}")

            # Execute search
            response = self.client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            # Parse results
            issues = []
            for hit in hits:
                source = hit.get("_source", {})
                issue = self._parse_jira_issue(source)
                if issue:
                    issues.append(issue)

            logger.info(f"Found {len(issues)} issues with flexible search")
            return issues

        except OpenSearchException as e:
            logger.error(f"OpenSearch error in flexible search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in flexible search: {e}")
            return []

    def get_issue_by_key(self, issue_key: str) -> Optional[JiraIssue]:
        """
        Retrieve a specific Jira issue by its key.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123")

        Returns:
            JiraIssue object if found, None otherwise
        """
        try:
            if not self.client:
                logger.error("OpenSearch client not available")
                return None

            # Get current year for index name
            current_year = datetime.now().year
            index_name = f"jira_issue_{current_year}"

            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return None

            # Build query
            query = {"query": {"term": {"issue_key.keyword": issue_key}}, "size": 1}

            logger.info(f"Searching for Jira issue with key: {issue_key}")

            # Execute search
            response = self.client.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            if hits:
                source = hits[0].get("_source", {})
                issue = self._parse_jira_issue(source)
                logger.info(f"Found Jira issue: {issue_key}")
                return issue
            else:
                logger.info(f"Jira issue not found: {issue_key}")
                return None

        except OpenSearchException as e:
            logger.error(f"OpenSearch error retrieving Jira issue: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving Jira issue: {e}")
            return None

    def _parse_jira_issue(self, source: Dict[str, Any]) -> Optional[JiraIssue]:
        """Parse OpenSearch document into JiraIssue object."""
        try:
            # Parse issue_time
            issue_time_str = source.get("issue_time")
            if issue_time_str:
                if isinstance(issue_time_str, str):
                    issue_time = datetime.fromisoformat(issue_time_str.replace("Z", "+00:00"))
                else:
                    issue_time = issue_time_str
            else:
                logger.warning("Missing issue_time in Jira issue")
                return None

            # Parse related_issue_keys
            related_issue_keys = source.get("related_issue_keys")
            if related_issue_keys and isinstance(related_issue_keys, str):
                related_issue_keys = [key.strip() for key in related_issue_keys.split(",") if key.strip()]

            # Parse related_traceback_hashes
            related_traceback_hashes = source.get("related_traceback_hashes")
            if related_traceback_hashes and isinstance(related_traceback_hashes, str):
                try:
                    related_traceback_hashes = [
                        int(hash_val.strip()) for hash_val in related_traceback_hashes.split(",") if hash_val.strip()
                    ]
                except ValueError:
                    related_traceback_hashes = None

            issue = JiraIssue(
                issue_key=source.get("issue_key", ""),
                issue_time=issue_time,
                error_message=source.get("error_message", ""),
                error_type=source.get("error_type"),
                traceback=source.get("traceback"),
                traceback_hash=source.get("traceback_hash"),
                modified_error_message=source.get("modified_error_message"),
                modified_error_message_hash=source.get("modified_error_message_hash"),
                modified_traceback=source.get("modified_traceback"),
                jira_summary=source.get("jira_summary"),
                jira_url=source.get("jira_url"),
                full_log_url=source.get("full_log_url"),
                site=source.get("site", "unknown"),
                log_group=source.get("log_group"),
                request_id=source.get("request_id"),
                related_issue_keys=related_issue_keys,
                related_traceback_hashes=related_traceback_hashes,
                count=source.get("count", 1),
            )

            return issue

        except Exception as e:
            logger.error(f"Error parsing Jira issue: {e}")
            return None

    def get_issue_statistics(self, days: int = 7, site: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about Jira issues for recent days.

        Args:
            days: Number of recent days to analyze (default: 7)
            site: Optional site filter

        Returns:
            Dictionary containing statistics
        """
        try:
            issues = self.get_recent_issues(days=days, site=site)

            if not issues:
                return {"total_issues": 0, "sites": {}, "error_types": {}, "date_range": {"start": None, "end": None}}

            # Calculate statistics
            sites = {}
            error_types = {}
            issue_times = []

            for issue in issues:
                # Count by site
                site_name = issue.site or "unknown"
                sites[site_name] = sites.get(site_name, 0) + 1

                # Count by error type
                if issue.error_type:
                    error_types[issue.error_type] = error_types.get(issue.error_type, 0) + 1

                # Collect issue times
                issue_times.append(issue.issue_time)

            # Calculate date range
            if issue_times:
                min_time = min(issue_times)
                max_time = max(issue_times)
            else:
                min_time = max_time = None

            statistics = {
                "total_issues": len(issues),
                "sites": sites,
                "error_types": error_types,
                "date_range": {
                    "start": min_time.isoformat() if min_time else None,
                    "end": max_time.isoformat() if max_time else None,
                },
            }

            logger.info(f"Generated statistics for {len(issues)} Jira issues")
            return statistics

        except Exception as e:
            logger.error(f"Error generating Jira issue statistics: {e}")
            return {"total_issues": 0, "sites": {}, "error_types": {}, "date_range": {"start": None, "end": None}}
