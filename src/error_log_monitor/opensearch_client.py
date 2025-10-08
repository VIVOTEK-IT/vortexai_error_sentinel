"""
OpenSearch client for error log retrieval and analysis.
"""

import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

from error_log_monitor.config import OpenSearchConfig, load_config

logger = logging.getLogger(__name__)


# Connect to OpenSearch
def connect_opensearch(config: OpenSearchConfig = None):
    """Connect to OpenSearch."""
    if not config:
        config = load_config().opensearch
    try:
        client = OpenSearch(
            hosts=[
                {
                    'host': config.host,
                    'port': config.port,
                }
            ],
            http_auth=(config.username, config.password) if config.username else None,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        logger.info(f"Connected to OpenSearch at {config.host}:{config.port}")
    except Exception as e:
        logger.error(f"Failed to connect to OpenSearch: {e}")
        raise
    return client


@dataclass
class ErrorLog:
    """Error log entry from OpenSearch."""

    message_id: str
    timestamp: datetime
    error_message: str
    error_message_hash: int
    traceback: Optional[str] = None
    traceback_hash: Optional[int] = None
    error_type: Optional[str] = None
    error_type_hash: Optional[int] = None
    site: str = "unknown"
    service: Optional[str] = None
    index_name: Optional[str] = None
    topic: Optional[str] = None
    count: int = 1
    request_id: Optional[str] = None
    category: Optional[str] = None
    log_group: Optional[str] = None
    module_name: Optional[str] = None
    version: Optional[str] = None
    jira_reference: Optional[str] = None

class OpenSearchClient:
    """OpenSearch client for error log operations."""

    def __init__(self, config: OpenSearchConfig):
        """Initialize OpenSearch client."""
        self.config = config
        self.client = connect_opensearch(self.config)

    def get_es_conn(self):
        return self.client

    def get_error_logs(self, site: str, start_date: datetime, end_date: datetime, page_size: int = 1000) -> List[ErrorLog]:
        """
        Retrieve error logs from OpenSearch using pagination.

        Args:
            site: Site name (dev, stage, prod)
            start_date: Start date for log retrieval
            end_date: End date for log retrieval
            page_size: Number of logs to retrieve per page (default: 1000)

        Returns:
            List of error log entries
        """
        if not self.client:
            raise ConnectionError("Not connected to OpenSearch")

        try:
            # Generate index name without leading zeros
            index_name = self.config.index_pattern.format(site=site, year=start_date.year, month=start_date.month)

            # Build query
            query = {
                "query": {
                    "bool": {
                        "must": [{"range": {"timestamp": {"gte": start_date.isoformat(), "lte": end_date.isoformat()}}}]
                    }
                },
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": page_size,
            }

            # Execute search with scroll for pagination
            response = self.client.search(index=index_name, body=query, scroll="2m")
            scroll_id = response.get("_scroll_id")
            
            # Parse results
            error_logs = []
            
            # Process first page
            for hit in response['hits']['hits']:
                error_log = self._parse_log_entry(hit, site, index_name)
                if error_log:
                    error_logs.append(error_log)

            # Process remaining pages using scroll
            while scroll_id:
                try:
                    response = self.client.scroll(scroll_id=scroll_id, scroll="2m")
                    hits = response.get("hits", {}).get("hits", [])
                    
                    if not hits:
                        break
                    
                    for hit in hits:
                        error_log = self._parse_log_entry(hit, site, index_name)
                        if error_log:
                            error_logs.append(error_log)
                    
                    scroll_id = response.get("_scroll_id")
                    
                except Exception as e:
                    logger.warning(f"Error during scroll pagination: {e}")
                    break
            
            # Clear scroll context
            if scroll_id:
                try:
                    self.client.clear_scroll(scroll_id=scroll_id)
                except Exception as e:
                    logger.warning(f"Failed to clear scroll context: {e}")

            logger.info(f"Retrieved {len(error_logs)} error logs from {index_name}")
            return error_logs

        except OpenSearchException as e:
            if "401" in str(e) or "AuthenticationException" in str(e):
                logger.warning(f"OpenSearch authentication failed. Please check credentials: {e}")
                logger.info("Continuing with sample data for demonstration purposes")
            else:
                logger.error(f"OpenSearch query failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return []

    def _parse_log_entry(self, hit: Dict[str, Any], site: str, index_name: str) -> Optional[ErrorLog]:
        """Parse OpenSearch hit into ErrorLog object."""
        try:
            source = hit['_source']

            # Extract basic fields
            message_id = hit['_id']
            timestamp_str = source.get('timestamp', source.get('@timestamp', ''))

            # Parse timestamp
            try:
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str[:-1]).replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            except ValueError:
                timestamp = datetime.now(timezone.utc)

            error_message = source.get('error_message', source.get('message', ''))
            error_message_hash = self._calculate_hash(error_message)

            # Extract traceback
            traceback = source.get('traceback', source.get('stack_trace'))
            traceback_hash = self._calculate_hash(traceback) if traceback else None

            # Extract error type
            error_type = source.get('error_type', source.get('exception_type'))
            error_type_hash = self._calculate_hash(error_type) if error_type else None

            # Compose service name from log_group and module_name
            log_group = source.get('log_group', None)
            module_name = source.get('module_name', None)

            if log_group and module_name:
                service = f"{log_group}.{module_name}"
            elif log_group:
                service = log_group
            elif module_name:
                service = module_name
            else:
                service = 'unknown'

            return ErrorLog(
                message_id=message_id,
                timestamp=timestamp,
                error_message=error_message,
                error_message_hash=error_message_hash,
                traceback=traceback,
                traceback_hash=traceback_hash,
                error_type=error_type,
                error_type_hash=error_type_hash,
                site=site,
                service=service,
                index_name=index_name,
                topic=source.get('topic', None),
                count=source.get('count', 1),
                request_id=source.get('request_id', None),
                category=source.get('category', None),
                log_group=log_group,
                module_name=module_name,
                version=source.get('version', None),
                jira_reference=source.get('jira_reference', None),
            )

        except Exception as e:
            logger.error(f"Error parsing log entry: {e}")
            return None

    def _calculate_hash(self, text: str) -> int:
        """Calculate hash for text content."""
        if not text:
            return 0
        return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

    def test_connection(self) -> bool:
        """Test OpenSearch connection."""
        try:
            if not self.client:
                return False
            self.client.info()
            return True
        except Exception as e:
            logger.error(f"OpenSearch connection test failed: {e}")
            return False
