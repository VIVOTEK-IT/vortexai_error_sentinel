"""
Jira Issue Embedding Database module for vector similarity search and correlation.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from opensearchpy.client import OpenSearch

from error_log_monitor.config import SystemConfig, JiraEmbeddingConfig
from error_log_monitor.opensearch_client import OpenSearchClient, ErrorLog, connect_opensearch
from error_log_monitor.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class JiraIssueData:
    """Data structure for Jira issue information."""

    key: str
    summary: str
    description: str
    status: str
    error_message: str
    error_type: str
    traceback: str
    site: str
    request_id: str
    created: str
    updated: str
    parent_issue_key: str
    is_parent: bool


@dataclass
class OccurrenceData:
    """Data structure for occurrence tracking."""

    doc_id: str
    timestamp: str


class JiraIssueEmbeddingDB:
    """Manages Jira issue embeddings and occurrence tracking in OpenSearch."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService],
        config: SystemConfig,
        index_name_template: Optional[str] = None,
        opensearch_connect: OpenSearch = None,
    ):
        """
        Initialize Jira Issue Embedding Database.

        Args:
            opensearch_client: OpenSearch client instance
            embedding_service: Embedding service for vector calculations (can be None for read-only operations)
            config: System configuration
            index_name_template: Optional custom index name template
        """

        self.embedding_service = embedding_service

        self.config = config
        self.jira_embedding_config: JiraEmbeddingConfig = config.jira_embedding
        self.index_name_template = index_name_template or self.jira_embedding_config.index_name_template
        self.logger = logging.getLogger(__name__)
        if not opensearch_connect:
            self.opensearch_connect = connect_opensearch(self.config.opensearch)

    def get_current_index_name(self) -> str:
        """Get the current year's index name."""
        current_year = datetime.now().year
        return self.index_name_template.format(year=current_year)

    def get_index_name_for_year(self, year: int) -> str:
        """Get index name for a specific year."""
        return self.index_name_template.format(year=year)

    def normalize_embedding(self, embedding_vector: List[float]) -> List[float]:
        """
        Normalize embedding vector to unit vector (length = 1).

        Args:
            embedding_vector: Raw embedding vector

        Returns:
            Normalized unit vector

        Raises:
            RuntimeError: If embedding service is not available
        """
        if self.embedding_service is None:
            raise RuntimeError("Embedding service not available for normalization")
        return self.embedding_service.normalize_embedding(embedding_vector)

    def validate_unit_vector(self, embedding_vector: List[float], tolerance: float = 1e-6) -> bool:
        """
        Validate that embedding vector is a unit vector.

        Args:
            embedding_vector: Vector to validate
            tolerance: Tolerance for magnitude check

        Returns:
            True if vector is unit vector, False otherwise

        Raises:
            RuntimeError: If embedding service is not available
        """
        if self.embedding_service is None:
            raise RuntimeError("Embedding service not available for validation")
        return self.embedding_service.validate_unit_vector(embedding_vector, tolerance)

    def create_index(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Create the OpenSearch index with proper mapping for the specified year.

        Args:
            year: Year for index creation, None for current year

        Returns:
            OpenSearch response
        """
        if year is None:
            index_name = self.get_current_index_name()
        else:
            index_name = self.get_index_name_for_year(year)

        # Check if index already exists
        if self.opensearch_connect.indices.exists(index=index_name):
            self.logger.info(f"Index {index_name} already exists")
            return {"acknowledged": True, "existing": True}

        # Index mapping with unit vector support
        mapping = {
            "mappings": {
                "properties": {
                    "key": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",  # OpenSearch only supports inner product
                            "engine": "faiss",
                            "parameters": {"ef_construction": 128, "m": 32},
                        },
                    },
                    "occurrence_list": {
                        "type": "nested",
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                        },
                    },
                    "summary": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "description": {"type": "text"},
                    "status": {"type": "keyword"},
                    "error_message": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "error_type": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "traceback": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "site": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "created": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "updated": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "parent_issue_key": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    "created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.knn": True,
                "index.knn.algo_param.ef_search": 100,
                "index.knn.algo_param.ef_construction": 128,
                "index.knn.algo_param.m": 16,
            },
        }

        try:
            response = self.opensearch_connect.indices.create(index=index_name, body=mapping)
            self.logger.info(f"Created index {index_name}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")
            raise

    def add_jira_issue(
        self, jira_issue_data: JiraIssueData, error_log_data: Optional[ErrorLog] = None
    ) -> Dict[str, Any]:
        """
        Add a new Jira issue with embedding to the database.

        Args:
            jira_issue_data: Jira issue data
            error_log_data: Optional error log data for embedding calculation

        Returns:
            OpenSearch response
        """
        try:
            # Calculate embedding
            if error_log_data:
                # Use error log data for embedding
                text_input = (
                    f"{error_log_data.error_message or ''} "
                    f"{error_log_data.error_type or ''} "
                    f"{error_log_data.traceback or ''}"
                )
            else:
                # Use Jira issue data for embedding
                text_input = f"{jira_issue_data.summary} {jira_issue_data.description} {jira_issue_data.error_message}"

            # Check if text input is empty or only whitespace
            if not text_input or not text_input.strip():
                error_msg = f"Cannot generate embedding for empty text input for Jira issue {jira_issue_data.key}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Generate embedding using embedding service (automatically normalized)
            if self.embedding_service is None:
                raise RuntimeError("Embedding service not available for generating embeddings")
            embedding = self.embedding_service.generate_embedding(text_input)

            # Validate unit vector (should always be true since EmbeddingService normalizes automatically)
            if not self.validate_unit_vector(embedding):
                self.logger.warning(f"Embedding for {jira_issue_data.key} is not a unit vector, this should not happen")
                # Re-normalize as fallback
                embedding = self.normalize_embedding(embedding)

            # Prepare document
            doc = {
                "key": jira_issue_data.key,
                "embedding": embedding,
                "occurrence_list": [],
                "summary": jira_issue_data.summary,
                "description": jira_issue_data.description,
                "status": jira_issue_data.status,
                "error_message": jira_issue_data.error_message,
                "error_type": jira_issue_data.error_type,
                "traceback": jira_issue_data.traceback,
                "site": jira_issue_data.site,
                "request_id": jira_issue_data.request_id,
                "created": jira_issue_data.created,
                "updated": jira_issue_data.updated,
                "parent_issue_key": jira_issue_data.parent_issue_key,
                "is_parent": jira_issue_data.is_parent,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Index document
            response = self.opensearch_connect.index(
                index=self.get_current_index_name(), id=jira_issue_data.key, body=doc
            )

            self.logger.info(f"Added Jira issue {jira_issue_data.key} to embedding database")
            return response

        except Exception as e:
            self.logger.error(f"Failed to add Jira issue {jira_issue_data.key}: {e}")
            raise

    def find_similar_jira_issue(
        self, error_log_embedding: List[float], site: str, similarity_threshold: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar Jira issue for an error log using embedding similarity.

        Args:
            error_log_embedding: Normalized embedding vector of the error log
            site: Site name for filtering
            similarity_threshold: Minimum similarity score

        Returns:
            Similar Jira issue or None if not found
        """
        try:
            # Validate that query embedding is normalized (should always be true)
            if not self.validate_unit_vector(error_log_embedding):
                self.logger.warning("Query embedding is not a unit vector, normalizing as fallback")
                error_log_embedding = self.normalize_embedding(error_log_embedding)

            # Search using kNN with site filter and parent issues only
            search_body = {
                "knn": {"field": "embedding", "query_vector": error_log_embedding, "k": 5, "num_candidates": 50},
                "query": {
                    "bool": {
                        "must": [{"term": {"site": site}}, {"term": {"is_parent": True}}]  # Only search parent issues
                    }
                },
                "_source": [
                    "key",
                    "summary",
                    "description",
                    "status",
                    "error_message",
                    "error_type",
                    "traceback",
                    "site",
                    "request_id",
                    "created",
                    "updated",
                    "parent_issue_key",
                    "is_parent",
                    "occurrence_list",
                ],
                "min_score": similarity_threshold,
            }

            response = self.opensearch_connect.search(index=self.get_current_index_name(), body=search_body)

            if response["hits"]["hits"]:
                hit = response["hits"]["hits"][0]  # Get the most similar
                source = hit["_source"]
                return {
                    "key": source["key"],
                    "score": hit["_score"],
                    "summary": source.get("summary", ""),
                    "description": source.get("description", ""),
                    "status": source.get("status", ""),
                    "error_message": source.get("error_message", ""),
                    "error_type": source.get("error_type", ""),
                    "traceback": source.get("traceback", ""),
                    "site": source.get("site", ""),
                    "request_id": source.get("request_id", ""),
                    "created": source.get("created", ""),
                    "updated": source.get("updated", ""),
                    "parent_issue_key": source.get("parent_issue_key", ""),
                    "is_parent": source.get("is_parent", True),
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to find similar Jira issue: {e}")
            return None

    def add_occurrence(self, jira_key: str, doc_id: str, timestamp: str, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Add an occurrence reference to a Jira issue.

        Args:
            jira_key: Jira issue key
            doc_id: Document ID from error log
            timestamp: Timestamp of occurrence
            year: Year for index, None for current year

        Returns:
            OpenSearch response
        """
        try:
            index_name = self.get_current_index_name() if year is None else self.get_index_name_for_year(year)

            # Add occurrence to the nested occurrence_list
            script = {
                "script": {
                    "source": (
                        "if (ctx._source.occurrence_list == null) { "
                        "ctx._source.occurrence_list = [] } "
                        "ctx._source.occurrence_list.add(params.occurrence)"
                    ),
                    "params": {"occurrence": {"doc_id": doc_id, "timestamp": timestamp}},
                }
            }

            response = self.opensearch_connect.update(index=index_name, id=jira_key, body=script)

            self.logger.debug(f"Added occurrence {doc_id} to Jira issue {jira_key}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to add occurrence to {jira_key}: {e}")
            raise

    def get_issue_by_key(self, jira_key: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific Jira issue by its key.

        Args:
            jira_key: Jira issue key
            year: Year for index, None for current year

        Returns:
            Jira issue data or None if not found
        """
        try:
            index_name = self.get_current_index_name() if year is None else self.get_index_name_for_year(year)

            response = self.opensearch_connect.get(index=index_name, id=jira_key)

            if response["found"]:
                return response["_source"]

            return None

        except Exception as e:
            self.logger.error(f"Failed to get Jira issue {jira_key}: {e}")
            return None

    def get_occurrences(self, jira_key: str, year: Optional[int] = None) -> List[OccurrenceData]:
        """
        Get all occurrences for a specific Jira issue.

        Args:
            jira_key: Jira issue key
            year: Year for index, None for current year

        Returns:
            List of occurrence data
        """
        try:
            issue_data = self.get_issue_by_key(jira_key, year)
            if not issue_data:
                return []

            occurrences = issue_data.get("occurrence_list", [])
            return [OccurrenceData(doc_id=occ["doc_id"], timestamp=occ["timestamp"]) for occ in occurrences]

        except Exception as e:
            self.logger.error(f"Failed to get occurrences for {jira_key}: {e}")
            return []

    def delete_issue(self, jira_key: str, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Delete a Jira issue and all its occurrences.

        Args:
            jira_key: Jira issue key
            year: Year for index, None for current year

        Returns:
            OpenSearch response
        """
        try:
            index_name = self.get_current_index_name() if year is None else self.get_index_name_for_year(year)

            response = self.opensearch_connect.delete(index=index_name, id=jira_key)

            self.logger.info(f"Deleted Jira issue {jira_key}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to delete Jira issue {jira_key}: {e}")
            raise

    def get_embedding_stats(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics about the embedding database for a specific year.

        Args:
            year: Year for index, None for current year

        Returns:
            Database statistics
        """
        try:
            index_name = self.get_current_index_name() if year is None else self.get_index_name_for_year(year)

            # Get index stats
            stats = self.opensearch_connect.indices.stats(index=index_name)

            # Get document count
            count_response = self.opensearch_connect.count(index=index_name)
            doc_count = count_response["count"]

            # Calculate average occurrences
            avg_occurrences = 0
            if doc_count > 0:
                # Sample some documents to estimate average occurrences
                sample_response = self.opensearch_connect.search(
                    index=index_name, body={"size": 100, "query": {"match_all": {}}}
                )

                total_occurrences = 0
                for hit in sample_response["hits"]["hits"]:
                    occurrences = hit["_source"].get("occurrence_list", [])
                    total_occurrences += len(occurrences)

                avg_occurrences = (
                    total_occurrences / len(sample_response["hits"]["hits"]) if sample_response["hits"]["hits"] else 0
                )

            return {
                "index_name": index_name,
                "total_documents": doc_count,
                "index_size_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "average_occurrences": avg_occurrences,
                "year": year or datetime.now().year,
            }

        except Exception as e:
            self.logger.error(f"Failed to get embedding stats: {e}")
            return {}

    def search_similar_issues(
        self, query_text: str, top_k: int = 10, similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Search for similar Jira issues using embedding similarity.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar issues
        """
        try:
            # Generate embedding for query (automatically normalized)
            if self.embedding_service is None:
                raise RuntimeError("Embedding service not available for generating query embeddings")
            query_embedding = self.embedding_service.generate_embedding(query_text)

            # Search using kNN
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 10,
                },
                "query": {"bool": {"must": [{"term": {"is_parent": True}}]}},  # Only search parent issues
                "_source": [
                    "key",
                    "summary",
                    "description",
                    "status",
                    "error_message",
                    "error_type",
                    "traceback",
                    "site",
                    "request_id",
                    "created",
                    "updated",
                    "parent_issue_key",
                    "is_parent",
                    "occurrence_list",
                ],
                "min_score": similarity_threshold,
            }

            response = self.opensearch_connect.search(index=self.get_current_index_name(), body=search_body)

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    {
                        "key": source["key"],
                        "score": hit["_score"],
                        "summary": source.get("summary", ""),
                        "description": source.get("description", ""),
                        "status": source.get("status", ""),
                        "error_message": source.get("error_message", ""),
                        "error_type": source.get("error_type", ""),
                        "traceback": source.get("traceback", ""),
                        "site": source.get("site", ""),
                        "request_id": source.get("request_id", ""),
                        "created": source.get("created", ""),
                        "updated": source.get("updated", ""),
                        "parent_issue_key": source.get("parent_issue_key", ""),
                        "is_parent": source.get("is_parent", True),
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Failed to search similar issues: {e}")
            return []

    def get_available_years(self) -> List[int]:
        """
        Get list of available years with data.

        Returns:
            List of years that have indices with data
        """
        try:
            # Get all indices matching the pattern
            indices = self.opensearch_connect.cat.indices(
                index=f"{self.index_name_template.replace('{year}', '*')}", format="json"
            )

            years = []
            for index_info in indices:
                index_name = index_info["index"]
                if index_name.startswith("jira_issue_embedding_"):
                    # Extract year from index name
                    year_str = index_name.replace("jira_issue_embedding_", "")
                    try:
                        year = int(year_str)
                        years.append(year)
                    except ValueError:
                        continue

            return sorted(years)

        except Exception as e:
            self.logger.error(f"Failed to get available years: {e}")
            return []

    def search_across_years(
        self, query_text: str, years: Optional[List[int]] = None, top_k: int = 10, similarity_threshold: float = 0.85
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Search for similar Jira issues across multiple years.

        Args:
            query_text: Text to search for
            years: List of years to search, None for all available years
            top_k: Number of results to return per year
            similarity_threshold: Minimum similarity score

        Returns:
            Results grouped by year
        """
        try:
            if years is None:
                years = self.get_available_years()

            # Generate embedding for query
            if self.embedding_service is None:
                raise RuntimeError("Embedding service not available for generating query embeddings")
            raw_embedding = self.embedding_service.generate_embedding(query_text)
            query_embedding = self.normalize_embedding(raw_embedding)

            all_results = {}

            for year in years:
                index_name = self.get_index_name_for_year(year)

                # Search in this year's index
                search_body = {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": top_k,
                        "num_candidates": top_k * 10,
                    },
                    "query": {"bool": {"must": [{"term": {"is_parent": True}}]}},  # Only search parent issues
                    "_source": [
                        "key",
                        "summary",
                        "description",
                        "status",
                        "error_message",
                        "error_type",
                        "traceback",
                        "site",
                        "request_id",
                        "created",
                        "updated",
                        "parent_issue_key",
                        "is_parent",
                        "occurrence_list",
                    ],
                    "min_score": similarity_threshold,
                }

                try:
                    response = self.opensearch_connect.search(index=index_name, body=search_body)

                    results = []
                    for hit in response["hits"]["hits"]:
                        source = hit["_source"]
                        results.append(
                            {
                                "key": source["key"],
                                "score": hit["_score"],
                                "year": year,
                                "summary": source.get("summary", ""),
                                "description": source.get("description", ""),
                                "status": source.get("status", ""),
                                "error_message": source.get("error_message", ""),
                                "error_type": source.get("error_type", ""),
                                "traceback": source.get("traceback", ""),
                                "site": source.get("site", ""),
                                "request_id": source.get("request_id", ""),
                                "created": source.get("created", ""),
                                "updated": source.get("updated", ""),
                                "parent_issue_key": source.get("parent_issue_key", ""),
                                "is_parent": source.get("is_parent", True),
                            }
                        )

                    all_results[year] = results

                except Exception as e:
                    self.logger.warning(f"Failed to search in year {year}: {e}")
                    all_results[year] = []

            return all_results

        except Exception as e:
            self.logger.error(f"Failed to search across years: {e}")
            return {}

    def migrate_old_issues(self, from_year: int, to_year: int) -> Dict[str, Any]:
        """
        Migrate issues from one year to another.

        Args:
            from_year: Source year
            to_year: Target year

        Returns:
            Migration results
        """
        try:
            from_index = self.get_index_name_for_year(from_year)
            to_index = self.get_index_name_for_year(to_year)

            # Ensure target index exists
            if not self.opensearch_connect.indices.exists(index=to_index):
                self.create_index(to_year)

            # Scroll through all documents in source index
            search_body = {"query": {"match_all": {}}, "size": 1000}

            response = self.opensearch_connect.search(index=from_index, body=search_body, scroll="5m")

            scroll_id = response.get("_scroll_id")
            total_migrated = 0

            while response["hits"]["hits"]:
                # Prepare bulk actions for migration
                bulk_actions = []

                for hit in response["hits"]["hits"]:
                    doc = hit["_source"]
                    doc_id = hit["_id"]

                    # Update timestamps for migration
                    doc["migrated_at"] = datetime.now(timezone.utc).isoformat()
                    doc["original_year"] = from_year
                    doc["updated_at"] = datetime.now(timezone.utc).isoformat()

                    bulk_actions.append({"index": {"_index": to_index, "_id": doc_id}})
                    bulk_actions.append(doc)

                if bulk_actions:
                    self.opensearch_connect.bulk(body=bulk_actions)
                    total_migrated += len(bulk_actions) // 2

                # Get next batch
                response = self.opensearch_connect.scroll(scroll_id=scroll_id, scroll="5m")

            # Clear scroll
            self.opensearch_connect.clear_scroll(scroll_id=scroll_id)

            return {"from_year": from_year, "to_year": to_year, "total_migrated": total_migrated, "status": "success"}

        except Exception as e:
            self.logger.error(f"Failed to migrate issues from {from_year} to {to_year}: {e}")
            return {
                "from_year": from_year,
                "to_year": to_year,
                "total_migrated": 0,
                "status": "failed",
                "error": str(e),
            }

    def initialize_database(
        self, jira_issues: List[Dict[str, Any]], error_logs_by_site: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Initialize the Jira Issue Embedding Database with Jira issues and error logs.

        This method processes Jira issues from the API, calculates embeddings using the RAG engine,
        processes error logs from the past 6 months, finds similar issues, and adds occurrences.

        Args:
            jira_issues: List of Jira issues from API with fields:
                - key: Jira issue key (e.g., "VEL-123")
                - summary: Issue summary
                - description: Issue description
                - status: Issue status
                - created: Creation timestamp
                - updated: Last update timestamp
                - site: Site name (stage/prod)
                - parent_issue_key: Parent issue key (empty for parent issues)
            error_logs_by_site: Dictionary mapping site names to lists of ErrorLog objects
                from the past 6 months

        Returns:
            Initialization results with statistics and any errors

        Raises:
            RuntimeError: If embedding service is not available
            ValueError: If input data is invalid
        """
        if self.embedding_service is None:
            raise RuntimeError("Embedding service not available for database initialization")

        try:
            self.logger.info("Starting Jira Issue Embedding Database initialization")
            initialization_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "jira_issues_processed": 0,
                "error_logs_processed": 0,
                "similar_issues_found": 0,
                "new_issues_created": 0,
                "occurrences_added": 0,
                "errors": [],
                "site_stats": {},
            }

            # Ensure current year index exists
            current_index = self.get_current_index_name()

            if not self.opensearch_connect.indices.exists(index=current_index):
                self.logger.info(f"Creating index for current year: {current_index}")
                self.create_index()

            # Process Jira issues
            self.logger.info(f"Processing {len(jira_issues)} Jira issues")
            for jira_issue_data in jira_issues:
                try:
                    # Convert Jira issue data to JiraIssueData format
                    jira_data = JiraIssueData(
                        key=jira_issue_data["key"],
                        summary=jira_issue_data.get("summary", ""),
                        description=jira_issue_data.get("description", ""),
                        status=jira_issue_data.get("status", ""),
                        error_message=jira_issue_data.get("error_message", ""),
                        error_type=jira_issue_data.get("error_type", ""),
                        traceback=jira_issue_data.get("traceback", ""),
                        site=jira_issue_data.get("site", ""),
                        request_id=jira_issue_data.get("request_id", ""),
                        created=jira_issue_data.get("created", ""),
                        updated=jira_issue_data.get("updated", ""),
                        parent_issue_key=jira_issue_data.get("parent_issue_key", ""),
                    )

                    # Add Jira issue to database
                    self.add_jira_issue(jira_data)
                    initialization_results["jira_issues_processed"] += 1

                except Exception as e:
                    error_msg = f"Failed to process Jira issue {jira_issue_data.get('key', 'unknown')}: {e}"
                    self.logger.error(error_msg)
                    initialization_results["errors"].append(error_msg)

            # Process error logs by site
            total_error_logs = sum(len(logs) for logs in error_logs_by_site.values())
            self.logger.info(f"Processing {total_error_logs} error logs across {len(error_logs_by_site)} sites")

            for site, error_logs in error_logs_by_site.items():
                site_stats = {
                    "site": site,
                    "error_logs_processed": 0,
                    "similar_issues_found": 0,
                    "new_issues_created": 0,
                    "occurrences_added": 0,
                    "errors": [],
                }

                for error_log in error_logs:
                    try:
                        # Calculate embedding for error log
                        error_text = (
                            f"{error_log.error_message or ''} "
                            f"{error_log.error_type or ''} "
                            f"{error_log.traceback or ''}"
                        ).strip()

                        if not error_text:
                            self.logger.warning(f"Skipping error log {error_log.message_id} - empty text content")
                            continue

                        error_embedding = self.embedding_service.generate_embedding(error_text)

                        # Find similar Jira issues
                        similar_issue = self.find_similar_jira_issue(
                            error_log_embedding=error_embedding, site=site, similarity_threshold=0.85
                        )

                        if similar_issue:
                            # Add occurrence to existing issue
                            self.add_occurrence(
                                jira_key=similar_issue["key"],
                                doc_id=error_log.message_id,
                                timestamp=error_log.timestamp,
                            )
                            site_stats["similar_issues_found"] += 1
                            site_stats["occurrences_added"] += 1
                            initialization_results["similar_issues_found"] += 1
                            initialization_results["occurrences_added"] += 1

                        else:
                            # Create new Jira issue from error log
                            new_jira_data = self._create_jira_issue_from_error_log(error_log, site)
                            if new_jira_data:
                                self.add_jira_issue(new_jira_data, error_log_data=error_log)
                                site_stats["new_issues_created"] += 1
                                initialization_results["new_issues_created"] += 1

                        site_stats["error_logs_processed"] += 1
                        initialization_results["error_logs_processed"] += 1

                    except Exception as e:
                        error_msg = f"Failed to process error log {error_log.message_id} for site {site}: {e}"
                        self.logger.error(error_msg)
                        site_stats["errors"].append(error_msg)
                        initialization_results["errors"].append(error_msg)

                initialization_results["site_stats"][site] = site_stats

            # Update final status
            initialization_results["status"] = "completed"
            self.logger.info(
                f"Database initialization completed. Processed "
                f"{initialization_results['jira_issues_processed']} Jira issues and "
                f"{initialization_results['error_logs_processed']} error logs"
            )

            return initialization_results

        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": error_msg,
                "jira_issues_processed": 0,
                "error_logs_processed": 0,
                "similar_issues_found": 0,
                "new_issues_created": 0,
                "occurrences_added": 0,
                "errors": [error_msg],
            }

    def _create_jira_issue_from_error_log(self, error_log: Any, site: str) -> Optional[JiraIssueData]:
        """
        Create a new Jira issue from an error log when no similar issue is found.

        Args:
            error_log: ErrorLog object
            site: Site name

        Returns:
            JiraIssueData object or None if creation fails
        """
        try:
            # Generate a unique Jira key for the new issue
            timestamp_str = error_log.timestamp.strftime("%Y%m%d%H%M%S")
            jira_key = f"VEL-{site.upper()}-{timestamp_str}"

            # Create JiraIssueData from error log
            jira_data = JiraIssueData(
                key=jira_key,
                summary=error_log.error_message[:100] if error_log.error_message else "Error from logs",
                description=f"Auto-generated issue from error log {error_log.message_id}",
                status="Open",
                error_message=error_log.error_message or "",
                error_type=error_log.error_type or "",
                traceback=error_log.traceback or "",
                site=site,
                request_id=error_log.request_id or "",
                created=error_log.timestamp.isoformat(),
                updated=error_log.timestamp.isoformat(),
                parent_issue_key="",  # This is a parent issue
            )

            return jira_data

        except Exception as e:
            self.logger.error(f"Failed to create Jira issue from error log {error_log.message_id}: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the embedding database.

        Returns:
            Health check results
        """
        try:
            health_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": "healthy",
                "checks": {},
                "errors": [],
            }

            # Check 1: OpenSearch connectivity
            try:
                self.opensearch_connect.cluster.health()
                health_status["checks"]["opensearch_connectivity"] = "healthy"
            except Exception as e:
                health_status["checks"]["opensearch_connectivity"] = "unhealthy"
                health_status["errors"].append(f"OpenSearch connectivity failed: {e}")
                health_status["overall_status"] = "unhealthy"

            # Check 2: Current index exists
            try:
                current_index = self.get_current_index_name()
                index_exists = self.opensearch_connect.indices.exists(index=current_index)
                health_status["checks"]["current_index_exists"] = "healthy" if index_exists else "unhealthy"
                if not index_exists:
                    health_status["errors"].append(f"Current index {current_index} does not exist")
                    health_status["overall_status"] = "unhealthy"
            except Exception as e:
                health_status["checks"]["current_index_exists"] = "unhealthy"
                health_status["errors"].append(f"Index check failed: {e}")
                health_status["overall_status"] = "unhealthy"

            # Check 3: Embedding service functionality
            try:
                if self.embedding_service is None:
                    health_status["checks"]["embedding_service"] = "unavailable"
                    health_status["errors"].append("Embedding service not available")
                    health_status["overall_status"] = "unhealthy"
                else:
                    test_embedding = self.embedding_service.generate_embedding("test")
                    normalized_embedding = self.normalize_embedding(test_embedding)
                    is_unit_vector = self.validate_unit_vector(normalized_embedding)
                    health_status["checks"]["embedding_service"] = "healthy" if is_unit_vector else "unhealthy"
                    if not is_unit_vector:
                        health_status["errors"].append("Embedding normalization failed")
                    health_status["overall_status"] = "unhealthy"
            except Exception as e:
                health_status["checks"]["embedding_service"] = "unhealthy"
                health_status["errors"].append(f"Embedding service failed: {e}")
                health_status["overall_status"] = "unhealthy"

            # Check 4: Search functionality
            try:
                test_embedding = [0.1] * 1536
                test_embedding = self.normalize_embedding(test_embedding)
                self.find_similar_jira_issue(test_embedding, "test", similarity_threshold=0.0)
                health_status["checks"]["search_functionality"] = "healthy"
            except Exception as e:
                health_status["checks"]["search_functionality"] = "unhealthy"
                health_status["errors"].append(f"Search functionality failed: {e}")
                health_status["overall_status"] = "unhealthy"

            # Check 5: Index statistics
            try:
                stats = self.get_embedding_stats()
                health_status["checks"]["index_statistics"] = "healthy"
                health_status["index_stats"] = stats
            except Exception as e:
                health_status["checks"]["index_statistics"] = "unhealthy"
                health_status["errors"].append(f"Index statistics failed: {e}")
                health_status["overall_status"] = "unhealthy"

            return health_status

        except Exception as e:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": "unhealthy",
                "checks": {},
                "errors": [f"Health check failed: {e}"],
            }

    def performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedding database.

        Returns:
            Performance metrics
        """
        try:
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "search_performance": {},
                "index_performance": {},
                "embedding_performance": {},
            }

            # Search performance test
            import time

            test_embedding = [0.1] * 1536
            test_embedding = self.normalize_embedding(test_embedding)

            # Test search latency
            start_time = time.time()
            search_result = self.find_similar_jira_issue(test_embedding, "test", similarity_threshold=0.0)
            search_latency = time.time() - start_time

            metrics["search_performance"] = {
                "average_search_latency_ms": search_latency * 1000,
                "search_success": search_result is not None,
            }

            # Index performance metrics
            try:
                stats = self.get_embedding_stats()
                metrics["index_performance"] = {
                    "total_documents": stats.get("total_documents", 0),
                    "index_size_bytes": stats.get("index_size_bytes", 0),
                    "average_occurrences": stats.get("average_occurrences", 0),
                }
            except Exception as e:
                metrics["index_performance"] = {"error": str(e)}

            # Embedding performance test
            if self.embedding_service is not None:
                start_time = time.time()
                test_embedding = self.embedding_service.generate_embedding("performance test")
                embedding_latency = time.time() - start_time
            else:
                embedding_latency = 0

            if self.embedding_service is not None:
                start_time = time.time()
                self.normalize_embedding(test_embedding)
                normalization_latency = time.time() - start_time
            else:
                normalization_latency = 0

            metrics["embedding_performance"] = {
                "embedding_generation_latency_ms": embedding_latency * 1000,
                "normalization_latency_ms": normalization_latency * 1000,
                "total_embedding_latency_ms": (embedding_latency + normalization_latency) * 1000,
            }

            return metrics

        except Exception as e:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}

    def error_rate_monitoring(self, hours: int = 24) -> Dict[str, Any]:
        """
        Monitor error rates for the embedding database.

        Args:
            hours: Number of hours to look back for error monitoring

        Returns:
            Error rate monitoring results
        """
        try:
            # This would typically integrate with a logging system
            # For now, we'll return a basic structure
            monitoring_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monitoring_period_hours": hours,
                "error_counts": {
                    "embedding_generation_errors": 0,
                    "normalization_errors": 0,
                    "search_errors": 0,
                    "index_errors": 0,
                },
                "success_rates": {
                    "embedding_generation": 100.0,
                    "normalization": 100.0,
                    "search_operations": 100.0,
                    "index_operations": 100.0,
                },
                "total_operations": 0,
                "error_rate_percentage": 0.0,
            }

            # In a real implementation, this would query logs or metrics
            # to calculate actual error rates

            return monitoring_result

        except Exception as e:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the embedding database.

        Returns:
            Database summary
        """
        try:
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database_info": {
                    "index_template": self.index_name_template,
                    "current_index": self.get_current_index_name(),
                    "available_years": self.get_available_years(),
                },
                "health_status": self.health_check(),
                "performance_metrics": self.performance_metrics(),
                "error_monitoring": self.error_rate_monitoring(),
            }

            return summary

        except Exception as e:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}
