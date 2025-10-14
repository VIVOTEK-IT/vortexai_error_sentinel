"""
Jira Issue Embedding Database module for vector similarity search and correlation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

from opensearchpy.client import OpenSearch

from error_log_monitor.config import SystemConfig, JiraEmbeddingConfig
from error_log_monitor.opensearch_client import ErrorLog, connect_opensearch
from error_log_monitor.embedding_service import EmbeddingService, ExceptionEmbeddingContextExceeded
from error_log_monitor.jira_cloud_client import JiraIssueDetails, JiraCloudClient


logger = logging.getLogger(__name__)


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
        self.jira_custom_fields_config = config.jira_custom_fields
        self.index_name_template = index_name_template or self.jira_embedding_config.index_name_template
        self.logger = logging.getLogger(__name__)
        if not opensearch_connect:
            self.opensearch_connect = connect_opensearch(self.config.opensearch)

        # Custom field mapping cache
        self._custom_field_mapping: Dict[str, str] = {}
        self._reverse_custom_field_mapping: Dict[str, str] = {}
        self._field_mapping_cache_time: Optional[datetime] = None

    def get_current_index_name(self) -> str:
        return self.index_name_template

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

    def _is_field_mapping_cache_valid(self) -> bool:
        """Check if custom field mapping cache is still valid."""
        if not self._field_mapping_cache_time:
            return False

        cache_age = (datetime.now(timezone.utc) - self._field_mapping_cache_time).total_seconds()
        return cache_age < self.jira_custom_fields_config.cache_ttl

    def _update_custom_field_mapping(self) -> None:
        """Update custom field mapping from Jira Cloud."""
        try:
            field_id_index = self.config.JiraConfig.field_id_index

            # Filter only known fields
            known_fields = self.jira_custom_fields_config.known_fields
            self._custom_field_mapping = {
                field_name: field_id for field_name, field_id in field_id_index.items() if field_name in known_fields
            }
            self._reverse_custom_field_mapping = {
                field_id: field_name for field_name, field_id in self._custom_field_mapping.items()
            }

            self._field_mapping_cache_time = datetime.now(timezone.utc)
            self.logger.info(f"Updated custom field mapping: {self._custom_field_mapping}")

        except Exception as e:
            self.logger.error(f"Failed to update custom field mapping: {e}")
            if not self.jira_custom_fields_config.fallback_mode:
                raise

    def get_custom_field_mapping(self) -> Dict[str, str]:
        """Get current custom field mapping (field_name -> field_id)."""
        if not self._is_field_mapping_cache_valid():
            self._update_custom_field_mapping()
        return self._custom_field_mapping.copy()

    def get_reverse_custom_field_mapping(self) -> Dict[str, str]:
        """Get reverse custom field mapping (field_id -> field_name)."""
        if not self._is_field_mapping_cache_valid():
            self._update_custom_field_mapping()
        return self._reverse_custom_field_mapping.copy()

    def map_custom_fields_from_jira_issue(self, jira_issue_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map custom fields from Jira issue raw data to human-readable names."""
        if not self.jira_custom_fields_config.enabled:
            return {}

        field_mapping = self.get_custom_field_mapping()
        mapped_fields = {}

        for field_name, field_id in field_mapping.items():
            if field_id in jira_issue_raw.get('fields', {}):
                field_value = jira_issue_raw['fields'][field_id]
                # Handle different field value types
                if isinstance(field_value, dict):
                    # Handle complex field types (e.g., user, select, etc.)
                    if 'value' in field_value:
                        mapped_fields[field_name] = field_value['value']
                    elif 'name' in field_value:
                        mapped_fields[field_name] = field_value['name']
                    elif 'key' in field_value:
                        mapped_fields[field_name] = field_value['key']
                    else:
                        mapped_fields[field_name] = str(field_value)
                else:
                    mapped_fields[field_name] = field_value

        return mapped_fields

    def _generate_embedding_from_data(
        self,
        data: Union[JiraIssueDetails, ErrorLog],
        reduce_length: Optional[bool] = False,
        retry_count: Optional[int] = 0,
    ) -> Optional[List[float]]:
        """
        Generate embedding from JiraIssueDetails or ErrorLog object.

        Args:
            data: Either JiraIssueDetails or ErrorLog object

        Returns:
            Normalized embedding vector or None if generation fails
        """
        try:
            # Extract text content based on data type
            if isinstance(data, JiraIssueDetails) or isinstance(data, ErrorLog):
                text_input = f"{data.error_message or ''} {data.error_type or ''} {data.traceback or ''}"
                if reduce_length:
                    text_input = (
                        f"{data.error_message[:500] or ''} {data.error_type[:200] or ''} {data.traceback[:200] or ''}"
                    )
            else:
                return None

            # Check if text input is empty or only whitespace
            if not text_input or not text_input.strip():
                return None

            # Generate embedding using embedding service (automatically normalized)
            if self.embedding_service is None:
                return None

            embedding = self.embedding_service.generate_embedding(text_input)

            # Validate unit vector (should always be true since EmbeddingService normalizes automatically)
            if not self.validate_unit_vector(embedding):
                # Re-normalize as fallback
                embedding = self.normalize_embedding(embedding)

            return embedding
        except ExceptionEmbeddingContextExceeded as e:
            if retry_count == 0:
                return self._generate_embedding_from_data(data, reduce_length=True, retry_count=1)
            else:
                logger.error(f"Failed to generate embedding for Jira issue {data.key}: {e}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Failed to generate embedding for Jira issue {data.key}: {e}", exc_info=True)
            return None

    def _generate_embedding_from_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding from raw text.

        Args:
            text: Raw text to generate embedding from

        Returns:
            Normalized embedding vector or None if generation fails
        """
        try:
            # Check if text input is empty or only whitespace
            if not text or not text.strip():
                return None

            # Generate embedding using embedding service (automatically normalized)
            if self.embedding_service is None:
                return None

            embedding = self.embedding_service.generate_embedding(text)

            # Validate unit vector (should always be true since EmbeddingService normalizes automatically)
            if not self.validate_unit_vector(embedding):
                # Re-normalize as fallback
                embedding = self.normalize_embedding(embedding)

            return embedding

        except Exception:
            return None

    def create_index(self) -> Dict[str, Any]:
        """
        Create the OpenSearch index with proper mapping.

        Args:
            None

        Returns:
            OpenSearch response
        """
        index_name = self.get_current_index_name()

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
                    "log_group": {"type": "keyword"},
                    "count": {"type": "integer"},
                    "parent_issue": {"type": "keyword"},
                    "created": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "updated": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "parent_issue_key": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    "not_commit_to_jira": {"type": "boolean"},
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

    def find_jira_issue_by_key(self, jira_key: str) -> Optional[Dict[str, Any]]:

        try:

            # Search using jira key
            search_body = {
                "size": 50,
                "query": {"term": {"key": jira_key}},
                "_source": [
                    "key",
                    "status",
                    "parent_issue_key",
                    "is_parent",
                    "error_message",
                    "error_type",
                    "traceback",
                    "site",
                    "request_id",
                    "log_group",
                    "count",
                    "created",
                    "updated",
                    "description",
                    "is_parent",
                    "not_commit_to_jira",
                    "created_at",
                    "updated_at",
                ],
            }
            logger.info(f"Searching for Jira issue with key: {jira_key}")
            search_body_str = json.dumps(search_body)
            response = self.opensearch_connect.search(index=self.get_current_index_name(), body=search_body)
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                source['doc_id'] = hit["_id"]
                source["score"] = hit["_score"]
                logger.info(f"Jira issue found: {source['key']}, score: {hit['_score']}")

                return source
                # {
                #     "key": source["key"],
                #     "status": source.get("status", ""),
                #     "parent_issue_key": source.get("parent_issue_key", ""),
                #     "is_parent": source.get("is_parent", True),
                #     "error_message": source.get("error_message", ""),
                #     "error_type": source.get("error_type", ""),
                #     "traceback": source.get("traceback", ""),
                #     "site": source.get("site", ""),
                #     "request_id": source.get("request_id", ""),
                #     "log_group": source.get("log_group", ""),
                #     "count": source.get("count", ""),
                #     "created": source.get("created", ""),
                #     "updated": source.get("updated", ""),
                #     "description": source.get("description", ""),
                #     "is_parent": source.get("is_parent", True),
                #     "not_commit_to_jira": source.get("not_commit_to_jira", False),
                #     "created_at": source.get("created_at", ""),
                #     "updated_at": source.get("updated_at", ""),
                # }
            return None

        except Exception as e:
            self.logger.error(f"Failed to find similar Jira issue: {e}")
            return None

    def add_jira_issue_from_jira_issue_detail(self, jira_issue_data: JiraIssueDetails) -> Dict[str, Any]:
        """Add or upsert a JiraIssueDetails-backed document to the embedding DB."""
        try:
            # Check if issue already exists by exact key match
            try:
                if jira_issue_data.key:
                    existing_doc = self.opensearch_connect.get(
                        index=self.get_current_index_name(), id=jira_issue_data.key
                    )
                    if not existing_doc:
                        find_by_key = self.find_jira_issue_by_key(jira_issue_data.key)
                        if find_by_key:
                            existing_doc = True
                    if existing_doc and existing_doc.get("found", False):
                        self.logger.info(f"Jira issue {jira_issue_data.key} already exists in database, skipping")
                        return {"result": "skipped", "reason": "already_exists", "id": jira_issue_data.key}
            except Exception:
                # Document doesn't exist, continue with adding
                pass

            # Calculate embedding from Jira issue data
            embedding = self._generate_embedding_from_data(jira_issue_data)

            if embedding is None:
                error_msg = f"Cannot generate embedding for Jira issue {jira_issue_data.key}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            # Check if key is already in the database
            if jira_issue_data.key:
                try:
                    result = self.find_jira_issue_by_key(jira_issue_data.key)
                    if result:
                        self.logger.info(f"Jira issue {jira_issue_data.key} already exists in database, skipping")
                        return {
                            "result": "skipped",
                            "reason": "already_exists",
                            "similar_issue_key": jira_issue_data.key,
                            "similar_issue": result,
                        }
                except Exception:
                    # Document doesn't exist, continue with adding
                    pass
            else:
                # If no key, protect against near-duplicates with a strict check
                similar_issue = self.find_similar_jira_issue(
                    error_log_embedding=embedding, site=(jira_issue_data.site or "unknown"), similarity_threshold=0.9999
                )
                if similar_issue:
                    self.logger.info(
                        f"Similar Jira issue found with 99%+ similarity: {similar_issue.get('key', 'unknown')}, skipping"
                    )
                    return {
                        "result": "skipped",
                        "reason": "similar_issue_found",
                        "similar_issue": similar_issue,
                    }

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
                "not_commit_to_jira": jira_issue_data.not_commit_to_jira,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "log_group": jira_issue_data.log_group,
            }

            # Save to embedding database
            response = self.opensearch_connect.index(index=self.get_current_index_name(), body=doc)
            assigned_id = response.get("_id", None)

            self.logger.info(f"Added Jira issue {assigned_id} to embedding database")
            return {"result": "added", "id": assigned_id, "opensearch_response": response}

        except Exception as e:
            self.logger.error(f"Failed to add Jira issue {jira_issue_data.key}: {e}", exc_info=True)
            raise

    def add_jira_issue_from_error_log(self, error_log_data: ErrorLog) -> Dict[str, Any]:
        """Create and add a Jira issue derived from an ErrorLog to the embedding DB."""
        try:
            details = self._create_jira_issue_from_error_log(error_log_data, getattr(error_log_data, "site", "unknown"))
            jira_cloud_client = JiraCloudClient(self.config.jira)
            if not details.key:
                details.key = jira_cloud_client.create_issue(details)

            if not details:
                raise ValueError("Failed to build JiraIssueDetails from error log", exc_info=True)
            return self.add_jira_issue_from_jira_issue_detail(details)
        except Exception as e:
            self.logger.error(
                f"Failed to add Jira issue from error log {getattr(error_log_data, 'message_id', '')}: {e}"
            )
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

            # Search using knn with site filter and parent issues only
            search_body = {
                "size": 10,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": error_log_embedding,
                            "k": 32,
                        }
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
                "min_score": 2 * similarity_threshold,
            }
            # logger.info(f"Searching for similar Jira issue with body: {search_body}")
            search_body_str = json.dumps(search_body)
            response = self.opensearch_connect.search(index=self.get_current_index_name(), body=search_body)
            candidates = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                logger.info(f"Similar Jira issue found: {source['key']}, score: {hit['_score']}")
                doc_candidate = source
                doc_candidate["doc_id"] = hit["_id"]
                doc_candidate["score"] = hit["_score"]
                # {
                #     "doc_id": hit["_id"],
                #     "key": source["key"],
                #     "score": hit["_score"],
                #     "summary": source.get("summary", ""),
                #     "description": source.get("description", ""),
                #     "status": source.get("status", ""),
                #     "error_message": source.get("error_message", ""),
                #     "error_type": source.get("error_type", ""),
                #     "traceback": source.get("traceback", ""),
                #     "site": source.get("site", ""),
                #     "request_id": source.get("request_id", ""),
                #     "created": source.get("created", ""),
                #     "updated": source.get("updated", ""),
                #     "parent_issue_key": source.get("parent_issue_key", ""),
                #     "is_parent": source.get("is_parent", True),
                # }
                if source["site"] == site and source["is_parent"] == True:
                    candidates.append(doc_candidate)
                    if doc_candidate["key"]:
                        return doc_candidate
            if len(candidates) > 0:
                return candidates[0]
            return None

        except Exception as e:
            self.logger.error(f"Failed to find similar Jira issue: {e}")
            return None

    # def add_occurrence(self, source_doc_id: str, doc_id: str, timestamp: str) -> Dict[str, Any]:
    #     """
    #     Add an occurrence reference to a Jira issue.

    #     Args:
    #         source_doc_id: Jira issue key
    #         doc_id: Document ID from error log
    #         timestamp: Timestamp of occurrence

    #     Returns:
    #         OpenSearch response
    #     """
    #     try:
    #         # TODO: check doc_id is not in source_doc_id's occurrence_list
    #         source_doc = self.opensearch_connect.get(index=self.get_current_index_name(), id=source_doc_id)
    #         if source_doc:
    #             if '_source' in source_doc:
    #                 source_doc = source_doc['_source']
    #             if doc_id in source_doc.get("occurrence_list"):
    #                 for occ in source_doc.get("occurrence_list"):
    #                     if occ["doc_id"] == doc_id:
    #                         self.logger.info(f"Doc {doc_id} already in {source_doc_id}'s occurrence_list, skipping")
    #                         return {"result": "skipped", "reason": "already_exists", "id": source_doc_id}

    #         index_name = self.get_current_index_name()

    #         # Add occurrence to the nested occurrence_list
    #         script = {
    #             "script": {
    #                 "source": (
    #                     "if (ctx._source.occurrence_list == null) { "
    #                     "ctx._source.occurrence_list = [] } "
    #                     "ctx._source.occurrence_list.add(params.occurrence)"
    #                 ),
    #                 "params": {"occurrence": {"doc_id": doc_id, "timestamp": timestamp}},
    #             }
    #         }

    #         response = self.opensearch_connect.update(index=index_name, id=source_doc_id, body=script)

    #         self.logger.debug(f"Added occurrence {doc_id} to Jira issue {source_doc_id}", exc_info=True)
    #         return response

    #     except Exception as e:
    #         self.logger.error(f"Failed to add occurrence to {source_doc_id}: {e}", exc_info=True)
    #         raise

    # def remove_duplicate_occurrences(self, doc_key: str) -> bool:
    #     """
    #     Remove duplicate occurrences from a Jira issue.
    #     """
    #     try:
    #         doc_data = self.opensearch_connect.get(index=self.get_current_index_name(), id=doc_key)
    #         if not doc_data:
    #             return False
    #         if '_source' in doc_data:
    #             doc_data = doc_data['_source']
    #         occurrences = doc_data.get("occurrence_list", [])
    #         occ_set = set()
    #         occ_data = {}
    #         for occ in occurrences:
    #             occ_set.add(occ["doc_id"])
    #             occ_data[occ["doc_id"]] = occ
    #         if len(occurrences) != len(occ_set):
    #             target = []
    #             for occ_id in occ_set:
    #                 target.append(occ_data[occ_id])
    #             self.opensearch_connect.update(
    #                 index=self.get_current_index_name(), id=doc_key, body={"doc": {"occurrence_list": target}}
    #             )

    #     except Exception as e:
    #         self.logger.error(f"Failed to remove duplicate occurrences for {doc_key}: {e}")
    #         return False
    #     return True

    # def get_occurrences(self, jira_key: str) -> List[Dict[str, str]]:
    #     """
    #     Get all occurrences for a specific Jira issue.

    #     Args:
    #         jira_key: Jira issue key

    #     Returns:
    #         List of occurrence data
    #     """
    #     try:
    #         issue_data = self.get_issue_by_key(jira_key)
    #         if not issue_data:
    #             return []

    #         occurrences = issue_data.get("occurrence_list", [])
    #         return [{"doc_id": occ.get("doc_id"), "timestamp": occ.get("timestamp")} for occ in occurrences]

    #     except Exception as e:
    #         self.logger.error(f"Failed to get occurrences for {jira_key}: {e}")
    #         return []

    def delete_issue(self, jira_key: str) -> Dict[str, Any]:
        """
        Delete a Jira issue and all its occurrences.

        Args:
            jira_key: Jira issue key


        Returns:
            OpenSearch response
        """
        try:
            index_name = self.get_current_index_name()

            response = self.opensearch_connect.delete(index=index_name, id=jira_key)

            self.logger.info(f"Deleted Jira issue {jira_key}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to delete Jira issue {jira_key}: {e}")
            raise

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding database

        Args:


        Returns:
            Database statistics
        """
        try:
            index_name = self.get_current_index_name()

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
            query_embedding = self._generate_embedding_from_text(query_text)

            if query_embedding is None:
                self.logger.error("Cannot generate embedding for query text")
                return []

            # Search using knn with parent issues only
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k,
                            "num_candidates": top_k * 10,
                            "filter": [{"term": {"is_parent": True}}],
                        }
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

    def _update_jira(self, jira_issues: List[JiraIssueDetails]) -> Dict[str, Any]:
        # Process Jira issues
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jira_issues_processed": 0,
            "jira_issues_added": 0,
            "jira_issues_skipped": 0,
            "errors": [],
        }
        logger.info(f"Processing {len(jira_issues)} Jira issues")
        for jira_issue_data in jira_issues:
            try:
                # Skip issues that are not committed to Jira
                if getattr(jira_issue_data, "not_commit_to_jira", False):
                    self.logger.info(f"Skipping Jira issue {jira_issue_data.key} - not committed to Jira")
                    continue

                # Add Jira issue to database
                result = self.add_jira_issue_from_jira_issue_detail(jira_issue_data)
                results["jira_issues_processed"] += 1

                if result.get("result") == "added":
                    results["jira_issues_added"] += 1
                elif result.get("result") == "skipped":
                    results["jira_issues_skipped"] += 1

            except Exception as e:
                error_msg = f"Failed to process Jira issue {jira_issue_data.key}: {e}"
                logger.error(error_msg, exc_info=True)
                results["errors"].append(error_msg)
        return results

    def _update_error_logs(
        self, error_logs: List[Any], skip_error_logs_count: int = 0, max_process_error_logs: int = 1000000
    ) -> Dict[str, Any]:
        """
        Update the error logs in the database.
        """

        initialization_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "error_logs_processed": skip_error_logs_count,
            "similar_issues_found": 0,
            "new_issues_created": 0,
            "occurrences_added": 0,
            "errors": [],
            "stats": {},
        }

        total_error_logs = len(error_logs)
        self.logger.info(f"Processing {total_error_logs} error logs")

        for error_log in error_logs:
            try:
                # Calculate embedding for error log
                error_embedding = self._generate_embedding_from_data(error_log)

                if error_embedding is None:
                    self.logger.warning(f"Skipping error log {error_log.message_id} - cannot generate embedding")
                    continue

                # Find similar Jira issues
                similar_issue = self.find_similar_jira_issue(
                    error_log_embedding=error_embedding, site=error_log.site, similarity_threshold=0.85
                )
                if not similar_issue:
                    error_embedding = self._generate_embedding_from_data(error_log, reduce_length=True)
                    similar_issue = self.find_similar_jira_issue(
                        error_log_embedding=error_embedding, site=error_log.site, similarity_threshold=0.85
                    )

                if similar_issue:
                    # Add occurrence to existing issue
                    self.add_occurrence(
                        source_doc_id=similar_issue["doc_id"],
                        doc_id=error_log.message_id,
                        timestamp=error_log.timestamp,
                    )
                    initialization_results["similar_issues_found"] += 1
                    initialization_results["occurrences_added"] += 1

                else:
                    # Create new Jira issue from error log
                    new_jira_data = self._create_jira_issue_from_error_log(error_log, error_log.site)
                    if new_jira_data:
                        result = self.add_jira_issue_from_jira_issue_detail(new_jira_data)
                        if result.get("result") == "added":
                            initialization_results["new_issues_created"] += 1
                        else:
                            logger.error(
                                f"Failed to add new Jira issue {new_jira_data.key}: {result.get('error')}",
                                exc_info=True,
                            )

                initialization_results["error_logs_processed"] += 1
            except Exception as e:
                error_msg = f"Failed to process error log {error_log.message_id} for site {error_log.site}: {e}"
                self.logger.error(error_msg)
                initialization_results["errors"].append(error_msg)

        return initialization_results

    def initialize_database(
        self,
        jira_issues: List[JiraIssueDetails],
        error_logs: List[Any],
        skip_error_logs_count: int = 0,
        max_process_error_logs: int = 1000000,
    ) -> Dict[str, Any]:
        """
        Initialize the Jira Issue Embedding Database with Jira issues and error logs.

        This method processes Jira issues from the API, calculates embeddings using the RAG engine,
        processes error logs from the past 6 months, finds similar issues, and adds occurrences.

        Args:
            jira_issues: List of JiraIssueDetails objects from API
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
        # Ensure current index exists
        current_index = self.get_current_index_name()

        if not self.opensearch_connect.indices.exists(index=current_index):
            self.logger.info("Creating index")
            self.create_index()
        try:
            self.logger.info("Starting Jira Issue Embedding Database initialization")
            initialization_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "jira_issues_processed": 0,
                "jira_issues_added": 0,
                "jira_issues_skipped": 0,
                "error_logs_processed": 0,
                "similar_issues_found": 0,
                "new_issues_created": 0,
                "occurrences_added": 0,
                "errors": [],
                "site_stats": {},
            }
            if jira_issues:
                jira_issues_results = self._update_jira(jira_issues)
                initialization_results["jira_issues_processed"] += jira_issues_results["jira_issues_processed"]
                initialization_results["jira_issues_added"] += jira_issues_results["jira_issues_added"]
                initialization_results["jira_issues_skipped"] += jira_issues_results["jira_issues_skipped"]
                initialization_results["errors"].extend(jira_issues_results["errors"])

            # Update final status
            initialization_results["status"] = "completed"
            self.logger.info(
                f"Database initialization completed. Processed "
                f"{initialization_results['jira_issues_processed']} Jira issues and "
                f"{initialization_results['error_logs_processed']} error logs"
            )
            if error_logs:
                error_logs_results = self._update_error_logs(error_logs, skip_error_logs_count, max_process_error_logs)
                initialization_results["error_logs_processed"] += error_logs_results["error_logs_processed"]
                initialization_results["similar_issues_found"] += error_logs_results["similar_issues_found"]
                initialization_results["new_issues_created"] += error_logs_results["new_issues_created"]
                initialization_results["occurrences_added"] += error_logs_results["occurrences_added"]
                initialization_results["errors"].extend(error_logs_results["errors"])
            return initialization_results

        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": error_msg,
                "jira_issues_processed": 0,
                "jira_issues_added": 0,
                "jira_issues_skipped": 0,
                "error_logs_processed": 0,
                "similar_issues_found": 0,
                "new_issues_created": 0,
                "occurrences_added": 0,
                "errors": [error_msg],
            }

    def _create_jira_issue_from_error_log(self, error_log: ErrorLog, site: str) -> Optional[JiraIssueDetails]:
        """
        Create a new Jira issue from an error log when no similar issue is found.

        Args:
            error_log: ErrorLog object
            site: Site name

        Returns:
            JiraIssueDetails object or None if creation fails
        """
        # if not skip_create_jira:
        #     # Add to Jira Cloud
        #     self.add_jira_issue_from_error_log(error_log)
        try:
            # Create JiraIssueDetails from error log
            jira_data = JiraIssueDetails(
                key=None,
                summary=error_log.error_message[:100] if error_log.error_message else "Error from logs",
                status="Open",
                parent_issue_key=None,
                child_issue_keys=[],
                error_message=error_log.error_message or "",
                error_type=error_log.error_type or "",
                traceback=error_log.traceback or "",
                site=site,
                request_id=error_log.request_id or "",
                log_group=error_log.log_group if hasattr(error_log, "log_group") else None,
                count=error_log.count if hasattr(error_log, "count") else None,
                created=error_log.timestamp.isoformat(),
                updated=error_log.timestamp.isoformat(),
                is_parent=True,
                not_commit_to_jira=True,
                description=f"Auto-generated issue from error log {error_log.message_id}",
            )

            return jira_data

        except Exception as e:
            self.logger.error(f"Failed to create Jira issue from error log {error_log.message_id}: {e}", exc_info=True)
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
                    test_embedding = self._generate_embedding_from_text("test")
                    if test_embedding is not None:
                        is_unit_vector = self.validate_unit_vector(test_embedding)
                        health_status["checks"]["embedding_service"] = "healthy" if is_unit_vector else "unhealthy"
                    else:
                        health_status["checks"]["embedding_service"] = "unhealthy"
                        health_status["errors"].append("Embedding service test failed: cannot generate embedding")
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
                test_embedding = self._generate_embedding_from_text("performance test")
                embedding_latency = time.time() - start_time

                if test_embedding is None:
                    self.logger.warning("Embedding performance test failed: cannot generate embedding")
                    embedding_latency = 0
            else:
                embedding_latency = 0

            if self.embedding_service is not None and test_embedding is not None:
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
                },
                "health_status": self.health_check(),
                "performance_metrics": self.performance_metrics(),
                "error_monitoring": self.error_rate_monitoring(),
            }

            return summary

        except Exception as e:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}
