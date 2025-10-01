"""
Test suite for Jira Issue Embedding Database functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB, JiraIssueData, OccurrenceData
from src.error_log_monitor.config import SystemConfig, JiraEmbeddingConfig
from src.error_log_monitor.opensearch_client import OpenSearchClient, ErrorLog
from src.error_log_monitor.embedding_service import EmbeddingService


class TestJiraIssueEmbeddingDB:
    """Test cases for JiraIssueEmbeddingDB class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock system configuration."""
        config = Mock(spec=SystemConfig)
        config.jira_embedding = JiraEmbeddingConfig(
            index_name_template="test_jira_issue_embedding",
            similarity_threshold=0.85,
            top_k=10,
            batch_size=100,
            retention_years=3,
        )
        return config

    @pytest.fixture
    def mock_opensearch_client(self):
        """Create mock OpenSearch client."""
        client = Mock(spec=OpenSearchClient)
        client.client = Mock()
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock(spec=EmbeddingService)
        service.generate_embedding.return_value = [0.1] * 1536
        return service

    @pytest.fixture
    def jira_embedding_db(self, mock_config, mock_opensearch_client, mock_embedding_service):
        """Create JiraIssueEmbeddingDB instance for testing."""
        return JiraIssueEmbeddingDB(
            opensearch_client=mock_opensearch_client, embedding_service=mock_embedding_service, config=mock_config
        )

    @pytest.fixture
    def sample_jira_issue_data(self):
        """Create sample Jira issue data for testing."""
        return JiraIssueData(
            key="TEST-123",
            summary="Test Issue Summary",
            description="Test Issue Description",
            status="Open",
            error_message="Test error message",
            error_type="TestError",
            traceback="Test traceback",
            site="test",
            request_id="req-123",
            created="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
            parent_issue_key="",
            is_parent=True,
        )

    @pytest.fixture
    def sample_error_log(self):
        """Create sample error log for testing."""
        return ErrorLog(
            doc_id="doc-123",
            timestamp="2024-01-01T00:00:00Z",
            site="test",
            error_message="Test error message",
            error_type="TestError",
            traceback="Test traceback",
            module_name="test_module",
            function_name="test_function",
            line_number=123,
        )

    def test_get_current_index_name(self, jira_embedding_db):

        expected = f"test_jira_issue_embedding"
        assert jira_embedding_db.get_current_index_name() == expected

    def test_normalize_embedding(self, jira_embedding_db, mock_embedding_service):
        """Test embedding normalization to unit vector."""
        # Mock the embedding service to return a normalized vector
        mock_embedding_service.normalize_embedding.return_value = [0.6, 0.8, 0.0]

        # Test with a simple vector
        embedding = [3.0, 4.0, 0.0]  # Should normalize to [0.6, 0.8, 0.0]
        normalized = jira_embedding_db.normalize_embedding(embedding)

        # Check that it's a unit vector
        magnitude = np.linalg.norm(normalized)
        assert abs(magnitude - 1.0) < 1e-6

        # Check specific values
        expected = [0.6, 0.8, 0.0]
        for i, val in enumerate(expected):
            assert abs(normalized[i] - val) < 1e-6

    def test_normalize_embedding_zero_vector(self, jira_embedding_db, mock_embedding_service):
        """Test normalization with zero vector."""
        # Mock the embedding service to return the original vector for zero vector
        mock_embedding_service.normalize_embedding.return_value = [0.0, 0.0, 0.0]

        embedding = [0.0, 0.0, 0.0]
        normalized = jira_embedding_db.normalize_embedding(embedding)

        # Should return original vector for zero vector
        assert normalized == embedding

    def test_validate_unit_vector(self, jira_embedding_db, mock_embedding_service):
        """Test unit vector validation."""
        # Mock the embedding service validation
        mock_embedding_service.validate_unit_vector.side_effect = lambda vec, tol: abs(np.linalg.norm(vec) - 1.0) <= tol

        # Test with unit vector
        unit_vector = [0.6, 0.8, 0.0]
        assert jira_embedding_db.validate_unit_vector(unit_vector) is True

        # Test with non-unit vector
        non_unit_vector = [1.0, 2.0, 3.0]
        assert jira_embedding_db.validate_unit_vector(non_unit_vector) is False

    def test_create_index(self, jira_embedding_db, mock_opensearch_client):
        """Test index creation."""
        # Mock index doesn't exist
        mock_opensearch_client.client.indices.exists.return_value = False
        mock_opensearch_client.client.indices.create.return_value = {"acknowledged": True}

        result = jira_embedding_db.create_index()

        assert result["acknowledged"] is True
        mock_opensearch_client.client.indices.create.assert_called_once()

    def test_create_index_exists(self, jira_embedding_db, mock_opensearch_client):
        """Test index creation when index already exists."""
        # Mock index exists
        mock_opensearch_client.client.indices.exists.return_value = True

        result = jira_embedding_db.create_index()

        assert result["existing"] is True
        mock_opensearch_client.client.indices.create.assert_not_called()

    def test_add_jira_issue(
        self, jira_embedding_db, sample_jira_issue_data, mock_opensearch_client, mock_embedding_service
    ):
        """Test adding Jira issue to database."""
        # Mock embedding service response (now returns normalized vectors)
        normalized_embedding = [0.1] * 1536
        mock_embedding_service.generate_embedding.return_value = normalized_embedding
        mock_embedding_service.validate_unit_vector.return_value = True
        mock_opensearch_client.client.index.return_value = {"result": "created"}

        result = jira_embedding_db.add_jira_issue(sample_jira_issue_data)

        assert result["result"] == "created"
        mock_embedding_service.generate_embedding.assert_called_once()
        mock_embedding_service.validate_unit_vector.assert_called_once()
        mock_opensearch_client.client.index.assert_called_once()

    def test_add_jira_issue_with_error_log(
        self,
        jira_embedding_db,
        sample_jira_issue_data,
        sample_error_log,
        mock_opensearch_client,
        mock_embedding_service,
    ):
        """Test adding Jira issue with error log data."""
        # Mock embedding service response (now returns normalized vectors)
        normalized_embedding = [0.1] * 1536
        mock_embedding_service.generate_embedding.return_value = normalized_embedding
        mock_embedding_service.validate_unit_vector.return_value = True
        mock_opensearch_client.client.index.return_value = {"result": "created"}

        result = jira_embedding_db.add_jira_issue(sample_jira_issue_data, sample_error_log)

        assert result["result"] == "created"
        mock_embedding_service.generate_embedding.assert_called_once()
        mock_embedding_service.validate_unit_vector.assert_called_once()
        mock_opensearch_client.client.index.assert_called_once()

    def test_find_similar_jira_issue(self, jira_embedding_db, mock_opensearch_client, mock_embedding_service):
        """Test finding similar Jira issues."""
        # Mock search response
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "key": "TEST-123",
                            "summary": "Test Summary",
                            "description": "Test Description",
                            "status": "Open",
                            "error_message": "Test error",
                            "error_type": "TestError",
                            "traceback": "Test traceback",
                            "site": "test",
                            "request_id": "req-123",
                            "created": "2024-01-01T00:00:00Z",
                            "updated": "2024-01-01T00:00:00Z",
                            "parent_issue_key": "",
                            "is_parent": True,
                        },
                    }
                ]
            }
        }
        mock_opensearch_client.client.search.return_value = mock_response
        mock_embedding_service.validate_unit_vector.return_value = True

        embedding = [0.1] * 1536
        result = jira_embedding_db.find_similar_jira_issue(embedding, "test")

        assert result is not None
        assert result["key"] == "TEST-123"
        assert result["score"] == 0.9
        mock_embedding_service.validate_unit_vector.assert_called_once()
        mock_opensearch_client.client.search.assert_called_once()

    def test_find_similar_jira_issue_no_results(
        self, jira_embedding_db, mock_opensearch_client, mock_embedding_service
    ):
        """Test finding similar Jira issues with no results."""
        # Mock empty search response
        mock_response = {"hits": {"hits": []}}
        mock_opensearch_client.client.search.return_value = mock_response
        mock_embedding_service.validate_unit_vector.return_value = True

        embedding = [0.1] * 1536
        result = jira_embedding_db.find_similar_jira_issue(embedding, "test")

        assert result is None
        mock_embedding_service.validate_unit_vector.assert_called_once()
        mock_opensearch_client.client.search.assert_called_once()

    def test_add_occurrence(self, jira_embedding_db, mock_opensearch_client):
        """Test adding occurrence to Jira issue."""
        mock_opensearch_client.client.update.return_value = {"result": "updated"}

        result = jira_embedding_db.add_occurrence("TEST-123", "doc-456", "2024-01-01T00:00:00Z")

        assert result["result"] == "updated"
        mock_opensearch_client.client.update.assert_called_once()

    def test_get_issue_by_key(self, jira_embedding_db, mock_opensearch_client):
        """Test getting Jira issue by key."""
        # Mock get response
        mock_response = {"found": True, "_source": {"key": "TEST-123", "summary": "Test Summary", "status": "Open"}}
        mock_opensearch_client.client.get.return_value = mock_response

        result = jira_embedding_db.get_issue_by_key("TEST-123")

        assert result is not None
        assert result["key"] == "TEST-123"
        mock_opensearch_client.client.get.assert_called_once()

    def test_get_issue_by_key_not_found(self, jira_embedding_db, mock_opensearch_client):
        """Test getting Jira issue by key when not found."""
        # Mock not found response
        mock_response = {"found": False}
        mock_opensearch_client.client.get.return_value = mock_response

        result = jira_embedding_db.get_issue_by_key("TEST-123")

        assert result is None
        mock_opensearch_client.client.get.assert_called_once()

    def test_get_occurrences(self, jira_embedding_db, mock_opensearch_client):
        """Test getting occurrences for Jira issue."""
        # Mock get response with occurrences
        mock_response = {
            "found": True,
            "_source": {
                "key": "TEST-123",
                "occurrence_list": [
                    {"doc_id": "doc-1", "timestamp": "2024-01-01T00:00:00Z"},
                    {"doc_id": "doc-2", "timestamp": "2024-01-02T00:00:00Z"},
                ],
            },
        }
        mock_opensearch_client.client.get.return_value = mock_response

        result = jira_embedding_db.get_occurrences("TEST-123")

        assert len(result) == 2
        assert result[0].doc_id == "doc-1"
        assert result[1].doc_id == "doc-2"

    def test_delete_issue(self, jira_embedding_db, mock_opensearch_client):
        """Test deleting Jira issue."""
        mock_opensearch_client.client.delete.return_value = {"result": "deleted"}

        result = jira_embedding_db.delete_issue("TEST-123")

        assert result["result"] == "deleted"
        mock_opensearch_client.client.delete.assert_called_once()

    def test_get_embedding_stats(self, jira_embedding_db, mock_opensearch_client):
        """Test getting embedding database statistics."""
        # Mock stats response
        mock_stats_response = {
            "indices": {"test_jira_issue_embedding_2024": {"total": {"store": {"size_in_bytes": 1024000}}}}
        }
        mock_count_response = {"count": 100}
        mock_sample_response = {
            "hits": {
                "hits": [
                    {"_source": {"occurrence_list": [{"doc_id": "doc1"}, {"doc_id": "doc2"}]}},
                    {"_source": {"occurrence_list": [{"doc_id": "doc3"}]}},
                ]
            }
        }

        mock_opensearch_client.client.indices.stats.return_value = mock_stats_response
        mock_opensearch_client.client.count.return_value = mock_count_response
        mock_opensearch_client.client.search.return_value = mock_sample_response

        result = jira_embedding_db.get_embedding_stats()

        assert result["total_documents"] == 100
        assert result["index_size_bytes"] == 1024000
        assert result["average_occurrences"] == 1.5  # (2 + 1) / 2

    def test_search_similar_issues(self, jira_embedding_db, mock_opensearch_client, mock_embedding_service):
        """Test searching for similar issues."""
        # Mock search response
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "key": "TEST-123",
                            "summary": "Test Summary",
                            "status": "Open",
                            "site": "test",
                            "is_parent": True,
                        },
                    }
                ]
            }
        }
        mock_opensearch_client.client.search.return_value = mock_response
        mock_embedding_service.generate_embedding.return_value = [0.1] * 1536

        result = jira_embedding_db.search_similar_issues("test query")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-123"
        assert result[0]["score"] == 0.9
        mock_embedding_service.generate_embedding.assert_called_once()
        mock_opensearch_client.client.search.assert_called_once()
    

    def test_error_handling_in_add_jira_issue(self, jira_embedding_db, sample_jira_issue_data, mock_embedding_service):
        """Test error handling in add_jira_issue method."""
        # Mock embedding service to raise exception
        mock_embedding_service.generate_embedding.side_effect = Exception("Embedding service error")

        with pytest.raises(Exception, match="Embedding service error"):
            jira_embedding_db.add_jira_issue(sample_jira_issue_data)

    def test_error_handling_in_find_similar_jira_issue(self, jira_embedding_db, mock_opensearch_client):
        """Test error handling in find_similar_jira_issue method."""
        # Mock search to raise exception
        mock_opensearch_client.client.search.side_effect = Exception("Search error")

        embedding = [0.1] * 1536
        result = jira_embedding_db.find_similar_jira_issue(embedding, "test")

        assert result is None

    def test_error_handling_in_get_embedding_stats(self, jira_embedding_db, mock_opensearch_client):
        """Test error handling in get_embedding_stats method."""
        # Mock stats to raise exception
        mock_opensearch_client.client.indices.stats.side_effect = Exception("Stats error")

        result = jira_embedding_db.get_embedding_stats()

        assert result == {}


class TestJiraIssueData:
    """Test cases for JiraIssueData dataclass."""

    def test_jira_issue_data_creation(self):
        """Test creating JiraIssueData instance."""
        data = JiraIssueData(
            key="TEST-123",
            summary="Test Summary",
            description="Test Description",
            status="Open",
            error_message="Test error",
            error_type="TestError",
            traceback="Test traceback",
            site="test",
            request_id="req-123",
            created="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
            parent_issue_key="",
            is_parent=True,
        )

        assert data.key == "TEST-123"
        assert data.summary == "Test Summary"
        assert data.is_parent is True


class TestOccurrenceData:
    """Test cases for OccurrenceData dataclass."""

    def test_occurrence_data_creation(self):
        """Test creating OccurrenceData instance."""
        data = OccurrenceData(doc_id="doc-123", timestamp="2024-01-01T00:00:00Z")

        assert data.doc_id == "doc-123"
        assert data.timestamp == "2024-01-01T00:00:00Z"


if __name__ == "__main__":
    pytest.main([__file__])
