import json
import os
import time
import unittest
from typing import List

import requests


VECTOR_DIM = 8


def build_find_similar_body(vector: List[float], site: str) -> dict:
    return {
        "size": 5,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": 5,
                    "num_candidates": 50,
                    "filter": [
                        {"term": {"site": site}},
                        {"term": {"is_parent": True}},
                    ],
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
        "min_score": 0.5,
    }


def build_search_similar_body(vector: List[float], top_k: int) -> dict:
    return {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": top_k,
                    "num_candidates": top_k * 10,
                    "filter": [
                        {"term": {"is_parent": True}},
                    ],
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
        "min_score": 0.5,
    }


def build_multi_year_body(vector: List[float], top_k: int) -> dict:
    return {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": top_k,
                    "num_candidates": top_k * 10,
                    "filter": [
                        {"term": {"is_parent": True}},
                    ],
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
        "min_score": 0.5,
    }


class KnnQueryStructureTests(unittest.TestCase):
    def test_knn_query_structures_valid_json(self) -> None:
        vec = [0.1] * VECTOR_DIM
        body1 = build_find_similar_body(vec, "vel")
        body2 = build_search_similar_body(vec, 10)
        body3 = build_multi_year_body(vec, 5)

        # Should dump to JSON without errors
        json.dumps(body1)
        json.dumps(body2)
        json.dumps(body3)

    def test_knn_filter_is_array(self) -> None:
        vec = [0.2] * VECTOR_DIM
        body = build_find_similar_body(vec, "vel")
        embedding = body["query"]["knn"]["embedding"]
        self.assertIsInstance(embedding["filter"], list)
        self.assertEqual(len(embedding["filter"]), 2)

    def test_knn_required_fields_present(self) -> None:
        vec = [0.3] * VECTOR_DIM
        body = build_search_similar_body(vec, 7)
        embedding = body["query"]["knn"]["embedding"]
        self.assertIn("vector", embedding)
        self.assertIn("k", embedding)
        self.assertIn("num_candidates", embedding)
        self.assertIn("filter", embedding)
        self.assertIsInstance(embedding["filter"], list)


class KnnQueryOpenSearchIntegrationTest(unittest.TestCase):
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    INDEX_NAME = "test_knn_queries"

    @classmethod
    def setUpClass(cls) -> None:
        try:
            response = requests.get(f"{cls.OPENSEARCH_URL}/_cluster/health", timeout=5)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - skip when OpenSearch unavailable
            raise unittest.SkipTest(f"OpenSearch not available: {exc}")

        cls._delete_index()
        cls._create_index()
        cls._index_documents()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._delete_index()

    @classmethod
    def _delete_index(cls) -> None:
        requests.delete(f"{cls.OPENSEARCH_URL}/{cls.INDEX_NAME}", timeout=5)

    @classmethod
    def _create_index(cls) -> None:
        index_definition = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": VECTOR_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "key": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "site": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    "error_message": {"type": "text"},
                }
            },
        }

        resp = requests.put(f"{cls.OPENSEARCH_URL}/{cls.INDEX_NAME}", json=index_definition, timeout=10)
        resp.raise_for_status()

    @classmethod
    def _index_documents(cls) -> None:
        docs = [
            {
                "key": "TEST-1",
                "summary": "Connection timeout",
                "error_message": "Timed out",
                "site": "integration_site",
                "is_parent": True,
                "embedding": [0.1] * VECTOR_DIM,
            },
            {
                "key": "TEST-2",
                "summary": "Database failure",
                "error_message": "DB down",
                "site": "integration_site",
                "is_parent": True,
                "embedding": [0.12] * VECTOR_DIM,
            },
            {
                "key": "TEST-CHILD",
                "summary": "Child issue",
                "error_message": "Child error",
                "site": "integration_site",
                "is_parent": False,
                "embedding": [0.9] * VECTOR_DIM,
            },
        ]

        for doc in docs:
            resp = requests.put(
                f"{cls.OPENSEARCH_URL}/{cls.INDEX_NAME}/_doc/{doc['key']}",
                json=doc,
                timeout=5,
            )
            resp.raise_for_status()

        requests.post(f"{cls.OPENSEARCH_URL}/{cls.INDEX_NAME}/_refresh", timeout=5)
        time.sleep(0.5)

    def test_knn_query_returns_expected_hit(self) -> None:
        query_vector = [0.1] * VECTOR_DIM
        body = build_find_similar_body(query_vector, "integration_site")

        response = requests.get(f"{self.OPENSEARCH_URL}/{self.INDEX_NAME}/_search", json=body, timeout=10)
        self.assertEqual(response.status_code, 200, response.text)

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        self.assertGreater(len(hits), 0, data)

        returned_keys = {hit["_source"]["key"] for hit in hits}
        self.assertIn("TEST-1", returned_keys)
        self.assertNotIn("TEST-CHILD", returned_keys)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
