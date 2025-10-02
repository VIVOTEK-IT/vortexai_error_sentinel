import sys
from datetime import datetime
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from error_log_monitor.config import (
    SystemConfig,
    OpenSearchConfig,
    VectorDBConfig,
    RAGConfig,
    JiraConfig,
    JiraEmbeddingConfig,
    JiraCustomFieldConfig,
    RDSConfig,
    ModelConfig,
    ModelType,
)
from error_log_monitor.weekly_report_2 import WeeklyReportGenerator2


def create_dummy_config() -> SystemConfig:
    models = {
        ModelType.GPT5_MINI: ModelConfig(
            name="gpt-5-mini",
            input_cost_per_1m=0.25,
            output_cost_per_1m=2.0,
            max_tokens=4096,
            use_cases=[],
        )
    }

    return SystemConfig(
        openai_api_key="",
        models=models,
        rag=RAGConfig(),
        opensearch=OpenSearchConfig(host="localhost", port=9200, use_ssl=False, verify_certs=False),
        rds=RDSConfig(),
        jira=JiraConfig(),
        jira_embedding=JiraEmbeddingConfig(),
        jira_custom_fields=JiraCustomFieldConfig(),
        vector_db=VectorDBConfig(),
    )


class WeeklyReport2Tests(unittest.TestCase):
    def test_initialization(self) -> None:
        config = create_dummy_config()
        generator = WeeklyReportGenerator2(config)
        self.assertIsNotNone(generator.jira_embedding_db)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
