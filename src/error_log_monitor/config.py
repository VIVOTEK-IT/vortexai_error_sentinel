"""
Configuration management for the Error Log Monitoring System.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available OpenAI models for error analysis."""

    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


@dataclass
class ModelConfig:
    """Configuration for each model type."""

    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_tokens: int
    use_cases: List[str]


@dataclass
class RAGConfig:
    """RAG system configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieved_chunks: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "text-embedding-3-small"


@dataclass
class OpenSearchConfig:
    """OpenSearch connection configuration."""

    host: str
    port: int = 9200
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    verify_certs: bool = False
    index_pattern: str = "error_log_{site}_{year}_{month}"


@dataclass
class RDSConfig:
    """RDS database configuration for Vortexai."""

    host: str = ""
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    ssl_mode: str = "require"


@dataclass
class JiraConfig:
    """Jira Cloud API configuration."""

    server_url: str = ""
    username: str = ""
    api_token: str = ""
    project_key: str = ""


@dataclass
class JiraEmbeddingConfig:
    """Jira Issue Embedding Database configuration."""

    index_name_template: str = "jira_issue_embedding"
    similarity_threshold: float = 0.85
    top_k: int = 10
    batch_size: int = 100
    retention_years: int = 3


@dataclass
class JiraCustomFieldConfig:
    """Jira Custom Fields configuration."""

    enabled: bool = True
    cache_ttl: int = 3600  # seconds
    auto_update: bool = True
    fallback_mode: bool = True
    known_fields: List[str] = None

    def __post_init__(self):
        if self.known_fields is None:
            self.known_fields = [
                "error_message",
                "error_type",
                "request_id",
                "log_group",
                "site",
                "count",
                "traceback",
                "parent_issue",
            ]


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    persist_directory: str = "./data/chroma_db"
    collection_name: str = "error_log_vectors"
    distance_metric: str = "cosine"
    embedding_model: str = "text-embedding-3-small"


@dataclass
class SystemConfig:
    """Main system configuration."""

    # OpenAI Configuration
    openai_api_key: str
    models: Dict[ModelType, ModelConfig]

    # RAG Configuration
    rag: RAGConfig

    # OpenSearch Configuration
    opensearch: OpenSearchConfig

    # RDS Configuration
    rds: RDSConfig

    # Jira Configuration
    jira: JiraConfig

    # Jira Embedding Configuration
    jira_embedding: JiraEmbeddingConfig

    # Jira Custom Fields Configuration
    jira_custom_fields: JiraCustomFieldConfig

    # Vector Database Configuration
    vector_db: VectorDBConfig

    # Processing Configuration
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: int = 5

    # Cost Optimization
    cost_threshold_daily: float = 100.0
    enable_cost_tracking: bool = True

    # Alerting Configuration
    webhook_url: Optional[str] = None
    alert_severity_threshold: str = "HIGH"

    # Scheduling
    cron_schedule: str = "0 6 * * *"
    timezone: str = "UTC"


def load_config() -> SystemConfig:
    """Load configuration from environment variables."""

    # Model configurations
    models = {
        ModelType.GPT5_MINI: ModelConfig(
            name="gpt-5-mini",
            input_cost_per_1m=0.25,
            output_cost_per_1m=2.00,
            max_tokens=4096,
            use_cases=["Standard error analysis", "Complex error patterns"],
        ),
        ModelType.GPT5_NANO: ModelConfig(
            name="gpt-5-nano",
            input_cost_per_1m=0.05,
            output_cost_per_1m=0.40,
            max_tokens=2048,
            use_cases=["Simple error classification", "High-volume processing"],
        ),
    }

    # RAG configuration
    rag_config = RAGConfig(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        max_retrieved_chunks=int(os.getenv("RAG_MAX_CHUNKS", "5")),
        similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7")),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    # OpenSearch configuration
    opensearch_config = OpenSearchConfig(
        host=os.getenv("OPENSEARCH_HOST", "43.207.106.51"),
        port=int(os.getenv("OPENSEARCH_PORT", "443")),
        username=os.getenv("OPENSEARCH_USERNAME", ""),
        password=os.getenv("OPENSEARCH_PASSWORD", ""),
        use_ssl=os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true",
        verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        index_pattern=os.getenv("OPENSEARCH_INDEX_PATTERN", "error_log_{site}_{year}_{month}"),
    )

    # RDS configuration
    rds_config = RDSConfig(
        host=os.getenv("RDS_HOST", ""),
        port=int(os.getenv("RDS_PORT", "5432")),
        database=os.getenv("RDS_DATABASE", ""),
        username=os.getenv("RDS_USER", ""),
        password=os.getenv("RDS_PASSWORD", ""),
        ssl_mode=os.getenv("RDS_SSL_MODE", "require"),
    )

    # Jira configuration
    jira_config = JiraConfig(
        server_url=os.getenv("JIRA_SERVER_URL", ""),
        username=os.getenv("JIRA_USERNAME", ""),
        api_token=os.getenv("JIRA_API_TOKEN", ""),
        project_key=os.getenv("JIRA_PROJECT_KEY", ""),
    )

    # Jira embedding configuration
    jira_embedding_config = JiraEmbeddingConfig(
        index_name_template=os.getenv("JIRA_EMBEDDING_INDEX_TEMPLATE", "jira_issue_embedding"),
        similarity_threshold=float(os.getenv("JIRA_EMBEDDING_SIMILARITY_THRESHOLD", "0.85")),
        top_k=int(os.getenv("JIRA_EMBEDDING_TOP_K", "10")),
        batch_size=int(os.getenv("JIRA_EMBEDDING_BATCH_SIZE", "100")),
        retention_years=int(os.getenv("JIRA_EMBEDDING_RETENTION_YEARS", "3")),
    )

    # Jira custom fields configuration
    jira_custom_fields_config = JiraCustomFieldConfig(
        enabled=os.getenv("JIRA_CUSTOM_FIELDS_ENABLED", "true").lower() == "true",
        cache_ttl=int(os.getenv("JIRA_CUSTOM_FIELDS_CACHE_TTL", "3600")),
        auto_update=os.getenv("JIRA_CUSTOM_FIELDS_AUTO_UPDATE", "true").lower() == "true",
        fallback_mode=os.getenv("JIRA_CUSTOM_FIELDS_FALLBACK_MODE", "true").lower() == "true",
    )

    # Vector database configuration
    vector_db_config = VectorDBConfig(
        persist_directory=os.getenv("VECTOR_DB_PERSIST_DIRECTORY", "./data/chroma_db"),
        collection_name=os.getenv("VECTOR_DB_COLLECTION_NAME", "error_log_vectors"),
        distance_metric=os.getenv("VECTOR_DB_DISTANCE_METRIC", "cosine"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    return SystemConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        models=models,
        rag=rag_config,
        opensearch=opensearch_config,
        rds=rds_config,
        jira=jira_config,
        jira_embedding=jira_embedding_config,
        jira_custom_fields=jira_custom_fields_config,
        vector_db=vector_db_config,
        batch_size=int(os.getenv("BATCH_SIZE", "10")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("RETRY_DELAY", "5")),
        cost_threshold_daily=float(os.getenv("COST_THRESHOLD_DAILY", "100.0")),
        enable_cost_tracking=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true",
        webhook_url=os.getenv("WEBHOOK_URL"),
        alert_severity_threshold=os.getenv("ALERT_SEVERITY_THRESHOLD", "HIGH"),
        cron_schedule=os.getenv("CRON_SCHEDULE", "0 6 * * *"),
        timezone=os.getenv("TIMEZONE", "UTC"),
    )


def get_model_for_complexity(complexity_score: float, config: SystemConfig) -> ModelType:
    """
    Select the most cost-effective model based on error complexity.

    Args:
        complexity_score: Score from 0.0 (simple) to 1.0 (complex)
        config: System configuration

    Returns:
        Selected model type
    """
    if complexity_score >= 0.45:
        return ModelType.GPT5_MINI
    else:
        return ModelType.GPT5_NANO


def calculate_estimated_cost(
    input_tokens: int, output_tokens: int, model_type: ModelType, config: SystemConfig
) -> float:
    """
    Calculate estimated cost for API call.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_type: Selected model type
        config: System configuration

    Returns:
        Estimated cost in USD
    """
    model_config = config.models[model_type]
    input_cost = (input_tokens / 1_000_000) * model_config.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model_config.output_cost_per_1m
    return input_cost + output_cost
