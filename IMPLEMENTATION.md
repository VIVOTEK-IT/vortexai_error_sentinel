# Error Log Monitoring System - Implementation Specification

## Overview

This document provides a comprehensive implementation specification for the Error Log Monitoring System designed to protect the Vortexai production service. The system implements RAG-based error similarity merging, OpenAI-powered impact analysis, and real-time RDS data integrity verification.

## System Architecture

### Core Components

#### 1. Configuration Management (`src/error_log_monitor/config.py`)
- **Purpose**: Centralized configuration management with environment variable support
- **Key Classes**:
  - `SystemConfig`: Main configuration container
  - `ModelConfig`: OpenAI model configuration (GPT-5-mini, GPT-5-nano)
  - `RAGConfig`: RAG system parameters
  - `OpenSearchConfig`: OpenSearch connection settings
  - `RDSConfig`: RDS database configuration
  - `JiraConfig`: Jira Cloud API configuration
  - `JiraEmbeddingConfig`: Jira Issue Embedding Database configuration
  - `VectorDBConfig`: Vector database settings

#### 2. OpenSearch Client (`src/error_log_monitor/opensearch_client.py`)
- **Purpose**: Read-only access to error logs from OpenSearch
- **Key Features**:
  - SSL/TLS connection support
  - Index pattern-based log retrieval
  - Error log parsing and normalization
  - Service name composition from log_group and module_name
- **Methods**:
  - `get_error_logs()`: Retrieve error logs by site and date range
  - `test_connection()`: Validate OpenSearch connectivity

#### 3. RDS Client (`src/error_log_monitor/rds_client.py`)
- **Purpose**: Data integrity verification for Vortexai RDS database
- **Key Features**:
  - Read-only PostgreSQL access
  - CAMERA_INFO table integrity checks
  - OBJECT_TRACE table integrity checks
  - Performance-compliant queries (filtered by mac and partition_id)
- **Methods**:
  - `get_camera_info(thingName)`: Retrive specified row of camera_info
  - `get_partition_info(partition_id)`: Retrive specified row of partition_info
  - `get_object_trace(mac, partition_id)`: Retrive specified row of object_trace

#### 4. Embedding Service (`src/error_log_monitor/embedding_service.py`)
- **Purpose**: Centralized service for text embedding generation and similarity calculations
- **Key Features**:
  - OpenAI embedding integration with configurable models
  - Batch embedding generation for efficiency
  - Cosine similarity calculations
  - Text similarity checking with configurable thresholds

#### 5. Jira Helper (`src/error_log_monitor/jira_helper.py`)
- **Purpose**: Retrieve and analyze Jira issues from OpenSearch with flexible querying options
- **Key Features**:
  - OpenSearch integration for Jira issue retrieval
  - Flexible querying by date range, error message patterns, and issue keys
  - Site-based filtering for environment-specific analysis
  - Comprehensive statistics and reporting
  - Support for related issue tracking and traceback analysis
- **Key Classes**:
  - `JiraIssue`: Data class representing a Jira issue with all relevant fields
  - `JiraHelper`: Main helper class for Jira operations
- **Methods**:
  - `get_recent_issues()`: Retrieve issues from recent days
  - `get_issues_by_date_range()`: Query issues within specific date range
  - `get_issues_by_error_message()`: Search issues by error message patterns
  - `get_issue_by_key()`: Retrieve specific issue by Jira key
  - `get_issue_statistics()`: Generate comprehensive issue statistics

#### 6. Jira Cloud Client (`src/error_log_monitor/jira_cloud_client.py`)
- **Purpose**: Real-time access to Jira Cloud API for detailed issue information
- **Key Features**:
  - Jira Cloud API integration with authentication
  - Real-time issue details fetching
  - Parent-child relationship tracking
  - Comprehensive issue metadata retrieval
- **Key Classes**:
  - `JiraIssueDetails`: Data class for detailed Jira issue information
  - `JiraCloudClient`: Main client for Jira Cloud operations
- **Methods**:
  - `get_issue_details()`: Fetch detailed information for a specific issue
  - `get_multiple_issue_details()`: Batch fetch details for multiple issues
  - `test_connection()`: Validate Jira Cloud connectivity
- **Additional Fields**:
  - `jira_status`: Current issue status from Jira Cloud
  - `parent_issue_key`: Parent issue key for epic/story relationships
  - `child_issue_keys`: List of child issue keys (subtasks)
  - `assignee`: Issue assignee
  - `priority`: Issue priority level
  - `issue_type`: Issue type (Bug, Story, Task, etc.)
  - `created`, `updated`: Timestamps
  - `description`: Issue description

#### 7. Vector Database Client (`src/error_log_monitor/vector_db_client.py`)
- **Purpose**: ChromaDB integration for vector storage and similarity search
- **Key Features**:
  - ChromaDB collection management
  - Vector chunk storage and retrieval
  - Similarity search with configurable parameters
  - Embedding service integration
- **Key Classes**:
  - `VectorDBClient`: Main client for vector database operations
- **Methods**:
  - `generate_embeddings()`: Generate embeddings for multiple texts
  - `generate_embedding()`: Generate embedding for single text
  - `cosine_similarity()`: Calculate similarity between vectors
  - `are_texts_similar()`: Check if two texts are similar
  - `batch_similarity_check()`: Check similarity between text pairs

#### 8. RAG Engine (`src/error_log_monitor/rag_engine.py`)
- **Purpose**: Context generation and similarity merging using RAG techniques
- **Key Features**:
  - EmbeddingService integration for similarity calculations
  - Error log chunking and vectorization
  - Similarity-based error grouping
  - Context retrieval for analysis
- **Key Classes**:
  - `MergedIssue`: Represents grouped similar errors
  - `VectorChunk`: Vector storage unit
- **Methods**:
  - `merge_similar_issues()`: Group similar errors using EmbeddingService
  - `generate_context()`: Create analysis context from similar logs
  - `retrieve_context_for_analysis()`: Get relevant context for analysis

#### 9. LLM Helper (`src/error_log_monitor/llm_helper.py`)
- **Purpose**: Generic function calling system for OpenAI integration with extensible function registry
- **Key Features**:
  - Function registry system for easy function management
  - Dynamic function definitions generation
  - Handler pattern for clean separation of concerns
  - Decorator support for easy function registration
  - Runtime function management (add/remove functions)
  - Round-trip conversation support
- **Key Classes**:
  - `FUNCTION_REGISTRY`: Central registry for all available functions
- **Key Functions**:
  - `register_function()`: Register new functions at runtime
  - `@function_tool`: Decorator for easy function registration
  - `execute_tool_call()`: Generic tool execution using registry
  - `llm_chat()`: Round-trip conversation management
- **Methods**:
  - `get_function_definitions()`: Generate OpenAI function definitions
  - `list_registered_functions()`: List all available functions
  - `unregister_function()`: Remove functions from registry

#### 10. Error Analyzer (`src/error_log_monitor/error_analyzer.py`)
- **Purpose**: OpenAI-powered error impact analysis with intelligent model selection
- **Key Features**:
  - 3-level error classification (LEVEL_1, LEVEL_2, LEVEL_3)
  - Intelligent model selection (GPT-5-mini vs GPT-5-nano)
  - RDS data integrity integration via function calling
  - Round-trip conversation with tool execution
  - Cost tracking and optimization
- **Key Classes**:
  - `ErrorAnalysis`: Complete analysis result
  - `ErrorScope`: Impact scope assessment
  - `RemediationPlan`: Action plan and guidelines
- **Methods**:
  - `analyze_merged_issues()`: Analyze grouped error issues
  - `analyze_issue()`: Analyze individual merged issue

#### 11. Main Application (`src/error_log_monitor/main.py`)
- **Purpose**: CLI interface and orchestration
- **Commands**:
  - `test`: Test database connections
  - `analyze`: Run error analysis for specified site and date range
- **Features**:
  - JSON and HTML report generation
  - Web dashboard integration
  - Comprehensive logging

## Data Flow

### 1. Error Log Retrieval
```
OpenSearch → OpenSearchClient → ErrorLog objects
```
- Retrieves error logs from OpenSearch using site and date range
- Parses and normalizes log entries
- Composes service names from log_group and module_name

### 2. RAG-Based Similarity Merging
```
ErrorLog[] → RAGEngine → MergedIssue[]
```
- Creates vector chunks from error logs
- Stores chunks in ChromaDB with metadata
- Groups similar errors using vector similarity
- Generates context for each merged issue

### 3. Impact Analysis
```
MergedIssue[] → ErrorAnalyzer → LLMHelper → OpenAI → ErrorAnalysis[]
```
- Selects appropriate OpenAI model based on complexity
- Uses LLM Helper for round-trip conversation with tool calling
- Provides RDS access functions (get_camera_info, get_partition_info, get_object_trace) to LLM
- LLM can execute tool calls to verify data integrity
- The analysis will be done by OpenAI model with tool access:
  - Have OpenAI to determine the damaged service or module
  - Have OpenAI to determine if data is damaged or not via RDS queries
  - Have OpenAI to determine if human remedial is needed, and provide guidance for human


### 4. Report Generation
```
ErrorAnalysis[] → Main Application → JSON/HTML Reports
```
- Creates machine-readable JSON reports
- Generates human-readable HTML reports
- Saves latest.json for web dashboard

## Error Classification System

### LEVEL_1 (Low Priority)
- **Criteria**: Single API/service broken, system working, no data damage
- **Action**: No human remedial action needed
- **Color Code**: Blue (#1976d2)

### LEVEL_2 (Medium Priority)
- **Criteria**: Part of services broken, no data damage
- **Action**: No human remedial action needed
- **Color Code**: Orange (#f57c00)

### LEVEL_3 (High Priority)
- **Criteria**: Data was damaged
- **Action**: Human remedial action needed
- **Color Code**: Red (#d32f2f)

## RDS Data Integrity Verification

### CAMERA_INFO Table Checks
- **Purpose**: Verify UPDATE_CAMERAINFO lambda data integrity
- **Checks**:
  - Record existence validation
  - Required field null checks (thingname, group_id, organization_id)
  - Data consistency verification
- **Performance**: Filtered by mac and partition_group_id

### OBJECT_TRACE Table Checks
- **Purpose**: Verify PARSE_METADATA lambda data integrity
- **Checks**:
  - Partition table existence
  - Record count and null value detection
  - Data consistency validation
- **Performance**: Filtered by mac and partition_id

## Function Calling System

### Overview
The system implements a generic, extensible function calling system that allows the LLM to interact with external systems (primarily RDS) during analysis. This enables real-time data integrity verification and more accurate impact assessment.

### Function Registry
- **Central Registry**: All available functions are registered in `FUNCTION_REGISTRY`
- **Dynamic Management**: Functions can be added/removed at runtime
- **Handler Pattern**: Each function has a dedicated handler for clean separation
- **Auto-Generation**: OpenAI function definitions are automatically generated from registry

### Available Functions
- **get_camera_info(mac)**: Retrieve camera information by MAC address
- **get_partition_info(partition_id)**: Get partition information
- **get_object_trace(mac, partition_id)**: Retrieve object trace data
- **check_object_trace_partition_exists(partition_id)**: Verify partition table existence
- **get_system_status(component)**: Get system health status

### Adding New Functions

#### Method 1: Using Decorator (Recommended)
```python
@function_tool(
    name="my_new_function",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"}
        },
        "required": ["param1"]
    }
)
def my_new_function_handler(function_args: dict, rds_client) -> str:
    param1 = function_args.get("param1")
    # Implementation here
    return json.dumps({"result": "success"})
```

#### Method 2: Using register_function()
```python
def my_handler(function_args: dict, rds_client) -> str:
    # Implementation here
    pass

register_function(
    name="my_function",
    description="Does something",
    parameters={...},
    handler_func_name="my_handler"
)
```

### Round-Trip Conversation
- **Multi-Turn Support**: LLM can make multiple tool calls in sequence
- **Context Preservation**: Tool results are fed back into conversation
- **Error Handling**: Robust error handling for failed tool calls
- **Cost Optimization**: Intelligent retry logic to minimize API costs

## Model Selection Strategy

### Complexity Scoring
- **Occurrence Count**: Higher frequency increases complexity
- **Affected Services**: Multiple services increase complexity
- **Error Message Length**: Longer messages indicate complexity
- **Traceback Presence**: Stack traces increase complexity
- **Keywords**: Database, connection, timeout, critical keywords

### Model Selection Logic
```python
if complexity_score >= 0.4:
    return ModelType.GPT5_MINI  # $0.25/$2.00 per 1M tokens
else:
    return ModelType.GPT5_NANO  # $0.05/$0.40 per 1M tokens
```

## Configuration Management

### Environment Variables

#### Required
- `OPENAI_API_KEY`: OpenAI API key for analysis
- `OPENSEARCH_HOST`: OpenSearch server host
- `OPENSEARCH_USERNAME`: OpenSearch authentication
- `OPENSEARCH_PASSWORD`: OpenSearch authentication

#### Optional
- `RDS_HOST`: RDS database host for integrity checks
- `RDS_USER`: RDS authentication
- `RDS_PASSWORD`: RDS authentication
- `RDS_DATABASE`: RDS database name (default: vsaas_postsearch)

#### RAG Configuration
- `RAG_CHUNK_SIZE`: Chunk size for vectorization (default: 1000)
- `RAG_CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `RAG_MAX_CHUNKS`: Maximum chunks to retrieve (default: 5)
- `RAG_SIMILARITY_THRESHOLD`: Similarity threshold (default: 0.7)
- `RAG_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)

## Database Schemas

### OpenSearch Index Schema
```json
{
  "mappings": {
    "properties": {
      "message_id": {"type": "keyword"},
      "timestamp": {"type": "date"},
      "error_message": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
      "error_message_hash": {"type": "long"},
      "traceback": {"type": "text"},
      "traceback_hash": {"type": "long"},
      "error_type": {"type": "keyword"},
      "error_type_hash": {"type": "long"},
      "site": {"type": "keyword"},
      "service": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
      "index_name": {"type": "keyword"},
      "topic": {"type": "keyword"},
      "count": {"type": "integer"},
      "request_id": {"type": "keyword"},
      "category": {"type": "keyword"},
      "log_group": {"type": "keyword"},
      "module_name": {"type": "keyword"},
      "version": {"type": "keyword"}
    }
  }
}
```

### RDS Table Schemas
- **CAMERA_INFO**: Camera information with mac, thingname, group_id, organization_id
- **OBJECT_TRACE**: Object trace data with oid, mac, obj_type, partition_id

## API Specifications

### CLI Interface

#### Test Command
```bash
python -m error_log_monitor.main test --site <site_name>
```
- Tests all database connections
- Validates configuration
- Returns connection status

#### Analyze Command
```bash
python -m error_log_monitor.main analyze --site <site_name> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
```
- Retrieves error logs for specified site and date range
- Performs RAG-based similarity merging
- Analyzes merged issues with OpenAI
- Generates JSON and HTML reports

### Web Dashboard API

#### Endpoints
- `GET /`: Main dashboard interface
- `GET /reports/latest.json`: Latest analysis results

#### Response Format
```json
{
  "site": "dev",
  "analysis_date": "2024-01-01T00:00:00Z",
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-02T00:00:00Z"
  },
  "summary": {
    "total_issues": 10,
    "level_1_issues": 5,
    "level_2_issues": 3,
    "level_3_issues": 2,
    "human_action_needed": 2,
    "total_cost": 0.0250
  },
  "issues": [...]
}
```

## Deployment Architecture

### Docker Configuration

#### Services
1. **error-monitor**: Main application container
2. **web-dashboard**: Nginx web server for dashboard

#### Volume Mappings
- `./data:/app/data`: Vector database persistence
- `./reports:/app/reports`: Report storage
- `./logs:/app/logs`: Log file storage
- `./web:/usr/share/nginx/html`: Web dashboard files

#### Network
- **error-monitor-network**: Bridge network for service communication

### Environment Setup

#### Development
```bash
# Clone repository
git clone <repository-url>
cd vortex_error_sentinel

# Configure environment
cp env.example .env
# Edit .env with your credentials

# Run with Docker
docker-compose up -d
```

#### Production
- Configure production environment variables
- Set up proper SSL certificates
- Configure monitoring and alerting
- Set up log rotation and retention

## Performance Considerations

### OpenSearch Queries
- Use filtered queries to avoid full table scans
- Implement proper indexing on timestamp and site fields
- Limit result sets to prevent memory issues

### RDS Queries
- Always filter by mac and partition_id
- Avoid full table reads of OBJECT_TRACE
- Use read-only connections to prevent data modification

### Vector Database
- Batch chunk storage operations
- Implement similarity threshold filtering
- Monitor collection size and performance

### OpenAI API
- Implement intelligent model selection
- Use batch processing where possible
- Monitor token usage and costs
- Implement retry logic for rate limits

## Security Considerations

### API Key Management
- Store OpenAI API key in environment variables
- Use secure key rotation policies
- Monitor API key usage

### Database Access
- Use read-only connections where possible
- Implement proper authentication
- Use SSL/TLS for all connections

### Data Privacy
- Implement log data retention policies
- Ensure sensitive data is not logged
- Use proper data anonymization

## Monitoring and Alerting

### Health Checks
- Database connection monitoring
- API endpoint availability
- Processing queue monitoring

### Metrics
- Error processing rate
- Analysis accuracy
- Cost tracking
- Performance metrics

### Alerts
- Connection failures
- High error rates
- Cost threshold breaches
- Critical error detection

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Test error handling scenarios

### Integration Tests
- End-to-end workflow testing
- Database integration testing
- API integration testing

### Performance Tests
- Load testing with large datasets
- Memory usage monitoring
- Response time validation

## Maintenance and Operations

### Log Management
- Implement log rotation
- Set up log aggregation
- Monitor log levels

### Data Management
- Vector database cleanup
- Report retention policies
- Backup and recovery procedures

### Updates and Patches
- Dependency updates
- Security patches
- Feature updates

## Troubleshooting Guide

### Common Issues

#### Connection Failures
- Check network connectivity
- Verify credentials
- Check firewall settings

#### Analysis Failures
- Verify OpenAI API key
- Check model availability
- Review error logs

#### Performance Issues
- Monitor resource usage
- Check database performance
- Review query optimization

### Debug Commands
```bash
# Test connections
docker exec -it error-monitor-app python -m error_log_monitor.main test --site dev

# Check logs
docker logs error-monitor-app

# Monitor resources
docker stats error-monitor-app
```

## Future Enhancements

### Planned Features
- Machine learning-based error pattern recognition
- Automated remediation suggestions
- Advanced analytics and reporting
- Multi-tenant support

### Scalability Improvements
- Horizontal scaling support
- Load balancing
- Caching strategies
- Database sharding

### Integration Opportunities
- Slack/Teams notifications
- JIRA integration
- Prometheus metrics
- Grafana dashboards

## 8. Jira Issue Embedding Database

### Purpose
The `JiraIssueEmbeddingDB` class provides a year-based vector database for Jira issues with embedding similarity search and error log correlation. It enables efficient similarity search between Jira issues and error logs using OpenAI embeddings with unit vector normalization for optimal OpenSearch performance.

### Key Features
- **Unit Vector Normalization**: All embeddings are normalized to unit vectors for cosine similarity via OpenSearch inner product
- **Occurrence Tracking**: Tracks which error logs reference specific Jira issues
- **Parent-Child Relationships**: Supports hierarchical issue structures with parent-only search
- **OpenSearch Integration**: Uses kNN search with inner product similarity for optimal performance
- **Health Monitoring**: Comprehensive health checks and performance metrics

### Classes and Functions

#### JiraIssueData
```python
@dataclass
class JiraIssueData:
    key: str                    # Jira issue key (e.g., "VEL-123")
    summary: str                # Jira issue summary
    description: str            # Jira issue description
    status: str                 # Current status
    error_message: str          # Error message
    error_type: str             # Error type
    traceback: str              # Traceback
    site: str                   # Site name
    request_id: str             # Request ID
    created: str                # Creation timestamp
    updated: str                # Last update timestamp
    parent_issue_key: str       # Parent issue key (if any)
    is_parent: bool             # True if parent_issue_key is empty
```

#### OccurrenceData
```python
@dataclass
class OccurrenceData:
    doc_id: str                 # Document ID from error_log OpenSearch database
    timestamp: str              # ISO 8601 timestamp of the occurrence
```

#### JiraIssueEmbeddingDB
```python
class JiraIssueEmbeddingDB:
    def __init__(self, opensearch_client: OpenSearchClient, embedding_service: EmbeddingService, config: SystemConfig, index_name_template: Optional[str] = None)
    def get_current_index_name(self) -> str
    def get_index_name_for_year(self, year: int) -> str
    def create_index(self, year: Optional[int] = None) -> Dict[str, Any]
    def add_jira_issue(self, jira_issue_data: JiraIssueData, error_log_data: Optional[ErrorLog] = None) -> Dict[str, Any]
    def find_similar_jira_issue(self, error_log_embedding: List[float], site: str, similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]
    def add_occurrence(self, jira_key: str, doc_id: str, timestamp: str, year: Optional[int] = None) -> Dict[str, Any]
    def get_issue_by_key(self, jira_key: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]
    def get_occurrences(self, jira_key: str, year: Optional[int] = None) -> List[OccurrenceData]
    def delete_issue(self, jira_key: str, year: Optional[int] = None) -> Dict[str, Any]
    def search_similar_issues(self, query_text: str, top_k: int = 10, similarity_threshold: float = 0.85) -> List[Dict[str, Any]]
    def get_embedding_stats(self, year: Optional[int] = None) -> Dict[str, Any]    
    def normalize_embedding(self, embedding_vector: List[float]) -> List[float]
    def validate_unit_vector(self, embedding_vector: List[float], tolerance: float = 1e-6) -> bool
    def health_check(self) -> Dict[str, Any]
    def performance_metrics(self) -> Dict[str, Any]
    def error_rate_monitoring(self, hours: int = 24) -> Dict[str, Any]
    def get_database_summary(self) -> Dict[str, Any]
```

### Configuration

#### JiraEmbeddingConfig
```python
@dataclass
class JiraEmbeddingConfig:
    index_name_template: str = "jira_issue_embedding_{year}"
    similarity_threshold: float = 0.85
    top_k: int = 10
    batch_size: int = 100
    retention_years: int = 3
```

#### Environment Variables
```bash
# Jira Issue Embedding Database
JIRA_EMBEDDING_INDEX_TEMPLATE=jira_issue_embedding_{year}
JIRA_EMBEDDING_SIMILARITY_THRESHOLD=0.85
JIRA_EMBEDDING_TOP_K=10
JIRA_EMBEDDING_BATCH_SIZE=100
JIRA_EMBEDDING_RETENTION_YEARS=3
JIRA_EMBEDDING_AUTO_CREATE_INDEX=true
```

### OpenSearch Index Schema

#### Index Mapping
```json
{
  "mappings": {
    "properties": {
      "key": {"type": "keyword"},
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "innerproduct",
          "engine": "faiss",
          "parameters": {
            "ef_construction": 128,
            "m": 32
          }
        }
      },
      "occurrence_list": {
        "type": "nested",
        "properties": {
          "doc_id": {"type": "keyword"},
          "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
        }
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
      "updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
    }
  },
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.knn": true,
    "index.knn.algo_param.ef_search": 100,
    "index.knn.algo_param.ef_construction": 128,
    "index.knn.algo_param.m": 16
  }
}
```

### Integration with Weekly Report

The `WeeklyReportGenerator` integrates with `JiraIssueEmbeddingDB` to find correlated Jira issues using embedding similarity:

- **Initialization**: `JiraIssueEmbeddingDB` is initialized in the `WeeklyReportGenerator` constructor
- **Correlation**: Uses `find_similar_jira_issue()` method to correlate merged issues with Jira issues
- **Embedding Calculation**: Calculates embeddings for merged issues using `rag_engine._calculate_error_log_embeddings`
- **Fallback Support**: Falls back to original correlation method if embedding database fails

### Monitoring Utilities

#### Monitoring Script (`scripts/monitor_jira_embeddings.py`)
- **Purpose**: Real-time monitoring and health checks
- **Commands**:
  - `health`: Check database health
  - `performance`: Get performance metrics
  - `summary`: Get database summary
  - `monitor`: Start continuous monitoring
  - `report`: Generate monitoring report
  - `alerts`: Check for alerts
  - `years`: List available years
  - `stats`: Get statistics for a year

### Technical Specifications

#### Unit Vector Requirements
- **OpenSearch Limitation**: OpenSearch only supports inner product similarity, not cosine similarity
- **Unit Vector Solution**: Unit vectors enable cosine similarity via inner product: `cos(θ) = a·b` for unit vectors
- **Normalization Process**: All embeddings are normalized to unit vectors before storage and search
- **Validation**: Built-in validation ensures all vectors are unit vectors with configurable tolerance

#### Performance Considerations
- **Year-Based Indexing**: Smaller indices per year for better performance and easier management
- **Index Size**: Monitor index size as embeddings are large (1536 dimensions × 4 bytes = ~6KB per issue)
- **Search Performance**: Use appropriate `ef_search` and `ef_construction` parameters
- **Batch Operations**: Use bulk operations for large data imports
- **Memory Usage**: Year-based indices reduce memory usage for single-year operations
- **OpenSearch Optimization**: Inner product with unit vectors provides optimal performance for cosine similarity

## 9. Weekly Report Generator

### Purpose
The `WeeklyReportGenerator` class provides comprehensive weekly report generation that combines error log analysis with Jira issue tracking, with separate processing for each site.

### Key Features
- **Site-Specific Processing**: Processes stage and production sites independently
- **Error Log Merging**: Uses RAG engine to merge similar error logs for each site separately
- **Embedding-Based Jira Correlation**: Advanced similarity matching using OpenAI embeddings with 0.85 threshold for precise issue correlation
- **Jira Cloud API Integration**: Real-time access to Jira Cloud for parent-child relationships, status, and detailed issue metadata
- **Root Cause Analysis**: Uses LLM to generate root cause descriptions for each issue
- **Comprehensive Reporting**: Generates both site-specific and combined Excel/HTML reports
- **Issue Tracking**: Links primary Jira issues with child issues for complete context

### Classes and Functions

#### WeeklyReportIssue
```python
@dataclass
class WeeklyReportIssue:
    key: str  # Jira issue URL (parent issue if available)
    site: str
    count: int  # Occurrence count within the week
    summary: str  # Jira issue summary
    error_message: str  # Error message preview (first 100 characters)
    status: str  # Jira issue status
    log_group: str
    latest_update: datetime  # Latest error timestamp
    note: str  # Root cause description from LLM
    child_issues: List[str]  # Child Jira issue keys
    primary_jira_issue: Optional[JiraIssue] = None
    merged_issue: Optional[MergedIssue] = None
```

#### WeeklyReportGenerator
```python
class WeeklyReportGenerator:
    def __init__(self, config: SystemConfig)
    def generate_weekly_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]
    def _correlate_with_jira_issues(self, merged_issues: List[MergedIssue], start_date: datetime, end_date: datetime) -> List[WeeklyReportIssue]
    def _generate_root_cause_analysis(self, weekly_issues: List[WeeklyReportIssue]) -> List[WeeklyReportIssue]
    def _generate_excel_report(self, weekly_issues: List[WeeklyReportIssue], start_date: datetime, end_date: datetime, site: str = "combined") -> str
    def _generate_html_report(self, weekly_issues: List[WeeklyReportIssue], start_date: datetime, end_date: datetime, site: str = "combined") -> str
    def _generate_combined_excel_report(self, site_reports: Dict[str, Any], start_date: datetime, end_date: datetime) -> str
    def _generate_combined_html_report(self, site_reports: Dict[str, Any], start_date: datetime, end_date: datetime) -> str
    def _create_empty_site_report(self, site: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]
```

### Report Structure
- **Individual Site Reports**: `weekly_report_{site}_{date}.xlsx/html`
- **Combined Reports**: `weekly_report_combined_{date}.xlsx/html`
- **Site-Specific Analysis**: Each site is processed independently with its own RAG merging and Jira correlation
- **Comprehensive Overview**: Combined reports provide cross-site analysis and comparison

### Correlation Process
The embedding-based correlation system uses a sophisticated 3-step process:

1. **Grab All Jira Issues**: Retrieves all Jira issues from the past 0.5 years (180 days) for the specific site
2. **Calculate Embeddings**: Generates embeddings for each Jira issue using combined `error_message`, `error_type`, and `traceback` fields
3. **Find Similar Issues**: Uses cosine similarity with 0.85 threshold to find matching issues from the same site
4. **Jira Cloud Enhancement**: Fetches additional details from Jira Cloud API including:
   - Real-time status information
   - Parent-child relationships (epic/story/subtask hierarchies)
   - Assignee, priority, issue type, and timestamps
   - Issue descriptions and metadata

### Integration
- **RAG Engine**: Uses `RAGEngine.merge_similar_issues()` for site-specific error log merging
- **Jira Helper**: Uses `JiraHelper.get_recent_issues()` for retrieving Jira issues from OpenSearch
- **Jira Cloud Client**: Uses `JiraCloudClient.get_multiple_issue_details()` for real-time Jira data
- **Embedding Service**: Uses `EmbeddingService.generate_embeddings()` for similarity calculations
- **Error Analyzer**: Uses `ErrorAnalyzer.analyze_issue()` for root cause analysis
- **OpenSearch Client**: Uses `OpenSearchClient.get_error_logs()` for error log retrieval

## Conclusion

This implementation specification provides a comprehensive guide for the Error Log Monitoring System. The system successfully implements all requirements from the design plan, including RAG-based similarity merging, OpenAI-powered analysis, RDS data integrity verification, comprehensive reporting capabilities, and site-separated weekly reports.

The modular architecture ensures maintainability and extensibility, while the Docker-based deployment provides easy setup and scaling. The system is specifically designed to protect the Vortexai production service with real-time monitoring and intelligent analysis capabilities.
