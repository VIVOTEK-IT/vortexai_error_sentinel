# Jira Issue Embedding Database Implementation Plan

## 📋 Project Overview

**Project Name**: Jira Issue Embedding Database  
**Purpose**: Implement a year-based vector database for Jira issues with embedding similarity search and error log correlation  
**Target**: Integrate with existing error log monitoring system for enhanced issue correlation  

## 🎯 Objectives

### Primary Goals
- **Vector Similarity Search**: Enable efficient similarity search between Jira issues and error logs using OpenAI embeddings
- **Occurrence Tracking**: Track which error logs reference specific Jira issues
- **Parent-Child Relationships**: Support hierarchical issue structures

### Success Criteria
- **Correlation Rate**: >60% of error logs successfully correlated to Jira issues
- **Performance**: <100ms response time for similarity searches
- **Scalability**: Handle 10,000+ Jira issues per year
- **Reliability**: 99.9% uptime with proper error handling

## 🏗️ Architecture Overview

### Core Components
1. **JiraIssueEmbeddingDB Class**: Main database interface
2. **OpenSearch Integration**: Vector storage and search
3. **RAG Engine Integration**: Embedding calculation
4. **Year-Based Indexing**: Data organization strategy
5. **Migration Utilities**: Data management tools

### Data Flow
```
Jira Issues → Embedding Calculation → OpenSearch Storage
Error Logs → Embedding Calculation → Similarity Search → Correlation
```

## 📅 Implementation Phases

### Phase 1: Foundation Setup (Week 1)
**Priority**: Critical | **Duration**: 3-4 days

#### 1.1 Configuration Framework
- [ ] **Add JiraEmbeddingConfig to config.py**
  ```python
  @dataclass
  class JiraEmbeddingConfig:
      index_name_template: str = "jira_issue_embedding_{year}"
      similarity_threshold: float = 0.85
      top_k: int = 10
      batch_size: int = 100
      retention_years: int = 3      
  ```

- [ ] **Update SystemConfig**
  - Add `jira_embedding: JiraEmbeddingConfig` field
  - Update `load_config()` method

- [ ] **Environment Variables**
  - `JIRA_EMBEDDING_INDEX_TEMPLATE=jira_issue_embedding_{year}`
  - `JIRA_EMBEDDING_SIMILARITY_THRESHOLD=0.85`
  - `JIRA_EMBEDDING_TOP_K=10`
  - `JIRA_EMBEDDING_BATCH_SIZE=100`
  - `JIRA_EMBEDDING_RETENTION_YEARS=3`
  - `JIRA_EMBEDDING_AUTO_CREATE_INDEX=true`

#### 1.2 OpenSearch Index Mapping
- [ ] **Create index mapping schema**
  - Dense vector field for embeddings (1536 dimensions)
  - Nested field for occurrence_list
  - Proper field types for all metadata
  - Optimized settings for kNN search

#### 1.3 Basic Class Structure
- [ ] **Create JiraIssueEmbeddingDB class skeleton**
  - Constructor with dependency injection
  - Year-based index name helpers
  - Basic method signatures

### Phase 2: Core Database Operations (Week 2)
**Priority**: Critical | **Duration**: 4-5 days

#### 2.1 Index Management
- [ ] **Implement index operations**
  - `create_index(year=None)`: Create year-specific index
  - `delete_index(year)`: Remove specific year index
  - `index_exists(year)`: Check index existence
  - `get_index_mapping()`: Retrieve index schema

#### 2.2 CRUD Operations
- [ ] **Basic CRUD methods**
  
  - `get_issue_by_key(jira_key, year=None)`
  - `update_issue(jira_key, update_data, year=None)`
  - `delete_issue(jira_key, year=None)`

#### 2.3 Vector Normalization
- [ ] **Unit vector methods**
  - `normalize_embedding(embedding_vector)`: L2 normalization to unit vector
  - `validate_unit_vector(embedding_vector, tolerance=1e-6)`: Validation
  - Integration with all embedding calculations
  - Error handling for normalization failures

**OpenSearch Limitation Note:**
- OpenSearch only supports inner product similarity, not cosine similarity
- Unit vectors enable cosine similarity via inner product: `cos(θ) = a·b` for unit vectors
- This provides optimal performance while maintaining mathematical correctness

#### 2.4 Occurrence Management
- [ ] **Occurrence tracking methods**
  - `add_occurrence(jira_key, doc_id, timestamp, year=None)`
  - `get_occurrences(jira_key, year=None)`
  - `remove_occurrence(jira_key, doc_id, year=None)`
  - `cleanup_old_occurrences(days_threshold=90)`

#### 2.5 Search Operations
- [ ] **Similarity search methods**
  - `find_similar_jira_issue(error_log_embedding, site, similarity_threshold=0.85)`
  - `search_similar_issues(query_text, top_k=10, similarity_threshold=0.85)`
  - `search_by_site(site, year=None)`
  - `search_by_status(status, year=None)`

### Phase 3: Data Processing Workflows (Week 3)
**Priority**: High | **Duration**: 4-5 days

#### 3.1 Initialization Workflow
- [ ] **Implement `initialize_database(jira_issues, error_logs_by_site)`**
  - Process Jira issues from API
  - Calculate embeddings using `rag_engine._calculate_error_log_embeddings`
    - Refactor RAGEngine and EmbeddingService:
      - Convert all vectors to unit vector before use or storage
  - Process error logs (past 6 months)
  - Find similar issues and add occurrences
  - Handle batch processing for large datasets

#### 3.2 Daily Update Workflow
- [ ] **Implement `daily_update(error_logs_by_site)`**
  - Process new error logs (past 24 hours)
  - Find similar parent Jira issues
  - Add occurrences or create new issues
  - Handle child issue creation
  - Update statistics and metrics

#### 3.3 Issue Creation
- [ ] **Issue creation methods**
  - `_create_jira_issue_from_error_log(error_log, site)`
  - `create_child_issue(parent_key, error_log, site)`
  - `generate_issue_key(site, error_log)`
  - `validate_issue_data(issue_data)`

### Phase 4: Advanced Features (Week 4)
**Priority**: Medium | **Duration**: 3-4 days

### Phase 5: Integration & Testing (Week 5)
**Priority**: High | **Duration**: 3-4 days

#### 5.1 Weekly Report Integration
- [ ] **Update WeeklyReportGenerator**
  - Initialize `JiraIssueEmbeddingDB` instance
  - Use `find_similar_jira_issue()` for correlation
  - Update `_find_correlated_jira_issues_embedding()` method
  - Add embedding-based correlation strategy

#### 5.2 Comprehensive Testing
- [ ] **Unit Tests**
  - Test all CRUD operations
  - Test search and similarity functions
  - Test year management operations
  - Test error handling and edge cases

- [ ] **Integration Tests**
  - Test with real OpenSearch instance
  - Test with RAG engine integration
  - Test with Jira API integration
  - Test performance with large datasets

- [ ] **End-to-End Tests**
  - Test complete initialization workflow
  - Test daily update workflow
  - Test cross-year operations
  - Test migration scenarios

### Phase 6: Monitoring & Documentation (Week 6)
**Priority**: Medium | **Duration**: 2-3 days

#### 6.1 Monitoring & Health Checks
- [ ] **Health monitoring**
  - `get_embedding_stats(year=None)`: Database statistics
  - `health_check()`: Overall system health
  - `performance_metrics()`: Performance monitoring
  - `error_rate_monitoring()`: Error tracking

#### 6.2 Documentation
- [ ] **Update project documentation**
  - `IMPLEMENTATION.md`: Add JiraIssueEmbeddingDB details
  - `README.md`: Update with new features
  - API documentation for new methods
  - Usage examples and best practices

#### 6.3 Deployment Preparation
- [ ] **Deployment readiness**
  - Docker configuration updates
  - Environment variable documentation
  - Migration scripts for existing data
  - Rollback procedures

## 🔧 Technical Specifications

### OpenSearch Index Schema (current)
```json
{
  "mappings": {
    "properties": {
      "key": { "type": "keyword" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "innerproduct",
          "engine": "faiss",
          "parameters": { "ef_construction": 128, "m": 32 }
        }
      },
      "occurrence_list": {
        "type": "nested",
        "properties": {
          "doc_id": { "type": "keyword" },
          "timestamp": { "type": "date", "format": "strict_date_optional_time||epoch_millis" }
        }
      },
      "summary": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "description": { "type": "text" },
      "status": { "type": "keyword" },
      "error_message": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "error_type": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "traceback": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "site": { "type": "keyword" },
      "request_id": { "type": "keyword" },
      "created": { "type": "date" },
      "updated": { "type": "date" },
      "parent_issue_key": { "type": "keyword" },
      "is_parent": { "type": "boolean" },
      "created_at": { "type": "date" },
      "updated_at": { "type": "date" }
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

### Dependencies (updated)
- **OpenSearch**: kNN inner product with unit vectors (cosine-equivalent)
- **EmbeddingService**: Text embeddings (text-embedding-3-small) and normalization
- **Jira API**: Issue data retrieval
- **NumPy**: Vector validation/normalization

### Changes from earlier drafts
- Removed year-based index rotation and `dense_vector` + cosine mapping in favor of single index template with `knn_vector` and innerproduct.
- Replaced `RAGEngine` dependency with `EmbeddingService` for embedding generation.
- Occurrence maintenance is considered backlog; correlation currently updates error logs with a Jira reference.

### Performance Requirements
- **Response Time**: <100ms for similarity searches
- **Throughput**: Process 1,000+ error logs daily
- **Storage**: Handle 10,000+ Jira issues per year
- **Memory**: Efficient embedding storage and retrieval
- **Vector Normalization**: <1ms per vector normalization operation

## 📊 Success Metrics

### Functional Metrics
- **Correlation Rate**: >60% of error logs correlated to Jira issues
- **Search Accuracy**: >90% relevant results in top 5 matches
- **Data Integrity**: 100% data consistency across operations
- **Error Handling**: <1% error rate in production

### Performance Metrics
- **Search Latency**: <100ms average response time
- **Index Size**: <1GB per year of data
- **Memory Usage**: <500MB for single-year operations
- **Throughput**: Process 1,000+ operations per minute

### Operational Metrics
- **Uptime**: 99.9% availability
- **Migration Success**: 100% successful data migrations
- **Monitoring**: Real-time health checks and alerts
- **Documentation**: Complete API and usage documentation

## 🚨 Risk Mitigation

### Technical Risks
- **OpenSearch Performance**: Implement proper indexing and query optimization
- **Memory Usage**: Use year-based indexing to limit memory footprint
- **Data Consistency**: Implement proper transaction handling and validation
- **Migration Failures**: Create comprehensive rollback procedures

### Operational Risks
- **Data Loss**: Implement backup and recovery procedures
- **Performance Degradation**: Monitor and optimize queries regularly
- **Integration Issues**: Thorough testing with existing systems
- **Scalability**: Design for horizontal scaling

## 📋 Deliverables (aligned)

### Code Deliverables
- `src/error_log_monitor/jira_issue_embedding_db.py`
- Updated `src/error_log_monitor/config.py` (index template, embedding config)
- Updated report flows to rely on EmbeddingService-based similarity
- Tests under `test/` for kNN search, normalization, and add/find flows

### Documentation Deliverables
- Updated `IMPLEMENTATION.md`
- Updated `README.md`
- API usage and examples consistent with current code

### Configuration Deliverables
- [ ] Updated `docker-compose.yml`
- [ ] Updated `.env` template
- [ ] Environment variable documentation
- [ ] OpenSearch index templates

## 🎯 Next Steps

1. **Start Phase 1**: Begin with configuration setup and basic class structure
2. **Set up development environment**: Ensure OpenSearch and dependencies are ready
3. **Create initial tests**: Set up test framework for TDD approach
4. **Implement core functionality**: Focus on CRUD operations first
5. **Iterate and test**: Regular testing and validation throughout development

## 📞 Support & Resources

### Development Resources
- **OpenSearch Documentation**: Vector search and kNN queries
- **RAG Engine**: Existing embedding calculation methods
- **Jira API**: Issue data retrieval and management
- **OpenAI API**: Text embedding generation

### Team Responsibilities
- **Backend Development**: Core database implementation
- **Integration**: Weekly report and existing system integration
- **Testing**: Comprehensive test suite development
- **DevOps**: Deployment and monitoring setup

This implementation plan provides a structured approach to building the Jira Issue Embedding Database with clear phases, deliverables, and success metrics. The year-based indexing strategy will provide excellent performance and scalability for the error log monitoring system.
