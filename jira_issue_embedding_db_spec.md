# Jira Issue Embedding Database Specification

## Overview

This specification defines a Jira Issue Embedding Database module that integrates with OpenSearch to store Jira issues with their embeddings and occurrence tracking. The database enables efficient similarity search and correlation between Jira issues and error logs.

## Database Schema

### OpenSearch Index: `jira_issue_embedding_{year}`

#### Document Structure

```json
{
  "key": "string",                    // Jira issue key (e.g., "VEL-123")
  "embedding": [float],               // Vector embedding array (1536 dimensions for text-embedding-3-small). 
                                      // The embedding is calculated by rag_engine._calculate_error_log_embeddings
                                      // IMPORTANT: All embedding vectors MUST be unit vectors (normalized to length 1)
  "occurrence_list": [                // List of all occurrences where this issue was referenced
    {
      "doc_id": "string",             // Document ID from error_log OpenSearch database
      "timestamp": "string"           // ISO 8601 timestamp of the occurrence
    }
  ],
  "summary": "string",                // Jira issue summary
  "description": "string",            // Jira issue description
  "status": "string",                 // Current status
  "error_message": "string",          // Error message
  "error_type": "string",             // Error type
  "traceback": "string",              // Traceback
  "site": "string",                   // Site name
  "request_id": "string",             // Request ID
  "created": "string",                // Creation timestamp
  "updated": "string",                // Last update timestamp
  "parent_issue_key": "string",       // Parent issue key (if any)
  "is_parent": "bool",                // True if parent_issue_key is empty, false otherwise
  "not_commit_to_jira": "bool",       // True if this issue is not yet committed to Jira, false otherwise
  "created_at": "string",             // When this embedding record was created
  "updated_at": "string"             // When this embedding record was last updated
}
```

### Index Mapping

```json
{
  "mappings": {
    "properties": {
      "key": {
        "type": "keyword"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "innerproduct",  // OpenSearch only supports inner product, unit vectors required for cosine similarity
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
          "doc_id": {
            "type": "keyword"
          },
          "timestamp": {
            "type": "date",
            "format": "strict_date_optional_time||epoch_millis"
          }
        }
      },
      "summary": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "description": {
        "type": "text"
      },
      "status": {
        "type": "keyword"
      },
      "error_message": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "error_type": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "traceback": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "site": {
        "type": "keyword"
      },
      "request_id": {
        "type": "keyword"
      },
      "created": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "updated": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "parent_issue_key": {
        "type": "keyword"
      },
      "is_parent": {
        "type": "boolean"
      },
      "not_commit_to_jira": {
        "type": "boolean"
      },
      "created_at": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "updated_at": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      }
    }
  }
}
```
## Operational Workflows

### Initialization (`scripts/init_jira_db.py`)
1. Fetch issues from Jira Cloud (`get_all_jira_issues`).
2. Compute embeddings and upsert documents into the `jira_issue_embedding` index.
3. Backfill occurrences from historical error logs.

### Daily Sync
1. Ingest new error logs, compute embeddings, and correlate with existing Jira issues.
2. Update occurrence lists accordingly.

### Log Group Synchronization (`scripts/sync_jira_log_groups.py`)
1. Fetch current Jira issues.
2. Iterate all embedding documents and align `log_group` values from Jira.

## Module Components

### 1. JiraIssueEmbeddingDB Class

```python
class JiraIssueEmbeddingDB:
    """Manages Jira issue embeddings and occurrence tracking in OpenSearch."""
    
    def __init__(self, opensearch_client, embedding_service, rag_engine, index_name_template="jira_issue_embedding_{year}"):
        self.client = opensearch_client
        self.embedding_service = embedding_service
        self.rag_engine = rag_engine
        self.index_name_template = index_name_template
    
    def get_current_index_name(self):
        """Get the current year's index name."""
        from datetime import datetime
        current_year = datetime.now().year
        return self.index_name_template.format(year=current_year)
    
    def get_index_name_for_year(self, year):
        """Get index name for a specific year."""
        return self.index_name_template.format(year=year)
    
    def create_index(self, year=None):
        """Create the OpenSearch index with proper mapping for the specified year."""
        if year is None:
            index_name = self.get_current_index_name()
        else:
            index_name = self.get_index_name_for_year(year)
        
    def initialize_database(self, jira_issues, error_logs_by_site):
        """Initialize the database with all Jira issues and error logs."""
        
    def daily_update(self, error_logs_by_site):
        """Daily update process for new error logs."""
        
    def find_similar_jira_issue(self, error_log_embedding, site, similarity_threshold=0.85):
        """Find similar Jira issue for an error log (parent issues only)."""
        
    def create_child_issue(self, parent_key, error_log, site):
        """Create a child issue for an existing parent issue."""
        
    def add_occurrence(self, jira_key, doc_id, timestamp):
        """Add an occurrence reference to a Jira issue."""
        
    def search_similar_issues(self, query_text, top_k=10, similarity_threshold=0.85):
        """Search for similar Jira issues using embedding similarity."""
        
    def get_issue_by_key(self, jira_key):
        """Retrieve a specific Jira issue by its key."""
        
    def get_occurrences(self, jira_key):
        """Get all occurrences for a specific Jira issue."""
        
    def delete_issue(self, jira_key):
        """Delete a Jira issue and all its occurrences."""
        
    def get_embedding_stats(self, year=None):
        """Get statistics about the embedding database for a specific year."""
                
    def normalize_embedding(self, embedding_vector):
        """Normalize embedding vector to unit vector (length = 1)."""
        
    def validate_unit_vector(self, embedding_vector, tolerance=1e-6):
        """Validate that embedding vector is a unit vector."""
```

### 2. Core Methods Interface

#### `initialize_database(jira_issues, error_logs_by_site)`
Initialize the database with all Jira issues and error logs.

**Process:**
1. Calculate embeddings for all Jira issues using `rag_engine._calculate_error_log_embeddings`
2. Calculate embeddings for all error logs (past 6 months)
3. For each Jira issue, find similar error logs and add to occurrence_list

**Important Requirements:**
- All embedding vectors MUST be unit vectors (normalized to length 1)
- Use cosine similarity for vector comparisons
- Ensure consistent normalization across all embedding calculations

#### `daily_update(error_logs_by_site)`
Daily update process for new error logs.

**Process:**
1. Calculate embeddings for new error logs
2. Find similar Jira issues (parent issues only)
3. Add occurrences or create new issues

<!-- #### `add_jira_issue(jira_issue_data, error_log_data=None)`
Add a new Jira issue with embedding to the database. -->

**Returns:** Created document response

#### `find_similar_jira_issue(error_log_embedding, site, similarity_threshold=0.85)`
Find similar Jira issue for an error log using embedding similarity.

**Returns:** Similar Jira issue or None if not found

#### `create_child_issue(parent_key, error_log, site)`
Create a child issue for an existing parent issue.

**Returns:** Created child issue response

#### `_create_jira_issue_from_error_log(error_log, site)`
Create a new Jira issue from an error log.

**Returns:** Jira issue data structure

#### `normalize_embedding(embedding_vector)`
Normalize embedding vector to unit vector (length = 1).

**Process:**
1. Calculate L2 norm of the vector
2. Divide each component by the norm
3. Return normalized vector

**Returns:** Normalized unit vector

#### `validate_unit_vector(embedding_vector, tolerance=1e-6)`
Validate that embedding vector is a unit vector.

**Process:**
1. Calculate vector magnitude
2. Check if magnitude is within tolerance of 1.0
3. Return validation result

**Returns:** Boolean indicating if vector is unit vector

### 3. Integration with Existing System

#### Integration with Weekly Report

The `WeeklyReportGenerator` should integrate with `JiraIssueEmbeddingDB` to find correlated Jira issues using embedding similarity.

**Integration Points:**
- Initialize `JiraIssueEmbeddingDB` with OpenSearch client, embedding service, and RAG engine
- Use `find_similar_jira_issue()` method to correlate merged issues with Jira issues
- Calculate embeddings for merged issues using `rag_engine._calculate_error_log_embeddings`

### 4. Configuration

#### OpenSearch Settings

```json
{
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

#### Environment Variables

```bash
# Jira Issue Embedding Database
JIRA_EMBEDDING_INDEX_TEMPLATE=jira_issue_embedding_{year}
JIRA_EMBEDDING_SIMILARITY_THRESHOLD=0.85
JIRA_EMBEDDING_TOP_K=10
JIRA_EMBEDDING_BATCH_SIZE=100
```

### 5. Usage Examples

#### Initialize Database
1. Initialize components: OpenSearch client, embedding service, RAG engine
2. Create `JiraIssueEmbeddingDB` instance with year-based index template
3. Create index for current year or specific year
4. Initialize with Jira issues and error logs data

#### Daily Update
1. Get recent error logs by site (past 24 hours)
2. Call `daily_update()` method to process new error logs

### 7. Unit Vector Requirements

#### Vector Normalization
All embedding vectors stored in the database MUST be unit vectors (normalized to length 1). This requirement is critical for:

- **OpenSearch Limitation**: OpenSearch only supports inner product similarity, not cosine similarity
- **Cosine Similarity via Inner Product**: Unit vectors enable cosine similarity calculation using inner product (cos(θ) = a·b for unit vectors)
- **Consistent Similarity Calculations**: Unit vectors ensure cosine similarity calculations are accurate and consistent
- **Optimal Search Performance**: Normalized vectors provide better kNN search results
- **Mathematical Correctness**: Cosine similarity is most effective with unit vectors

#### Normalization Process
1. **Calculate Raw Embedding**: Use `rag_engine._calculate_error_log_embeddings()` to get raw embedding
2. **Normalize to Unit Vector**: Apply L2 normalization to ensure vector length = 1
3. **Validate Normalization**: Verify vector magnitude is approximately 1.0 before storage
4. **Consistent Application**: Apply normalization to all embeddings (Jira issues and error logs)

#### OpenSearch Technical Limitation
OpenSearch's kNN search only supports **inner product** similarity, not direct cosine similarity. However, we can achieve cosine similarity using unit vectors:

**Mathematical Relationship:**
- For unit vectors: `cos(θ) = a·b` (inner product equals cosine similarity)
- For non-unit vectors: `cos(θ) = (a·b) / (||a|| × ||b||)` (requires magnitude calculation)

**Why Unit Vectors are Required:**
- OpenSearch inner product: `a·b` (fast, optimized)
- Cosine similarity with unit vectors: `cos(θ) = a·b` (same as inner product)
- Cosine similarity without normalization: Requires magnitude calculation (slower, not supported)

#### Implementation Requirements
- **Pre-storage Normalization**: Normalize all embeddings before storing in OpenSearch
- **Search-time Normalization**: Ensure query embeddings are also normalized
- **Validation Checks**: Implement validation to reject non-unit vectors
- **Error Handling**: Handle normalization failures gracefully
- **OpenSearch Compatibility**: Use inner product with unit vectors for cosine similarity

### 8. Jira Custom Fields Requirements

#### Dynamic Custom Field Handling

Jira Cloud uses custom fields for all non-standard columns, and these fields must be dynamically queried and handled during data retrieval from Jira Cloud, not at the OpenSearch level.

#### Custom Field Query Process

**Field Discovery:**
1. Use `get_jira_field_index()` method to query available custom fields from Jira Cloud
2. Map custom field IDs to human-readable names
3. Dynamically construct field queries based on discovered custom fields

**Known VEL Custom Fields:**
- `error_message` - Error message content
- `error_type` - Type of error (e.g., Exception, Error, Warning)
- `request_id` - Unique request identifier
- `log_group` - Log group classification
- `site` - Site name or identifier
- `count` - Occurrence count
- `traceback` - Full stack traceback
- `parent_issue` - Parent issue reference

#### Implementation Requirements

**Jira Cloud Integration:**
- Query custom fields dynamically using `get_jira_field_index()`
- Map custom field IDs to field names during issue retrieval
- Handle missing or undefined custom fields gracefully
- Support field name changes without code modifications

**Data Processing:**
- Extract custom field values during Jira issue retrieval
- Store custom field data in OpenSearch document structure
- Maintain field mapping for consistent data access
- Handle custom field value variations and formats

**Field Mapping Strategy:**
```python
# Example custom field mapping
custom_field_mapping = {
    "error_message": "customfield_10001",
    "error_type": "customfield_10002", 
    "request_id": "customfield_10003",
    "log_group": "customfield_10004",
    "site": "customfield_10005",
    "count": "customfield_10006",
    "traceback": "customfield_10007",
    "parent_issue": "customfield_10008"
}
```

**Dynamic Field Handling:**
- Query field index on initialization
- Update field mapping when fields change
- Handle field additions/removals gracefully
- Maintain backward compatibility for field changes

#### OpenSearch Integration

**Document Structure:**
- Store custom field values as regular document fields
- Use consistent field naming (human-readable names)
- Maintain field type consistency in OpenSearch mapping
- Support dynamic field addition without index recreation

**Field Mapping in OpenSearch:**
```json
{
  "error_message": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
  "error_type": {"type": "keyword"},
  "request_id": {"type": "keyword"},
  "log_group": {"type": "keyword"},
  "site": {"type": "keyword"},
  "count": {"type": "integer"},
  "traceback": {"type": "text"},
  "parent_issue": {"type": "keyword"}
}
```

#### Error Handling

**Missing Fields:**
- Handle cases where custom fields are not defined
- Provide default values for missing fields
- Log warnings for undefined expected fields
- Continue processing with available fields

**Field Changes:**
- Detect field ID changes in Jira
- Update field mapping automatically
- Maintain data consistency across updates
- Handle field type changes gracefully

#### Configuration

**Environment Variables:**
```bash
# Jira Custom Fields
JIRA_CUSTOM_FIELDS_ENABLED=true
JIRA_CUSTOM_FIELDS_CACHE_TTL=3600
JIRA_CUSTOM_FIELDS_AUTO_UPDATE=true
JIRA_CUSTOM_FIELDS_FALLBACK_MODE=true
```

**Python Configuration:**
```python
class JiraCustomFieldConfig:
    enabled: bool = True
    cache_ttl: int = 3600  # seconds
    auto_update: bool = True
    fallback_mode: bool = True
    known_fields: List[str] = [
        "error_message", "error_type", "request_id", 
        "log_group", "site", "count", "traceback", "parent_issue"
    ]
```

### 9. Performance Considerations

- **Year-Based Indexing**: Smaller indices per year for better performance and easier management
- **Index Size**: Monitor index size as embeddings are large (1536 dimensions × 4 bytes = ~6KB per issue)
- **Search Performance**: Use appropriate `ef_search` and `ef_construction` parameters
- **Batch Operations**: Use bulk operations for large data imports
- **Memory Usage**: Year-based indices reduce memory usage for single-year operations
- **Site Filtering**: Efficient site-based filtering for better performance
- **Cross-Year Queries**: Use `search_across_years` for multi-year searches instead of single large index
- **Index Lifecycle**: Implement retention policies to archive/delete old year indices
- **Vector Normalization**: Unit vector requirement adds minimal computational overhead
- **OpenSearch Optimization**: Inner product with unit vectors provides optimal performance for cosine similarity
- **Custom Field Caching**: Cache custom field mappings to reduce Jira API calls
- **Dynamic Field Performance**: Optimize custom field query and mapping operations

### 10. Monitoring and Maintenance

#### Health Checks
- Get statistics about the embedding database
- Monitor total documents, index size, and average occurrence count
- Track performance metrics and health status

#### Configuration

**Python Configuration:**
- `index_name_template`: Template for year-based index names
- `similarity_threshold`: Minimum similarity score for matches
- `top_k`: Number of results to return
- `batch_size`: Batch size for bulk operations
- `retention_years`: Number of years to keep data

**Environment Variables:**
- `JIRA_EMBEDDING_INDEX_TEMPLATE`: Index name template
- `JIRA_EMBEDDING_RETENTION_YEARS`: Data retention period

This specification provides a comprehensive foundation for implementing a Jira issue embedding database that integrates seamlessly with the existing error log monitoring system and uses the RAG engine's embedding calculation method with year-based index organization.
