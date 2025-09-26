# VortexAI Lambda API Documentation

This document describes all the AWS Lambda functions available in the VortexAI system.

## System Overview

The VortexAI Lambda system provides serverless functions for:
- Data processing and parsing
- Database maintenance and recycling
- Milvus vector database operations
- S3 data pipeline management
- User feedback processing
- Archive management
- Case vault operations
- Multi-location data forwarding

## Base Configuration

All Lambda functions are deployed on AWS Lambda and can be triggered by:
- S3 events
- SQS messages
- Direct invocation
- Step Functions
- CloudWatch Events

---

## Data Processing Lambdas

### S3 Upload Parser

#### lambda_handler_s3uploadparser
- **Function**: `lambda_handler_s3uploadparser`
- **Description**: Parse S3 metadata and process uploaded files
- **Module Name**: lambda_handler_s3uploadparser
- **Trigger**: S3 events
- **Purpose**: Process uploaded files and extract metadata for further processing
- **Input**: S3 event with bucket, key, and metadata
- **Output**: Parsed metadata and processing status

### Deep Search Processing

#### lambda_handler_ivstestcmd
- **Function**: `lambda_handler_ivstestcmd`
- **Description**: Test command handler for IVS deep search functionality
- **Module Name**: lambda_handler_ivstestcmd
- **Trigger**: Direct invocation
- **Purpose**: Execute test commands for deep search operations
- **Input**: Test command parameters
- **Output**: Test execution results

---

## Database Management Lambdas

### Database Recycler

#### lambda_handler_ivsdatabaserecycler
- **Function**: `lambda_handler_ivsdatabaserecycler`
- **Description**: Recycle and clean up database records
- **Module Name**: lambda_handler_ivsdatabaserecycler
- **Trigger**: Scheduled events
- **Purpose**: Remove expired data and maintain database performance
- **Input**: Recycling parameters and time ranges
- **Output**: Cleanup statistics and status

### Camera Info Update

#### lambda_handler_ivsupdatecamerainfo
- **Function**: `lambda_handler_ivsupdatecamerainfo`
- **Description**: Update camera information in the database
- **Module Name**: lambda_handler_ivsupdatecamerainfo
- **Trigger**: S3 events or direct invocation
- **Purpose**: Sync camera metadata and configuration changes
- **Input**: Camera information updates
- **Output**: Update confirmation and status

### AI Control Setting Update

#### lambda_handler_ivsupdate_ai_control_setting
- **Function**: `lambda_handler_ivsupdate_ai_control_setting`
- **Description**: Update AI control settings for organizations
- **Module Name**: lambda_handler_ivsupdate_ai_control_setting
- **Trigger**: Direct invocation
- **Purpose**: Manage AI feature toggles and configurations
- **Input**: AI control setting parameters
- **Output**: Update status and configuration

---

## Milvus Vector Database Lambdas

### Milvus Data Processing

#### lambda_handler (Milvus)
- **Function**: `lambda_handler`
- **Description**: Process object data and insert into Milvus vector database
- **Module Name**: lambda_handler
- **Trigger**: S3 events or Kinesis
- **Purpose**: Index object data for similarity search and research
- **Input**: Object metadata and embeddings
- **Output**: Milvus insertion status
- **Features**:
  - Batch processing (10 objects per batch)
  - AI control setting application
  - CLIP image encoding support
  - Research data indexing

### Milvus Maintenance

#### lambda_handler (Milvus Maintenance)
- **Function**: `lambda_handler`
- **Description**: Maintain and update Milvus database
- **Module Name**: lambda_handler
- **Trigger**: Step Functions or SQS
- **Purpose**: Update license periods and maintain database health
- **Input**: Maintenance tasks and parameters
- **Output**: Maintenance status and progress
- **Features**:
  - Task queue processing
  - Cumulative count tracking
  - Automatic retry on failure

---

## S3 Data Pipeline Lambdas

### S3 to Kinesis Writer

#### lambda_handler (S3 to Kinesis)
- **Function**: `lambda_handler`
- **Description**: Forward S3 data to Kinesis for streaming processing
- **Module Name**: lambda_handler
- **Trigger**: S3 events
- **Purpose**: Stream data from S3 to Kinesis for real-time processing
- **Input**: S3 event with object metadata
- **Output**: Kinesis write status
- **Features**:
  - MAC address extraction
  - License validation
  - Expired report filtering
  - Amplification factor support

### Multi-Location Handler

#### lambda_handler (Multi-Location)
- **Function**: `lambda_handler`
- **Description**: Forward S3 metadata between different AWS regions
- **Module Name**: lambda_handler
- **Trigger**: S3 events
- **Purpose**: Handle multi-region data forwarding
- **Input**: S3 event with region information
- **Output**: Forwarding status and metadata
- **Features**:
  - Cross-region Lambda invocation
  - Metadata validation
  - Error handling and retry

---

## Archive Management Lambdas

### User Feedback Helper

#### lambda_handler_ivsuserfeedbackhelper
- **Function**: `lambda_handler_ivsuserfeedbackhelper`
- **Description**: Comprehensive user feedback and archive management
- **Module Name**: lambda_handler_ivsuserfeedbackhelper
- **Trigger**: Direct invocation or IoT events
- **Purpose**: Manage user feedback, archive operations, and case vault maintenance
- **Input**: Command-based requests with various parameters
- **Output**: Operation status and results
- **Commands**:
  - `s3_copy`: Copy S3 objects between buckets
  - `copy_archive`: Recursively copy archive data
  - `archive_report`: Process archive job reports
  - `sync_archive_status`: Synchronize archive status across systems
  - `collect_case_objects_from_cloudbackup`: Collect case objects from cloud backup
  - `archive_from_cloudbackup`: Archive data from cloud backup
  - `remove_unused_archived_case_objects`: Clean up unused archived objects
  - `rearchive_case_object`: Re-archive specific case objects

---

## Case Vault Management Lambdas

### Case Vault Operations

#### lambda_handler (Case Vault)
- **Function**: `lambda_handler`
- **Description**: Handle case vault operations and queries
- **Module Name**: lambda_handler
- **Trigger**: Direct invocation
- **Purpose**: Manage case vault data and provide case information
- **Input**: Case vault commands and parameters
- **Output**: Case vault data and status
- **Commands**:
  - `check_remain_case`: Check remaining case count for organization

---

## Infrastructure Management Lambdas

### OpenSearch Alias Creation

#### lambda_handler (OpenSearch Alias)
- **Function**: `lambda_handler`
- **Description**: Create OpenSearch aliases for time-based indexing
- **Module Name**: lambda_handler
- **Trigger**: Scheduled events (monthly)
- **Purpose**: Pre-create OpenSearch aliases for upcoming time periods
- **Input**: Time parameters for alias creation
- **Output**: Alias creation status

### Model Management

#### lambda_handler_ivsmodelupload
- **Function**: `lambda_handler_ivsmodelupload`
- **Description**: Upload and manage AI models
- **Module Name**: lambda_handler_ivsmodelupload
- **Trigger**: Direct invocation
- **Purpose**: Handle AI model uploads and updates
- **Input**: Model data and metadata
- **Output**: Upload status and model information

#### lambda_handler_ivscleanmodelhistory
- **Function**: `lambda_handler_ivscleanmodelhistory`
- **Description**: Clean up old model history and versions
- **Module Name**: lambda_handler_ivscleanmodelhistory
- **Trigger**: Scheduled events
- **Purpose**: Remove outdated model versions and history
- **Input**: Cleanup parameters and time ranges
- **Output**: Cleanup status and statistics

### Partition Management

#### lambda_handler_ivscreatepartition
- **Function**: `lambda_handler_ivscreatepartition`
- **Description**: Create database partitions for data organization
- **Module Name**: lambda_handler_ivscreatepartition
- **Trigger**: Scheduled events or direct invocation
- **Purpose**: Create time-based or other partitions for database optimization
- **Input**: Partition parameters and time ranges
- **Output**: Partition creation status

### Cache Management

#### lambda_handler_ivscleanopensearchcache
- **Function**: `lambda_handler_ivscleanopensearchcache`
- **Description**: Clean up OpenSearch cache and temporary data
- **Module Name**: lambda_handler_ivscleanopensearchcache
- **Trigger**: Scheduled events
- **Purpose**: Maintain OpenSearch performance by cleaning cache
- **Input**: Cache cleanup parameters
- **Output**: Cache cleanup status

---

## Error Handling Lambdas

### Error Message Handler

#### lambda_handler (Error Handler)
- **Function**: `lambda_handler`
- **Description**: Handle and process error messages from various sources
- **Module Name**: lambda_handler
- **Trigger**: SQS or direct invocation
- **Purpose**: Process error messages and notifications
- **Input**: Error messages and context
- **Output**: Error processing status

---

## Common Request/Response Patterns

### Lambda Event Structure
```json
{
  "Records": [
    {
      "s3": {
        "bucket": {"name": "bucket-name"},
        "object": {"key": "object-key", "size": 12345}
      },
      "awsRegion": "us-west-2"
    }
  ]
}
```

### Direct Invocation Structure
```json
{
  "cmd": "command_name",
  "parameters": {
    "key": "value"
  }
}
```

### Standard Response Format
```json
{
  "statusCode": 200,
  "body": "OK",
  "additional_data": {}
}
```

---

## Error Codes

- **200**: Success
- **400**: Bad Request - Invalid parameters or missing required fields
- **500**: Internal Server Error - Processing failure or system error

---

## Environment Variables

Common environment variables used across Lambda functions:

- `APP_ENV`: Environment (dev, preprod, prod)
- `REGION`: AWS region
- `LOCATION`: Geographic location identifier
- `PEER_REGION`: Target region for cross-region operations
- `SITE`: Site identifier
- `AMPLIFY_FACTOR`: Kinesis write amplification factor
- `DSLM_ORG`: DSLM organization configuration
- `WITH_LOCALSTACK`: LocalStack environment flag

---

## Monitoring and Logging

All Lambda functions include:
- Execution time logging
- Error tracking and reporting
- Performance metrics
- CloudWatch integration
- Elasticsearch logging (where configured)

---

## Notes

- All Lambda functions support graceful error handling
- Functions are designed for serverless execution with appropriate timeouts
- Batch processing is implemented where applicable for efficiency
- Cross-region operations are supported for multi-location deployments
- Archive operations include dry-run capabilities for testing
- Database operations include connection pooling and proper cleanup
