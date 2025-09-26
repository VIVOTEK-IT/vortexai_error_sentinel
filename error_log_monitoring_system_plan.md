# Error Log Monitoring System Design Plan
## System Overview
The monitoring system is designed to protect a service named Vortexai. The error log of Vortexai are stored in a Opensearch database. This system is going grab error logs from the Opensearch database, and analysis the downloaded issues. Additionally, the system provides comprehensive Jira issue management capabilities to track and analyze error-related tickets stored in OpenSearch.

## User Roles and Access Control
- **DevOps Engineers**: Full access to all features including error log analysis, Jira integration, system configuration, and real-time monitoring
- **Developers**: Read-only access to error logs, analysis results, and Jira issue details for debugging and issue resolution
- **Managers**: Access to reports and dashboards only, including weekly reports and high-level analytics

## Clarifications
### Session 2025-01-27
- Q: What should happen when critical components fail? → A: Retry with exponential backoff - automatically retry failed operations up to 3 times
- Q: Who are the primary users of this monitoring system? → A: Multiple roles - DevOps (full access), developers (read-only), managers (reports only)

### Overvew of Vortexai
Vortexai is an online service going to protect. It is composed by AWS Lambdas, ECS services, RDS database, Milvus database, Opensearch database. ECS services is built by flask.
#### Important APIs:
Below is the API you must be concern carefuly:
 - **UPDATE_CAMERAINFO**: It is a lambda function to update table CAMERA_INFO in the RDS. The function of this lambda is dertermined by the argument *functionName*. It has following functions
   - delete: delete a camera specified by thingName ( composed by "mac-timestamp" ) .
   - update: update a camera specified by thingName, mac, ororganizationID and deviceGroupID
   - create: create a camera specified by thingName, mac, ororganizationID and deviceGroupID
 - **PARSE_METADATA**: It is a lambda function to insert object's metadata into OBJECT_TRACE. This lambda first load and parse a series of objects from a file located at S3. The parsed data is finally instered into OBJECT_TRACE.
#### Database Schema
##### RDS
 - **Database**: This is a online AWS RDS postgres database. It MUST BE read only.
 - **Schema**: defined in db_schema/rds.sql
 - **Credential**: RDS_HOST, RDS_PASSWORD, RDS_PORT, RDS_DATABASE, RDS_USER are stored in evn. variable.
 - **Performance Requirement**: It MUST be very careful to access this database. Below performace rules MUST BE followed:
   - Always avoid full table read of OBJECT_TRACE. You can only get specified rows filtered by mac and partition_id.
   - System must support real-time processing with <1 second response time for critical errors

#### Error Log Database Schema
##### Opensearch
- **Database**: OpenSearch instance (43.207.106.51). It MUST BE read only.
- **Index Pattern**: error_log_{site}_{YYYY_M}
- **Schema**: Defined in db_schema/error_issue_schema.json
- **Fields**: message_id, timestamp, error_message, error_message_hash, traceback, traceback_hash, error_type, error_type_hash, site, service (composed from log_group.module_name), index_name, topic, count, request_id, category, log_group, module_name, version
- **Credentials**:OPENSEARCH_URL, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD are stored in env. variable.

#### Jira Issue Database Schema
##### Opensearch
- **Database**: Same OpenSearch instance (43.207.106.51). It MUST BE read only.
- **Index Pattern**: jira_issue{year} (e.g., jira_issue2025)
- **Schema**: Defined in db_schema/jira_issue.json
- **Fields**: issue_key, issue_time, error_message, error_type, traceback, traceback_hash, modified_error_message, modified_error_message_hash, modified_traceback, jira_summary, jira_url, full_log_url, site, log_group, request_id, related_issue_keys, related_traceback_hashes, count, jira_status, parent_issue_key, child_issue_keys, assignee, priority, issue_type, created, updated, description
- **Purpose**: Track and analyze Jira issues related to error logs for comprehensive issue management
- **Jira Cloud Integration**: Real-time access to Jira Cloud API for additional metadata including parent-child relationships, status, assignee, priority, and issue type


## Analysis flow
The aim of analysis is to determine the important error log that implies a damaged service or damaged data in the database. The analys flow is described as below:
 - **Merge Similar Issue**: 
  - Use RAG technique generate the context of each error log.  
  - Use OpenAI's text-embedding-3-small for cost efficiency (1536 dimensions)
  - Merge the similar issue into a reduce issue set. Further analysis is performed on this set. 
  - Store merged issues ant its context into a vector database.
 - **Jira Issue Correlation**:
  - Use embedding-based similarity matching to correlate error logs with Jira issues
  - Retrieve all Jira issues from the past 0.5 years (180 days) for comprehensive coverage
  - Calculate embeddings for each Jira issue using combined error_message, error_type, and traceback fields
  - Use cosine similarity with 0.85 threshold to find matching issues from the same site
  - Fetch additional details from Jira Cloud API including parent-child relationships, status, assignee, priority, and issue type
 - **Impact Analysis**: 
  - The analysis performed daily by manual
  - Use OpenAI to analysis each issue to see the impact of the error
   - Provide APIs for OpenAI to access Vortexai's database to check if related data is damaged or not.
   - OpenAI can execute tool calls to query RDS database in real-time during analysis
   - OPENAI_API_KEY is stoed in the env. variable.
  - The system can use gpt-5-mini or gpt-5-nano. Use gpt-5-nano for simple analysis to reduce the cost
  - Round-trip conversation allows OpenAI to make multiple tool calls for comprehensive analysis
  - The output of each issue should cover below points
   - Human remedial action needed?: if yes, provide guide line for further human tasks.
   - Damaged module/API: list the impact module/API. Point out the possible root cause
 - **Analysis Report**:
  - Provide a single static web page to list all the analysed issues
   - Provide the log pattern of each issue
   - Provided the occurance count of each issue

## Jira Issue Management
The system includes comprehensive Jira issue management capabilities for tracking and analyzing error-related tickets:

### Jira Issue Retrieval
- **Recent Issues**: Retrieve Jira issues from recent days with configurable time ranges
- **Date Range Queries**: Search for issues within specific date ranges
- **Error Message Search**: Find issues by error message patterns using text matching
- **Specific Issue Lookup**: Retrieve individual issues by Jira issue key
- **Site Filtering**: Filter issues by environment (dev, stage, prod)

### Jira Issue Analysis
- **Statistics Dashboard**: Generate comprehensive statistics about issue distribution
- **Site-based Analytics**: Analyze issues by environment and service
- **Error Type Analysis**: Track and categorize issues by error types
- **Trend Analysis**: Monitor issue patterns over time

### Integration Benefits
- **Comprehensive Tracking**: Link error logs with their corresponding Jira tickets
- **Issue Correlation**: Identify related issues through traceback and error message analysis
- **Historical Context**: Access complete issue history and resolution status
- **Workflow Integration**: Support for existing Jira-based issue management workflows

## Weekly Report
Generate comprehensive weekly reports with site-separated analysis and combined overview.
### Process
 1. First, grab error logs from Opensearch for stage and prod sites separately.
 2. For each site, merge error logs using "merge_similar_issues" in RAG engine. The merged issues are the base for further steps.
 3. For each site, check each issue if it lies in Jira database
  - If multiple issues matched, pick the latest one as the primary issue, and put others as child issues.
 4. For each site, create a LLM conversation model like the one used in error log monitoring system, conclude a short description of root cause of this issue.
 5. Generate site-specific tables of the issues with following columns:
  - Key: the url of primary jira issue( File the parent_issue if it is not empty )
  - Site: site
  - Count: occurance with-in the week
  - Summary: the summary of primary jira issue
  - Error_Message: error message[0:100]
  - Status: the status of primary jira issue (from Jira Cloud API)
  - Log Group: log_group
  - Latest Update: the timestamp of the latest error
  - Note: the short description of the root cause
  - Child Issues: the jira key of the child issues (from Jira Cloud API)
  
 6. Export above tables into:
  - Site-specific Excel files: `weekly_report_{site}_{date}.xlsx`
  - Site-specific HTML pages: `weekly_report_{site}_{date}.html`
  - Combined Excel file: `weekly_report_combined_{date}.xlsx`
  - Combined HTML page: `weekly_report_combined_{date}.html`

