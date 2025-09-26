#!/usr/bin/env python3
"""
Acceptance Test for RAG Engine
Tests similarity merging functionality based on test cases defined in acceptance_test_for_RAG.md
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from error_log_monitor.config import SystemConfig, RAGConfig, VectorDBConfig
from error_log_monitor.opensearch_client import ErrorLog
from error_log_monitor.vector_db_client import VectorDBClient
from error_log_monitor.rag_engine import RAGEngine, MergedIssue
from error_log_monitor.config import ModelConfig, OpenSearchConfig, RDSConfig, ModelType, load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_error_logs() -> List[ErrorLog]:
    """Create test error logs based on the acceptance test cases."""

    # Test Case 1: API_FACE_PROFILE_SEARCH errors (should be merged)
    error_log_1 = ErrorLog(
        message_id="449416772_2025-09-03T14:54:00",
        timestamp=datetime(2025, 9, 3, 14, 54, 0, tzinfo=timezone.utc),
        error_message="Exception in function API_FACE_PROFILE_SEARCH (1728, ['0002D1A18076', '0002D1A6ABCD', '0002D1A41FBB', '0002D199DAFC', '0002D19B5F96', '0002D1987F1D', '0002D1AF44CD', '0002D1A18090', '0002D1AA819A', '0002D1AA819B', '0002D1AA814B', '0002D1AA810E', '0002D1AA8152', '0002D1AA143F', '0002D1A8650C', '0002D1AA815B', '0002D1A3D60A', '0002D1A86568', '0002D1A77AB0', '0002D1A714C3', '0002D1A86566', '0002D1AF44D9', '0002D1B5C977', '0002D1A714AC', '0002D1A86564', '0002D1C1CF9C', '0002D1BBE553', '0002D1BCF1BE', '0002D1A3D60F', '0002D1A714D2', '0002D1BEA0FF', '0002D1BBDE2A', '0002D1BBD908', '0002D1BFC23A', '0002D1A64E5B', '0002D1BB11FB', '0002D1B28F38', '0002D1B28F37', '0002D1BDBB87', '0002D1BEA190', '0002D1C1251E', '0002D1A3C2F8', '0002D1AA144B', '0002D1BBDDC5', '0002D1C1254D', '0002D1A2E3AC', '0002D1A44C28', '0002D1A8651E', '0002D1AAEA78', '0002D1A6DE4C', '0002D1A86520', '0002D1A41FB4', '0002D1A86517', '0002D1B0D40E', '0002D1B353D9', '0002D1B99F3E', '0002D1B68B75', '0002D1B4E6B5'], 2025-08-04 14:54:52, 2025-09-03 14:54:52)",
        error_message_hash=758585383,
        traceback="Traceback (most recent call last):;  File \"/app/vortexai/ivs_deepsearch/upload_parser/api_face_profile_search.py\", line 111, in lambda_handler;    result_list = execute_knn_query_plan(;                  ^^^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/profile_helper.py\", line 511, in execute_knn_query_plan;    result = elk_helper.query_knn(;             ^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 498, in query_knn;    result_array = self.es.search(;                   ^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/utils.py\", line 177, in _wrapped;    return func(*args, params=params, headers=headers, **kwargs);           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/__init__.py\", line 1544, in search;    return self.transport.perform_request(;           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 407, in perform_request;    raise e;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 368, in perform_request;    status, headers_response, data = connection.perform_request(;                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/http_urllib3.py\", line 275, in perform_request;    self._raise_error(;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/base.py\", line 300, in _raise_error;    raise HTTP_EXCEPTIONS.get(status_code, TransportError)(",
        traceback_hash=449416772,
        error_type="opensearchpy.exceptions.RequestError: RequestError(400, 'too_long_http_line_exception', 'An HTTP line is larger than 10240 bytes.')",
        error_type_hash=1663138015,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=15,
        request_id="82ae64d8-8838-4e51-aa95-4a338127290b",
        category="error",
        log_group="/ecs/vortex-ai-vortexai",
        service="API_FACE_PROFILE_SEARCH",
    )

    error_log_2 = ErrorLog(
        message_id="449416772_2025-09-03T14:07:00",
        timestamp=datetime(2025, 9, 3, 14, 7, 0, tzinfo=timezone.utc),
        error_message="Exception in function API_FACE_PROFILE_SEARCH (2572, ['0002D1A18076', '0002D1A6ABCD', '0002D1A41FBB', '0002D199DAFC', '0002D19B5F96', '0002D1987F1D', '0002D1AF44CD', '0002D1A18090', '0002D1AA819A', '0002D1AA819B', '0002D1AA814B', '0002D1AA810E', '0002D1AA8152', '0002D1AA143F', '0002D1A8650C', '0002D1AA815B', '0002D1A3D60A', '0002D1A86568', '0002D1A77AB0', '0002D1A714C3', '0002D1A86566', '0002D1AF44D9', '0002D1B5C977', '0002D1A714AC', '0002D1A86564', '0002D1C1CF9C', '0002D1BBE553', '0002D1BCF1BE', '0002D1A3D60F', '0002D1A714D2', '0002D1BEA0FF', '0002D1BBDE2A', '0002D1BBD908', '0002D1BFC23A', '0002D1A64E5B', '0002D1BB11FB', '0002D1B28F38', '0002D1B28F37', '0002D1BDBB87', '0002D1BEA190', '0002D1C1251E', '0002D1A3C2F8', '0002D1AA144B', '0002D1BBDDC5', '0002D1C1254D', '0002D1A2E3AC', '0002D1A44C28', '0002D1A8651E', '0002D1AAEA78', '0002D1A6DE4C', '0002D1A86520', '0002D1A41FB4', '0002D1A86517', '0002D1B0D40E', '0002D1B353D9', '0002D1B99F3E', '0002D1B68B75', '0002D1B4E6B5'], 2025-08-04 14:07:47, 2025-09-03 14:07:47)",
        error_message_hash=2349449461,
        traceback="Traceback (most recent call last):;  File \"/app/vortexai/ivs_deepsearch/upload_parser/api_face_profile_search.py\", line 111, in lambda_handler;    result_list = execute_knn_query_plan(;                  ^^^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/profile_helper.py\", line 511, in execute_knn_query_plan;    result = elk_helper.query_knn(;             ^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 498, in query_knn;    result_array = self.es.search(;                   ^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/utils.py\", line 177, in _wrapped;    return func(*args, params=params, headers=headers, **kwargs);           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/__init__.py\", line 1544, in search;    return self.transport.perform_request(;           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 407, in perform_request;    raise e;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 368, in perform_request;    status, headers_response, data = connection.perform_request(;                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/http_urllib3.py\", line 275, in perform_request;    self._raise_error(;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/base.py\", line 300, in _raise_error;    raise HTTP_EXCEPTIONS.get(status_code, TransportError)(",
        traceback_hash=449416772,
        error_type="opensearchpy.exceptions.RequestError: RequestError(400, 'too_long_http_line_exception', 'An HTTP line is larger than 10240 bytes.')",
        error_type_hash=1663138015,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=3,
        request_id="416d0df8-1708-4887-93e1-3d447398ca5b",
        category="error",
        log_group="/ecs/vortex-ai-vortexai",
        service="API_FACE_PROFILE_SEARCH",
    )

    # Test Case 2: Skip zero vector errors (should be merged)
    error_log_3 = ErrorLog(
        message_id="3983379513_2025-09-02T17:00:06.847000+00:00",
        timestamp=datetime(2025, 9, 2, 17, 0, 6, 847000, tzinfo=timezone.utc),
        error_message="Skip zero vector for oid:1756830780468, mac:0002D1B25EDC_ch5, s3 bucket: banana-devices-push-storage-the-greate-one-vsaas-vortex-prod, s3_key: 0002D1B25EDC-1753386287274/ch6/object_thumbnail/2025/09/02/1756832160507_1756832161508_1756831762509_1756830780468_1756832167510.objmetadata",
        error_message_hash=1048633025,
        traceback="  File \"/var/runtime/bootstrap.py\", line 63, in <module>\n    main()\n;  File \"/var/runtime/bootstrap.py\", line 60, in main\n    awslambdaricmain.main([os.environ[\"LAMBDA_TASK_ROOT\"], os.environ[\"_HANDLER\"]])\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/__main__.py\", line 21, in main\n    bootstrap.run(app_root, handler, lambda_runtime_api_addr)\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 529, in run\n    handle_event_request(\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 177, in handle_event_request\n    response = request_handler(event, lambda_context)\n;  File \"/var/task/lambdas/utils.py\", line 43, in wrapper\n    result = func(*args, **kwargs)\n;  File \"/var/task/lambdas/lambda_s3_uploader.py\", line 7, in lambda_handler_s3uploadparser\n    return lambda_parse_s3_metadata.lambda_handler(event, context)\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/lambda_parse_s3_metadata.py\", line 211, in lambda_handler\n    result = parser.parse_s3report(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 582, in parse_s3report\n    thumbnail_dict['Thumbnail'], thumb_size = parse_thumbnail_item(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 946, in parse_thumbnail_item\n    r = elk_helper.insert_action(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 236, in insert_action\n    logger.error(f'Skip zero vector for oid:{oid}, mac:{mac}, s3 bucket: {s3_bucket}, s3_key: {s3_key}')\n;  File \"/var/task/vortexai/ivs_logger.py\", line 40, in error\n    _msg = prepare_error_message(msg, LogCategory.ERROR)\n",
        traceback_hash=3983379513,
        error_type="NoneType: None",
        error_type_hash=1393578982,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=2,
        request_id="e6aad46c-62d6-478d-a6b7-6ab34aea78a4",
        category="error",
        log_group="/aws/lambda/vortex-ai-S3UploadParser",
        service="S3UploadParser",
    )

    error_log_4 = ErrorLog(
        message_id="3983379513_2025-09-02T16:50:07.067000+00:00",
        timestamp=datetime(2025, 9, 2, 16, 50, 7, 67000, tzinfo=timezone.utc),
        error_message="Skip zero vector for oid:1756830780467, mac:0002D1B25EDC_ch6, s3 bucket: banana-devices-push-storage-the-greate-one-vsaas-vortex-prod, s3_key: 0002D1B25EDC-1753386287274/ch6/object_thumbnail/2025/09/02/1756831365494_1756831523495_1756830780468_1756831496493.objmetadata",
        error_message_hash=945301040,
        traceback="  File \"/var/runtime/bootstrap.py\", line 63, in <module>\n    main()\n;  File \"/var/runtime/bootstrap.py\", line 60, in main\n    awslambdaricmain.main([os.environ[\"LAMBDA_TASK_ROOT\"], os.environ[\"_HANDLER\"]])\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/__main__.py\", line 21, in main\n    bootstrap.run(app_root, handler, lambda_runtime_api_addr)\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 529, in run\n    handle_event_request(\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 177, in handle_event_request\n    response = request_handler(event, lambda_context)\n;  File \"/var/task/lambdas/utils.py\", line 43, in wrapper\n    result = func(*args, **kwargs)\n;  File \"/var/task/lambdas/lambda_s3_uploader.py\", line 7, in lambda_handler_s3uploadparser\n    return lambda_parse_s3_metadata.lambda_handler(event, context)\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/lambda_parse_s3_metadata.py\", line 211, in lambda_handler\n    result = parser.parse_s3report(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 582, in parse_s3report\n    thumbnail_dict['Thumbnail'], thumb_size = parse_thumbnail_item(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 946, in parse_thumbnail_item\n    r = elk_helper.insert_action(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 236, in insert_action\n    logger.error(f'Skip zero vector for oid:{oid}, mac:{mac}, s3 bucket: {s3_bucket}, s3_key: {s3_key}')\n;  File \"/var/task/vortexai/ivs_logger.py\", line 40, in error\n    _msg = prepare_error_message(msg, LogCategory.ERROR)\n",
        traceback_hash=3983379513,
        error_type="NoneType: None",
        error_type_hash=1393578982,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=2,
        request_id="d9711e48-a6b3-4797-88a2-3d036bdd068f",
        category="error",
        log_group="/aws/lambda/vortex-ai-S3UploadParser",
        service="S3UploadParser",
    )

    # Test Case 3: Different error (should NOT be merged with others)
    error_log_5 = ErrorLog(
        message_id="different_error_123",
        timestamp=datetime(2025, 9, 2, 15, 0, 0, tzinfo=timezone.utc),
        error_message="Database connection timeout after 30 seconds",
        error_message_hash=123456789,
        traceback="Traceback (most recent call last):;  File \"/app/database.py\", line 45, in connect;    connection = psycopg.connect(;  File \"/usr/local/lib/python3.11/site-packages/psycopg/connection.py\", line 123, in connect;    raise TimeoutError(\"Connection timeout\")",
        traceback_hash=987654321,
        error_type="TimeoutError: Connection timeout",
        error_type_hash=555666777,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=1,
        request_id="different-request-id",
        category="error",
        log_group="/aws/lambda/vortex-ai-S3UploadParser",
        service="S3UploadParser",
    )
    # Test Case 3: Different error (should NOT be merged with others)
    error_log_6 = ErrorLog(
        message_id="different_error_123",
        timestamp=datetime(2025, 9, 2, 15, 0, 0, tzinfo=timezone.utc),
        error_message="Database connection timeout after 19 seconds",
        error_message_hash=123456789,
        traceback="Traceback (most recent call last):;  File \"/app/database.py\", line 45, in connect;    connection = psycopg.connect(;  File \"/usr/local/lib/python3.11/site-packages/psycopg/connection.py\", line 123, in connect;    raise TimeoutError(\"Connection timeout\")",
        traceback_hash=987654321,
        error_type="TimeoutError: Connection timeout",
        error_type_hash=555666777,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=1,
        request_id="different-request-id",
        category="error",
        log_group="/ecs/vortex-ai-vortexai",
        service="API_FACE_PROFILE_SEARCH",
    )

    return [error_log_1, error_log_2, error_log_3, error_log_4, error_log_5, error_log_6]


def create_test_config() -> SystemConfig:
    """Create test configuration."""

    _config = load_config()

    _config.vector_db = (
        VectorDBConfig(
            persist_directory="./test_chroma_db",
            collection_name="test_error_log_vectors",
            distance_metric="cosine",
            embedding_model="text-embedding-3-small",
        ),
    )
    return _config


def test_similarity_merging_1():
    config = create_test_config()
    print("Created test configuration")

    # Create vector database client
    vector_db_client = VectorDBClient(config.vector_db)
    print("Created vector database client")

    # Create RAG engine
    rag_engine = RAGEngine(config, vector_db_client)
    print("Created RAG engine")

    # Create test error logs
    error_logs = create_test_error_logs()
    print(f"Created {len(error_logs)} test error logs")
    logger.info(f"Created {len(error_logs)} test error logs")
    test_error_logs = [error_logs[0], error_logs[1], error_logs[2], error_logs[5]]
    # Test merging
    merged_issues = rag_engine.merge_similar_issues(test_error_logs)
    assert len(merged_issues) == 3
    # make sure there are 3 different error_message in the merged_issue
    error_msg_set = set()
    for issue in merged_issues:
        error_msg_set.add(issue.representative_log.error_message)
    assert len(error_msg_set) == 3

    # Output detailed information about merged issues
    print(f"\n📊 Test 1 - Merged Issues Details:")
    print(f"Total input logs: {len(test_error_logs)}")
    print(f"Total merged issues: {len(merged_issues)}")
    print("-" * 60)

    found_error_log_1 = False
    found_error_log_2 = False

    # Debug: Print what we're looking for
    print(f"\n🔍 DEBUG - Looking for:")
    print(f"  error_logs[0][:55]: {error_logs[0].error_message[:55]}")
    print(f"  error_logs[1][:55]: {error_logs[1].error_message[:55]}")

    for i, issue in enumerate(merged_issues, 1):
        print(f"\n🔍 Issue {i}:")
        print(f"  ID: {issue.issue_id}")
        print(f"  Representative Error: {issue.representative_log.error_message[:100]}...")
        print(f"  Error Type: {issue.representative_log.error_type}")
        print(f"  Service: {issue.representative_log.service}")
        print(f"  Occurrence Count: {issue.occurrence_count}")
        print(f"  Similar Logs Count: {len(issue.similar_logs)}")
        print(f"  Affected Services: {issue.affected_services}")
        print(f"  Time Span: {issue.time_span}")
        print(f"  Context Length: {len(issue.context)} characters")

        # Check representative log
        if error_logs[0].error_message[:55] in issue.representative_log.error_message:
            found_error_log_1 = True
            print(f"      ✅ Found error_logs[0] in representative log")
        if error_logs[1].error_message[:55] in issue.representative_log.error_message:
            found_error_log_2 = True
            print(f"      ✅ Found error_logs[1] in representative log")

        # Show similar logs details
        if issue.similar_logs:
            print(f"  Similar Logs:")
            for j, similar_log in enumerate(issue.similar_logs, 1):
                print(f"    {j}. {similar_log.error_message[:55]}... (Service: {similar_log.service})")
                if error_logs[0].error_message[:55] in similar_log.error_message:
                    found_error_log_1 = True
                    print(f"      ✅ Found error_logs[0] in similar log {j}")
                if error_logs[1].error_message[:55] in similar_log.error_message:
                    found_error_log_2 = True
                    print(f"      ✅ Found error_logs[1] in similar log {j}")

    print(f"\n🔍 DEBUG - Results:")
    print(f"  found_error_log_1: {found_error_log_1}")
    print(f"  found_error_log_2: {found_error_log_2}")

    assert found_error_log_1
    assert found_error_log_2


def test_similarity_merging_2():
    config = create_test_config()
    print("Created test configuration")

    # Create vector database client
    vector_db_client = VectorDBClient(config.vector_db)
    print("Created vector database client")

    # Create RAG engine
    rag_engine = RAGEngine(config, vector_db_client)
    print("Created RAG engine")

    # Create test error logs
    error_logs = create_test_error_logs()
    print(f"Created {len(error_logs)} test error logs")
    logger.info(f"Created {len(error_logs)} test error logs")
    test_error_logs = [error_logs[0], error_logs[2], error_logs[3], error_logs[4]]
    # Test merging
    merged_issues = rag_engine.merge_similar_issues(test_error_logs)
    print(f"\n🔍 DEBUG - Test 2 - Expected 3 merged issues, got {len(merged_issues)}")
    if len(merged_issues) != 3:
        for row in merged_issues:
            print(f"Merged issue: {row.representative_log.error_message[:60]}")
    # The RAG engine correctly identifies 4 distinct error patterns
    assert len(merged_issues) == 3
    # make sure there are 4 different error_message in the merged_issue
    error_msg_set = set()
    for issue in merged_issues:
        error_msg_set.add(issue.representative_log.error_message)
    print(f"\n🔍 DEBUG - Test 2 - Unique error messages: {len(error_msg_set)}")
    for i, msg in enumerate(error_msg_set, 1):
        print(f"  {i}. {msg[:100]}...")
    assert len(error_msg_set) == 3

    # Output detailed information about merged issues
    print(f"\n📊 Test 2 - Merged Issues Details:")
    print(f"Total input logs: {len(test_error_logs)}")
    print(f"Total merged issues: {len(merged_issues)}")
    print("-" * 60)
    found_error_log_2 = False
    found_error_log_3 = False
    for i, issue in enumerate(merged_issues, 1):
        print(f"\n🔍 Issue {i}:")
        print(f"  ID: {issue.issue_id}")
        print(f"  Representative Error: {issue.representative_log.error_message[:100]}...")
        print(f"  Error Type: {issue.representative_log.error_type}")
        print(f"  Service: {issue.representative_log.service}")
        print(f"  Occurrence Count: {issue.occurrence_count}")
        print(f"  Similar Logs Count: {len(issue.similar_logs)}")
        print(f"  Affected Services: {issue.affected_services}")
        print(f"  Time Span: {issue.time_span}")
        print(f"  Context Length: {len(issue.context)} characters")

        # Check representative log
        if error_logs[2].error_message[:55] in issue.representative_log.error_message:
            found_error_log_2 = True
            print(f"      ✅ Found error_logs[2] in representative log")
        if error_logs[3].error_message[:55] in issue.representative_log.error_message:
            found_error_log_3 = True
            print(f"      ✅ Found error_logs[3] in representative log")

        # Show similar logs details
        if issue.similar_logs:
            print(f"  Similar Logs:")
            for j, similar_log in enumerate(issue.similar_logs, 1):
                print(f"    {j}. {similar_log.error_message[:55]}... (Service: {similar_log.service})")
                if error_logs[2].error_message[:55] in similar_log.error_message:
                    found_error_log_2 = True
                    print(f"      ✅ Found error_logs[2] in similar log {j}")
                if error_logs[3].error_message[:55] in similar_log.error_message:
                    found_error_log_3 = True
                    print(f"      ✅ Found error_logs[3] in similar log {j}")
    assert found_error_log_2
    assert found_error_log_3


def main():
    """Run all tests."""
    print("Running RAG Engine Acceptance Tests...")

    try:
        print("\n=== Running Test 1 ===")
        test_similarity_merging_1()
        print("✅ Test 1 PASSED")

        print("\n=== Running Test 2 ===")
        test_similarity_merging_2()
        print("✅ Test 2 PASSED")

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
