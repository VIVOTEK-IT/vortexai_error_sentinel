#!/usr/bin/env python3
"""
Acceptance Test for Error Analysis
Tests error analysis functionality based on test cases defined in acceptance_test_for_error_analysis.md
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add src to path
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
except NameError:
    # Handle case when __file__ is not defined (e.g., when exec'd)
    sys.path.insert(0, '/app/src')

from error_log_monitor.config import SystemConfig, VectorDBConfig
from error_log_monitor.opensearch_client import ErrorLog
from error_log_monitor.vector_db_client import VectorDBClient
from error_log_monitor.rag_engine import RAGEngine, MergedIssue
from error_log_monitor.error_analyzer import ErrorAnalyzer, ErrorSeverity
from error_log_monitor.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config() -> SystemConfig:
    """Create test configuration."""

    _config = load_config()

    _config.vector_db = VectorDBConfig(
        persist_directory="./test_chroma_db",
        collection_name="test_error_log_vectors",
        distance_metric="cosine",
        embedding_model="text-embedding-3-small",
    )
    return _config


def create_test_error_log() -> ErrorLog:
    """Create test error log based on acceptance test case."""
    return ErrorLog(
        message_id="test_skip_zero_vector_1758210277886",
        timestamp=datetime(2025, 9, 18, 10, 30, 0, tzinfo=timezone.utc),
        error_message="Skip zero vector for oid:1758210277886, mac:0002D1AF8A57_ch9, s3 bucket: banana-devices-push-storage-the-greate-one-vsaas-vortex-prod, s3_key: 0002D1AF8A57-1740688267072/ch9/object_thumbnail/2025/09/18/1758210558900_1758210277886_1758210591902_1758210604903_1758210560901_1758210630905.objmetadata",
        error_message_hash=1234567890,
        traceback="  File \"/var/runtime/bootstrap.py\", line 63, in <module>\n    main()\n;  File \"/var/runtime/bootstrap.py\", line 60, in main\n    awslambdaricmain.main([os.environ[\"LAMBDA_TASK_ROOT\"], os.environ[\"_HANDLER\"]])\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/__main__.py\", line 21, in main\n    bootstrap.run(app_root, handler, lambda_runtime_api_addr)\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 529, in run\n    handle_event_request(\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 177, in handle_event_request\n    response = request_handler(event, lambda_context)\n;  File \"/var/task/lambdas/utils.py\", line 43, in wrapper\n    result = func(*args, **kwargs)\n;  File \"/var/task/lambdas/lambda_s3_uploader.py\", line 7, in lambda_handler_s3uploadparser\n    return lambda_parse_s3_metadata.lambda_handler(event, context)\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/lambda_parse_s3_metadata.py\", line 211, in lambda_handler\n    result = parser.parse_s3report(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 582, in parse_s3report\n    thumbnail_dict['Thumbnail'], thumb_size = parse_thumbnail_item(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 946, in parse_thumbnail_item\n    r = elk_helper.insert_action(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 236, in insert_action\n    logger.error(f'Skip zero vector for oid:{oid}, mac:{mac}, s3 bucket: {s3_bucket}, s3_key: {s3_key}')\n;  File \"/var/task/vortexai/ivs_logger.py\", line 40, in error\n    _msg = prepare_error_message(msg, LogCategory.ERROR)\n",
        traceback_hash=9876543210,
        error_type="NoneType: None",
        error_type_hash=1111111111,
        site="prod",
        index_name="error_log_prod_2025_9",
        topic="",
        count=1,
        request_id="test-request-id-12345",
        category="error",
        log_group="/aws/lambda/vortex-ai-S3UploadParser",
        service="S3UploadParser",
    )


def test_skip_zero_vector_analysis():
    """Test Case 1: Skip zero vector error analysis."""
    print("=== Running Test Case 1: Skip Zero Vector Error Analysis ===")

    # Create test configuration
    config = create_test_config()
    print("Created test configuration")

    # Create vector database client
    vector_db_client = VectorDBClient(config.vector_db)
    print("Created vector database client")

    # Create RAG engine
    rag_engine = RAGEngine(config, vector_db_client)
    print("Created RAG engine")

    # Create error analyzer
    error_analyzer = ErrorAnalyzer(config, rag_engine)
    print("Created error analyzer")

    # Create test error log
    error_log = create_test_error_log()
    print(f"Created test error log: {error_log.error_message[:100]}...")

    # Create merged issue for analysis
    merged_issue = MergedIssue(
        issue_id="test_issue_skip_zero_vector",
        representative_log=error_log,
        similar_logs=[],
        context="Skip zero vector error occurred during S3 metadata processing. This typically happens when the vector data is empty or invalid, causing the system to skip processing that particular object.",
        occurrence_count=1,
        time_span=str(error_log.timestamp),
        affected_services=["S3UploadParser"],
    )

    print(f"Created merged issue: {merged_issue.issue_id}")

    # Perform error analysis
    print("Performing error analysis...")
    analysis_result = error_analyzer.analyze_issue(merged_issue)

    print(f"\n📊 Analysis Results:")
    print(f"Error ID: {analysis_result.error_id}")
    print(f"Error Message: {analysis_result.error_message[:100]}...")
    print(f"Severity: {analysis_result.severity}")
    print(f"Service: {analysis_result.service}")
    print(f"Confidence Score: {analysis_result.confidence_score}")
    print(f"Analysis Model: {analysis_result.analysis_model}")
    print(f"Analysis Time: {analysis_result.analysis_time:.2f}s")

    print(f"\n Scope Analysis:")
    print(f"  Affected Services: {analysis_result.scope.affected_services}")
    print(f"  Technical Impact: {analysis_result.scope.technical_impact}")
    print(f"  Affected Users: {analysis_result.scope.affected_users}")
    print(f"  Estimated Downtime: {analysis_result.scope.estimated_downtime}")

    print(f"\n🛠️ Remediation Plan:")
    print(f"  Human Action Needed: {analysis_result.remediation_plan.human_action_needed}")
    print(f"  Action Guidelines: {analysis_result.remediation_plan.action_guidelines}")
    print(f"  Damaged Modules: {analysis_result.remediation_plan.damaged_modules}")
    print(f"  Root Cause: {analysis_result.remediation_plan.root_cause}")
    print(f"  Urgency: {analysis_result.remediation_plan.urgency}")

    if analysis_result.data_damage_assessment:
        print(f"\n💾 Data Damage Assessment:")
        print(f"  Data Damaged: {analysis_result.data_damage_assessment.get('data_damaged', 'Unknown')}")
        print(f"  Damaged Tables: {analysis_result.data_damage_assessment.get('damaged_tables', [])}")
        print(f"  Damage Description: {analysis_result.data_damage_assessment.get('damage_description', 'Unknown')}")
        print(f"  Affected Records: {analysis_result.data_damage_assessment.get('affected_records', 'Unknown')}")

    # Verify expected results based on acceptance test
    print(f"\n✅ Verification:")

    # Check severity
    expected_severity = ErrorSeverity.LEVEL_1
    if analysis_result.severity == expected_severity:
        print(f"  ✅ Severity: {analysis_result.severity} (Expected: {expected_severity})")
    else:
        print(f"  ❌ Severity: {analysis_result.severity} (Expected: {expected_severity})")

    # Check affected services - should contain S3UploadParser
    expected_services_contain = "S3UploadParser"
    if expected_services_contain in str(analysis_result.scope.affected_services):
        print(
            f"  ✅ Affected Services: {analysis_result.scope.affected_services} (contains {expected_services_contain})"
        )
    else:
        print(
            f"  ❌ Affected Services: {analysis_result.scope.affected_services} (Expected to contain: {expected_services_contain})"
        )

    # Check technical impact - should indicate non-fatal or benign
    expected_impact_keywords = ["non-fatal", "benign", "skip", "zero vector"]
    impact_text = analysis_result.scope.technical_impact.lower()
    if any(keyword in impact_text for keyword in expected_impact_keywords):
        print(f"  ✅ Technical Impact: {analysis_result.scope.technical_impact}")
    else:
        print(
            f"  ❌ Technical Impact: {analysis_result.scope.technical_impact} (Expected to contain keywords: {expected_impact_keywords})"
        )

    # Check human action needed
    expected_human_action = False
    if analysis_result.remediation_plan.human_action_needed == expected_human_action:
        print(f"  ✅ Human Action Needed: {analysis_result.remediation_plan.human_action_needed}")
    else:
        print(
            f"  ❌ Human Action Needed: {analysis_result.remediation_plan.human_action_needed} (Expected: {expected_human_action})"
        )

    # Check action guidelines - should have some guidelines
    if len(analysis_result.remediation_plan.action_guidelines) > 0:
        print(f"  ✅ Action Guidelines: {analysis_result.remediation_plan.action_guidelines}")
    else:
        print(
            f"  ❌ Action Guidelines: {analysis_result.remediation_plan.action_guidelines} (Expected: non-empty list)"
        )

    # Check damaged modules - should be empty or contain S3UploadParser (both are acceptable for this error)
    damaged_modules = analysis_result.remediation_plan.damaged_modules
    if len(damaged_modules) == 0 or "S3UploadParser" in str(damaged_modules):
        print(f"  ✅ Damaged Modules: {damaged_modules} (empty or contains S3UploadParser)")
    else:
        print(f"  ❌ Damaged Modules: {damaged_modules} (Expected: empty or contains S3UploadParser)")

    # Check root cause - should mention zero vector or skip
    expected_root_cause_keywords = ["zero vector", "zero-valued vector", "skip", "benign"]
    root_cause_text = analysis_result.remediation_plan.root_cause.lower()
    if any(keyword in root_cause_text for keyword in expected_root_cause_keywords):
        print(f"  ✅ Root Cause: {analysis_result.remediation_plan.root_cause}")
    else:
        print(
            f"  ❌ Root Cause: {analysis_result.remediation_plan.root_cause} (Expected to contain keywords: {expected_root_cause_keywords})"
        )

    # Check urgency - should be LOW or MEDIUM
    expected_urgency_values = ["LOW", "MEDIUM"]
    if analysis_result.remediation_plan.urgency in expected_urgency_values:
        print(f"  ✅ Urgency: {analysis_result.remediation_plan.urgency}")
    else:
        print(f"  ❌ Urgency: {analysis_result.remediation_plan.urgency} (Expected one of: {expected_urgency_values})")

    # Overall test result - check all criteria
    all_checks_passed = (
        analysis_result.severity == expected_severity
        and expected_services_contain in str(analysis_result.scope.affected_services)
        and any(keyword in analysis_result.scope.technical_impact.lower() for keyword in expected_impact_keywords)
        and analysis_result.remediation_plan.human_action_needed == expected_human_action
        and len(analysis_result.remediation_plan.action_guidelines) > 0
        and (
            len(analysis_result.remediation_plan.damaged_modules) == 0
            or "S3UploadParser" in str(analysis_result.remediation_plan.damaged_modules)
        )
        and any(
            keyword in analysis_result.remediation_plan.root_cause.lower() for keyword in expected_root_cause_keywords
        )
        and analysis_result.remediation_plan.urgency in expected_urgency_values
    )

    if all_checks_passed:
        print(f"\n🎉 Test Case 1 PASSED - All acceptance criteria met!")
        return True
    else:
        print(f"\n❌ Test Case 1 FAILED - Some acceptance criteria not met")
        return False


def main():
    """Run error analysis acceptance tests."""
    print("Running Error Analysis Acceptance Tests...")
    print("=" * 60)

    # Clear vector database to ensure clean test
    try:
        config = create_test_config()
        vector_db_client = VectorDBClient(config.vector_db)
        vector_db_client.clear_collection()
        print("Vector database cleared for clean test")
    except Exception as e:
        print(f"Warning: Could not clear vector database: {e}")

    # Run test cases
    test_results = []

    # Test Case 1: Skip zero vector error analysis
    result1 = test_skip_zero_vector_analysis()
    test_results.append(("Test Case 1: Skip Zero Vector Analysis", result1))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(" All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
