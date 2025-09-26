# Acceptance Test
## RAG Engine
### Test Case 1
Following two error logs are similar, and should be merged into one issue.

* Error Log 1:
```
{
  "_index": "error_log_prod_2025_9",
  "_id": "449416772_2025-09-03T14:54:00",
  "_version": 1,
  "_score": null,
  "_source": {
    "timestamp": "2025-09-03T14:54:00",
    "error_message": "Exception in function API_FACE_PROFILE_SEARCH (1728, ['0002D1A18076', '0002D1A6ABCD', '0002D1A41FBB', '0002D199DAFC', '0002D19B5F96', '0002D1987F1D', '0002D1AF44CD', '0002D1A18090', '0002D1AA819A', '0002D1AA819B', '0002D1AA814B', '0002D1AA810E', '0002D1AA8152', '0002D1AA143F', '0002D1A8650C', '0002D1AA815B', '0002D1A3D60A', '0002D1A86568', '0002D1A77AB0', '0002D1A714C3', '0002D1A86566', '0002D1AF44D9', '0002D1B5C977', '0002D1A714AC', '0002D1A86564', '0002D1C1CF9C', '0002D1BBE553', '0002D1BCF1BE', '0002D1A3D60F', '0002D1A714D2', '0002D1BEA0FF', '0002D1BBDE2A', '0002D1BBD908', '0002D1BFC23A', '0002D1A64E5B', '0002D1BB11FB', '0002D1B28F38', '0002D1B28F37', '0002D1BDBB87', '0002D1BEA190', '0002D1C1251E', '0002D1A3C2F8', '0002D1AA144B', '0002D1BBDDC5', '0002D1C1254D', '0002D1A2E3AC', '0002D1A44C28', '0002D1A8651E', '0002D1AAEA78', '0002D1A6DE4C', '0002D1A86520', '0002D1A41FB4', '0002D1A86517', '0002D1B0D40E', '0002D1B353D9', '0002D1B99F3E', '0002D1B68B75', '0002D1B4E6B5', '0002D1AAAFDA_ch1', '0002D1AAAFDA_ch0', '0002D1A35CE9_ch11', '0002D1A35D03_ch0', '0002D1B9DC98_ch0', '0002D1A35CE9_ch0', '0002D1A35CC3_ch6', '0002D1AAAFF6_ch0', '0002D1AAAFB2_ch0', '0002D1B180A1_ch0', '0002D1A35CF7_ch0', '0002D1A35CDF_ch0', '0002D1AAAFB2_ch1', '0002D1A35CDF_ch1', '0002D1B180A1_ch1', '0002D1A35CF7_ch1', '0002D1AAAFF6_ch1', '0002D1AAAFB2_ch2', '0002D1A35CDF_ch2', '0002D1A35CF7_ch2', '0002D1B180A1_ch2', '0002D1A35CDF_ch3', '0002D1AAAFB2_ch3', '0002D1B180A1_ch3', '0002D1A35CF7_ch3', '0002D1A35CF7_ch4', '0002D1B180A1_ch4', '0002D1AAAFB2_ch4', '0002D1A35CDF_ch4', '0002D1B180A1_ch5', '0002D1A35CF7_ch5', '0002D1A35CF7_ch6', '0002D1B180A1_ch6', '0002D1A35CF7_ch7', '0002D1A35CF7_ch8', '0002D1A35CF7_ch9', '0002D1A35CF7_ch10', '0002D1A35CF7_ch11', '0002D1A35CF7_ch12', '0002D1A35CF7_ch13', '0002D1A35CF7_ch14', '0002D1A35CF7_ch15', '0002D1A35CF7_ch16', '0002D1A35CF7_ch17', '0002D1A35CF7_ch18', '0002D1A35CF7_ch19', '0002D1A35CC3_ch18', '0002D1A35CE9_ch14', '0002D1AAAFDA_ch4', '0002D1BAE063_ch3', '0002D1AAAFDA_ch3', '0002D1A35D03_ch4', '0002D1A35CC3_ch2', '0002D1A35D03_ch7', '0002D1A35D03_ch6', '0002D1A35CC3_ch14', '0002D1AAB022_ch1', '0002D1A35CC3_ch4', '0002D1A35CE9_ch13', '0002D1AAAFDA_ch5', '0002D1C1880C_ch1', '0002D1A35CE9_ch7', '0002D1A35D03_ch11', '0002D1BAE063_ch4', '0002D1AF90BB_ch1', '0002D1A35CE9_ch9', '0002D1A35CE9_ch5', '0002D1AAAFD2_ch1', '0002D1A35CC3_ch5', '0002D1A35CE9_ch3', '0002D1A35CE9_ch12', '0002D1A35CC3_ch17', '0002D1A35D03_ch3', '0002D1B9DC98_ch1', '0002D1A35CC3_ch15', '0002D1A35D03_ch1', '0002D1A35CE9_ch6', '0002D1A35CC3_ch13', '0002D1BAE063_ch0', '0002D1AAAFDA_ch2', '0002D1A35CC3_ch3', '0002D1AF90BB_ch0', '0002D1A35CE9_ch10', '0002D1A35D03_ch9', '0002D1A35CC3_ch8', '0002D1C1880C_ch0', '0002D1A35CE9_ch2', '0002D1A35D03_ch8', '0002D1A35D03_ch2', '0002D1BAE063_ch7', '0002D1AAB022_ch0', '0002D1A35CC3_ch11', '0002D1A35D03_ch10', '0002D1A35CC3_ch7', '0002D1AAB022_ch3', '0002D1A35CC3_ch9', '0002D1B9DC98_ch4', '0002D1A35D03_ch5', '0002D1A35CC3_ch16', '0002D1C1880C_ch4', '0002D1A35CC3_ch10', '0002D1BAE063_ch5', '0002D1C1880C_ch2', '0002D1C1880C_ch3', '0002D1A35CC3_ch12', '0002D1A35CE9_ch4', '0002D1A35CC3_ch0', '0002D1A35D03_ch12', '0002D1BAE063_ch1', '0002D1B9DC98_ch2', '0002D1BAE063_ch2', '0002D1A35CE9_ch1', '0002D1AAB022_ch2', '0002D1A35CC3_ch1', '0002D1B9DC98_ch3', '0002D1AAAFD2_ch0', '0002D1A35CE9_ch8', '0002D1BAE063_ch6', '0002D1AAB022_ch4', '0002D1A7D4D0_ch15', '0002D1A7D4D0_ch4', '0002D1A7D4D0_ch6', '0002D1A7D4D0_ch14', '0002D1A7D4D0_ch7', '0002D1A7D4D0_ch8', '0002D1A7D4D0_ch5', '0002D1A7D4D0_ch1', '0002D1A7D4D0_ch12', '0002D1A7D4D0_ch9', '0002D1A7D4D0_ch2', '0002D1A7D4D0_ch10', '0002D1A7D4D0_ch11', '0002D1A7D4D0_ch13', '0002D1A7D4D0_ch3', '0002D1A7D4D0_ch0', '0002D1B5C97A', '0002D178F84B', '0002D1AF90C2', '0002D1B68FED'], 2025-08-04 14:54:52, 2025-09-03 14:54:52)",
    "error_message_hash": 758585383,
    "traceback": "Traceback (most recent call last):;  File \"/app/vortexai/ivs_deepsearch/upload_parser/api_face_profile_search.py\", line 111, in lambda_handler;    result_list = execute_knn_query_plan(;                  ^^^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/profile_helper.py\", line 511, in execute_knn_query_plan;    result = elk_helper.query_knn(;             ^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 498, in query_knn;    result_array = self.es.search(;                   ^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/utils.py\", line 177, in _wrapped;    return func(*args, params=params, headers=headers, **kwargs);           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/__init__.py\", line 1544, in search;    return self.transport.perform_request(;           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 407, in perform_request;    raise e;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 368, in perform_request;    status, headers_response, data = connection.perform_request(;                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/http_urllib3.py\", line 275, in perform_request;    self._raise_error(;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/base.py\", line 300, in _raise_error;    raise HTTP_EXCEPTIONS.get(status_code, TransportError)(",
    "traceback_hash": 449416772,
    "error_type": "opensearchpy.exceptions.RequestError: RequestError(400, 'too_long_http_line_exception', 'An HTTP line is larger than 10240 bytes.')",
    "error_type_hash": 1663138015,
    "site": "prod",
    "index_name": "error_log_prod_2025_9",
    "topic": "",
    "count": 15,
    "request_id": "82ae64d8-8838-4e51-aa95-4a338127290b",
    "category": "error",
    "log_group": "/ecs/vortex-ai-vortexai"
  },
  "fields": {
    "timestamp": [
      "2025-09-03T14:54:00.000Z"
    ]
  },
  "highlight": {
    "site": [
      "@opensearch-dashboards-highlighted-field@prod@/opensearch-dashboards-highlighted-field@"
    ]
  },
  "sort": [
    1756911240000
  ]
}
```

* Error Log 2:
```
{
  "_index": "error_log_prod_2025_9",
  "_id": "449416772_2025-09-03T14:07:00",
  "_version": 1,
  "_score": null,
  "_source": {
    "timestamp": "2025-09-03T14:07:00",
    "error_message": "Exception in function API_FACE_PROFILE_SEARCH (2572, ['0002D1A18076', '0002D1A6ABCD', '0002D1A41FBB', '0002D199DAFC', '0002D19B5F96', '0002D1987F1D', '0002D1AF44CD', '0002D1A18090', '0002D1AA819A', '0002D1AA819B', '0002D1AA814B', '0002D1AA810E', '0002D1AA8152', '0002D1AA143F', '0002D1A8650C', '0002D1AA815B', '0002D1A3D60A', '0002D1A86568', '0002D1A77AB0', '0002D1A714C3', '0002D1A86566', '0002D1AF44D9', '0002D1B5C977', '0002D1A714AC', '0002D1A86564', '0002D1C1CF9C', '0002D1BBE553', '0002D1BCF1BE', '0002D1A3D60F', '0002D1A714D2', '0002D1BEA0FF', '0002D1BBDE2A', '0002D1BBD908', '0002D1BFC23A', '0002D1A64E5B', '0002D1BB11FB', '0002D1B28F38', '0002D1B28F37', '0002D1BDBB87', '0002D1BEA190', '0002D1C1251E', '0002D1A3C2F8', '0002D1AA144B', '0002D1BBDDC5', '0002D1C1254D', '0002D1A2E3AC', '0002D1A44C28', '0002D1A8651E', '0002D1AAEA78', '0002D1A6DE4C', '0002D1A86520', '0002D1A41FB4', '0002D1A86517', '0002D1B0D40E', '0002D1B353D9', '0002D1B99F3E', '0002D1B68B75', '0002D1B4E6B5', '0002D1AAAFDA_ch1', '0002D1AAAFDA_ch0', '0002D1A35CE9_ch11', '0002D1A35D03_ch0', '0002D1B9DC98_ch0', '0002D1A35CE9_ch0', '0002D1A35CC3_ch6', '0002D1AAAFF6_ch0', '0002D1AAAFB2_ch0', '0002D1B180A1_ch0', '0002D1A35CF7_ch0', '0002D1A35CDF_ch0', '0002D1AAAFB2_ch1', '0002D1A35CDF_ch1', '0002D1B180A1_ch1', '0002D1A35CF7_ch1', '0002D1AAAFF6_ch1', '0002D1AAAFB2_ch2', '0002D1A35CDF_ch2', '0002D1A35CF7_ch2', '0002D1B180A1_ch2', '0002D1A35CDF_ch3', '0002D1AAAFB2_ch3', '0002D1B180A1_ch3', '0002D1A35CF7_ch3', '0002D1A35CF7_ch4', '0002D1B180A1_ch4', '0002D1AAAFB2_ch4', '0002D1A35CDF_ch4', '0002D1B180A1_ch5', '0002D1A35CF7_ch5', '0002D1A35CF7_ch6', '0002D1B180A1_ch6', '0002D1A35CF7_ch7', '0002D1A35CF7_ch8', '0002D1A35CF7_ch9', '0002D1A35CF7_ch10', '0002D1A35CF7_ch11', '0002D1A35CF7_ch12', '0002D1A35CF7_ch13', '0002D1A35CF7_ch14', '0002D1A35CF7_ch15', '0002D1A35CF7_ch16', '0002D1A35CF7_ch17', '0002D1A35CF7_ch18', '0002D1A35CF7_ch19', '0002D1A35CC3_ch18', '0002D1A35CE9_ch14', '0002D1AAAFDA_ch4', '0002D1BAE063_ch3', '0002D1AAAFDA_ch3', '0002D1A35D03_ch4', '0002D1A35CC3_ch2', '0002D1A35D03_ch7', '0002D1A35D03_ch6', '0002D1A35CC3_ch14', '0002D1AAB022_ch1', '0002D1A35CC3_ch4', '0002D1A35CE9_ch13', '0002D1AAAFDA_ch5', '0002D1C1880C_ch1', '0002D1A35CE9_ch7', '0002D1A35D03_ch11', '0002D1BAE063_ch4', '0002D1AF90BB_ch1', '0002D1A35CE9_ch9', '0002D1A35CE9_ch5', '0002D1AAAFD2_ch1', '0002D1A35CC3_ch5', '0002D1A35CE9_ch3', '0002D1BAE063_ch6', '0002D1A35CE9_ch12', '0002D1A35CC3_ch17', '0002D1A35D03_ch3', '0002D1B9DC98_ch1', '0002D1A35CC3_ch15', '0002D1A35D03_ch1', '0002D1A35CE9_ch6', '0002D1A35CC3_ch13', '0002D1BAE063_ch0', '0002D1AAAFDA_ch2', '0002D1A35CC3_ch3', '0002D1AF90BB_ch0', '0002D1A35CE9_ch10', '0002D1A35D03_ch9', '0002D1A35CC3_ch8', '0002D1C1880C_ch0', '0002D1A35CE9_ch2', '0002D1A35D03_ch8', '0002D1A35D03_ch2', '0002D1BAE063_ch7', '0002D1AAB022_ch0', '0002D1A35CC3_ch11', '0002D1A35D03_ch10', '0002D1A35CC3_ch7', '0002D1AAB022_ch3', '0002D1A35CC3_ch9', '0002D1B9DC98_ch4', '0002D1A35D03_ch5', '0002D1A35CC3_ch16', '0002D1C1880C_ch4', '0002D1A35CC3_ch10', '0002D1BAE063_ch5', '0002D1C1880C_ch2', '0002D1C1880C_ch3', '0002D1A35CC3_ch12', '0002D1A35CE9_ch4', '0002D1A35CC3_ch0', '0002D1A35D03_ch12', '0002D1BAE063_ch1', '0002D1B9DC98_ch2', '0002D1BAE063_ch2', '0002D1A35CE9_ch1', '0002D1AAB022_ch2', '0002D1A35CC3_ch1', '0002D1B9DC98_ch3', '0002D1AAAFD2_ch0', '0002D1A35CE9_ch8', '0002D1AAB022_ch4', '0002D1A7D4D0_ch15', '0002D1A7D4D0_ch4', '0002D1A7D4D0_ch6', '0002D1A7D4D0_ch14', '0002D1A7D4D0_ch7', '0002D1A7D4D0_ch8', '0002D1A7D4D0_ch5', '0002D1A7D4D0_ch1', '0002D1A7D4D0_ch12', '0002D1A7D4D0_ch9', '0002D1A7D4D0_ch2', '0002D1A7D4D0_ch10', '0002D1A7D4D0_ch11', '0002D1A7D4D0_ch13', '0002D1A7D4D0_ch3', '0002D1A7D4D0_ch0', '0002D1B5C97A', '0002D178F84B', '0002D1AF90C2', '0002D1B68FED'], 2025-08-04 14:07:47, 2025-09-03 14:07:47)",
    "error_message_hash": 2349449461,
    "traceback": "Traceback (most recent call last):;  File \"/app/vortexai/ivs_deepsearch/upload_parser/api_face_profile_search.py\", line 111, in lambda_handler;    result_list = execute_knn_query_plan(;                  ^^^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/profile_helper.py\", line 511, in execute_knn_query_plan;    result = elk_helper.query_knn(;             ^^^^^^^^^^^^^^^^^^^^^;  File \"/app/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 498, in query_knn;    result_array = self.es.search(;                   ^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/utils.py\", line 177, in _wrapped;    return func(*args, params=params, headers=headers, **kwargs);           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/client/__init__.py\", line 1544, in search;    return self.transport.perform_request(;           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 407, in perform_request;    raise e;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/transport.py\", line 368, in perform_request;    status, headers_response, data = connection.perform_request(;                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/http_urllib3.py\", line 275, in perform_request;    self._raise_error(;  File \"/usr/local/lib/python3.11/site-packages/opensearchpy/connection/base.py\", line 300, in _raise_error;    raise HTTP_EXCEPTIONS.get(status_code, TransportError)(",
    "traceback_hash": 449416772,
    "error_type": "opensearchpy.exceptions.RequestError: RequestError(400, 'too_long_http_line_exception', 'An HTTP line is larger than 10240 bytes.')",
    "error_type_hash": 1663138015,
    "site": "prod",
    "index_name": "error_log_prod_2025_9",
    "topic": "",
    "count": 3,
    "request_id": "416d0df8-1708-4887-93e1-3d447398ca5b",
    "category": "error",
    "log_group": "/ecs/vortex-ai-vortexai"
  },
  "fields": {
    "timestamp": [
      "2025-09-03T14:07:00.000Z"
    ]
  },
  "highlight": {
    "site": [
      "@opensearch-dashboards-highlighted-field@prod@/opensearch-dashboards-highlighted-field@"
    ]
  },
  "sort": [
    1756908420000
  ]
}
```

### Test Case 2
Following two error logs are similar, and should be merged into one issue.

* Error Log 1:
```
{
  "_index": "error_log_prod_2025_9",
  "_id": "3983379513_2025-09-02T17:00:06.847000+00:00",
  "_version": 1,
  "_score": null,
  "_source": {
    "timestamp": "2025-09-02T17:00:06.847000+00:00",
    "error_message": "Skip zero vector for oid:1756830780468, mac:0002D1B25EDC_ch6, s3 bucket: banana-devices-push-storage-the-greate-one-vsaas-vortex-prod, s3_key: 0002D1B25EDC-1753386287274/ch6/object_thumbnail/2025/09/02/1756832160507_1756832161508_1756831762509_1756830780468_1756832167510.objmetadata",
    "error_message_hash": 1048633025,
    "traceback": "  File \"/var/runtime/bootstrap.py\", line 63, in <module>\n    main()\n;  File \"/var/runtime/bootstrap.py\", line 60, in main\n    awslambdaricmain.main([os.environ[\"LAMBDA_TASK_ROOT\"], os.environ[\"_HANDLER\"]])\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/__main__.py\", line 21, in main\n    bootstrap.run(app_root, handler, lambda_runtime_api_addr)\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 529, in run\n    handle_event_request(\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 177, in handle_event_request\n    response = request_handler(event, lambda_context)\n;  File \"/var/task/lambdas/utils.py\", line 43, in wrapper\n    result = func(*args, **kwargs)\n;  File \"/var/task/lambdas/lambda_s3_uploader.py\", line 7, in lambda_handler_s3uploadparser\n    return lambda_parse_s3_metadata.lambda_handler(event, context)\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/lambda_parse_s3_metadata.py\", line 211, in lambda_handler\n    result = parser.parse_s3report(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 582, in parse_s3report\n    thumbnail_dict['Thumbnail'], thumb_size = parse_thumbnail_item(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 946, in parse_thumbnail_item\n    r = elk_helper.insert_action(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 236, in insert_action\n    logger.error(f'Skip zero vector for oid:{oid}, mac:{mac}, s3 bucket: {s3_bucket}, s3_key: {s3_key}')\n;  File \"/var/task/vortexai/ivs_logger.py\", line 40, in error\n    _msg = prepare_error_message(msg, LogCategory.ERROR)\n",
    "traceback_hash": 3983379513,
    "error_type": "NoneType: None",
    "error_type_hash": 1393578982,
    "site": "prod",
    "index_name": "error_log_prod_2025_9",
    "topic": "",
    "count": 2,
    "request_id": "e6aad46c-62d6-478d-a6d7-6ab34aea78a4",
    "category": "error",
    "log_group": "/aws/lambda/vortex-ai-S3UploadParser"
  },
  "fields": {
    "timestamp": [
      "2025-09-02T17:00:06.847Z"
    ]
  },
  "highlight": {
    "site": [
      "@opensearch-dashboards-highlighted-field@prod@/opensearch-dashboards-highlighted-field@"
    ]
  },
  "sort": [
    1756832406847
  ]
}
```

* Error Log 2:
```
{
  "_index": "error_log_prod_2025_9",
  "_id": "3983379513_2025-09-02T16:50:07.067000+00:00",
  "_version": 1,
  "_score": null,
  "_source": {
    "timestamp": "2025-09-02T16:50:07.067000+00:00",
    "error_message": "Skip zero vector for oid:1756830780468, mac:0002D1B25EDC_ch6, s3 bucket: banana-devices-push-storage-the-greate-one-vsaas-vortex-prod, s3_key: 0002D1B25EDC-1753386287274/ch6/object_thumbnail/2025/09/02/1756831365494_1756831523495_1756830780468_1756831496493.objmetadata",
    "error_message_hash": 945301040,
    "traceback": "  File \"/var/runtime/bootstrap.py\", line 63, in <module>\n    main()\n;  File \"/var/runtime/bootstrap.py\", line 60, in main\n    awslambdaricmain.main([os.environ[\"LAMBDA_TASK_ROOT\"], os.environ[\"_HANDLER\"]])\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/__main__.py\", line 21, in main\n    bootstrap.run(app_root, handler, lambda_runtime_api_addr)\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 529, in run\n    handle_event_request(\n;  File \"/var/lang/lib/python3.11/site-packages/awslambdaric/bootstrap.py\", line 177, in handle_event_request\n    response = request_handler(event, lambda_context)\n;  File \"/var/task/lambdas/utils.py\", line 43, in wrapper\n    result = func(*args, **kwargs)\n;  File \"/var/task/lambdas/lambda_s3_uploader.py\", line 7, in lambda_handler_s3uploadparser\n    return lambda_parse_s3_metadata.lambda_handler(event, context)\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/lambda_parse_s3_metadata.py\", line 211, in lambda_handler\n    result = parser.parse_s3report(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 582, in parse_s3report\n    thumbnail_dict['Thumbnail'], thumb_size = parse_thumbnail_item(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/parse_metadata.py\", line 946, in parse_thumbnail_item\n    r = elk_helper.insert_action(\n;  File \"/var/task/vortexai/ivs_deepsearch/upload_parser/elk_helper.py\", line 236, in insert_action\n    logger.error(f'Skip zero vector for oid:{oid}, mac:{mac}, s3 bucket: {s3_bucket}, s3_key: {s3_key}')\n;  File \"/var/task/vortexai/ivs_logger.py\", line 40, in error\n    _msg = prepare_error_message(msg, LogCategory.ERROR)\n",
    "traceback_hash": 3983379513,
    "error_type": "NoneType: None",
    "error_type_hash": 1393578982,
    "site": "prod",
    "index_name": "error_log_prod_2025_9",
    "topic": "",
    "count": 2,
    "request_id": "d9711e48-a6b3-4797-88a2-3d036bdd068f",
    "category": "error",
    "log_group": "/aws/lambda/vortex-ai-S3UploadParser"
  },
  "fields": {
    "timestamp": [
      "2025-09-02T16:50:07.067Z"
    ]
  },
  "highlight": {
    "site": [
      "@opensearch-dashboards-highlighted-field@prod@/opensearch-dashboards-highlighted-field@"
    ]
  },
  "sort": [
    1756831807067
  ]
}
```
