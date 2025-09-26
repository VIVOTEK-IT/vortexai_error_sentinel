# VortexAI ECS API Documentation

This document describes all the Flask API endpoints available in the VortexAI ECS system.

## System Overview

The VortexAI ECS system is a comprehensive video analytics platform that provides various APIs for:
- User authentication and management
- Deep search and object detection
- Face profile management
- Case vault management
- User feedback and reporting
- Archive management
- Research and analytics

## Base URL

The API is typically served at the root domain with various endpoints.

## Authentication

Most endpoints require JWT authentication via the `Authorization` header with Bearer token format.

---

## Core System APIs

### Health and Version

#### GET /
- **Endpoint**: `GET /`
- **Description**: Redirects to API documentation
- **Authentication**: None required
- **Module Name**: root
- **Response**: Redirect to `/apidocs/`

#### GET /version
- **Endpoint**: `GET /version`
- **Description**: Returns the backend version information
- **Authentication**: None required
- **Module Name**: version
- **Response**: Version string

#### GET /health
- **Endpoint**: `GET /health`
- **Description**: Health check endpoint
- **Authentication**: None required
- **Module Name**: health
- **Response**: "ok"

#### GET /clean_tmp
- **Endpoint**: `GET /clean_tmp`
- **Description**: Cleans temporary files
- **Authentication**: None required
- **Module Name**: clean_tmp
- **Response**: Status message

---

## Authentication APIs

### POST /login
- **Description**: Authenticate user and get JWT token
- **Authentication**: None required
- **Module Name**: login
- **Request Body**:
  - `username` (string): Username
  - `password` (string): Password
- **Response**: JWT token

---

## Deep Search APIs

### POST /deepsearch
- **Description**: Main deep search functionality
- **Authentication**: Required
- **Module Name**: deepsearchapi
- **Functionality**: Search for objects in video data

### POST /v1/deepsearch
- **Description**: Version 1 deep search API
- **Authentication**: Required
- **Module Name**: deepsearchv1api
- **Functionality**: Enhanced search with version 1 features

### GET /v1/deepsearch/count
- **Description**: Get count of search results
- **Authentication**: Required
- **Module Name**: deepsearchcountingv1api
- **Functionality**: Returns count of matching records

### POST /deepsearch_knn
- **Description**: K-Nearest Neighbors search
- **Authentication**: Required
- **Module Name**: deepsearchknn
- **Functionality**: Similarity-based search

### POST /deepsearch_first
- **Description**: First stage of deep search
- **Authentication**: Required
- **Module Name**: deepsearchapi
- **Functionality**: Initial search processing

### POST /deepsearch_second
- **Description**: Second stage of deep search
- **Authentication**: Required
- **Module Name**: deepsearchapi
- **Functionality**: Refined search processing

### POST /v2/deepsearch
- **Description**: Version 2 deep search API
- **Authentication**: Required
- **Module Name**: deepsearchv2
- **Functionality**: Latest deep search features

### GET /v2/deepsearch/history
- **Description**: Get search history
- **Authentication**: Required
- **Module Name**: deepsearchv2
- **Functionality**: Retrieve past search queries

### GET /v2/deepsearch/descriptions
- **Description**: Get search descriptions
- **Authentication**: Required
- **Module Name**: deepsearchv2
- **Functionality**: Retrieve search metadata

### POST /thinksearch
- **Description**: AI-powered search
- **Authentication**: Required
- **Module Name**: deepsearchapi
- **Functionality**: Intelligent search with AI assistance

---

## Object Detection APIs

### POST /v1/object-detection-records/async-search
- **Description**: Asynchronous object detection search
- **Authentication**: Required
- **Module Name**: deepsearchv1api
- **Functionality**: Start async search job

### GET /v1/object-detection-records/async-search/{task_id}
- **Description**: Get async search job status
- **Authentication**: Required
- **Module Name**: deepsearchv1api
- **Functionality**: Check job progress and results

### POST /v1/object_trace/query
- **Description**: Object tracking query
- **Authentication**: Required
- **Module Name**: objecttracequeryapi
- **Functionality**: Track objects across time

---

## Face Profile APIs

### POST /faceprofile_search
- **Description**: Search face profiles
- **Authentication**: Required
- **Module Name**: faceprofilesearchapi
- **Functionality**: Find matching face profiles

### POST /faceprofile_create
- **Description**: Create new face profile
- **Authentication**: Required
- **Module Name**: faceprofilecreateapi
- **Functionality**: Add new face to database

### GET /faceprofile_list
- **Description**: List face profiles
- **Authentication**: Required
- **Module Name**: faceprofilelistapi
- **Functionality**: Get all face profiles

### PUT /faceprofile_rename
- **Description**: Rename face profile
- **Authentication**: Required
- **Module Name**: faceprofileupdateapi
- **Functionality**: Update profile name

### DELETE /faceprofile_delete
- **Description**: Delete face profile
- **Authentication**: Required
- **Module Name**: faceprofiledeleteapi
- **Functionality**: Remove profile from database

### PUT /faceprofile_update
- **Description**: Update face profile
- **Authentication**: Required
- **Module Name**: faceprofileupdateapi
- **Functionality**: Modify profile information

### POST /faceprofile_grant
- **Description**: Grant profile access
- **Authentication**: Required
- **Module Name**: faceprofilegrantprofiletokenapi
- **Functionality**: Manage profile permissions

### POST /v1/faceprofile_grant
- **Description**: Version 1 profile grant
- **Authentication**: Required
- **Module Name**: faceprofilegrantprofiletokenapi
- **Functionality**: Grant access with v1 features

### POST /faceprofile_feedback
- **Description**: Provide face profile feedback
- **Authentication**: Required
- **Module Name**: faceprofilefeedbackapi
- **Functionality**: Submit user feedback

### POST /faceprofile_upload
- **Description**: Upload face images
- **Authentication**: Required
- **Module Name**: faceprofileuploadapi
- **Functionality**: Add face images to profiles

### V2 Face Profile APIs

### GET /v2/profile/face
- **Description**: Get face profile collection
- **Authentication**: Required
- **Module Name**: faceprofilelistapi
- **Functionality**: Retrieve face profiles (RESTful)

### GET /v2/profile/face/{profile_id}
- **Description**: Get specific face profile
- **Authentication**: Required
- **Module Name**: faceprofilelistapi
- **Functionality**: Retrieve individual profile

### POST /v2/profile/face/{profile_id}/images
- **Description**: Add images to face profile
- **Authentication**: Required
- **Module Name**: faceprofileuploadapi
- **Functionality**: Upload images for profile

### GET /v2/profile/face/{profile_id}/images/{img_id}
- **Description**: Get specific face image
- **Authentication**: Required
- **Module Name**: faceprofilelistapi
- **Functionality**: Retrieve individual image

---

## Case Vault APIs

### POST /caseVaults
- **Description**: Create new case vault
- **Authentication**: Required
- **Module Name**: casevaultresouce
- **Functionality**: Create case for evidence management

### GET /caseVaults
- **Description**: Get all case vaults
- **Authentication**: Required
- **Module Name**: casevaultlistresouce
- **Functionality**: List all cases with details

### DELETE /caseVaults
- **Description**: Delete all case vaults
- **Authentication**: Required
- **Module Name**: casevaultresouce
- **Functionality**: Clear all cases in organization

### GET /listCaseVaults
- **Description**: List case vaults (brief)
- **Authentication**: Required
- **Module Name**: casevaultlistresouce
- **Functionality**: Get brief case list

### GET /caseVault/{case_id}
- **Description**: Get specific case vault
- **Authentication**: Required
- **Module Name**: casevaultresouce
- **Functionality**: Retrieve case details

### PUT /caseVault/{case_id}
- **Description**: Update case vault
- **Authentication**: Required
- **Module Name**: casevaultresouce
- **Functionality**: Modify case information

### DELETE /caseVault/{case_id}
- **Description**: Delete specific case vault
- **Authentication**: Required
- **Module Name**: casevaultresouce
- **Functionality**: Remove individual case

### GET /caseVaultExportStatus
- **Description**: Get export status
- **Authentication**: Required
- **Module Name**: casevaultexportresource
- **Functionality**: Check export progress

### GET /caseVault/{caseId}/videoStatistics
- **Description**: Get video statistics
- **Authentication**: Required
- **Module Name**: casevaultobjectstatresource
- **Functionality**: Retrieve case video stats

### GET /export/caseVault/{caseId}/{export_type}
- **Description**: Download exported case
- **Authentication**: Required
- **Module Name**: casevaultexportresource
- **Functionality**: Get download link

### POST /export/caseVault/{caseId}/{export_type}
- **Description**: Export case
- **Authentication**: Required
- **Module Name**: casevaultexportresource
- **Functionality**: Start export process

### DELETE /export/caseVault/{caseId}/{export_type}
- **Description**: Stop export
- **Authentication**: Required
- **Module Name**: casevaultexportresource
- **Functionality**: Cancel export job

---

## User Feedback APIs

### POST /userFeedback
- **Description**: Create user feedback
- **Authentication**: Required
- **Module Name**: userfeedback
- **Functionality**: Submit feedback for events

### GET /userFeedbackStatus
- **Description**: Check feedback status
- **Authentication**: Required
- **Module Name**: userfeedbackstatus
- **Functionality**: Verify if event was feedbacked

### GET /query_userfeedback
- **Description**: Query user feedback
- **Authentication**: Required
- **Module Name**: userfeedbackquery
- **Functionality**: Search feedback records

---

## Search Feedback APIs

### POST /deepsearch_report
- **Description**: Report false search
- **Authentication**: Required
- **Module Name**: falsesearchreport
- **Functionality**: Submit false positive reports

### GET /deepsearch_report
- **Description**: Query false search reports
- **Authentication**: Required
- **Module Name**: falsesearchreport
- **Functionality**: Retrieve false positive data

---

## Archive APIs

### GET /feedback/archive/{thingname}
- **Description**: Get feedback archive
- **Authentication**: Required
- **Module Name**: feedbackarchive
- **Functionality**: Retrieve device feedback shadow

### POST /feedback/archive/{thingname}
- **Description**: Update feedback archive
- **Authentication**: Required
- **Module Name**: feedbackarchive
- **Functionality**: Refactor device feedback shadow

---

## Research APIs

### POST /v1/research
- **Description**: Research functionality
- **Authentication**: Required
- **Module Name**: researchv1api
- **Functionality**: Advanced research capabilities

---

## VSLS Integration APIs

### GET /api/deepsearch/getaggregates
- **Description**: Get aggregated data
- **Authentication**: Required
- **Module Name**: vslcapigetdictionary
- **Functionality**: Retrieve aggregated search results

### GET /api/deepsearch/getrecords
- **Description**: Get search records
- **Authentication**: Required
- **Module Name**: vslcapigetrecords
- **Functionality**: Retrieve search result records

### GET /api/deepsearch/getdictionary
- **Description**: Get data dictionary
- **Authentication**: Required
- **Module Name**: vslcapigetdictionary
- **Functionality**: Retrieve field definitions

---

## Development APIs (Dev Environment Only)

### POST /messagesearch/search_by_type
- **Description**: Message search by type
- **Authentication**: Required
- **Module Name**: deepsearchapi
- **Functionality**: Search messages by type (dev only)

### GET /beta/faces/clusters/summary
- **Description**: Face cluster summary
- **Authentication**: Required
- **Module Name**: faceprofilelistapi
- **Functionality**: Get face clustering data (dev only)

### POST /grant
- **Description**: Grant permissions
- **Authentication**: Required
- **Module Name**: faceprofilegrantprofiletokenapi
- **Functionality**: Manage system permissions (dev only)

---

## Error Codes

- **200**: Success
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **409**: Conflict
- **488**: Exceed Max Object limitation
- **489**: Exceed Max Case limitation
- **490**: Duplicate alias name
- **500**: Internal Server Error

---

## Common Request/Response Patterns

### Authentication Header
```
Authorization: Bearer <jwt_token>
```

### Pagination
```json
{
  "page": 1,
  "nrOfRecords": 50
}
```

### Filtering
```json
{
  "filters": {
    "type": "and",
    "filterClause": [
      {
        "field": "Date",
        "condition": "gt",
        "value": "2023-01-01"
      }
    ]
  }
}
```

### Sorting
```json
{
  "sorting": [
    {
      "columnName": "Date",
      "ascending": true
    }
  ]
}
```

---

## Notes

- All timestamps are in ISO 8601 format
- File uploads support various image formats
- Case vault operations are organization-scoped
- Face profile operations support batch processing
- Deep search supports complex filtering and aggregation
- All APIs return JSON responses unless otherwise specified
