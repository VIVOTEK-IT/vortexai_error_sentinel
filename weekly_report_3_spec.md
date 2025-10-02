# Weekly Report 3 Specification

## Goal
Produce a consolidated Jira issue weekly report by reconciling data from Jira Cloud, the Jira Issue Embedding Database, and recent error logs.

## High-Level Workflow
1. **Fetch Jira Issues (6 months)**
   - Use `init_jira_db.get_all_jira_issues` to retrieve issues.
   - Partition issues by `site` (e.g., `stage`, `prod`).
2. **Fetch Embedding Issues (6 months)**
   - Read all issues from `jira_issue_embedding_db` within the same window.
   - Partition by `site`.
3. **Fetch Error Logs (7 days)**
   - Obtain error logs (grouped by `site`) covering the past week.
4. **Issue Mapping & Consolidation**
   - Map error logs to embedding issues via similarity search (see Issue Mapping).
   - Merge duplicate or child issues as required (see Cleanup requirements).
5. **Report Generation**
   - Output both Excel and web (HTML) weekly reports with the required columns.

## Non-Functional Requirements
- Maintain compatibility with existing `embedding_service.py`, `error_analyzer.py`, and `jira_cloud_client.py` modules.
- Reuse existing functions/APIs wherever possible to minimize new surface area.
- Refactor is permitted when it reduces duplication or improves clarity.

## Issue Mapping Rules
- `jira_issue_embedding_db` documents contain a `key` field used to link back to Jira issues.
- Error logs can be embedded and compared against the embedding index.
  - If similarity exceeds the defined threshold, treat the log and issue as referring to the same Jira issue.

## Jira Embedding DB Cleanup
1. **Status Synchronization**
   - Update each embedding document’s `status` to match its Jira counterpart.
2. **Merge Orphaned Issues (no `key` value)**
   - For documents lacking `key`, search for a similar issue whose `key` is known.
   - If similarity exceeds the threshold:
     - Merge occurrence lists into the keyed issue.
     - Remove the orphaned document from the index.

## Error Log Updates in Embedding DB
- For every error log within the target window:
  1. Compute an embedding.
  2. Find the closest embedding issue (above the similarity threshold).
  3. Append the log’s `doc_id` and timestamp to the matched issue’s occurrence list.

## Reporting Outputs
Generate Excel and web (HTML) reports containing the columns:
- `key` – Jira issue key.
- `site` – e.g., `stage` or `prod`.
- `count` – number of occurrences within the reporting period.
- `error_message` – representative error message.
- `status` – Jira workflow status.
- `log_group` – logging group/category.
- `latest update` – timestamp of the most recent error log (UTC).
- `note` – root-cause summary.
