# Weekly Report 3 Specification

## Goal
Based ob weekly_report_3_spec.md, this is a daily report version. Here only list new requirement or requirement change in daily report.

## Functional requirements
 - Only fetch 24 hours Jira issues, embedding issue or error logs
 - Generate a lambda API for running daily report
 - Send daily report by email via AWS SES
   - Email recipents is stored in env EMAIL_RECIPENTS
   - Email sender is vortexai.dashboard@vortex.vivotek.com
   - Email subject should be composed by [Y-m-d][Vortexai Error Issue]  daily issue report
   - the body of daily report mail should be html formated document
   
 

 - 