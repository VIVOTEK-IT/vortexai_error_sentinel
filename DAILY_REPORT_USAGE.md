# 日報系統使用說明

## 概述

日報系統是基於週報系統的簡化版本，專門用於生成過去24小時的錯誤報告並通過 AWS SES 發送郵件通知。

## 功能特性

### ✅ 已實現功能

1. **日報生成器** (`DailyReportGenerator`)
   - 基於週報系統架構
   - 獲取過去24小時的 Jira 問題
   - 獲取過去24小時的錯誤日誌
   - 使用嵌入相似性匹配關聯問題
   - 生成 Excel 和 HTML 報告

2. **郵件服務** (`EmailService`)
   - AWS SES 集成
   - HTML 郵件發送
   - 郵件驗證狀態檢查
   - 發送配額查詢

3. **HTML 郵件模板** (`email_templates.py`)
   - 響應式設計
   - 美觀的統計卡片
   - 問題狀態標籤
   - 站點分組顯示

4. **Lambda API** (`lambda_daily_report.py`)
   - `lambda_handler`: 完整日報生成和郵件發送
   - `generate_daily_report_only`: 僅生成報告
   - `test_email_service`: 測試郵件服務

## 環境配置

### 必要環境變數

```bash
# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key_here

# OpenSearch 配置
OPENSEARCH_HOST=43.207.106.51
OPENSEARCH_PORT=443
OPENSEARCH_USERNAME=vsaas-admin
OPENSEARCH_PASSWORD=your_password

# Jira 配置
JIRA_SERVER_URL=https://your-jira-instance.atlassian.net
JIRA_USERNAME=your_email@domain.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=your_project_key

# AWS SES 配置
AWS_REGION=ap-northeast-1
EMAIL_SENDER=vortexai.dashboard@vortex.vivotek.com
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com

# RDS 配置
RDS_HOST=your_rds_host
RDS_PORT=5432
RDS_DATABASE=vsaas_postsearch
RDS_USER=your_rds_user
RDS_PASSWORD=your_rds_password
```

## 使用方法

### 1. 基本使用

```bash
# 載入虛擬環境
source /home/jerry/venv311/bin/activate

# 進入專案目錄
cd /home/jerry/venv311/prj_error_sentinel/vortexai_error_sentinel

# 運行日報（包含郵件發送）
python scripts/run_daily_report.py

# 運行日報（不發送郵件）
python scripts/run_daily_report.py --no-email

# 指定結束日期
python scripts/run_daily_report.py --end-date 2025-10-03
```

### 2. 測試功能

```bash
# 運行簡化測試（不依賴外部服務）
python scripts/test_daily_report_simple.py

# 運行完整測試（需要所有外部服務）
python scripts/test_daily_report.py
```

### 3. Docker 使用

```bash
# 使用 Docker Compose
docker-compose up -d

# 在容器中運行日報
docker-compose exec error-monitor python scripts/run_daily_report.py

# 在容器中運行測試
docker-compose exec error-monitor python scripts/test_daily_report_simple.py
```

## 文件結構

```
src/error_log_monitor/
├── daily_report.py              # 日報生成器
├── email_service.py             # AWS SES 郵件服務
├── email_templates.py           # HTML 郵件模板
├── lambda_daily_report.py       # Lambda API 處理器
└── config.py                    # 配置管理（已更新）

scripts/
├── run_daily_report.py          # 日報運行腳本
├── test_daily_report.py         # 完整測試腳本
└── test_daily_report_simple.py  # 簡化測試腳本

reports/                         # 生成的報告文件
├── daily_report_stage_*.xlsx
├── daily_report_prod_*.xlsx
└── daily_report_combined_*.xlsx
```

## 報告格式

### Excel 報告欄位

| 欄位 | 描述 |
|------|------|
| Key | Jira 問題鍵值 |
| Site | 站點（stage/prod） |
| Count | 發生次數 |
| Error_Message | 錯誤訊息 |
| Status | Jira 狀態 |
| Log Group | 日誌群組 |
| Latest Update | 最新更新時間 |
| Note | 備註 |

### HTML 郵件特性

- **響應式設計**：適配各種設備
- **統計卡片**：顯示總問題數、各站點問題數
- **狀態標籤**：不同顏色的狀態指示器
- **問題表格**：詳細的問題列表
- **時間範圍**：清晰的報告期間顯示

## Lambda 部署

### 1. 準備部署包

```bash
# 創建部署目錄
mkdir lambda_deployment
cd lambda_deployment

# 複製源代碼
cp -r ../src .
cp ../requirements.txt .

# 安裝依賴
pip install -r requirements.txt -t .

# 創建部署包
zip -r daily_report_lambda.zip .
```

### 2. Lambda 函數配置

- **運行時**：Python 3.11
- **記憶體**：512 MB（建議）
- **超時**：15 分鐘
- **環境變數**：設定所有必要的環境變數

### 3. 觸發器設定

- **CloudWatch Events**：每日定時觸發
- **API Gateway**：手動觸發
- **S3 事件**：基於文件上傳觸發

## 故障排除

### 常見問題

1. **OpenAI API Key 錯誤**
   ```
   ValueError: OpenAI API key is required
   ```
   **解決方案**：檢查 `.env` 文件中的 `OPENAI_API_KEY` 設定

2. **AWS 憑證過期**
   ```
   ExpiredToken: The security token included in the request is expired
   ```
   **解決方案**：更新 AWS 憑證或重新配置 AWS CLI

3. **OpenSearch 連接失敗**
   ```
   ConnectionError: Failed to connect to OpenSearch
   ```
   **解決方案**：檢查 OpenSearch 主機和憑證設定

4. **Jira API 錯誤**
   ```
   JiraError: Authentication failed
   ```
   **解決方案**：檢查 Jira 用戶名和 API Token

### 測試建議

1. **先運行簡化測試**：`python scripts/test_daily_report_simple.py`
2. **檢查配置**：確認所有環境變數正確設定
3. **測試外部服務**：逐一測試 OpenSearch、Jira、AWS SES 連接
4. **逐步測試**：先測試報告生成，再測試郵件發送

## 監控和日誌

### 日誌位置

- **應用日誌**：`logs/error_monitor.log`
- **Docker 日誌**：`docker-compose logs error-monitor`

### 監控指標

- 日報生成成功率
- 郵件發送成功率
- 處理時間
- 錯誤日誌數量

## 擴展功能

### 可能的改進

1. **多語言支援**：支援中文和英文報告
2. **自定義模板**：允許用戶自定義郵件模板
3. **報告排程**：更靈活的排程選項
4. **告警整合**：與 Slack、Teams 等整合
5. **報告歸檔**：自動歸檔歷史報告

## 支援

如有問題，請聯繫：
- **郵件**：vortexai.dashboard@vortex.vivotek.com
- **技術支援**：yenjie.chen@vivotek.com


