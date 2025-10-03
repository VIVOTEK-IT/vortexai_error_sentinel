#!/usr/bin/env python3
"""
簡化的日報測試腳本 - 不依賴外部服務
"""

import os
import sys
import logging
from datetime import datetime, timezone

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_log_monitor.email_templates import generate_daily_report_html_email
from error_log_monitor.daily_report import DailyReportRow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_html_email_template():
    """測試 HTML 郵件模板生成"""
    logger.info("Testing HTML email template generation...")
    
    try:
        # 創建模擬報告數據
        mock_issues = [
            DailyReportRow(
                key="TEST-123",
                site="stage",
                count=5,
                error_message="Test error message for stage environment - Database connection timeout",
                status="Open",
                log_group="test-service",
                latest_update=datetime.now(timezone.utc)
            ),
            DailyReportRow(
                key="TEST-456",
                site="prod",
                count=3,
                error_message="Test error message for production environment - Memory allocation failed",
                status="In Progress",
                log_group="prod-service",
                latest_update=datetime.now(timezone.utc)
            ),
            DailyReportRow(
                key="TEST-789",
                site="stage",
                count=1,
                error_message="Test error message for stage environment - API rate limit exceeded",
                status="Resolved",
                log_group="api-service",
                latest_update=datetime.now(timezone.utc)
            )
        ]
        
        mock_site_reports = {
            "stage": {"issues": [mock_issues[0], mock_issues[2]]},
            "prod": {"issues": [mock_issues[1]]}
        }
        
        start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now(timezone.utc)
        
        # 生成 HTML 郵件
        html_content = generate_daily_report_html_email(
            site_reports=mock_site_reports,
            start_date=start_date,
            end_date=end_date,
            total_issues=3
        )
        
        # 保存 HTML 到文件供檢查
        output_file = os.path.join(os.path.dirname(__file__), "..", "test_daily_report_email.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"\n📧 HTML 郵件模板測試結果:")
        print(f"  HTML 內容生成成功")
        print(f"  輸出保存到: {output_file}")
        print(f"  內容長度: {len(html_content)} 字符")
        print("  ✅ HTML 郵件模板測試通過")
        
        return True
        
    except Exception as e:
        logger.error(f"HTML 郵件模板測試失敗: {str(e)}", exc_info=True)
        return False


def test_config_loading():
    """測試配置載入"""
    logger.info("Testing configuration loading...")
    
    try:
        from error_log_monitor.config import load_config
        
        config = load_config()
        
        print(f"\n⚙️ 配置載入測試結果:")
        print(f"  OpenAI API Key: {'已設定' if config.openai_api_key else '未設定'}")
        print(f"  AWS Region: {config.email.aws_region}")
        print(f"  Email Sender: {config.email.sender_email}")
        print(f"  Email Recipients: {config.email.recipients}")
        print(f"  OpenSearch Host: {config.opensearch.host}")
        print(f"  Jira Project Key: {config.jira.project_key}")
        print("  ✅ 配置載入測試通過")
        
        return True
        
    except Exception as e:
        logger.error(f"配置載入測試失敗: {str(e)}", exc_info=True)
        return False


def test_imports():
    """測試模組導入"""
    logger.info("Testing module imports...")
    
    try:
        # 測試主要模組導入
        from error_log_monitor.daily_report import DailyReportGenerator, DailyReportRow
        from error_log_monitor.email_service import EmailService
        from error_log_monitor.email_templates import generate_daily_report_html_email
        from error_log_monitor.lambda_daily_report import lambda_handler, generate_daily_report_only
        
        print(f"\n📦 模組導入測試結果:")
        print(f"  DailyReportGenerator: ✅")
        print(f"  DailyReportRow: ✅")
        print(f"  EmailService: ✅")
        print(f"  generate_daily_report_html_email: ✅")
        print(f"  lambda_handler: ✅")
        print(f"  generate_daily_report_only: ✅")
        print("  ✅ 模組導入測試通過")
        
        return True
        
    except Exception as e:
        logger.error(f"模組導入測試失敗: {str(e)}", exc_info=True)
        return False


def main():
    """運行所有測試"""
    print("🚀 開始日報系統簡化測試")
    print("=" * 50)
    
    tests = [
        ("模組導入", test_imports),
        ("配置載入", test_config_loading),
        ("HTML 郵件模板", test_html_email_template),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 運行 {test_name} 測試...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"測試 {test_name} 失敗，異常: {str(e)}", exc_info=True)
            results.append((test_name, False))
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("📋 測試摘要:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n總計: {passed}/{len(results)} 個測試通過")
    
    if passed == len(results):
        print("🎉 所有測試通過！日報系統核心功能正常。")
        print("\n📝 注意事項:")
        print("  - 需要設定有效的 OpenAI API Key 才能運行完整測試")
        print("  - 需要設定有效的 AWS 憑證才能測試郵件發送功能")
        print("  - 需要連接到 OpenSearch 和 Jira 才能生成實際報告")
        return 0
    else:
        print("⚠️  部分測試失敗。請檢查日誌了解詳情。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


