# Vortex Error Message Sentinel

A comprehensive error log monitoring and analysis system that integrates Jira issue tracking with OpenSearch-based error log analysis using RAG (Retrieval-Augmented Generation) and vector similarity search.

## 🚀 Overview

The Vortex Error Message Sentinel is designed to protect the Vortexai production service by providing intelligent error log analysis, automatic Jira issue correlation, and real-time data integrity verification. The system uses advanced AI techniques to identify similar errors, generate impact analysis, and maintain data consistency across multiple data sources.

## ✨ Key Features

### 🔍 **Intelligent Error Analysis**
- **RAG-based Error Similarity**: Automatically groups similar error logs using vector embeddings
- **OpenAI-Powered Impact Analysis**: Generates detailed impact assessments using GPT models
- **Real-time Correlation**: Links error logs with existing Jira issues for better tracking

### 🎯 **Jira Integration**
- **Dynamic Custom Field Support**: Automatically discovers and maps Jira custom fields
- **Issue Embedding Database**: Vector-based similarity search for Jira issues
- **Automated Issue Creation**: Creates new Jira issues for uncorrelated errors
- **Parent-Child Issue Management**: Organizes related issues hierarchically

### 📊 **Data Integrity Verification**
- **RDS Data Validation**: Verifies data integrity across multiple database tables
- **OpenSearch Integration**: Efficient error log retrieval and analysis
- **Multi-Site Support**: Handles production and staging environments

### 🛠 **Advanced Analytics**
- **Weekly Report Generation**: Automated comprehensive error analysis reports
- **Cost Tracking**: Monitors OpenAI API usage and costs
- **Performance Metrics**: Tracks system performance and efficiency

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenSearch    │    │   Jira Cloud    │    │   RDS Database  │
│   (Error Logs)  │    │   (Issues)      │    │   (Data)        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                Error Log Monitoring System                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  RAG Engine     │  Jira Embedding │  Data Integrity            │
│  (Similarity)   │  Database       │  Verification              │
└─────────────────┴─────────────────┴─────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Weekly Report Generator                           │
│              (Impact Analysis & Reports)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Jira Embedding Workflow

1. **Ingestion (`scripts/init_jira_db.py`)** – fetch Jira issues, compute embeddings, and populate the `jira_issue_embedding` index with normalized vectors and occurrence scaffolding.
2. **Log Group Synchronization (`scripts/sync_jira_log_groups.py`)** – periodically align `log_group` values in OpenSearch with the authoritative values in Jira.
3. **Reporting (`scripts/run_weekly_report_2.py`)** – read issues directly from the embedding index, aggregate occurrences within the requested window, and generate per-site/combined Excel reports.


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenSearch cluster
- Jira Cloud instance
- PostgreSQL database (RDS)
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vortex_error_sentinel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Set up environment variables**
   ```bash
   # OpenAI Configuration
   export OPENAI_API_KEY="your-openai-api-key"
   
   # OpenSearch Configuration
   export OPENSEARCH_HOST="your-opensearch-host"
   export OPENSEARCH_USERNAME="your-username"
   export OPENSEARCH_PASSWORD="your-password"
   
   # Jira Configuration
   export JIRA_SERVER_URL="https://your-domain.atlassian.net"
   export JIRA_USERNAME="your-email@domain.com"
   export JIRA_API_TOKEN="your-api-token"
   export JIRA_PROJECT_KEY="VEL"
   
   # RDS Configuration
   export RDS_HOST="your-rds-host"
   export RDS_DATABASE="your-database"
   export RDS_USERNAME="your-username"
   export RDS_PASSWORD="your-password"
   ```

### Running the System

1. **Initialize Jira Issue Embedding Database**
   ```bash
   python scripts/init_jira_db.py --dry-run
   python scripts/init_jira_db.py --project-key VEL
   ```

2. **Run Weekly Report Generation**
   ```bash
   python scripts/generate_weekly_report.py
   ```

3. **Start the monitoring service**
   ```bash
   python src/error_log_monitor/main.py
   ```

## 📁 Project Structure

```
vortex_error_sentinel/
├── src/error_log_monitor/          # Core system modules
│   ├── config.py                   # Configuration management
│   ├── opensearch_client.py        # OpenSearch integration
│   ├── rds_client.py               # RDS database client
│   ├── embedding_service.py        # AI embedding service
│   ├── rag_engine.py               # RAG similarity engine
│   ├── jira_cloud_client.py        # Jira Cloud API client
│   ├── jira_issue_embedding_db.py  # Jira embedding database
│   ├── weekly_report_generator.py  # Report generation
│   └── main.py                     # Main application entry
├── scripts/                        # Utility scripts
│   ├── init_jira_db.py            # Database initialization
│   └── generate_weekly_report.py  # Report generation
├── data/                          # Data storage
├── reports/                       # Generated reports
├── logs/                          # System logs
├── requirements.txt               # Python dependencies
├── docker-compose.yml            # Docker configuration
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for AI services | Required |
| `OPENSEARCH_HOST` | OpenSearch cluster host | Required |
| `OPENSEARCH_USERNAME` | OpenSearch username | Required |
| `OPENSEARCH_PASSWORD` | OpenSearch password | Required |
| `JIRA_SERVER_URL` | Jira Cloud server URL | Required |
| `JIRA_USERNAME` | Jira Cloud username | Required |
| `JIRA_API_TOKEN` | Jira Cloud API token | Required |
| `JIRA_PROJECT_KEY` | Jira project key | Required |
| `RDS_HOST` | RDS database host | Required |
| `RDS_DATABASE` | RDS database name | Required |
| `RDS_USERNAME` | RDS username | Required |
| `RDS_PASSWORD` | RDS password | Required |

### Jira Custom Fields

The system automatically discovers and maps Jira custom fields:

- `error_message` - Error message content
- `error_type` - Type of error (Exception, Error, Warning)
- `request_id` - Unique request identifier
- `log_group` - Log group classification
- `site` - Site name or identifier
- `count` - Occurrence count
- `traceback` - Full stack traceback
- `parent_issue` - Parent issue reference

## 🎯 Usage Examples

### Initialize Jira Issue Database

```bash
# Dry run to test configuration
python scripts/init_jira_db.py --dry-run

# Initialize with specific project
python scripts/init_jira_db.py --project-key VEL --max-issues 1000

# Initialize with custom time range
python scripts/init_jira_db.py --sites prod,stage --months 12
```

### Generate Weekly Reports

```bash
# Generate report for specific sites
python scripts/generate_weekly_report.py --sites prod,stage

# Generate report with custom date range
python scripts/generate_weekly_report.py --start-date 2024-01-01 --end-date 2024-01-07

# Simplified weekly report using Jira embeddings only
python scripts/run_weekly_report_2.py --start-date 2025-01-01 --end-date 2025-01-07
```

### Synchronize Jira Log Groups

```bash
# Preview updates without modifying data
python scripts/sync_jira_log_groups.py --dry-run

# Apply updates (removing --dry-run)
python scripts/sync_jira_log_groups.py
```

### Run Monitoring Service

```bash
# Start the monitoring service
python src/error_log_monitor/main.py

# Run with specific configuration
python src/error_log_monitor/main.py --config custom_config.yaml
```

## 🔍 API Reference

### Core Classes

#### `JiraIssueEmbeddingDB`
Manages Jira issue embeddings and occurrence tracking in OpenSearch.

```python
from error_log_monitor.jira_issue_embedding_db import JiraIssueEmbeddingDB

# Initialize database
db = JiraIssueEmbeddingDB(embedding_service, config)

# Add Jira issue
db.add_jira_issue(jira_issue_data, error_log_data)

# Find similar issues
similar_issues = db.find_similar_jira_issue(error_log_embedding, site)
```

#### `RAGEngine`
Handles error log similarity analysis using RAG techniques.

```python
from error_log_monitor.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine(embedding_service, opensearch_client)

# Find similar errors
similar_errors = rag.find_similar_errors(error_log, threshold=0.85)

# Generate impact analysis
analysis = rag.generate_impact_analysis(error_logs)
```

#### `WeeklyReportGenerator`
Generates comprehensive weekly error analysis reports.

```python
from error_log_monitor.weekly_report_generator import WeeklyReportGenerator

# Initialize generator
generator = WeeklyReportGenerator(config)

# Generate report
report = generator.generate_report(sites=['prod', 'stage'])
```

#### `WeeklyReportGenerator2`
Generates a simplified weekly report by pulling Jira issues directly from `JiraIssueEmbeddingDB`.

```python
from error_log_monitor.weekly_report_2 import WeeklyReportGenerator2
from error_log_monitor.config import load_config
from datetime import datetime, timedelta

config = load_config()
reporter = WeeklyReportGenerator2(config)
start = datetime.utcnow() - timedelta(days=7)
end = datetime.utcnow()
report = reporter.generate_weekly_report(start, end)
print(report["combined_excel_path"])
```

## 🐳 Docker Support

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Building Custom Images

```bash
# Build the application image
docker build -t vortex-error-sentinel .

# Run with custom configuration
docker run -d --env-file .env vortex-error-sentinel
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/error_log_monitor

# Run specific test file
pytest tests/test_rag_engine.py
```

### Test Configuration

```bash
# Test OpenSearch connection
python -c "from src.error_log_monitor.opensearch_client import OpenSearchClient; print('OpenSearch OK')"

# Test Jira connection
python -c "from src.error_log_monitor.jira_cloud_client import JiraCloudClient; print('Jira OK')"

# Test RDS connection
python -c "from src.error_log_monitor.rds_client import RDSClient; print('RDS OK')"
```

## 📊 Monitoring and Logging

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General system information
- `WARNING`: Warning messages
- `ERROR`: Error conditions
- `CRITICAL`: Critical system failures

### Log Files

- `logs/error_monitor.log` - Main application logs
- `logs/jira_embedding.log` - Jira embedding database logs
- `logs/rag_engine.log` - RAG engine logs
- `logs/weekly_report.log` - Report generation logs

### Metrics

The system tracks various metrics:

- Error log processing rate
- Jira issue correlation success rate
- OpenAI API usage and costs
- Database query performance
- Embedding generation time

## 🔒 Security

### API Key Management

- Store API keys in environment variables
- Use `.env` files for local development
- Never commit API keys to version control

### Database Security

- Use read-only database connections where possible
- Implement proper authentication and authorization
- Encrypt sensitive data in transit and at rest

### Network Security

- Use HTTPS/TLS for all external connections
- Implement proper firewall rules
- Monitor network traffic for anomalies

## 🚨 Troubleshooting

### Common Issues

1. **OpenSearch Connection Failed**
   - Check OpenSearch host and credentials
   - Verify network connectivity
   - Ensure SSL certificates are valid

2. **Jira API Authentication Failed**
   - Verify Jira credentials and API token
   - Check project key and permissions
   - Ensure custom fields are properly configured

3. **RDS Connection Timeout**
   - Check RDS host and port
   - Verify database credentials
   - Ensure security groups allow connections

4. **OpenAI API Rate Limits**
   - Check API key validity
   - Monitor usage and costs
   - Implement rate limiting if needed

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python src/error_log_monitor/main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful AI models
- Atlassian for Jira Cloud API
- OpenSearch community for excellent search capabilities
- PostgreSQL community for robust database support

## 📞 Support

For support and questions:

- Create an issue in the GitHub repository
- Check the documentation in the `docs/` directory
- Review the troubleshooting section above

---

**Vortex Error Message Sentinel** - Protecting your production services with intelligent error analysis and automated issue tracking.
