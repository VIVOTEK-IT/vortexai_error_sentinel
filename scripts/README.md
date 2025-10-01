# Jira Issue Embedding Database Scripts

This directory contains scripts for managing the Jira Issue Embedding Database.

## Scripts Overview

### 1. `init_jira_db.py` - Simple Initialization Script

A simple script to initialize the Jira Issue Embedding Database with sample data.

**Usage:**
```bash
# Dry run (preview only)
python scripts/init_jira_db.py --dry-run

# Initialize with specific sites
python scripts/init_jira_db.py --sites prod,stage --months 6

# Initialize with all default settings
python scripts/init_jira_db.py
```

**Options:**
- `--dry-run`: Run without making changes (preview only)
- `--sites SITES`: Comma-separated list of sites (default: prod,stage)
- `--months MONTHS`: Number of months of error logs to process (default: 6)

### 2. `initialize_jira_embedding_db.py` - Full-Featured Initialization Script

A comprehensive script with advanced features for production use.

**Usage:**
```bash
# Dry run with verbose logging
python scripts/initialize_jira_embedding_db.py --dry-run --verbose

# Initialize with specific configuration
python scripts/initialize_jira_embedding_db.py --sites prod --months 3 --batch-size 50

# Show help
python scripts/initialize_jira_embedding_db.py --help
```

**Options:**
- `--dry-run`: Run without making changes (preview only)
- `--sites SITES`: Comma-separated list of sites to process
- `--months MONTHS`: Number of months of error logs to process (default: 6)
- `--batch-size SIZE`: Batch size for processing (default: 100)
- `--verbose`: Enable verbose logging
- `--help`: Show help message

### 3. `migrate_jira_embeddings.py` - Data Migration Script

Script for migrating Jira issue embeddings between year-based indices.

**Usage:**
```bash
# Show help
python scripts/migrate_jira_embeddings.py --help
```

### 4. `monitor_jira_embeddings.py` - Monitoring Script

Script for monitoring and health checking the Jira Issue Embedding Database.

**Usage:**
```bash
# Check database health
python scripts/monitor_jira_embeddings.py health

# Get performance metrics
python scripts/monitor_jira_embeddings.py metrics

# Get database summary
python scripts/monitor_jira_embeddings.py summary

# Show help
python scripts/monitor_jira_embeddings.py --help
```

## Docker Usage

All scripts can be run inside the Docker container using the mounted `/app/scripts` directory:

```bash
# Run simple initialization
docker-compose exec error-monitor python /app/scripts/init_jira_db.py --dry-run

# Run full initialization
docker-compose exec error-monitor python /app/scripts/initialize_jira_embedding_db.py --dry-run --verbose

# Run migration
docker-compose exec error-monitor python /app/scripts/migrate_jira_embeddings.py list

# Run monitoring
docker-compose exec error-monitor python /app/scripts/monitor_jira_embeddings.py health
```

**Note**: The scripts directory is mounted at `/app/scripts` inside the container, so use the full path when running scripts.

## Prerequisites

Before running the scripts, ensure:

1. **Environment Variables**: All required environment variables are set in `.env`
2. **OpenSearch**: OpenSearch is running and accessible
3. **OpenAI API**: OpenAI API key is configured for embedding generation
4. **Jira API**: Jira API credentials are configured (for real data fetching)

## Sample Data

The simple initialization script (`init_jira_db.py`) uses sample data for testing. For production use, you would need to:

1. Implement real data fetching from Jira API
2. Implement real data fetching from OpenSearch error logs
3. Configure proper site detection logic
4. Set up proper error handling for API failures

## Error Handling

All scripts include comprehensive error handling:

- **Configuration Errors**: Clear messages for missing configuration
- **API Errors**: Graceful handling of API failures
- **Data Errors**: Validation and error reporting for invalid data
- **Network Errors**: Retry logic and timeout handling

## Logging

Scripts support different logging levels:

- **INFO**: Basic progress information
- **DEBUG**: Detailed debugging information (use `--verbose`)
- **WARNING**: Non-fatal issues
- **ERROR**: Fatal errors that stop execution

## Examples

### Initialize Database with Sample Data
```bash
# Preview what would be done (local)
python scripts/init_jira_db.py --dry-run

# Actually initialize the database (local)
python scripts/init_jira_db.py

# Or using Docker
docker-compose exec error-monitor python /app/scripts/init_jira_db.py --dry-run
```

### Initialize with Real Data (Production)
```bash
# Preview with real data (local)
python scripts/initialize_jira_embedding_db.py --dry-run --verbose

# Initialize with real data (local)
python scripts/initialize_jira_embedding_db.py --sites prod,stage --months 6

# Or using Docker
docker-compose exec error-monitor python /app/scripts/initialize_jira_embedding_db.py --dry-run --verbose
```

### Monitor Database Health
```bash
# Check overall health (local)
python scripts/monitor_jira_embeddings.py health

# Get detailed metrics (local)
python scripts/monitor_jira_embeddings.py metrics

# Or using Docker
docker-compose exec error-monitor python /app/scripts/monitor_jira_embeddings.py health
```

### Migrate Data Between Years
```bash
# List available years (local)
python scripts/migrate_jira_embeddings.py list

# Migrate 2023 data to 2024 (local)
python scripts/migrate_jira_embeddings.py migrate --from-year 2023 --to-year 2024

# Or using Docker
docker-compose exec error-monitor python /app/scripts/migrate_jira_embeddings.py list
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root or using Docker
2. **Configuration Errors**: Check that all environment variables are set
3. **API Errors**: Verify API credentials and network connectivity
4. **Permission Errors**: Ensure scripts have execute permissions

### Debug Mode

Use `--verbose` flag for detailed logging:

```bash
python scripts/initialize_jira_embedding_db.py --dry-run --verbose
```

### Docker Debug

For debugging inside Docker:

```bash
# Get shell access
docker-compose exec error-monitor bash

# Run script with debug
python scripts/init_jira_db.py --dry-run
```
