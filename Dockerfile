FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only db_schema (src will be mounted from outside)
COPY db_schema/ ./db_schema/

# Create necessary directories
RUN mkdir -p /app/data/chroma_db /app/reports /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check (optional - can be disabled)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -m error_log_monitor.main test --site dev || exit 1

# Default command - do nothing, wait for manual commands
CMD ["sleep", "infinity"]
