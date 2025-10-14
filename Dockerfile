FROM public.ecr.aws/lambda/python:3.11

# Optional: install minimal build tools if a package truly needs build (avoid where possible)
RUN yum -y install make && yum clean all || true

# Copy requirements early for caching
COPY requirements.txt  ./

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Preinstall numpy as wheel to avoid compiling
RUN python -m pip install --no-cache-dir --only-binary=:all: numpy==1.26.4 --target "${LAMBDA_TASK_ROOT}"

# Install the rest of dependencies (allow sdists for pure-Python packages like pypika)
RUN python -m pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code into Lambda task root
COPY src/ ${LAMBDA_TASK_ROOT}/
COPY db_schema/ ${LAMBDA_TASK_ROOT}/db_schema/

# Provide writable temp/report locations for Lambda
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV TMPDIR=/tmp

# Default Lambda handler (override in Lambda console if needed)
CMD ["error_log_monitor.lambda_daily_report.lambda_handler"]
