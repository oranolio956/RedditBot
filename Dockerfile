# Multi-stage Production Dockerfile for AI Conversation System
# Optimized for performance, security, and production scaling

# ==================== Build Stage ====================
FROM python:3.11-slim as builder

# Build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDARCH
ARG TARGETARCH

# Environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    # ML library dependencies
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    # Security and networking
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    # Database drivers
    libpq-dev \
    # Git for package installation
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and essential tools
RUN pip install --no-cache-dir --upgrade \
    pip==23.3.1 \
    wheel==0.41.2 \
    setuptools==68.2.2

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install ML dependencies with CPU optimization
RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Production Stage ====================
FROM python:3.11-slim as production

# Security labels
LABEL maintainer="AI Conversation System" \
      version="1.0.0" \
      description="Production-ready AI conversation system" \
      security.scan="enabled"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Performance optimization
    PYTHONHASHSEED=random \
    # Security
    PYTHONPATH=/app \
    # Production settings
    ENVIRONMENT=production \
    # OMP settings for ML performance
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Create non-root user with specific UID/GID for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser && \
    # Create app directory structure
    mkdir -p /app /app/logs /app/models /app/data /app/temp /app/static && \
    chown -R appuser:appuser /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libpq5 \
    libgomp1 \
    libopenblas0 \
    libhdf5-103 \
    # Networking and security
    curl \
    ca-certificates \
    # For health checks
    wget \
    # Process management
    tini \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # Security hardening
    && chmod u+s /bin/ping

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Create startup script for better process management
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

# Wait for database if DB_HOST is provided
if [ -n "$DB_HOST" ]; then
    echo "Waiting for database at $DB_HOST:$DB_PORT..."
    while ! timeout 1 bash -c "echo > /dev/tcp/$DB_HOST/$DB_PORT" 2>/dev/null; do
        sleep 1
    done
    echo "Database is ready!"
fi

# Wait for Redis if REDIS_HOST is provided
if [ -n "$REDIS_HOST" ]; then
    echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    while ! timeout 1 bash -c "echo > /dev/tcp/$REDIS_HOST/$REDIS_PORT" 2>/dev/null; do
        sleep 1
    done
    echo "Redis is ready!"
fi

# Run database migrations
if [ "$1" = "web" ] || [ "$1" = "uvicorn" ]; then
    echo "Running database migrations..."
    alembic upgrade head || echo "Migration failed or not needed"
fi

# Execute the main command
exec "$@"
EOF

# Make startup script executable
RUN chmod +x /app/start.sh && chown appuser:appuser /app/start.sh

# Switch to non-root user
USER appuser

# Health check with multiple endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health && \
        curl -f http://localhost:8000/ready || exit 1

# Expose ports
EXPOSE 8000 8001

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/app/start.sh"]

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ==================== Development Stage ====================
FROM production as development

# Switch back to root for development dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    # Development tools
    git \
    vim \
    htop \
    # Debugging tools
    strace \
    tcpdump \
    netcat \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest-xdist \
    pytest-benchmark

# Switch back to app user
USER appuser

# Development command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ==================== Worker Stage ====================
FROM production as worker

# Worker-specific environment
ENV WORKER_TYPE=celery

# Default worker command
CMD ["celery", "-A", "app.worker", "worker", "--loglevel=info", "--concurrency=4", "--prefetch-multiplier=1"]

# ==================== Scheduler Stage ====================
FROM production as scheduler

# Scheduler-specific environment
ENV WORKER_TYPE=scheduler

# Default scheduler command
CMD ["celery", "-A", "app.worker", "beat", "--loglevel=info"]

# ==================== Monitoring Stage ====================
FROM production as monitoring

# Install monitoring dependencies
USER root
RUN /opt/venv/bin/pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    statsd==4.0.1 \
    psutil==5.9.6

USER appuser

# Monitoring command
CMD ["python", "-m", "app.monitoring.server"]