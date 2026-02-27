# Multi-LLM Orchestrator - Multi-stage Dockerfile
# ================================================

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Builder
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim as builder

# Security: Run as non-root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install dependencies first (better caching)
COPY pyproject.toml README.md ./
COPY orchestrator/__init__.py orchestrator/
RUN pip install --no-cache-dir -e .

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Production
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim as production

# Security: Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup orchestrator/ ./orchestrator/
COPY --chown=appuser:appgroup README.md ./
COPY --chown=appuser:appgroup pyproject.toml ./

# Create necessary directories
RUN mkdir -p outputs logs data && chown -R appuser:appgroup outputs logs data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import orchestrator; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python", "-m", "orchestrator"]
CMD ["--help"]

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Development
# ═══════════════════════════════════════════════════════════════════════════════
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,security]"

# Install additional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Testing
# ═══════════════════════════════════════════════════════════════════════════════
FROM production as test

USER root

# Install test dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy tests
COPY --chown=appuser:appgroup tests/ ./tests/

USER appuser

# Run tests by default
CMD ["pytest", "-v", "--tb=short"]
