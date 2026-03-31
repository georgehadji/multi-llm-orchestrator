# AI Orchestrator — Deployment Guide

**Version:** 1.0.0 | **Date:** 2026-03-25

> **Production deployment guide** for AI Orchestrator v1.0.0

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Production Checklist](#production-checklist)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.12 |
| **RAM** | 4GB | 8GB |
| **Storage** | 10GB | 20GB |
| **CPU** | 2 cores | 4+ cores |

### Required Accounts

- xAI API account (for Grok models)
- At least one additional LLM provider (OpenAI, Anthropic, Google, DeepSeek)
- Optional: Nexus Search (self-hosted)

---

## Installation

### Option 1: PyPI (When Published)

```bash
# Install from PyPI
pip install ai-orchestrator

# Verify installation
python -m orchestrator --version
```

### Option 2: From Source

```bash
# Clone repository
git clone https://github.com/georrgehadji/multi-llm-orchestrator.git
cd multi-llm-orchestrator

# Install in editable mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev,security,tracing,dashboard]"
```

### Option 3: Docker (Recommended for Production)

```bash
# Build Docker image
docker build -t ai-orchestrator:1.0.0 .

# Run container
docker run -d \
  --name orchestrator \
  -p 8000:8000 \
  -e DEEPSEEK_API_KEY="sk-..." \
  -e OPENAI_API_KEY="sk-..." \
  ai-orchestrator:1.0.0
```

---

## Configuration

### Environment Variables

#### Required

```bash
# At least one API key required
export DEEPSEEK_API_KEY="sk-..."      # Recommended (best value)
export OPENAI_API_KEY="sk-..."        # GPT-4o, GPT-4o-mini
export ANTHROPIC_API_KEY="sk-ant-..." # Claude models
export GOOGLE_API_KEY="AIzaSy..."    # Gemini models
export XAI_API_KEY="xai-..."          # Grok models
```

#### Optional

```bash
# Nexus Search
export NEXUS_SEARCH_ENABLED=true
export NEXUS_API_URL="http://localhost:8080"

# X Search
export X_SEARCH_ENABLED=true

# Rate Limiting
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_TIER=auto  # auto, or specific tier (1-6)

# Provisioned Throughput
export PROVISIONED_THROUGHPUT_ENABLED=true
export PROVISIONED_UNITS=4

# Logging
export ORCHESTRATOR_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"  # OpenTelemetry
```

### Configuration File

Create `.env` file in project root:

```bash
# .env
DEEPSEEK_API_KEY="sk-..."
OPENAI_API_KEY="sk-..."
XAI_API_KEY="xai-..."
NEXUS_SEARCH_ENABLED=true
ORCHESTRATOR_LOG_LEVEL=INFO
```

Load with:

```bash
# Python
from dotenv import load_dotenv
load_dotenv()

# Or use python-dotenv
python -m dotenv run python -m orchestrator
```

---

## Deployment Options

### Option 1: Single Server

**Best for:** Small teams, development, testing

```bash
# Install
pip install -e .

# Set environment variables
export DEEPSEEK_API_KEY="sk-..."

# Run
python -m orchestrator --project "..." --criteria "..." --budget 5.0
```

---

### Option 2: Docker Compose

**Best for:** Production with dependencies

```yaml
# docker-compose.yml
version: '3.8'

services:
  orchestrator:
    image: ai-orchestrator:1.0.0
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - XAI_API_KEY=${XAI_API_KEY}
    volumes:
      - ./results:/app/results
      - ./cache:/app/.cache
    ports:
      - "8000:8000"
    restart: unless-stopped

  nexus-search:
    image: searxng/searxng:latest
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080
    ports:
      - "8080:8080"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

Deploy:

```bash
docker-compose up -d
```

---

### Option 3: Kubernetes

**Best for:** Large-scale production

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: ai-orchestrator:1.0.0
        env:
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: deepseek
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

Deploy:

```bash
kubectl apply -f k8s/
```

---

### Option 4: Cloud Platforms

#### AWS

```bash
# Using ECS
aws ecs create-cluster --cluster-name orchestrator
aws ecs create-service --cluster orchestrator --service-name orchestrator-service

# Using Lambda (for serverless)
# Package as Lambda layer
zip -r orchestrator.zip orchestrator/
aws lambda create-function --function-name orchestrator --runtime python3.12 ...
```

#### GCP

```bash
# Using Cloud Run
gcloud run deploy orchestrator \
  --image gcr.io/PROJECT_ID/orchestrator:1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure

```bash
# Using Azure Container Instances
az container create \
  --resource-group orchestrator-rg \
  --name orchestrator \
  --image ai-orchestrator:1.0.0 \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

---

## Production Checklist

### Pre-Deployment

- [ ] All API keys configured
- [ ] Environment variables set
- [ ] Database initialized (SQLite/PostgreSQL)
- [ ] Cache configured (Redis optional)
- [ ] Logging configured
- [ ] Monitoring enabled (OpenTelemetry optional)
- [ ] Rate limiting configured
- [ ] Provisioned throughput configured (if using)

### Security

- [ ] API keys stored in secrets manager
- [ ] HTTPS enabled
- [ ] CORS configured
- [ ] Rate limiting enabled
- [ ] Input validation enabled
- [ ] Output sanitization enabled

### Performance

- [ ] Caching enabled
- [ ] Connection pooling configured
- [ ] Timeout settings configured
- [ ] Load balancing configured (if multi-instance)

### Monitoring

- [ ] Health checks configured
- [ ] Metrics collection enabled
- [ ] Log aggregation configured
- [ ] Alert rules configured
- [ ] Dashboard configured

### Backup & Recovery

- [ ] Database backup configured
- [ ] Backup schedule defined
- [ ] Recovery procedure documented
- [ ] Disaster recovery plan documented

---

## Monitoring & Maintenance

### Health Checks

```bash
# Health endpoint
curl http://localhost:8000/health

# Ready endpoint
curl http://localhost:8000/ready

# Metrics endpoint (if enabled)
curl http://localhost:8000/metrics
```

### Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **API Latency (P95)** | <500ms | >1000ms |
| **Error Rate** | <1% | >5% |
| **Cache Hit Rate** | >70% | <50% |
| **Rate Limit Hits** | <10%/hour | >100/hour |
| **CPU Usage** | <70% | >90% |
| **Memory Usage** | <70% | >90% |

### Maintenance Tasks

#### Daily

- [ ] Check error logs
- [ ] Review rate limit hits
- [ ] Check API spend

#### Weekly

- [ ] Review performance metrics
- [ ] Check cache effectiveness
- [ ] Review rate limiter tiers

#### Monthly

- [ ] Update dependencies
- [ ] Review API costs
- [ ] Check provisioned capacity utilization
- [ ] Review and optimize queries

---

## Troubleshooting

### Common Issues

#### Issue: API Key Errors

**Symptoms:**
```
Error: Invalid API key
```

**Solution:**
```bash
# Verify API key is set
echo $DEEPSEEK_API_KEY

# Check .env file
cat .env

# Restart service
docker-compose restart
```

---

#### Issue: Rate Limit Exceeded

**Symptoms:**
```
Error: Rate limit exceeded (tier 1)
```

**Solution:**
```bash
# Check current tier
python -c "from orchestrator.rate_limiter import get_rate_limiter; print(get_rate_limiter().get_stats())"

# Wait for reset (60 seconds)
# Or upgrade tier by increasing spend
```

---

#### Issue: High Latency

**Symptoms:**
```
Warning: Request took 5000ms
```

**Solution:**
```bash
# Check Nexus Search health
curl http://localhost:8080/healthz

# Check cache hit rate
python -c "from orchestrator.nexus_search.optimization import get_query_cache; print(get_query_cache().get_stats())"

# Enable/verify caching
export NEXUS_CACHE_ENABLED=true
```

---

#### Issue: Out of Memory

**Symptoms:**
```
Error: MemoryError
```

**Solution:**
```bash
# Reduce concurrency
export ORCHESTRATOR_MAX_CONCURRENCY=3

# Clear cache
python -c "from orchestrator.nexus_search.optimization import get_query_cache; get_query_cache().clear()"

# Increase memory limit (Docker)
docker update --memory=4g orchestrator
```

---

### Getting Help

- **Documentation:** https://github.com/georrgehadji/multi-llm-orchestrator/docs
- **Issues:** https://github.com/georrgehadji/multi-llm-orchestrator/issues
- **Discussions:** https://github.com/georrgehadji/multi-llm-orchestrator/discussions

---

## Post-Deployment Verification

### Run Verification Script

```bash
# Run all checks
python scripts/verify_deployment.py

# Expected output:
# ✅ API keys configured
# ✅ Database initialized
# ✅ Cache working
# ✅ Health check passing
# ✅ All systems operational
```

### Test Basic Functionality

```bash
# Run a simple project
python -m orchestrator \
  --project "Hello World API" \
  --criteria "Returns 200 OK" \
  --budget 1.0

# Verify output in ./results/
```

---

## Rollback Procedure

If deployment fails:

```bash
# Docker rollback
docker-compose down
docker-compose up -d ai-orchestrator:0.9.0  # Previous version

# Kubernetes rollback
kubectl rollout undo deployment/ai-orchestrator

# Verify rollback
curl http://localhost:8000/health
```

---

**Deployment Complete!** ✅

For ongoing support, refer to the [Monitoring & Maintenance](#monitoring--maintenance) section.

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
