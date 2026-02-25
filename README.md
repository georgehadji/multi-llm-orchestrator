# multi-llm-orchestrator

Multi-provider LLM orchestrator with intelligent routing, cost optimization, and cross-provider resilience.

Decomposes project specs → Routes tasks to optimal providers → Executes generate→critique→revise cycles → Evaluates quality.

**Key Features:** 7 LLM providers • Cost-optimized routing • Budget hierarchy • Resume capability • Deterministic validation • Policy enforcement • Real-time telemetry

---

## Quick Start

### Install

```bash
pip install -e .                    # Core + aiosqlite
pip install pytest ruff jsonschema  # Optional validators
```

### Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."   # Recommended (best cost/quality ratio)
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create `.env` and load it.

### First Project

```bash
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT authentication" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 2.0
```

Output saved to `./results/`

---

## Providers & Models

| Provider | Models | Cost (per 1M tokens input) |
|----------|--------|--------------------------|
| **DeepSeek** | Chat (V3), Reasoner (R1) | $0.27–$0.55 ⭐ |
| **Kimi** | K2.5 (8K/32K/128K) | $0.14 |
| **Minimax** | Minimax-3 (frontier reasoning) | $0.50 |
| **Zhipu** | GLM-4 (general purpose) | $1.00 |
| **OpenAI** | GPT-4o, GPT-4o-mini | $0.15–$2.50 |
| **Google** | Gemini 2.5 Pro, Flash | $0.15–$1.25 |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | $0.80–$3.00 |

---

## Core Capabilities

- **Multi-Provider Routing:** Task-aware model selection with fallback chains
- **Cost Optimization:** EMA-tracked pricing, budget hierarchy (org/team/job)
- **Quality Assurance:** Deterministic validators + multi-round critique + LLM scoring
- **Resilience:** Circuit breaker (3-strike) + cross-provider fallback chains
- **Resume:** Auto-detect similar projects, resumable by project ID
- **Policy Control:** HARD/SOFT/MONITOR enforcement for compliance
- **Observability:** OpenTelemetry tracing, real-time telemetry, event hooks

---

## CLI Examples

### 1. Build a FastAPI Service (Default)

```bash
python -m orchestrator \
  --project "FastAPI authentication with JWT tokens" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 3.0
```

### 2. Build a Next.js App

```bash
python -m orchestrator \
  --project "Next.js e-commerce storefront with product listing and cart" \
  --criteria "npm build succeeds, no console errors" \
  --output-dir ./storefront
```

### 3. Resume Interrupted Run

```bash
python -m orchestrator --resume <project_id>
```

### 4. Skip Enhancement & Resume Detection

```bash
python -m orchestrator \
  --project "Build a React dashboard" \
  --criteria "npm build succeeds" \
  --no-enhance --new-project
```

---

## Python API

```python
import asyncio
from orchestrator import Orchestrator, Budget

async def main():
    budget = Budget(max_usd=5.0, max_time_seconds=3600)
    orch = Orchestrator(budget=budget)

    state = await orch.run_project(
        project_description="Implement a Python rate limiter using decorators",
        success_criteria="pytest suite passes, ruff linting clean",
    )

    print(f"Status: {state.status.value}")
    print(f"Spent: ${state.budget.spent_usd:.4f}")
    print(f"Score: {state.overall_quality_score:.3f}")

asyncio.run(main())
```

---

## Documentation

- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** — CLI reference, Python API examples, best practices
- **[CAPABILITIES.md](./CAPABILITIES.md)** — Feature deep-dive, architecture details, advanced recipes

---

## Architecture Overview

```
Project Description
       ↓
Auto-Resume Detect → Project Enhancer → Architecture Advisor
       ↓
Decompose into Tasks
       ↓
Route → Generate → Critique → Revise → Evaluate
       ↓
Cross-Provider Fallback Chain (quality escalation)
       ↓
Deterministic Validation (python_syntax, pytest, ruff, json_schema)
       ↓
Store Results + Telemetry + State Checkpoint (SQLite)
```

---

## Testing

```bash
pytest tests/                  # Full suite (152+ tests)
pytest tests/test_opus_removal.py -v  # Verify Opus removal
pytest --tb=short -q          # Summary
```

---

## Configuration

All configuration can be set via environment variables or Python API:

```bash
# Required: at least one provider key
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIzaSy..."
export KIMI_API_KEY="sk-..."
export MINIMAX_API_KEY="..."
export ZHIPU_API_KEY="..."

# Optional
export ORCHESTRATOR_CACHE_DIR="~/.orchestrator_cache"
export ORCHESTRATOR_LOG_LEVEL="INFO"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

---

## Project Status

- **Baseline:** 152+ tests passing, 0 regressions
- **Last Updated:** 2026-02-26

---

## Next Steps

1. Install & set API key (see Quick Start above)
2. Run first project: `python -m orchestrator --project "..."`
3. Check [USAGE_GUIDE.md](./USAGE_GUIDE.md) for CLI reference
4. See [CAPABILITIES.md](./CAPABILITIES.md) for advanced features

---

**Questions?** Check the docs or run `python -m orchestrator --help`
