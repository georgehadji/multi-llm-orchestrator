# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-29%2F29%20passing-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%7C%20black%20%7C%20mypy-blue.svg)]()

> **Autonomous Software Development Platform** with multi-provider LLM orchestration, intelligent routing, and cross-provider resilience.

Decomposes project specifications → Routes tasks to optimal providers → Executes generate→critique→revise cycles → Evaluates quality with deterministic validation.

---

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🎯 **Smart Routing** | Intelligent model selection with cost optimization (-35%) | ✅ |
| 💰 **Budget Management** | Hierarchical budgets (org/team/job) with mid-task enforcement | ✅ |
| 🔄 **Cross-Provider Fallback** | Automatic failover with native OpenRouter fallbacks | ✅ |
| 📊 **Real-time Telemetry** | Cost tracking, latency monitoring, quality metrics | ✅ |
| 🧪 **Deterministic Validation** | Syntax checkers, test runners, security scans | ✅ |
| 🚀 **OpenRouter Optimizations** | JSON Schema, model variants, provider sorting | ✅ NEW |
| 🔍 **Nexus Search** | Self-hosted web search with hybrid RRF ranking | ✅ |
| 🧠 **ARA Pipeline** | 12 Advanced Reasoning Methods (Debate, Jury, etc.) | ✅ |
| 📱 **iOS Suite** | App Store compliance with 6 enhancement modules | ✅ |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/georgehadji/multi-llm-orchestrator.git
cd multi-llm-orchestrator

# Install with all dependencies
pip install -e ".[dev,security,tracing]"

# Or minimal install
pip install -e .
```

### Configuration

Create `.env` file with your API keys:

```bash
# Required: At least one provider
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: OpenRouter optimizations
export USE_JSON_SCHEMA_RESPONSES="true"
export USE_MODEL_VARIANTS="true"
export USE_NATIVE_FALLBACKS="true"
```

### First Run

```bash
# Simple project
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT auth" \
  --budget 2.0

# With success criteria
python -m orchestrator \
  --project "Create a React dashboard" \
  --criteria "TypeScript, responsive, dark mode" \
  --budget 5.0
```

Results saved to `./results/`

---

## 📊 Dashboard

Launch the web UI for visual project management:

```bash
# Windows
start_dashboard.bat

# Linux/Mac
python start_dashboard.py
```

**Features:**
- ✅ Create projects from natural language
- ✅ Refactor existing codebases  
- ✅ Upload YAML/JSON specifications
- ✅ Real-time progress & logs
- ✅ Multiple concurrent projects

**URL:** http://localhost:8888

---

## 🏗️ Architecture

```
Project Description
       │
       ▼
┌─────────────────┐     ┌─────────────────┐
│ Auto-Resume     │────►│  Enhancement    │
│ Detection       │     │  & Advisory     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         DECOMPOSITION ENGINE            │
│  Breaks projects into atomic tasks      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           EXECUTION PIPELINE            │
│  Route → Generate → Critique → Revise   │
│         ↓                               │
│  Evaluate + Deterministic Validation    │
│         ↓                               │
│  Cross-Provider Fallback Chain          │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         RESULTS + TELEMETRY             │
│  Code • Tests • Docs • Metrics          │
└─────────────────────────────────────────┘
```

See [CODEBASE_MINDMAP.md](docs/CODEBASE_MINDMAP.md) for complete architecture.

---

## 🔧 OpenRouter Optimizations (NEW)

Phase 1 & 2 optimizations for OpenRouter API:

| Optimization | Impact | Env Var |
|--------------|--------|---------|
| **JSON Schema** | -50% parsing errors | `USE_JSON_SCHEMA_RESPONSES=true` |
| **Model Variants** | Cost savings with `:free` tier | `USE_MODEL_VARIANTS=true` |
| **Native Fallbacks** | +30% fallback success | `USE_NATIVE_FALLBACKS=true` |
| **Provider Sorting** | +40% throughput | `USE_PROVIDER_SORTING=true` |

### Canary Deployment

```python
from orchestrator.canary_deployment import get_canary_deployment

canary = get_canary_deployment()
await canary.start_rollout(
    optimization="json_schema",
    stages=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0],
    thresholds={"max_error_rate": 0.05}
)
```

See [OPENROUTER_IMPLEMENTATION_COMPLETE.md](OPENROUTER_IMPLEMENTATION_COMPLETE.md)

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=orchestrator --cov-report=html

# Specific test suites
pytest tests/test_openrouter_optimizations.py -v
pytest tests/test_openrouter_phase2.py -v
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Comprehensive usage examples |
| [CAPABILITIES.md](CAPABILITIES.md) | Feature capabilities matrix |
| [METHODS.md](METHODS.md) | API reference & methods |
| [DESIGN.md](DESIGN.md) | Architecture & design decisions |
| [AGENTS.md](AGENTS.md) | AI agent guidelines |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev,security,tracing]"

# Run linting
ruff check orchestrator/
black orchestrator/ tests/
mypy orchestrator/

# Run security scan
bandit -r orchestrator/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Author:** Georgios-Chrysovalantis Chatzivantsidis
- **Version:** 6.3.0
- **Last Updated:** 2026-04-05

---

<p align="center">
  <b>Built with ❤️ for autonomous software development</b>
</p>
