# Integration Guide — Multi-LLM Orchestrator

**Version:** 6.0.0  
**Updated:** 2026-05-27  
**Audience:** Developers integrating the orchestrator into existing workflows

---

## Table of Contents

1. [Python API Integration](#1-python-api-integration)
2. [CLI Integration](#2-cli-integration)
3. [IDE Integration](#3-ide-integration)
4. [CI/CD Integration](#4-cicd-integration)
5. [External Tool Integration](#5-external-tool-integration)
6. [Observability Integration](#6-observability-integration)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Python API Integration

### Basic Usage

```python
import asyncio
from orchestrator import Orchestrator
from orchestrator.budget import Budget

async def run_project():
    orch = Orchestrator(
        budget=Budget(max_usd=5.0),
        max_parallel_tasks=3,
    )

    async with orch:
        result = await orch.run_project(
            project_description="Build a FastAPI todo app with SQLite",
            success_criteria="All CRUD endpoints working",
        )
        print(f"Project ID: {result.project_id}")
        print(f"Status: {result.status}")
        print(f"Tasks completed: {len(result.results)}")
        return result

result = asyncio.run(run_project())
```

### With Custom Cache

```python
from orchestrator import Orchestrator
from orchestrator.cache import DiskCache

cache = DiskCache(cache_dir="/path/to/cache")
orch = Orchestrator(
    cache=cache,
    budget=Budget(max_usd=10.0),
)
```

### With ServiceContainer (for DI)

```python
from orchestrator import Orchestrator
from orchestrator.engine_core.container import ServiceContainer
from orchestrator.domain.ports import NullCache, NullState

container = ServiceContainer.build(
    budget=Budget(max_usd=5.0),
    cache=NullCache(),       # No-op cache for testing
    state_manager=NullState(),  # In-memory state
    max_concurrency=2,
)

orch = Orchestrator(container=container)
```

### Running a Single Task

```python
from orchestrator.models import Task, TaskType, Model
from orchestrator import Orchestrator

async def run_single_task():
    async with Orchestrator(budget=Budget(max_usd=1.0)) as orch:
        task = Task(
            id="t_001",
            type=TaskType.CODE_GEN,
            prompt="Write a Python function to sort a list",
            model=Model.DEEPSEEK_V4_FLASH,
        )
        result = await orch._execute_task(task)
        print(f"Output: {result.output}")
        print(f"Quality score: {result.score}")
        print(f"Cost: ${result.cost_usd:.4f}")
```

---

## 2. CLI Integration

### Scripting with Shell

```bash
# Run project and capture project ID
PROJECT_ID=$(python -m orchestrator \
    --project "Build a CLI tool" \
    --budget 2.0 \
    --dry-run \
    2>&1 | grep "Project ID" | awk '{print $NF}')

echo "Created project: $PROJECT_ID"

# Resume on failure
python -m orchestrator --resume "$PROJECT_ID" --budget 3.0
```

### JSON Output for Automation

```bash
# Get project list as JSON (pipe through custom scripts)
python -m orchestrator --list-projects 2>/dev/null

# Check status
python -m orchestrator dashboard --days 7
```

### Webhook Integration

The orchestrator supports webhook-style hooks:

```python
from orchestrator import Orchestrator
from orchestrator.events.core import EventType

async def run_with_hooks():
    async with Orchestrator(budget=Budget(max_usd=5.0)) as orch:
        # Register lifecycle hooks
        orch._hook_registry.on(EventType.TASK_COMPLETED, lambda **kw: print(f"Task done: {kw.get('task_id')}"))
        orch._hook_registry.on(EventType.BUDGET_WARNING, lambda **kw: print(f"Budget alert: {kw}"))

        result = await orch.run_project(
            project_description="Build a login system",
            success_criteria="Registration and login working",
        )
```

---

## 3. IDE Integration

### IDE Backend Server

The orchestrator provides a WebSocket-based IDE backend for integration with VS Code, JetBrains, or custom editors.

```bash
# Start the IDE backend
python -m orchestrator.ide_backend.server

# Default: WebSocket on port 8181, REST API on port 8000
```

### Programmatic IDE Connection

```python
import asyncio
import json
import websockets

async def connect_to_orchestrator():
    async with websockets.connect("ws://localhost:8181/ws") as ws:
        # Send a task
        await ws.send(json.dumps({
            "type": "execute",
            "task_id": "t_001",
            "description": "Refactor this function",
            "code": "def old(): pass",
        }))

        # Receive progress
        async for message in ws:
            data = json.loads(message)
            print(f"[{data.get('type')}] {data.get('content', '')}")

asyncio.run(connect_to_orchestrator())
```

---

## 4. CI/CD Integration

### GitHub Actions

The repository includes `.github/workflows/ci.yml` with:

- **Lint**: black + ruff formatting check
- **Typecheck**: mypy strict mode
- **Unit tests**: pytest (no API, no slow, no e2e)
- **Contract tests**: Protocol adherence verification

### Git Hooks

```bash
# Set up pre-commit hooks
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
hooks provide: ruff linting + pytest quick-check
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV ORCH_MAX_CONCURRENCY=3
ENV ORCH_DEFAULT_BUDGET_USD=10.0

CMD ["python", "-m", "orchestrator", "--project", "Auto-build"]
```

---

## 5. External Tool Integration

### OpenRouter (Default — No Setup Required)

The orchestrator uses OpenRouter as the single API gateway for all 52 models across 15 providers. Set `OPENROUTER_API_KEY` in `.env`:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

No other provider keys needed.

### Direct Provider Keys (Optional)

If you prefer to bypass OpenRouter and use providers directly, set individual keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export MISTRAL_API_KEY="..."
export XAI_API_KEY="..."
```

These are used as fallbacks or for specialized routing when the OpenRouter path is unavailable.

### Slack Integration

```python
from orchestrator.slack_integration import SlackNotifier

notifier = SlackNotifier(webhook_url="https://hooks.slack.com/...")
notifier.send("Orchestrator run completed: 12 tasks, $2.34 spent")
```

### Redis (for Distributed Event Bus)

```python
from orchestrator.events.core import RedisEventBus

# Requires: pip install redis
bus = RedisEventBus(redis_url="redis://localhost:6379/0")
await bus.connect()

bus.on("task_completed", lambda **kw: print(f"Remote event: {kw}"))
await bus.publish(AgentMessage(sender="orch_a", content="Hello from process A"))

await bus.disconnect()
```

### Tracing Observability

```python
from orchestrator.tracing import configure_tracing, TracingConfig

# Configure Jaeger/Grafana Tempo endpoint
configure_tracing(TracingConfig(
    service_name="orchestrator",
    otlp_endpoint="http://localhost:4317",
    sample_rate=0.1,
))
```

Then run with:

```bash
python -m orchestrator --project "..." --tracing
# Every LLM call emits a span visible in Jaeger
```

---

## 6. Observability Integration

### Structured Logging (structlog)

```python
from orchestrator.logging import configure_logging, get_logger

# Called at entry point
configure_logging(level="DEBUG", json_format=True)

# Used in any module
log = get_logger(__name__)
log.info("task_started", task_id="t_001", model="deepseek-v4-flash")
log.warning("budget_exceeded", phase="generation", spent=12.5, cap=10.0)
```

**Environment variables:**

```bash
export LOG_FORMAT=json          # JSON output (default: console)
export LOG_LEVEL=DEBUG          # Debug verbosity
```

### Telemetry Store

```python
# Query historical model performance
from orchestrator.telemetry_store import TelemetryStore
store = TelemetryStore()
profiles = await store.load_historical_profile(model, task_type)
```

---

## 7. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (recommended, enables all 52 models) |
| `OPENAI_API_KEY` | — | OpenAI direct API key (optional fallback) |
| `DEEPSEEK_API_KEY` | — | DeepSeek direct API key |
| `ANTHROPIC_API_KEY` | — | Anthropic direct API key |
| `GOOGLE_API_KEY` | — | Google direct API key |
| `MISTRAL_API_KEY` | — | Mistral direct API key |
| `XAI_API_KEY` | — | xAI/Grok direct API key |
| `ORCH_MAX_CONCURRENCY` | 3 | Max parallel API calls |
| `ORCH_MAX_PARALLEL_TASKS` | 3 | Max parallel task executions |
| `ORCH_DEFAULT_BUDGET_USD` | 10.0 | Default budget per project |
| `ORCH_CONTEXT_COMPRESSION` | false | Enable prompt compression |
| `LOG_FORMAT` | console | `json` for structured JSON output |
| `LOG_LEVEL` | INFO | Log verbosity (DEBUG, INFO, WARNING) |
| `DASHBOARD_PORT` | 8000 | Dashboard server port |
| `CACHE_TTL_HOURS` | 48 | Cache entry time-to-live |

### Configuration Hierarchy

```
crosscutting/config.py  ← FeatureFlags + OrchestratorSettings (env-backed)
config.py               ← Static defaults (Timeouts, TokenLimits, BudgetDefaults)
nexus_search/config.py  ← Nexus Search subsystem config
```

All configuration can be imported from a single target:

```python
from orchestrator.crosscutting.config import flags, settings, TIMEOUT_SECONDS
```

---

## 8. Troubleshooting

### Common Issues

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `No module named 'openai'` | Missing dependency | `pip install -e ".[dev]"` |
| `OPENROUTER_API_KEY not set` | Missing API key | Set in `.env` or environment |
| `ModuleNotFoundError: No module named 'orchestrator.xxx'` | Broken relative import after migration | Check `orchestrator/xxx` is a shim pointing to correct package |
| `database is locked` | SQLite contention | Already handled by WAL mode + pool (max_parallel_tasks=3) |
| `structlog not installed` | Optional dependency | `pip install structlog` (or `pip install -e ".[dev]"`) |
| `[Errno 111] Connection refused` | Redis not running (RedisEventBus) | Start Redis or use in-memory EventBus |

### Quick Health Check

```bash
# Verify core imports work
python -c "from orchestrator import Orchestrator; print('Core OK')"

# Verify domain ports
python -c "from orchestrator.domain.ports import CachePort, StatePort; print('Ports OK')"

# Verify contract tests pass
pytest tests/contracts/ -v --no-cov

# Test import chain
python -c "from orchestrator.agent_model_registry import AGENT_MODELS; print(f'{len(AGENT_MODELS)} agents')"
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python -m orchestrator --project "Hello World" --budget 1.0 --verbose

# Dry-run to verify decomposition
python -m orchestrator --project "Build API" --dry-run --verbose
```

---

*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
