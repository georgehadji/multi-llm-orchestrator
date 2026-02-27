# 🐛 Debugging & Troubleshooting Guide
## Multi-LLM Orchestrator - Complete Error Recovery Manual

---

## 📋 Περιεχόμενα

1. [Debugging Workflow](#1-debugging-workflow)
2. [Common Errors & Solutions](#2-common-errors--solutions)
3. [Log Analysis](#3-log-analysis)
4. [Testing & Validation](#4-testing--validation)
5. [Recovery Procedures](#5-recovery-procedures)
6. [Diagnostic Tools](#6-diagnostic-tools)

---

## 1. Debugging Workflow

### 🔍 Standard Debug Process

```
1. Παρατήρηση Συμπτωμάτων
   ↓
2. Συλλογή Logs & Telemetry
   ↓
3. Αναπαραγωγή Προβλήματος
   ↓
4. Root Cause Analysis
   ↓
5. Εφαρμογή Fix
   ↓
6. Επαλήθευση & Monitoring
```

### 🛠️ Debug Mode Activation

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable tracing
from orchestrator import TracingConfig, configure_tracing
configure_tracing(TracingConfig(
    enabled=True,
    console_exporter=True,
    detailed_spans=True,
))

# Run with debug hooks
from orchestrator import Orchestrator, Budget
from orchestrator.hooks import EventType

orch = Orchestrator(budget=Budget(max_usd=5.0))

# Add debug hooks
orch.add_hook(EventType.TASK_STARTED, 
    lambda task_id, model, **_: print(f"[DEBUG] Task {task_id} starting with {model}"))

orch.add_hook(EventType.TASK_FAILED,
    lambda task_id, error, **_: print(f"[DEBUG] Task {task_id} failed: {error}"))
```

---

## 2. Common Errors & Solutions

### ❌ Error Categories

#### A. Installation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependencies | `pip install -e ".[dev]"` |
| `ImportError` | Circular imports | Check `__init__.py` exports |
| `PermissionError` | File access denied | Run as admin / check permissions |
| `WinError 2` | Shell execution failed | Use Python file operations |

**Fix Example:**
```bash
# Full reinstall
pip uninstall multi-llm-orchestrator -y
pip install -e ".[dev,security,tracing]"

# Verify installation
python -c "from orchestrator import Orchestrator; print('✓ OK')"
```

#### B. API/Provider Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `AuthenticationError` | Invalid API key | Check env vars / regenerate keys |
| `RateLimitError` | Too many requests | Add rate limiting / use backoff |
| `ModelUnavailableError` | Model offline | Enable fallback chain |
| `TimeoutError` | Request timeout | Increase timeout / use async |

**Debug API Issues:**
```python
from orchestrator import UnifiedClient, Model

# Test individual provider
client = UnifiedClient(Model.GPT_4O)
try:
    response = await client.generate("Test prompt")
    print(f"✓ Provider works: {response.text[:100]}")
except Exception as e:
    print(f"✗ Provider error: {e}")
```

#### C. Task Execution Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `TaskValidationError` | Output validation failed | Check validator requirements |
| `TaskTimeoutError` | Task exceeded time limit | Optimize prompt / split task |
| `TaskRetryExhaustedError` | Max retries reached | Check model availability |
| `BudgetExceededError` | Cost limit reached | Increase budget / optimize routing |

**Debug Task Failures:**
```python
# Inspect failed task
from orchestrator.state import StateManager

state_mgr = StateManager()
state = state_mgr.load_state(project_id)

for task_id, result in state.results.items():
    if result.error:
        print(f"Task {task_id} failed:")
        print(f"  Error: {result.error}")
        print(f"  Attempts: {len(result.attempts)}")
        print(f"  Final score: {result.score}")
```

#### D. Dashboard Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `WebSocket 403` | Connection rejected | Switch to HTTP polling |
| `Port in use` | Port occupied | Change port: `--port 8889` |
| `ImportError: fastapi` | Missing dependencies | `pip install fastapi uvicorn` |
| `Redis connection failed` | Redis not running | Start redis or use memory cache |

**Fix Dashboard:**
```bash
# Install dashboard dependencies
pip install fastapi uvicorn websockets httpx

# Run with fallback mode
python run_optimized_dashboard.py --port 8888

# Check if working
curl http://localhost:8888/api/metrics
```

---

## 3. Log Analysis

### 📁 Log Locations

```
project_root/
├── logs/
│   ├── orchestrator.log          # Main application log
│   ├── audit.log                 # Audit trail (JSONL)
│   └── telemetry.log             # Telemetry data
├── .orchestrator/
│   └── cache/
│       └── state.db              # SQLite state database
└── results/
    └── {project_id}/
        ├── output/               # Generated files
        └── state.json            # Project state
```

### 🔍 Log Levels

```python
from orchestrator.logging import get_logger, configure_logging

# Configure detailed logging
configure_logging(
    level="DEBUG",
    format="detailed",
    file_path="logs/debug.log",
)

logger = get_logger(__name__)
logger.debug("Debug information")
logger.info("General info")
logger.warning("Warning")
logger.error("Error occurred")
```

### 📊 Analyzing Logs

```bash
# Filter errors
grep "ERROR" logs/orchestrator.log

# Find specific task
grep "task_123" logs/orchestrator.log

# Real-time monitoring
tail -f logs/orchestrator.log | grep -E "(ERROR|WARN|task_123)"

# JSON log parsing
jq 'select(.level == "ERROR")' logs/audit.log
```

### 🐍 Python Log Analysis

```python
import json
from pathlib import Path

def analyze_errors(log_file: Path):
    """Analyze error patterns in logs."""
    errors = []
    
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("level") == "ERROR":
                    errors.append({
                        "timestamp": entry["timestamp"],
                        "message": entry["message"],
                        "exception": entry.get("exception"),
                    })
            except json.JSONDecodeError:
                continue
    
    # Count by error type
    from collections import Counter
    error_types = Counter(e["message"].split(":")[0] for e in errors)
    
    print("Top Errors:")
    for error_type, count in error_types.most_common(10):
        print(f"  {error_type}: {count}")
    
    return errors

# Use it
errors = analyze_errors(Path("logs/orchestrator.log"))
```

---

## 4. Testing & Validation

### ✅ Pre-flight Checks

```python
# Validate environment
from orchestrator import validate_environment

errors = validate_environment()
if errors:
    print("Environment issues found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Environment OK")

# Test API keys
from orchestrator.api_clients import test_all_providers

results = await test_all_providers()
for provider, status in results.items():
    symbol = "✓" if status["ok"] else "✗"
    print(f"{symbol} {provider}: {status['message']}")
```

### 🧪 Unit Testing

```bash
# Run specific test
pytest tests/test_knowledge_base.py -v

# Run with debugging
pytest tests/test_project_manager.py -v --tb=long --log-cli-level=DEBUG

# Run performance tests
pytest tests/test_performance.py -v --benchmark-only

# Test specific functionality
pytest tests/ -k "test_cache" -v
```

### 🔬 Integration Testing

```python
# Dry run mode
from orchestrator import Orchestrator, Budget
from orchestrator.dry_run import DryRunRenderer

orch = Orchestrator(budget=Budget(max_usd=1.0))

# Dry run to see what would happen
plan = await orch.dry_run(
    project_description="Build a calculator",
    success_criteria="tests pass",
)

DryRunRenderer().render(plan)
```

### 🎯 Validation Testing

```python
# Test validators
from orchestrator.validators import python_syntax_validator

code = """
def add(a, b):
    return a + b
"""

result = python_syntax_validator(code)
print(f"Valid: {result.valid}")
if not result.valid:
    print(f"Error: {result.error}")
```

---

## 5. Recovery Procedures

### 🔄 Resume Failed Project

```bash
# List available projects
python -m orchestrator --list-projects

# Resume specific project
python -m orchestrator --resume <project_id>

# Or programmatically
from orchestrator import Orchestrator, ResumeDetector

detector = ResumeDetector()
candidates = await detector.find_similar_projects(
    project_description="Build API"
)

if candidates:
    best_match = candidates[0]
    print(f"Resuming: {best_match.project_id}")
    orch = Orchestrator.resume_from(best_match.project_id)
```

### 🛠️ Manual State Repair

```python
from orchestrator.state import StateManager
from orchestrator.models import ProjectState, TaskStatus

# Load corrupted state
state_mgr = StateManager()
state = state_mgr.load_state("project_123")

# Fix task statuses
for task_id in state.task_ids:
    if task_id not in state.results:
        # Mark as pending
        state.task_status[task_id] = TaskStatus.PENDING

# Save repaired state
state_mgr.save_state(state)
```

### 🚨 Emergency Recovery

```python
# Reset to clean state
from orchestrator.cache import DiskCache
from orchestrator.state import StateManager

# Clear cache
cache = DiskCache()
cache.clear()

# Archive old state
state_mgr = StateManager()
state_mgr.archive_old_projects(keep_last=5)

# Reset telemetry
from orchestrator.telemetry_store import TelemetryStore
telemetry = TelemetryStore()
telemetry.reset()
```

---

## 6. Diagnostic Tools

### 🔧 Built-in Diagnostics

```python
from orchestrator.diagnostics import SystemDiagnostic

diag = SystemDiagnostic()

# Full system check
report = await diag.run_full_check()

print(f"Overall Health: {report.health_status}")
print(f"Issues Found: {len(report.issues)}")

for issue in report.issues:
    print(f"\n[{issue.severity}] {issue.component}")
    print(f"  Problem: {issue.description}")
    print(f"  Solution: {issue.suggested_fix}")
```

### 📊 Health Check Dashboard

```python
# Quick health check
from orchestrator import health_checker

status = await health_checker.check()

print(f"System Status: {status['overall']}")
for check_name, check_result in status['checks'].items():
    symbol = "✓" if check_result['healthy'] else "✗"
    print(f"{symbol} {check_name}: {check_result['status']}")
```

### 🔍 Performance Profiling

```python
# Profile specific function
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run operation
await orch.run_project(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 🌐 Network Diagnostics

```python
# Test API connectivity
import asyncio
import aiohttp

async def test_connectivity():
    endpoints = {
        "openai": "https://api.openai.com/v1/models",
        "deepseek": "https://api.deepseek.com/v1/models",
        # ... etc
    }
    
    async with aiohttp.ClientSession() as session:
        for name, url in endpoints.items():
            try:
                async with session.get(url, timeout=5) as resp:
                    print(f"✓ {name}: {resp.status}")
            except Exception as e:
                print(f"✗ {name}: {e}")

asyncio.run(test_connectivity())
```

---

## 🎯 Common Debug Scenarios

### Scenario 1: Task Always Fails

```python
# Debug approach:

# 1. Check which model is being used
print(f"Model: {task.assigned_model}")

# 2. Test model directly
from orchestrator import UnifiedClient
client = UnifiedClient(task.assigned_model)
response = await client.generate(task.prompt)
print(f"Response: {response.text[:200]}")

# 3. Check validation
from orchestrator.validators import get_validator_for_task
validator = get_validator_for_task(task)
result = validator(response.text)
print(f"Validation: {result.valid} - {result.error}")

# 4. Try with different model
# In policy: allowed_models=["gpt-4o", "claude-sonnet"]
```

### Scenario 2: High Costs

```python
# Analyze cost breakdown
from orchestrator.cost import CostAnalyzer

analyzer = CostAnalyzer()
report = analyzer.analyze_project(project_id)

print("Cost Breakdown:")
for model, cost in report.by_model.items():
    print(f"  {model}: ${cost:.4f}")

print(f"\nMost expensive tasks:")
for task in report.most_expensive(5):
    print(f"  {task.id}: ${task.cost:.4f}")

# Optimize
if report.total > budget.max_usd * 0.8:
    print("⚠️ Over budget! Consider:")
    print("  - Using cheaper models")
    print("  - Reducing iterations")
    print("  - Enabling caching")
```

### Scenario 3: Slow Performance

```python
# Profile execution
from orchestrator.monitoring import metrics
import time

start = time.time()
state = await orch.run_project(...)
duration = time.time() - start

# Get metrics
all_metrics = await metrics.get_all_metrics()
print(f"Total duration: {duration:.1f}s")
print(f"Avg response time: {all_metrics['response_time_p50']}")
print(f"Cache hit rate: {all_metrics['cache_hit_rate']}")

# Optimize if slow
if duration > 300:  # 5 minutes
    print("Slow execution detected. Recommendations:")
    print("  - Enable caching: @cached()")
    print("  - Use async batching")
    print("  - Check network latency")
```

---

## 📚 Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) - Usage examples
- [CAPABILITIES.md](./CAPABILITIES.md) - Feature reference
- [MANAGEMENT_SYSTEMS.md](./MANAGEMENT_SYSTEMS.md) - Management systems
- [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md) - Performance

---

## 🆘 Getting Help

If issues persist:

1. **Check logs:** `logs/orchestrator.log`
2. **Run diagnostics:** `python -m orchestrator --diagnose`
3. **Enable debug mode:** Set `ORCHESTRATOR_LOG_LEVEL=DEBUG`
4. **Test environment:** `python -m orchestrator --test-env`
5. **Open issue:** https://github.com/gchatz22/multi-llm-orchestrator/issues

---

**Last Updated:** 2026-02-26 | **Version:** v5.1
