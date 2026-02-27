# 🆘 Troubleshooting Cheatsheet
## Quick fixes for common Multi-LLM Orchestrator issues

---

## 🔴 Critical Errors

### 1. "No API keys found" (ENV001)

**Symptoms:**
```
❌ No API keys found. At least one provider key is required.
```

**Quick Fix:**
```bash
# Set at least one API key
export OPENAI_API_KEY="sk-..."
# OR
export DEEPSEEK_API_KEY="sk-..."

# Verify
echo $OPENAI_API_KEY
```

---

### 2. "Authentication failed" (API001)

**Symptoms:**
```
AuthenticationError: Invalid API key
```

**Quick Fix:**
```bash
# Test API key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# If fails, regenerate key at:
# https://platform.openai.com/api-keys
```

---

### 3. "Module not found" (DEP001)

**Symptoms:**
```
ModuleNotFoundError: No module named 'orchestrator'
```

**Quick Fix:**
```bash
# Reinstall in editable mode
cd "E:\Documents\Vibe-Coding\Ai Orchestrator"
pip install -e ".[dev]"

# Verify
python -c "from orchestrator import Orchestrator; print('OK')"
```

---

## 🟡 Common Errors

### 4. Task Always Fails

**Debug:**
```python
# Check task details
from orchestrator.state import StateManager

state_mgr = StateManager()
state = state_mgr.load_state("project_id")

for task_id, result in state.results.items():
    if result.error:
        print(f"Task {task_id}:")
        print(f"  Error: {result.error}")
        print(f"  Model: {result.model_used}")
        print(f"  Attempts: {len(result.attempts)}")
```

**Fix:**
```python
# Try different model
from orchestrator import Policy, PolicySet

policies = PolicySet([
    Policy(allowed_models=["gpt-4o", "deepseek-coder"])
])
```

---

### 5. High Costs

**Check:**
```python
from orchestrator.cost import CostAnalyzer

analyzer = CostAnalyzer()
report = analyzer.analyze_project("project_id")

print(f"Total: ${report.total:.2f}")
for model, cost in report.by_model.items():
    print(f"  {model}: ${cost:.2f}")
```

**Fix:**
```python
# Use cheaper models
from orchestrator.optimization import WeightedSumBackend

orch = Orchestrator(
    optimization_backend=WeightedSumBackend(
        w_quality=0.3,
        w_cost=0.7,  # Prioritize cost
    )
)
```

---

### 6. Slow Performance

**Profile:**
```python
import time
from orchestrator import get_cache

# Check cache
cache = get_cache()
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']}")

# Time execution
start = time.time()
await orch.run_project(...)
print(f"Duration: {time.time() - start:.1f}s")
```

**Fix:**
```python
# Enable caching
from orchestrator import cached

@cached(ttl=300)
async def expensive_operation():
    return await fetch_data()
```

---

### 7. Dashboard Won't Start

**Error:**
```
ImportError: No module named 'fastapi'
```

**Fix:**
```bash
pip install fastapi uvicorn websockets httpx

# Run with HTTP polling (no WebSocket)
python run_optimized_dashboard.py --port 8888
```

**Test:**
```bash
curl http://localhost:8888/api/metrics
```

---

### 8. Project Won't Resume

**Check:**
```bash
# List projects
python -m orchestrator --list-projects

# Check specific project
python -c "
from orchestrator.state import StateManager
state_mgr = StateManager()
state = state_mgr.load_state('project_id')
print(f'Status: {state.status}')
print(f'Completion: {state.completion_percentage}%')
"
```

**Fix:**
```bash
# Force resume
python -m orchestrator --resume project_id

# Or manually repair
python -c "
from orchestrator.state import StateManager
from orchestrator.models import TaskStatus

state_mgr = StateManager()
state = state_mgr.load_state('project_id')

# Reset stuck tasks
for task_id in state.task_ids:
    if state.task_status.get(task_id) == TaskStatus.IN_PROGRESS:
        state.task_status[task_id] = TaskStatus.PENDING

state_mgr.save_state(state)
"
```

---

## 🟢 Optimization Tips

### 9. Low Quality Output

**Fix:**
```python
# Use premium models
policies = PolicySet([
    Policy(
        allowed_models=["deepseek-reasoner", "gpt-4o"],
        min_quality_score=0.9,
    )
])

# More iterations
orch = Orchestrator(max_iterations=5)
```

---

### 10. Out of Memory

**Fix:**
```bash
# Clear cache
python -c "
from orchestrator.cache import DiskCache
DiskCache().clear()
"

# Reduce concurrency
export ORCHESTRATOR_MAX_WORKERS=2
```

---

## 🛠️ Diagnostic Commands

### Full System Check
```bash
python -c "
import asyncio
from orchestrator.diagnostics import SystemDiagnostic, print_diagnostic_report

async def check():
    diag = SystemDiagnostic()
    report = await diag.run_full_check()
    print_diagnostic_report(report)

asyncio.run(check())
"
```

### Test All Providers
```bash
python -c "
import asyncio
from orchestrator.api_clients import test_all_providers

async def test():
    results = await test_all_providers()
    for provider, status in results.items():
        print(f'{provider}: {status}')

asyncio.run(test())
"
```

### Check Logs
```bash
# Recent errors
tail -50 logs/orchestrator.log | grep ERROR

# Specific project
grep "project_id" logs/orchestrator.log

# Real-time
tail -f logs/orchestrator.log
```

---

## 📋 Pre-flight Checklist

Before running a project:

- [ ] API keys set: `echo $OPENAI_API_KEY`
- [ ] Dependencies installed: `pip list | grep orchestrator`
- [ ] Disk space: `df -h .`
- [ ] Network: `ping api.openai.com`
- [ ] Cache dir exists: `ls ~/.orchestrator_cache`

---

## 🆘 Emergency Recovery

### Complete Reset
```bash
# 1. Clear all caches
rm -rf ~/.orchestrator_cache/*

# 2. Clear logs
rm -rf logs/*

# 3. Reset state (keeps projects)
python -c "
from orchestrator.state import StateManager
StateManager().archive_old_projects(keep_last=3)
"

# 4. Reinstall
pip uninstall multi-llm-orchestrator -y
pip install -e ".[dev]"
```

### Get Help
```bash
# Run diagnostics
python -m orchestrator.diagnostics

# Check version
python -c "from orchestrator import __version__; print(__version__)"

# Open issue
github issue create \
  --repo gchatz22/multi-llm-orchestrator \
  --title "Bug: ..." \
  --body "Description..."
```

---

**Last Updated:** 2026-02-26 | **Version:** v5.1
