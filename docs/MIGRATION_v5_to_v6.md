# Migration Guide: v5 → v6

**Version**: 6.0.0  
**Date**: 2026-03-07  
**Breaking Changes**: Yes  
**Migration Time**: 30-60 minutes

---

## OVERVIEW

Version 6.0 introduces significant architectural changes:

1. **Unified Dashboard** - 7 dashboards consolidated into 1
2. **Unified Events** - 4 event systems unified into 1 bus
3. **Plugin Architecture** - Core-only + optional plugins
4. **NASH Stability** - New reliability features

---

## BREAKING CHANGES

### 1. Dashboard Consolidation

**Before (v5)**:
```python
from orchestrator import run_live_dashboard
from orchestrator import run_mission_control
from orchestrator import run_ant_design_dashboard
```

**After (v6)**:
```python
from orchestrator import run_dashboard

# All-in-one dashboard
run_dashboard(view="mission-control")  # or "live", "ant-design"
```

**Migration**:
```python
# Old code
run_live_dashboard(port=8000)

# New code
run_dashboard(view="live", port=8000)
```

**Deprecated Dashboards** (removed in v7.0):
- `dashboard.py`
- `dashboard_optimized.py`
- `dashboard_real.py`
- `dashboard_enhanced.py`
- `dashboard_antd.py`
- `dashboard_mission_control.py`
- `dashboard_live.py`

---

### 2. Event System Unification

**Before (v5)**:
```python
from orchestrator.streaming import ProjectEventBus
from orchestrator.events import EventBus
from orchestrator.hooks import HookRegistry
```

**After (v6)**:
```python
from orchestrator.unified_events import UnifiedEventBus, get_event_bus

# Single event bus for all events
event_bus = get_event_bus()
await event_bus.publish(TaskCompletedEvent(task_id="task_001"))
```

**Event Mapping**:

| v5 Event | v6 Event |
|----------|----------|
| `ProjectStarted` | `ProjectStartedEvent` |
| `TaskStarted` | `TaskStartedEvent` |
| `TaskCompleted` | `TaskCompletedEvent` |
| `ProjectCompleted` | `ProjectCompletedEvent` |

**Migration**:
```python
# Old code
from orchestrator.streaming import ProjectEventBus, TaskCompleted

bus = ProjectEventBus()
await bus.publish(TaskCompleted(task_id="task_001"))

# New code
from orchestrator.unified_events import get_event_bus, TaskCompletedEvent

bus = get_event_bus()
await bus.publish(TaskCompletedEvent(task_id="task_001"))
```

---

### 3. Plugin Architecture

**Before (v5)**:
Plugins bundled with core.

**After (v6)**:
Core-only installation, plugins optional.

**Migration**:
```bash
# v5
pip install multi-llm-orchestrator

# v6 (core only)
pip install multi-llm-orchestrator

# v6 (with plugins)
pip install multi-llm-orchestrator[plugins]
```

---

### 4. NASH Stability Features

**New in v6**:
- `NashStableOrchestrator` - Integration of all stability features
- `knowledge_graph` - Model performance knowledge
- `adaptive_templates` - Dynamic prompt templates
- `pareto_frontier` - Cost-quality optimization
- `federated_learning` - Cross-org learning
- `nash_backup` - Backup/restore system
- `nash_auto_tuning` - Parameter auto-tuning

**Usage**:
```python
from orchestrator import NashStableOrchestrator

orch = NashStableOrchestrator()
result = await orch.run_project("Build an API")
```

---

## DEPENDENCY CHANGES

### Python Version

| Version | Python Required |
|---------|-----------------|
| v5 | 3.9+ |
| v6 | **3.10+** |

**Action**: Upgrade to Python 3.10 or later.

### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scipy` | >=1.10 | Drift detection (statistical tests) |
| `numpy` | >=1.24 | Performance optimizations |
| `tomli` | >=2.0 | TOML parsing (Python 3.10+) |

### Removed Dependencies

| Package | Reason |
|---------|--------|
| `legacy-dashboard-deps` | Consolidated into dashboard_core |

---

## CONFIGURATION CHANGES

### .env File

**New Required Variables**:
```bash
# NASH Stability features
NASH_STABILITY_ENABLED=true
KNOWLEDGE_GRAPH_ENABLED=true
ADAPTIVE_TEMPLATES_ENABLED=true
```

**Optional Variables**:
```bash
# Drift detection
DRIFT_DETECTION_ENABLED=true
DRIFT_BASELINE_DAYS=7

# Backup
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
```

### pyproject.toml

**New Tool Configuration**:
```toml
[tool.pytest.ini_options]
addopts = [
    "-v",
    "--benchmark-only",  # New: performance benchmarks
]

[tool.ruff]
# New rules
select = [
    "E", "F", "I", "W",
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
```

---

## CODE MIGRATION CHECKLIST

### Step 1: Update Imports

```python
# ❌ Remove these
from orchestrator.streaming import ProjectEventBus
from orchestrator.events import EventBus
from orchestrator.hooks import HookRegistry

# ✅ Add these
from orchestrator.unified_events import UnifiedEventBus, get_event_bus
```

### Step 2: Update Dashboard Calls

```python
# ❌ Remove these
run_live_dashboard()
run_mission_control()

# ✅ Add these
run_dashboard(view="live")
run_dashboard(view="mission-control")
```

### Step 3: Update Event Classes

```python
# ❌ Old event classes
TaskCompleted
ProjectStarted
BudgetWarning

# ✅ New event classes
TaskCompletedEvent
ProjectStartedEvent
BudgetWarningEvent
```

### Step 4: Enable NASH Features (Optional)

```python
from orchestrator import NashStableOrchestrator

orch = NashStableOrchestrator(
    enable_knowledge_graph=True,
    enable_adaptive_templates=True,
    enable_pareto_frontier=True,
)
```

### Step 5: Update Tests

```python
# Add benchmark tests
pytest tests/benchmarks/ --benchmark-only

# Update event tests
from orchestrator.unified_events import TaskCompletedEvent
```

---

## ROLLBACK PROCEDURE

If you need to rollback to v5:

```bash
# 1. Uninstall v6
pip uninstall multi-llm-orchestrator

# 2. Install v5
pip install multi-llm-orchestrator==5.0.0

# 3. Restore old imports
# Revert import changes from Step 1

# 4. Restore old dashboard calls
# Revert dashboard changes from Step 2
```

---

## KNOWN ISSUES

### Issue 1: Legacy Dashboard Import Errors

**Symptom**: `ImportError: cannot import name 'run_live_dashboard'`

**Fix**:
```python
# Use compatibility layer (temporary)
from orchestrator.compat import run_live_dashboard

# Or update to new API
from orchestrator import run_dashboard
run_dashboard(view="live")
```

### Issue 2: Event Bus Not Initialized

**Symptom**: `RuntimeError: Event bus not initialized`

**Fix**:
```python
# Ensure event bus is initialized before use
from orchestrator.unified_events import get_event_bus
event_bus = get_event_bus()
```

---

## TESTING

After migration, run:

```bash
# Full test suite
pytest tests/ -v

# Benchmarks
pytest tests/benchmarks/ --benchmark-only

# Coverage
pytest --cov=orchestrator --cov-report=html
```

---

## SUPPORT

- **Documentation**: https://georgehadji.github.io/multi-llm-orchestrator/
- **Issues**: https://github.com/georgehadji/multi-llm-orchestrator/issues
- **Discussions**: https://github.com/georgehadji/multi-llm-orchestrator/discussions

---

## TIMELINE

| Date | Milestone |
|------|-----------|
| 2026-03-07 | v6.0 released |
| 2026-06-07 | v6.1 (minor improvements) |
| 2026-09-07 | v7.0 (legacy dashboards removed) |

**Action**: Complete migration before September 2026 to avoid breaking changes.

---

*Migration guide last updated: 2026-03-07*
