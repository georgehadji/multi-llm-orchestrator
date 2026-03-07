# Refactoring Plan: Core vs Plugins

## Περίληψη Αλλαγών

```
Before:  4 modules, ~2,500 lines, όλα στο core
After:   2 core modules (~1,200 lines) + 2 official plugin packages (~1,300 lines)
```

---

## Phase 1: Core Consolidation

### Μένει στο Core

```
orchestrator/
├── plugins.py              # Μόνο τα interfaces
│   ├── Plugin (ABC)
│   ├── PluginRegistry
│   ├── PluginMetadata
│   ├── ValidationResult
│   └── *ΔΙΑΓΡΑΦΗ*: built-in plugins (PythonTypeCheckerValidator, TeamsIntegration)
│
├── feedback_loop.py        # Storage + API μόνο
│   ├── ProductionOutcome
│   ├── FeedbackLoop
│   ├── ModelPerformanceRecord
│   └── *ΔΙΑΓΡΑΦΗ*: Sentry/Datadog specific code
│
├── leaderboard.py          # Engine + standard tasks
│   ├── ModelLeaderboard
│   ├── BenchmarkTask
│   └── *ΚΡΑΤΑΜΕ*: ~10 standard tasks
│
└── outcome_router.py       # 100% core
    └── (no changes needed)
```

### Φεύγει από Core → Official Plugins

```
orchestrator-plugins/
├── validators/
│   ├── python_mypy.py      # από plugins.py
│   └── __init__.py
│
├── integrations/
│   ├── teams.py            # από plugins.py
│   ├── slack_extended.py   # enhanced από slack_integration.py
│   └── __init__.py
│
└── feedback_processors/
    ├── sentry.py
    ├── datadog.py
    └── __init__.py
```

---

## Phase 2: Code Changes

### Αλλαγή 1: plugins.py (Διαγραφή built-ins)

```python
# orchestrator/plugins.py - BEFORE
class PythonTypeCheckerValidator(ValidatorPlugin):
    """Built-in validator using mypy."""
    ...

class TeamsIntegration(IntegrationPlugin):
    """Microsoft Teams integration plugin."""
    ...

def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        # Register built-in plugins
        _registry.register(PythonTypeCheckerValidator())  # <-- DELETE
        _registry.register(TeamsIntegration())             # <-- DELETE
    return _registry
```

```python
# orchestrator/plugins.py - AFTER
# ... μόνο interfaces ...

def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        # NO built-in plugins registered here
        # They are loaded from orchestrator_plugins package if installed
    return _registry
```

### Αλλαγή 2: Νέο Official Plugin Package

```python
# orchestrator-plugins/orchestrator_plugins/validators/python_mypy.py
from orchestrator.plugins import ValidatorPlugin, PluginMetadata, PluginType, ValidationResult

class PythonTypeCheckerValidator(ValidatorPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="python-mypy",
            version="1.0.0",
            author="orchestrator-team",
            description="Type checking using mypy",
            plugin_type=PluginType.VALIDATOR,
        )
    
    def validate(self, code: str, context: dict) -> ValidationResult:
        # implementation...
        pass

# Auto-register on import
from orchestrator.plugins import get_plugin_registry
get_plugin_registry().register(PythonTypeCheckerValidator())
```

### Αλλαγή 3: feedback_loop.py (Plugin hooks)

```python
# orchestrator/feedback_loop.py - ADD

class FeedbackLoop:
    # ... existing code ...
    
    async def _run_plugin_processors(self, outcome: ProductionOutcome) -> None:
        """Run feedback plugin processors."""
        from .plugins import get_plugin_registry
        plugins = get_plugin_registry().get_feedback_processors()
        
        payload = FeedbackPayload(
            project_id=outcome.project_id,
            deployment_id=outcome.deployment_id,
            task_type=outcome.task_type,
            model_used=outcome.model_used,
            generated_code="",
            runtime_errors=[asdict(e) for e in outcome.runtime_errors],
            performance_metrics=asdict(outcome.performance_metrics),
            user_rating=outcome.user_feedback.rating if outcome.user_feedback else None,
        )
        
        for plugin in plugins:
            if plugin.should_process(payload):
                try:
                    await plugin.process_feedback(payload)
                except Exception as e:
                    logger.error(f"Plugin {plugin.metadata.name} failed: {e}")
```

### Αλλαγή 4: leaderboard.py (Plugin discovery)

```python
# orchestrator/leaderboard.py - ADD

class ModelLeaderboard:
    def __init__(self, ...):
        # ... existing ...
        self._load_custom_benchmarks()  # <-- NEW
    
    def _load_custom_benchmarks(self) -> None:
        """Load custom benchmark tasks from plugins."""
        from .plugins import get_plugin_registry
        
        # Discover BenchmarkProvider plugins
        registry = get_plugin_registry()
        for plugin in registry.get_by_type(PluginType.BENCHMARK):
            if hasattr(plugin, 'get_tasks'):
                try:
                    custom_tasks = plugin.get_tasks()
                    self.suite.tasks.extend(custom_tasks)
                    logger.info(f"Loaded {len(custom_tasks)} tasks from {plugin.metadata.name}")
                except Exception as e:
                    logger.error(f"Failed to load tasks from {plugin.metadata.name}: {e}")
```

---

## Phase 3: Installation Options

### setup.py / pyproject.toml structure

```toml
# Core package
[project]
name = "multi-llm-orchestrator"
version = "5.3.0"
dependencies = [
    "httpx>=0.24",
    "pydantic>=2.0",
    # ΜΟΝΟ απαραίτητα dependencies
]

[project.optional-dependencies]
# Official plugins bundle
all = [
    "orchestrator-plugins-validators>=1.0",
    "orchestrator-plugins-integrations>=1.0",
    "orchestrator-plugins-benchmarks>=1.0",
]

# Επιλεκτικά
validators = ["orchestrator-plugins-validators>=1.0"]
integrations = ["orchestrator-plugins-integrations>=1.0"]
slack = ["orchestrator-plugins-integrations[slack]>=1.0"]
sentry = ["orchestrator-plugins-feedback[sentry]>=1.0"]

# Development
dev = ["pytest", "ruff", "mypy", ...]
```

---

## Phase 4: Migration Guide για Χρήστες

### Breaking Changes

| Before | After | Migration |
|--------|-------|-----------|
| `from orchestrator.plugins import TeamsIntegration` | `from orchestrator_plugins.integrations import TeamsIntegration` | Update import |
| `pip install orchestrator` (with all validators) | `pip install orchestrator[all]` | Add `[all]` or specific extras |

### Backward Compatibility Layer (6 months)

```python
# orchestrator/plugins_compat.py
import warnings

def TeamsIntegration(*args, **kwargs):
    warnings.warn(
        "TeamsIntegration moved to orchestrator_plugins.integrations. "
        "Import will fail in v6.0. Update your imports.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from orchestrator_plugins.integrations import TeamsIntegration as RealTeams
        return RealTeams(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "TeamsIntegration is no longer in core. "
            "Install: pip install orchestrator-plugins-integrations"
        )
```

---

## Phase 5: Testing Strategy

### Core Tests (μένουν)
- Plugin registry functionality
- Feedback loop storage
- Leaderboard engine
- Router core logic

### Plugin Tests (μετακινούνται)
- MyPy validator tests → `orchestrator-plugins-validators/tests/`
- Teams integration tests → `orchestrator-plugins-integrations/tests/`

### Integration Tests (νέα)
```python
# tests/test_plugin_integration.py
@pytest.mark.skipif(not HAS_PLUGINS, reason="Plugins not installed")
def test_plugin_integration():
    """Test core + plugins work together."""
    from orchestrator_plugins.validators import PythonTypeCheckerValidator
    from orchestrator.plugins import get_plugin_registry
    
    registry = get_plugin_registry()
    assert "python-mypy" in [p.name for p in registry.list_plugins()]
```

---

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Extract plugin interfaces | `orchestrator/plugins.py` cleaned |
| 1 | Create plugin packages | `orchestrator-plugins-*` repos |
| 2 | Add compatibility layer | Backward compat imports |
| 2 | Update documentation | Migration guide |
| 3 | Testing & bug fixes | All tests pass |
| 3 | Release v5.4.0-beta | Community testing |
| 4 | Release v5.4.0 | Stable release |
| 6+ | Deprecation warnings | Prepare for v6.0 |

---

## Code Metrics

### Before
```
Core codebase: ~12,000 lines
Dependencies: 15
Installation size: ~5MB
Load time: ~0.8s
```

### After (Core + minimal plugins)
```
Core codebase: ~9,000 lines (-25%)
Dependencies: 8 (-47%)
Installation size: ~3MB (-40%)
Load time: ~0.5s (-37%)
```

### After (Core + all official plugins)
```
Total codebase: ~13,000 lines
Dependencies: 22
Installation size: ~7MB
Load time: ~1.0s
```

**Key Win**: Users που θέλουν μόνο το core, παίρνουν significantly leaner installation.
