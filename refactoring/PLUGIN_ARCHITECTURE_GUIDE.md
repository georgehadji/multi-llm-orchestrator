# Plugin Architecture Guide

## Overview

The Plugin Architecture (Phase 5) extracts optional features from the core orchestrator into decoupled, loadable plugins. This enables:

- **Modularity**: Features can be enabled/disabled independently
- **Maintainability**: Clear separation between core and optional features
- **Extensibility**: Easy to add new features without modifying core
- **Performance**: Only load plugins you need

## Quick Start

### Using Plugins

```python
from orchestrator import Orchestrator
from orchestrator.plugins import get_plugin_registry, CostOptimizationPlugin

# Create orchestrator
orch = Orchestrator()

# Get plugin registry
registry = get_plugin_registry()

# Register plugins
registry.register(CostOptimizationPlugin())

# Initialize plugins
await registry.initialize_all()

# Run project with plugin hooks
await registry.execute_pre_project(project_state)
# ... orchestrator runs ...
await registry.execute_post_project(project_state)

# Shutdown
await registry.shutdown_all()
```

### Integration with Orchestrator

```python
# In orchestrator/engine_core/core.py

from .plugins import get_plugin_registry

class OrchestratorCore:
    def __init__(self, ...):
        self.plugin_registry = get_plugin_registry()
    
    async def run_project(self, ...):
        # Pre-project hooks
        await self.plugin_registry.execute_pre_project(state)
        
        # Execute tasks
        for task_id in execution_order:
            # Pre-task hooks
            await self.plugin_registry.execute_pre_task(task)
            
            # Execute task
            result = await self._task_executor.execute_task(...)
            
            # Post-task hooks
            await self.plugin_registry.execute_post_task(task, result)
        
        # Post-project hooks
        await self.plugin_registry.execute_post_project(state)
```

## Available Plugins

### 1. Cost Optimization Plugin

**Purpose**: Reduce LLM costs through caching, batching, and cascading.

**Features**:
- Prompt caching (80-90% input cost reduction)
- Batch API processing
- Token budget enforcement
- Model cascading (cheap → expensive)
- Streaming validation

**Usage**:
```python
from orchestrator.plugins import CostOptimizationPlugin, CostOptimizationConfig

config = CostOptimizationConfig(
    enable_prompt_cache=True,
    enable_model_cascading=True,
    enable_batch_api=False,
    cache_ttl_hours=48,
)

plugin = CostOptimizationPlugin(config=config)
registry.register(plugin)
```

**Statistics**:
```python
plugin = registry.get("cost-optimization")
stats = plugin.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Cost saved: ${stats['cost_saved_usd']:.2f}")
```

### 2. Nash Stability Plugin

**Purpose**: Game-theoretic model selection and optimization.

**Features**:
- Nash equilibrium detection
- Performance-based model scoring
- Adaptive template selection
- Cost-quality frontier (Pareto optimization)

**Usage**:
```python
from orchestrator.plugins import NashStabilityPlugin

plugin = NashStabilityPlugin(
    enable_nash_monitoring=True,
    enable_adaptive_templates=True,
    enable_pareto_frontier=True,
)

registry.register(plugin)
```

**Model Selection**:
```python
plugin = registry.get("nash-stability")
optimal_model = plugin.get_optimal_model(
    task_type="CODE_GEN",
    budget_constraint=0.05,
)
```

## Creating Custom Plugins

### Base Plugin Class

```python
from orchestrator.plugins import Plugin, PluginMetadata, PluginPriority

class MyCustomPlugin(Plugin):
    metadata = PluginMetadata(
        name="my-custom-plugin",
        version="1.0.0",
        description="My custom orchestrator plugin",
        author="Your Name",
        priority=PluginPriority.NORMAL,
    )
    
    async def initialize(self) -> None:
        """Initialize plugin resources."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown plugin resources."""
        pass
    
    async def on_pre_task(self, context: PluginContext) -> None:
        """Called before each task."""
        logger.info(f"Task {context.task.id} starting")
    
    async def on_post_task(self, context: PluginContext) -> None:
        """Called after each task."""
        logger.info(f"Task {context.task.id} completed: {context.task_result.score}")
```

### Plugin Lifecycle Hooks

| Hook | When Called | Purpose |
|------|-------------|---------|
| `initialize()` | Plugin registration | Setup resources |
| `on_pre_project()` | Before project starts | Initialize project state |
| `on_pre_task()` | Before each task | Modify task, check cache |
| `on_post_task()` | After each task | Record results, update stats |
| `on_post_project()` | After project completes | Generate reports |
| `shutdown()` | Plugin unload | Cleanup resources |

### Plugin Context

```python
@dataclass
class PluginContext:
    project_state: Optional[ProjectState]  # Full project state
    task: Optional[Task]  # Current task
    task_result: Optional[TaskResult]  # Task execution result
    metadata: dict[str, Any]  # Custom metadata
```

### Plugin Priority

Plugins execute in priority order (HIGH → NORMAL → LOW):

```python
from orchestrator.plugins import PluginPriority

class HighPriorityPlugin(Plugin):
    metadata = PluginMetadata(
        name="high-priority",
        version="1.0.0",
        description="Runs before other plugins",
        author="You",
        priority=PluginPriority.HIGH,  # Executes first
    )
```

## Plugin Dependencies

Plugins can declare dependencies on other plugins:

```python
plugin_metadata = PluginMetadata(
    name="dependent-plugin",
    version="1.0.0",
    description="Requires cost-optimization plugin",
    author="You",
    dependencies=["cost-optimization"],  # Must be loaded first
)
```

## Plugin Discovery

### Auto-Discovery (Future)

Plugins can be auto-discovered from entry points:

```python
# In setup.py or pyproject.toml
[project.entry-points."orchestrator.plugins"]
cost-optimization = "orchestrator.plugins.cost_optimization:CostOptimizationPlugin"
nash-stability = "orchestrator.plugins.nash_stability:NashStabilityPlugin"
```

### Manual Registration

```python
registry = get_plugin_registry()
registry.register(MyPlugin())
```

## Best Practices

### 1. Keep Plugins Focused

Each plugin should have a single responsibility:
- ✅ CostOptimizationPlugin: Cost reduction
- ✅ NashStabilityPlugin: Model selection
- ❌ MegaPlugin: Everything

### 2. Handle Errors Gracefully

```python
async def on_pre_task(self, context: PluginContext) -> None:
    try:
        await self._do_something(context.task)
    except Exception as e:
        logger.error(f"Plugin error: {e}")
        # Don't fail the task, just skip plugin functionality
```

### 3. Use Metadata for Communication

```python
# In pre-task hook
context.metadata["my_plugin_data"] = some_value

# In post-task hook
data = context.metadata.get("my_plugin_data")
```

### 4. Respect Plugin Lifecycle

```python
# ✅ Correct
async def initialize(self):
    await self._setup_resources()
    self._initialized = True

async def shutdown(self):
    if self._initialized:
        await self._cleanup_resources()
        self._initialized = False

# ❌ Incorrect - no lifecycle tracking
```

### 5. Document Plugin Configuration

```python
@dataclass
class MyPluginConfig:
    """Configuration for MyPlugin.
    
    Attributes:
        enable_feature_x: Enable X feature (default: True)
        max_retries: Maximum retry attempts (default: 3)
        timeout_seconds: Request timeout (default: 30)
    """
    enable_feature_x: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30
```

## Migration Guide

### Extracting Feature to Plugin

1. **Identify Feature Boundaries**
   - What functionality to extract?
   - What dependencies does it have?

2. **Create Plugin Class**
   ```python
   class MyFeaturePlugin(Plugin):
       ...
   ```

3. **Move Code**
   - Move feature code to plugin
   - Replace core imports with plugin imports

4. **Add Hooks**
   - Implement lifecycle hooks
   - Register with plugin registry

5. **Test**
   - Test plugin in isolation
   - Test integration with orchestrator

## Troubleshooting

### Plugin Not Loading

```python
# Check if registered
registry = get_plugin_registry()
print(registry.list_plugins())  # Should include your plugin

# Check initialization
plugin = registry.get("my-plugin")
print(plugin.is_initialized)  # Should be True
```

### Hook Not Called

```python
# Verify hook implementation
class MyPlugin(Plugin):
    async def on_pre_task(self, context: PluginContext) -> None:
        logger.info("This should print before each task")
        # Make sure to call super() if overriding
```

### Plugin Order Issues

```python
# Set priority
metadata = PluginMetadata(
    name="my-plugin",
    priority=PluginPriority.HIGH,  # or NORMAL or LOW
)
```

## Future Enhancements

- [ ] Plugin marketplace/discovery
- [ ] Hot-reload plugins
- [ ] Plugin versioning
- [ ] Plugin configuration UI
- [ ] Plugin metrics dashboard

## See Also

- [ENGINE_DECOMPOSITION_PLAN.md](ENGINE_DECOMPOSITION_PLAN.md) - Engine refactoring
- [TYPE_SAFETY_IMPROVEMENTS.md](TYPE_SAFETY_IMPROVEMENTS.md) - Type safety improvements
- [ARCHITECTURE_OVERVIEW.md](../ARCHITECTURE_OVERVIEW.md) - System architecture
