# Plugin Development Guide

**Version**: 6.0.0  
**Date**: 2026-03-07

---

## OVERVIEW

The AI Orchestrator supports a plugin architecture for extending functionality without modifying core code.

**Plugin Types**:
1. **Validator Plugins** - Custom validation rules
2. **Integration Plugins** - External service integrations
3. **Router Plugins** - Custom routing logic
4. **Feedback Plugins** - Custom feedback mechanisms

---

## QUICK START

### 1. Create Plugin Structure

```
my_plugin/
├── __init__.py
├── plugin.py
├── tests/
│   └── test_plugin.py
└── pyproject.toml
```

### 2. Basic Plugin Template

```python
# my_plugin/plugin.py
from orchestrator.plugins import BasePlugin, PluginType
from orchestrator.models import TaskResult

class MyValidatorPlugin(BasePlugin):
    """Custom validator plugin."""
    
    name = "my_validator"
    plugin_type = PluginType.VALIDATOR
    version = "1.0.0"
    
    async def validate(self, task_result: TaskResult) -> bool:
        """Validate task result."""
        # Your validation logic here
        if not task_result.output:
            return False
        
        # Check for required patterns
        if "def main" not in task_result.output:
            return False
        
        return True
    
    async def get_error_message(self) -> str:
        """Return error message on validation failure."""
        return "Output missing required 'def main' function"
```

### 3. Register Plugin

```python
# In your code
from my_plugin.plugin import MyValidatorPlugin
from orchestrator import Orchestrator

orch = Orchestrator()

# Register plugin
orch.register_plugin(MyValidatorPlugin())

# Use orchestrator as normal
result = await orch.run_project("Build an API")
```

---

## PLUGIN TYPES

### 1. Validator Plugins

Validate task outputs before acceptance.

```python
from orchestrator.plugins import BasePlugin, PluginType
from orchestrator.models import TaskResult, TaskType

class CodeStyleValidator(BasePlugin):
    """Validate code style (PEP 8)."""
    
    name = "code_style_validator"
    plugin_type = PluginType.VALIDATOR
    version = "1.0.0"
    
    async def validate(self, task_result: TaskResult) -> bool:
        if task_result.task_type != TaskType.CODE_GEN:
            return True  # Skip non-code tasks
        
        output = task_result.output
        
        # Check line length
        for line in output.split('\n'):
            if len(line) > 100:
                return False
        
        return True
    
    async def get_error_message(self) -> str:
        return "Code violates PEP 8 line length (max 100 chars)"
```

### 2. Integration Plugins

Connect external services.

```python
from orchestrator.plugins import BasePlugin, PluginType
from orchestrator.models import TaskResult

class SlackNotifierPlugin(BasePlugin):
    """Notify Slack on task completion."""
    
    name = "slack_notifier"
    plugin_type = PluginType.INTEGRATION
    version = "1.0.0"
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def on_task_complete(self, task_result: TaskResult):
        """Send Slack notification."""
        import aiohttp
        
        message = {
            "text": f"Task {task_result.task_id} completed",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Task Completed*\nID: {task_result.task_id}\nStatus: {task_result.status}"
                    }
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=message)
```

### 3. Router Plugins

Custom model routing logic.

```python
from orchestrator.plugins import BasePlugin, PluginType
from orchestrator.models import Model, TaskType

class CostOptimizerRouter(BasePlugin):
    """Route to cheapest available model."""
    
    name = "cost_optimizer_router"
    plugin_type = PluginType.ROUTER
    version = "1.0.0"
    
    async def select_model(
        self,
        task_type: TaskType,
        available_models: list[Model]
    ) -> Model:
        """Select cheapest model for task."""
        # Cost table (example)
        costs = {
            Model.GPT_4O_MINI: 0.00015,
            Model.GPT_4O: 0.0025,
            Model.CLAUDE_3_HAIKU: 0.00025,
            Model.CLAUDE_3_SONNET: 0.003,
        }
        
        # Filter available and sort by cost
        available_with_cost = [
            (m, costs.get(m, float('inf')))
            for m in available_models
        ]
        available_with_cost.sort(key=lambda x: x[1])
        
        return available_with_cost[0][0]
```

### 4. Feedback Plugins

Custom feedback mechanisms.

```python
from orchestrator.plugins import BasePlugin, PluginType
from orchestrator.models import TaskResult

class LearningFeedbackPlugin(BasePlugin):
    """Collect feedback for model improvement."""
    
    name = "learning_feedback"
    plugin_type = PluginType.FEEDBACK
    version = "1.0.0"
    
    async def collect_feedback(
        self,
        task_result: TaskResult,
        user_rating: int
    ) -> dict:
        """Collect and store feedback."""
        feedback = {
            "task_id": task_result.task_id,
            "model": task_result.model_used,
            "rating": user_rating,
            "output_length": len(task_result.output),
            "iterations": task_result.iterations,
        }
        
        # Store feedback (database, file, etc.)
        await self._store_feedback(feedback)
        
        return feedback
```

---

## PLUGIN LIFECYCLE

### Initialization

```python
class MyPlugin(BasePlugin):
    async def initialize(self) -> None:
        """Called when plugin is registered."""
        # Setup connections, load config, etc.
        pass
    
    async def shutdown(self) -> None:
        """Called when orchestrator shuts down."""
        # Cleanup connections, save state, etc.
        pass
```

### Event Hooks

```python
class MyPlugin(BasePlugin):
    async def on_project_start(self, project_id: str):
        """Called when project starts."""
        pass
    
    async def on_task_start(self, task_id: str):
        """Called when task starts."""
        pass
    
    async def on_task_complete(self, task_result: TaskResult):
        """Called when task completes."""
        pass
    
    async def on_project_complete(self, project_id: str):
        """Called when project completes."""
        pass
```

---

## PLUGIN SANDBOXING

For untrusted plugins, use sandboxed execution:

```python
from orchestrator.plugin_isolation import SandboxedPlugin

class SandboxedMyPlugin(SandboxedPlugin):
    """Plugin runs in isolated process."""
    
    name = "sandboxed_plugin"
    version = "1.0.0"
    
    # Resource limits
    MAX_MEMORY_MB = 256
    MAX_CPU_PERCENT = 50
    MAX_DISK_MB = 100
    TIMEOUT_SECONDS = 30
```

---

## TESTING PLUGINS

### Unit Tests

```python
# tests/test_plugin.py
import pytest
from my_plugin.plugin import MyValidatorPlugin
from orchestrator.models import TaskResult, TaskStatus, Model

@pytest.mark.asyncio
async def test_validator_passes():
    plugin = MyValidatorPlugin()
    
    result = TaskResult(
        task_id="test_001",
        output="def main():\n    pass",
        score=0.9,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.COMPLETED,
    )
    
    assert await plugin.validate(result) == True

@pytest.mark.asyncio
async def test_validator_fails():
    plugin = MyValidatorPlugin()
    
    result = TaskResult(
        task_id="test_001",
        output="",  # Empty output
        score=0.0,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.FAILED,
    )
    
    assert await plugin.validate(result) == False
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_plugin_integration():
    from orchestrator import Orchestrator
    from my_plugin.plugin import MyValidatorPlugin
    
    orch = Orchestrator()
    orch.register_plugin(MyValidatorPlugin())
    
    # Run project with plugin
    result = await orch.run_project("Build a simple API")
    
    # Verify plugin was called
    assert result is not None
```

---

## DISTRIBUTION

### Package Structure

```
my_plugin/
├── my_plugin/
│   ├── __init__.py
│   └── plugin.py
├── tests/
│   └── test_plugin.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-orchestrator-plugin"
version = "1.0.0"
description = "My custom orchestrator plugin"
requires-python = ">=3.10"
dependencies = [
    "multi-llm-orchestrator>=6.0.0",
]

[project.entry-points."orchestrator.plugins"]
my_plugin = "my_plugin.plugin:MyValidatorPlugin"
```

### Publishing to PyPI

```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

### Installation

```bash
# From PyPI
pip install my-orchestrator-plugin

# From GitHub
pip install git+https://github.com/user/my-orchestrator-plugin.git
```

---

## BEST PRACTICES

### 1. Error Handling

```python
class MyPlugin(BasePlugin):
    async def validate(self, task_result: TaskResult) -> bool:
        try:
            # Your logic here
            return True
        except Exception as e:
            # Log error but don't crash
            logger.error(f"Plugin error: {e}")
            return False  # Fail closed
```

### 2. Configuration

```python
class MyPlugin(BasePlugin):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.timeout = self.config.get("timeout", 30)
```

### 3. Logging

```python
import logging

logger = logging.getLogger("orchestrator.plugins.my_plugin")

class MyPlugin(BasePlugin):
    async def validate(self, task_result: TaskResult) -> bool:
        logger.debug(f"Validating task {task_result.task_id}")
        # ...
        logger.info(f"Validation passed for {task_result.task_id}")
```

### 4. Performance

```python
class MyPlugin(BasePlugin):
    # Cache expensive operations
    _cache = {}
    
    async def validate(self, task_result: TaskResult) -> bool:
        # Check cache first
        if task_result.task_id in self._cache:
            return self._cache[task_result.task_id]
        
        # Expensive operation
        result = await self._expensive_check(task_result)
        
        # Cache result
        self._cache[task_result.task_id] = result
        
        return result
```

---

## EXAMPLE PLUGINS

### 1. Security Scanner Plugin

```python
class SecurityScannerPlugin(BasePlugin):
    """Scan code for security issues."""
    
    name = "security_scanner"
    plugin_type = PluginType.VALIDATOR
    
    async def validate(self, task_result: TaskResult) -> bool:
        import bandit
        from bandit.core import manager as b_manager
        
        # Run bandit security scan
        bm = b_manager.BanditManager(None, 'file')
        
        # Scan output
        # ... bandit logic ...
        
        return no_issues_found
```

### 2. License Checker Plugin

```python
class LicenseCheckerPlugin(BasePlugin):
    """Check for license headers."""
    
    name = "license_checker"
    plugin_type = PluginType.VALIDATOR
    
    async def validate(self, task_result: TaskResult) -> bool:
        output = task_result.output
        
        # Check for license header
        required_headers = [
            "MIT License",
            "Apache License",
            "BSD License",
        ]
        
        return any(h in output for h in required_headers)
```

### 3. Documentation Generator Plugin

```python
class DocGeneratorPlugin(BasePlugin):
    """Generate documentation from code."""
    
    name = "doc_generator"
    plugin_type = PluginType.INTEGRATION
    
    async def on_task_complete(self, task_result: TaskResult):
        if task_result.task_type != TaskType.CODE_GEN:
            return
        
        # Generate docs from code
        # ... doc generation logic ...
        
        # Save docs
        await self._save_docs(task_result.task_id, docs)
```

---

## TROUBLESHOOTING

### Plugin Not Loading

**Symptom**: Plugin not registered

**Fix**:
```python
# Check plugin name is unique
print(MyPlugin.name)

# Verify plugin type
print(MyPlugin.plugin_type)

# Check registration
orch.list_plugins()
```

### Plugin Crashes Orchestrator

**Symptom**: Unhandled exception crashes orchestrator

**Fix**:
```python
# Use sandboxed execution
from orchestrator.plugin_isolation import SandboxedPlugin

class MyPlugin(SandboxedPlugin):
    # Runs in isolated process
    pass
```

### Plugin Too Slow

**Symptom**: Validation takes too long

**Fix**:
```python
# Add timeout
class MyPlugin(BasePlugin):
    TIMEOUT_SECONDS = 10  # 10 second timeout
    
    async def validate(self, task_result: TaskResult) -> bool:
        return await asyncio.wait_for(
            self._validate_impl(task_result),
            timeout=self.TIMEOUT_SECONDS
        )
```

---

## SUPPORT

- **Documentation**: https://georgehadji.github.io/multi-llm-orchestrator/plugins/
- **Examples**: https://github.com/georgehadji/multi-llm-orchestrator/tree/main/examples/plugins
- **Issues**: https://github.com/georgehadji/multi-llm-orchestrator/issues

---

*Plugin development guide last updated: 2026-03-07*
