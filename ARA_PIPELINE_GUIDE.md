# ARA Pipeline Integration Guide

**Version:** 1.0.0  
**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Date:** 2026-03-23

**Purpose:** Integration guide and usage reference for ARA Pipeline. For model recommendations, see [ARA_MODEL_SELECTION_GUIDE.md](./ARA_MODEL_SELECTION_GUIDE.md). For phase-by-phase analysis, see [ARA_PHASE_MODEL_ANALYSIS.md](./ARA_PHASE_MODEL_ANALYSIS.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Available Reasoning Methods](#available-reasoning-methods)
4. [Method Selection](#method-selection)
5. [Configuration](#configuration)
6. [Python API](#python-api)
7. [CLI Usage](#cli-usage)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Overview

The **ARA Pipeline** (Advanced Reasoning & Analysis) integration brings 12 sophisticated reasoning methods to the AI Orchestrator. Each method implements a distinct cognitive strategy optimized for specific problem types.

### What's New

- **12 Reasoning Methods** from cognitive science and decision research
- **Intelligent Method Selection** — rule-based + LLM optimization
- **Automatic Complexity/Risk Assessment** — selects method based on task characteristics
- **Cost-Aware Routing** — respects budget constraints
- **Fallback Mechanism** — gracefully degrades to standard execution on errors

### Architecture

```
Task → Method Selector → Pipeline Factory → Execute → Result
         ↓                    ↓                  ↓
    Rule-based + LLM    Create pipeline    Run phases
    Complexity/Risk     with method        Return TaskResult
```

---

## Quick Start

### 1. Enable ARA Pipelines

```python
from orchestrator import Orchestrator, Budget
from orchestrator.ara_integration import create_ara_integration

# Create orchestrator
orch = Orchestrator(budget=Budget(max_usd=10.0))

# Create ARA integration
ara = create_ara_integration(
    client=orch.client,
    cache=orch.cache,
    telemetry=orch._telemetry,
    enabled=True,
    auto_select=True,
)

# Execute task with ARA pipeline
from orchestrator.models import Task, TaskType

task = Task(
    id="arch_001",
    type=TaskType.REASONING,
    prompt="Design authentication system for high-security financial app",
    max_output_tokens=4000,
)

result = await ara.execute_task_with_pipeline(task)
print(f"Method used: {result.metadata['ara_method']}")
print(f"Score: {result.score}")
```

### 2. Manual Method Selection

```python
from orchestrator.ara_pipelines import ReasoningMethod, PipelineFactory
from orchestrator.method_selector import select_method_for_task

# Auto-select method
selection = select_method_for_task(
    task=task,
    complexity="high",
    risk="high",
    use_llm=True,
    client=orch.client,
)

print(f"Recommended: {selection.method.value}")
print(f"Confidence: {selection.confidence}")
print(f"Cost multiplier: {selection.estimated_cost_multiplier}×")

# Execute with specific method
pipeline = PipelineFactory.create(
    method=ReasoningMethod.PRE_MORTEM,
    client=orch.client,
)

result = await pipeline.execute(task)
```

---

## Available Reasoning Methods

### Standard Methods (7)

| Method | Cost | Time | Best For |
|--------|------|------|----------|
| **Multi-Perspective** | 4.0× | 1.4× | General problem analysis |
| **Iterative** | 2.0× | 1.3× | Optimization, design |
| **Debate** | 2.5× | 1.6× | Strategic decisions, architecture |
| **Research** | 1.5× | 1.2× | Evidence-based, current events |
| **Jury** | 5.0× | 1.8× | High-stakes, critical code |
| **Scientific** | 2.0× | 1.5× | Technical decisions, algorithms |
| **Socratic** | 1.5× | 1.3× | Clarifying ambiguous requirements |

### Specialized Methods (5)

| Method | Cost | Time | Best For |
|--------|------|------|----------|
| **Pre-Mortem** ⭐ | 1.8× | 1.4× | Risk assessment, project planning |
| **Bayesian** | 2.2× | 1.5× | Decisions under uncertainty |
| **Dialectical** | 2.0× | 1.5× | Philosophical conflicts, policy |
| **Analogical** ⭐ | 1.9× | 1.4× | Innovation, cross-domain transfer |
| **Delphi** | 3.5× | 1.7× | Predictions, expert consensus |

⭐ **Recommended for most projects**

---

## Method Selection

### Automatic Selection

The `MethodSelector` uses a two-phase approach:

**Phase 1: Rule-Based Classification**
- Task type (CODE_GEN, REASONING, etc.)
- Complexity level (LOW, MEDIUM, HIGH, CRITICAL)
- Risk level (LOW, MEDIUM, HIGH, CRITICAL)

**Phase 2: LLM Optimization** (optional)
- Analyzes task prompt for keywords
- Considers cost/time constraints
- Provides confidence score

### Example

```python
from orchestrator.method_selector import MethodSelector, ComplexityLevel, RiskLevel

selector = MethodSelector(client=orch.client)

selection = selector.select_method(
    task=task,
    complexity=ComplexityLevel.HIGH,
    risk_level=RiskLevel.MEDIUM,
    use_llm_optimization=True,
)

print(f"Method: {selection.method.value}")
print(f"Rationale: {selection.rationale}")
print(f"Alternatives: {[m.value for m in selection.alternative_methods]}")
```

### Selection Rules (Examples)

| Task Type | Complexity | Risk | Recommended Methods |
|-----------|------------|------|---------------------|
| CODE_GEN | LOW | LOW | Multi-Perspective, Iterative |
| CODE_GEN | CRITICAL | HIGH | Jury, Pre-Mortem |
| REASONING | HIGH | HIGH | Bayesian, Delphi |
| CODE_REVIEW | HIGH | HIGH | Jury, Pre-Mortem |
| WRITING | MEDIUM | LOW | Analogical, Iterative |

---

## Configuration

### Environment Variables

```bash
# Enable/disable ARA pipelines
export ORCHESTRATOR_ARA_ENABLED=true

# Default method (if auto-select disabled)
export ORCHESTRATOR_ARA_DEFAULT_METHOD=multi_perspective

# Enable LLM optimization for method selection
export ORCHESTRATOR_ARA_LLM_OPTIMIZATION=true

# Cost constraints
export ORCHESTRATOR_ARA_MAX_COST_MULTIPLIER=5.0
export ORCHESTRATOR_ARA_MAX_TIME_MULTIPLIER=2.0

# Method-specific overrides (JSON)
export ORCHESTRATOR_ARA_METHOD_OVERRIDES='{"task_123": "jury", "task_456": "pre_mortem"}'
```

### Python Configuration

```python
from orchestrator.ara_integration import create_ara_integration

ara = create_ara_integration(
    client=orch.client,
    enabled=True,
    auto_select=True,
)

# Configure constraints
ara.configure(
    enabled=True,
    max_cost_multiplier=3.0,
    max_time_multiplier=1.5,
    method_overrides={
        "auth_module": "jury",
        "risk_assessment": "pre_mortem",
    },
)
```

---

## Python API

### Pipeline Execution

```python
from orchestrator.ara_pipelines import (
    ReasoningMethod,
    PipelineFactory,
    PipelineState,
)

# Create pipeline
pipeline = PipelineFactory.create(
    method=ReasoningMethod.BAYESIAN,
    client=orch.client,
    cache=orch.cache,
)

# Execute
result = await pipeline.execute(
    task=task,
    context="Previous context from dependencies",
)

# Access results
print(f"Output: {result.output}")
print(f"Score: {result.score}")
print(f"Metadata: {result.metadata}")
```

### Method Selection

```python
from orchestrator.method_selector import (
    MethodSelector,
    ComplexityLevel,
    RiskLevel,
    select_method_for_task,
)

# Convenience function
selection = select_method_for_task(
    task=task,
    complexity="high",
    risk="medium",
    use_llm=True,
    client=orch.client,
)

# Or use selector class
selector = MethodSelector(client=orch.client)
selection = selector.select_method(
    task=task,
    complexity=ComplexityLevel.HIGH,
    risk_level=RiskLevel.MEDIUM,
    budget_constraint=3.0,  # Max cost multiplier
)
```

### Integration with Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.ara_integration import ARAPipelineIntegration

# Create orchestrator
orch = Orchestrator(budget=Budget(max_usd=20.0))

# Create ARA integration
ara = ARAPipelineIntegration(
    client=orch.client,
    cache=orch.cache,
    telemetry=orch._telemetry,
)

# Override _execute_task in your workflow
async def execute_task_with_ara(task, context=""):
    return await ara.execute_task_with_pipeline(task, context)
```

---

## CLI Usage

### Run Project with ARA Pipelines

```bash
# Enable ARA pipelines
export ORCHESTRATOR_ARA_ENABLED=true
export ORCHESTRATOR_ARA_AUTO_SELECT=true

# Run project
python -m orchestrator \
  --project "Build secure authentication system" \
  --criteria "OAuth2, JWT, rate limiting" \
  --budget 20.0

# Use specific method for all tasks
export ORCHESTRATOR_ARA_DEFAULT_METHOD=jury
python -m orchestrator \
  --project "Mission-critical payment processor" \
  --criteria "PCI compliance, zero errors" \
  --budget 50.0
```

### Method-Specific Execution

```bash
# Pre-Mortem for risk assessment
export ORCHESTRATOR_ARA_METHOD_OVERRIDES='{"*": "pre_mortem"}'
python -m orchestrator \
  --project "Production deployment" \
  --criteria "Zero downtime" \
  --budget 10.0

# Analogical for innovation
export ORCHESTRATOR_ARA_DEFAULT_METHOD=analogical
python -m orchestrator \
  --project "Innovative UI interaction model" \
  --criteria "Novel, intuitive" \
  --budget 5.0
```

---

## Best Practices

### 1. Match Method to Task Type

```python
# Code generation → Multi-Perspective or Iterative
task = Task(type=TaskType.CODE_GEN, ...)
# → Multi-Perspective (balanced) or Iterative (optimization)

# Architecture decision → Debate or Dialectical
task = Task(type=TaskType.REASONING, prompt="Choose between microservices vs monolith")
# → Debate (trade-offs) or Dialectical (synthesis)

# Risk assessment → Pre-Mortem
task = Task(type=TaskType.REASONING, prompt="Production deployment plan")
# → Pre-Mortem (failure analysis)

# Innovation → Analogical
task = Task(type=TaskType.WRITING, prompt="Novel UI interaction")
# → Analogical (cross-domain transfer)
```

### 2. Use Auto-Selection with Constraints

```python
ara.configure(
    auto_select=True,
    max_cost_multiplier=3.0,  # Cap expensive methods
    max_time_multiplier=1.5,  # Ensure reasonable latency
)
```

### 3. Override for Critical Tasks

```python
ara.configure(
    method_overrides={
        "authentication": "jury",      # Highest quality
        "payment_processing": "jury",
        "deployment_plan": "pre_mortem",  # Risk assessment
        "architecture_decision": "debate",
    }
)
```

### 4. Monitor Statistics

```python
stats = ara.get_stats()
print(f"Tasks executed: {stats['tasks_executed']}")
print(f"Method distribution: {ara.get_method_distribution()}")
print(f"Avg cost multiplier: {stats['avg_cost_multiplier']:.2f}×")
```

### 5. Fallback Gracefully

ARA pipelines automatically fallback to standard execution on:
- Pipeline creation errors
- Execution failures
- Budget constraint violations

---

## Examples

### Example 1: High-Stakes Code Review

```python
from orchestrator.ara_pipelines import ReasoningMethod, PipelineFactory

task = Task(
    id="review_auth",
    type=TaskType.CODE_REVIEW,
    prompt="Review authentication module for security vulnerabilities",
    max_output_tokens=4000,
)

# Use Jury for maximum quality
pipeline = PipelineFactory.create(
    method=ReasoningMethod.JURY,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Security issues found: {result.output[:500]}")
```

### Example 2: Architecture Decision with Debate

```python
from orchestrator.ara_pipelines import ReasoningMethod

task = Task(
    id="arch_decision",
    type=TaskType.REASONING,
    prompt="Should we use microservices or monolith for our startup?",
    max_output_tokens=4000,
)

# Use Debate to explore trade-offs
pipeline = PipelineFactory.create(
    method=ReasoningMethod.DEBATE,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Decision rationale: {result.output}")
```

### Example 3: Risk Assessment with Pre-Mortem

```python
from orchestrator.ara_pipelines import ReasoningMethod

task = Task(
    id="deployment_risk",
    type=TaskType.REASONING,
    prompt="Deploy new payment system to production",
    max_output_tokens=4000,
)

# Use Pre-Mortem to identify failure modes
pipeline = PipelineFactory.create(
    method=ReasoningMethod.PRE_MORTEM,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Failure narrative: {result.metadata.get('failure_narrative', '')[:300]}")
print(f"Safeguards: {result.metadata.get('safeguards', [])}")
```

### Example 4: Innovation with Analogical Transfer

```python
from orchestrator.ara_pipelines import ReasoningMethod

task = Task(
    id="ui_innovation",
    type=TaskType.WRITING,
    prompt="Design innovative UI for music creation app",
    max_output_tokens=4000,
)

# Use Analogical for cross-domain inspiration
pipeline = PipelineFactory.create(
    method=ReasoningMethod.ANALOGICAL,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Source domains: {result.metadata.get('source_domains', [])}")
print(f"Solution: {result.output}")
```

### Example 5: Full Project with Mixed Methods

```python
from orchestrator import Orchestrator, Budget
from orchestrator.ara_integration import create_ara_integration

orch = Orchestrator(budget=Budget(max_usd=50.0))
ara = create_ara_integration(
    client=orch.client,
    auto_select=True,
)

# Configure method overrides for critical tasks
ara.configure(
    method_overrides={
        "auth": "jury",
        "payment": "jury",
        "deployment": "pre_mortem",
        "architecture": "debate",
    },
    max_cost_multiplier=4.0,
)

# Run project
state = await orch.run_project(
    project_description="Build e-commerce platform with payment processing",
    success_criteria="All tests pass, PCI compliant",
    analyze_on_complete=True,
)

# Check ARA statistics
print(f"ARA stats: {ara.get_stats()}")
print(f"Method distribution: {ara.get_method_distribution()}")
```

---

## Troubleshooting

### Issue: Method selection always returns Multi-Perspective

**Solution:** Enable auto-select and LLM optimization:
```python
ara.configure(
    auto_select=True,
    use_llm_for_selection=True,
)
```

### Issue: Pipeline execution fails

**Solution:** Check logs for specific error. Common causes:
- Model unavailable → Check `api_health`
- Budget exceeded → Increase `max_cost_multiplier`
- Invalid task type → Ensure task has valid `TaskType`

### Issue: High costs

**Solution:** Set cost constraints:
```python
ara.configure(
    max_cost_multiplier=2.0,  # Limit to 2× baseline
)
```

Or use cheaper methods:
```python
ara.configure(
    default_method=ReasoningMethod.ITERATIVE,  # 2.0× vs 4.0×
)
```

---

## API Reference

### ReasoningMethod Enum

```python
class ReasoningMethod(str, Enum):
    MULTI_PERSPECTIVE = "multi_perspective"
    ITERATIVE = "iterative"
    DEBATE = "debate"
    RESEARCH = "research"
    JURY = "jury"
    SCIENTIFIC = "scientific"
    SOCRATIC = "socratic"
    PRE_MORTEM = "pre_mortem"
    BAYESIAN = "bayesian"
    DIALECTICAL = "dialectical"
    ANALOGICAL = "analogical"
    DELPHI = "delphi"
```

### MethodSelection Dataclass

```python
@dataclass
class MethodSelection:
    method: ReasoningMethod
    confidence: float           # 0-1 confidence score
    rationale: str              # Human-readable explanation
    alternative_methods: List[ReasoningMethod]
    estimated_cost_multiplier: float
    estimated_time_multiplier: float
```

### PipelineFactory

```python
class PipelineFactory:
    @classmethod
    def create(
        cls,
        method: ReasoningMethod,
        client: UnifiedClient,
        cache: Optional[DiskCache] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> BasePipeline
```

---

## Migration Guide

### From Standard Orchestrator

**Before:**
```python
orch = Orchestrator()
result = await orch.run_project(...)
```

**After (with ARA):**
```python
orch = Orchestrator()
ara = create_ara_integration(client=orch.client)

# Override task execution
original_execute = orch._execute_task
orch._execute_task = lambda task, ctx: ara.execute_task_with_pipeline(task, ctx)

result = await orch.run_project(...)
```

---

## Future Enhancements

- [ ] Custom pipeline definitions (user-defined methods)
- [ ] A/B testing framework for method comparison
- [ ] Learning from execution outcomes (reinforcement learning)
- [ ] Real-time method switching based on intermediate results
- [ ] Ensemble methods (combine multiple methods)

---

## References

- **Multi-Perspective:** Evans (2018) - "Thinking Twice: System 1 and System 2"
- **Pre-Mortem:** Klein (1989) - "Recognition-Primed Decision Making"
- **Bayesian:** Jaynes (2003) - "Probability Theory: The Logic of Science"
- **Delphi:** Dalkey & Helmer (1963) - "An Experimental Application of the Delphi Method"
- **Analogical:** Gentner (1983) - "Structure-Mapping Theory"
- **Dialectical:** Hegel (1807) - "Phenomenology of Spirit"

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [ARA_PIPELINE_GUIDE.md](./ARA_PIPELINE_GUIDE.md) | **Integration & usage guide** (this file) |
| [ARA_MODEL_SELECTION_GUIDE.md](./ARA_MODEL_SELECTION_GUIDE.md) | **Model recommendations** — Top 3 models per method |
| [ARA_PHASE_MODEL_ANALYSIS.md](./ARA_PHASE_MODEL_ANALYSIS.md) | **Phase-by-phase analysis** — Detailed phase breakdowns |

---

*Last updated: 2026-03-23*  
*Author: Georgios-Chrysovalantis Chatzivantsidis*
