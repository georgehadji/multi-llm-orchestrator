# Meta-Optimization System

> **Inspired by Hyperagents** (arXiv:2603.19461) — Self-referential AI systems that improve their own improvement mechanisms.

## Overview

The Meta-Optimization System adds a self-improving layer to the AI Orchestrator. It analyzes execution history, identifies patterns, and proposes optimizations for:

- **Model routing strategies** — Which models work best for which task types
- **Budget allocation** — Optimal budget distribution across task types
- **Template configurations** — Best prompt templates per task/model combination
- **Concurrency settings** — Optimal parallelization levels

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Orchestrator                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Meta-Optimization Layer                   │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐ │  │
│  │  │ MetaOptimizer   │  │ SelfImprovingTemplates      │ │  │
│  │  │ - Analyze       │  │ - Track variant performance │ │  │
│  │  │ - Propose       │  │ - Propose improvements      │ │  │
│  │  │ - Evaluate      │  │ - Evolve templates          │ │  │
│  │  └────────┬────────┘  └──────────────┬──────────────┘ │  │
│  │           │                           │                │  │
│  │  ┌────────▼───────────────────────────▼────────────┐   │  │
│  │  │           ExecutionArchive                       │   │  │
│  │  │  - Store project trajectories                    │   │  │
│  │  │  - Pattern mining                                │   │  │
│  │  │  - Performance statistics                        │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from orchestrator.meta_orchestrator import MetaOptimizationIntegration, ExecutionArchive, ProjectTrajectory

# Initialize integration
integration = MetaOptimizationIntegration(orchestrator)

# After each project completion, record the execution
trajectory = ProjectTrajectory(
    project_id="my-project-001",
    project_description="Build a web API",
    total_cost=0.50,
    total_time=120.0,
    success=True,
    task_records=[...],  # List of ExecutionRecord
    model_sequence=["deepseek-chat", "claude-3-5-sonnet"],
)
await integration.record_execution(trajectory)

# Periodically optimize (e.g., every 10 projects)
proposals = await integration.maybe_optimize()
for proposal in proposals:
    print(f"Proposal: {proposal.description}")
    print(f"Expected improvement: {proposal.expected_improvement:.2%}")
```

### Self-Improving Templates

```python
from orchestrator.adaptive_templates import get_self_improving_templates

sit = get_self_improving_templates()

# Record template execution
sit.record_execution(
    task_type=TaskType.CODE_GEN,
    model=Model.DEEPSEEK_CHAT,
    variant_name="structured",
    score=0.92,
    success=True,
    cost_usd=0.005,
)

# Get improvement proposals
proposals = sit.propose_improvements(min_samples=10)
for proposal in proposals:
    print(f"{proposal['type']}: {proposal['reason']}")
```

## Components

### ExecutionArchive

Stores execution trajectories for pattern mining.

**Features:**
- Persistent storage (JSONL format)
- Performance statistics per model and task type
- Similarity-based project retrieval
- Pattern extraction for meta-optimization

**API:**
```python
archive = ExecutionArchive(archive_path=Path.home() / ".orchestrator_cache" / "archive")

# Store execution
archive.store(trajectory)

# Get statistics
stats = archive.get_model_performance("deepseek-chat")
print(f"Success rate: {stats['success_rate']:.2%}")

# Find similar projects
similar = archive.find_similar_projects("Build Python API", limit=5)

# Extract patterns
patterns = archive.get_patterns()
print(patterns["model_task_affinity"])  # Best model per task type
```

### MetaOptimizer

Analyzes archive and generates optimization proposals.

**Proposal Types:**
| Type | Description |
|------|-------------|
| `MODEL_ROUTING` | Change model routing for task types |
| `BUDGET_ALLOCATION` | Adjust budget partitions |
| `TEMPLATE_CONFIG` | Update template defaults |

**API:**
```python
optimizer = MetaOptimizer(archive, min_samples=10, improvement_threshold=0.05)

# Generate proposals
proposals = await optimizer.analyze_and_propose()

# Evaluate proposal
approved = await optimizer.evaluate_proposal(proposal)

# Apply proposal
await optimizer.apply_proposal(proposal)
```

### SelfImprovingTemplates

Evolves prompt templates based on performance.

**Features:**
- Track variant performance per task/model
- Propose retiring underperforming variants
- Promote high-performing variants to default

**API:**
```python
sit = SelfImprovingTemplates()

# Record execution
sit.record_execution(task_type, model, variant_name, score, success, cost)

# Get statistics
stats = sit.get_variant_stats("structured")

# Propose improvements
proposals = sit.propose_improvements()

# Get evolution report
report = sit.get_evolution_report()
```

## Configuration

### Optimization Settings

```python
optimizer = MetaOptimizer(
    archive,
    min_samples=10,              # Minimum executions before optimization
    improvement_threshold=0.05,  # 5% improvement required
)
```

### Archive Settings

```python
archive = ExecutionArchive(
    archive_path=Path.home() / ".orchestrator_cache" / "archive"
)
```

## Safety Mechanisms

### Staged Evaluation

Proposals go through staged evaluation:
1. **Fast simulation** — Based on historical data
2. **A/B testing** — Live execution comparison (future)
3. **Gradual rollout** — Percentage-based deployment (future)

### Rollback Capability

All proposals are tracked with status:
- `PENDING` → `EVALUATING` → `APPROVED`/`REJECTED` → `APPLIED`/`ROLLED_BACK`

### Budget Caps

Meta-optimization has separate budget allocation to prevent runaway costs.

## Metrics & Monitoring

### Get Optimization Status

```python
status = integration.get_status()
print(f"Total projects: {status['archive_stats']['total_projects']}")
print(f"Pending proposals: {status['pending_proposals']}")
print(f"Applied proposals: {status['applied_proposals']}")
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `total_projects` | Projects in archive |
| `total_executions` | Total task executions |
| `model_performance` | Per-model success rate, cost, score |
| `patterns.model_task_affinity` | Best model per task type |
| `patterns.failure_patterns` | Models with high failure rates |

## Best Practices

### When to Optimize

- **After every N projects** (recommended: N=10-20)
- **When performance degrades** (monitor success rate)
- **Before large projects** (ensure optimal configuration)

### Proposal Evaluation

- Review evidence before applying
- Start with high-confidence proposals (>0.8)
- Monitor impact after application

### Archive Management

- Archive grows indefinitely — consider periodic pruning
- Back up archive for production systems
- Archive is safe to delete (will rebuild from new executions)

## Troubleshooting

### "Insufficient data for optimization"

**Cause:** Less than `min_samples` executions recorded.

**Solution:** Run more projects or lower `min_samples` threshold.

### "No proposals generated"

**Cause:** No clear patterns detected in current data.

**Solution:** Continue executing projects; patterns emerge with more data.

### High proposal rejection rate

**Cause:** Improvement threshold too high.

**Solution:** Lower `improvement_threshold` (default: 5%).

## Advanced Topics

### Custom Pattern Extraction

Extend `ExecutionArchive.get_patterns()` for domain-specific patterns:

```python
class CustomArchive(ExecutionArchive):
    def get_patterns(self) -> Dict[str, Any]:
        patterns = super().get_patterns()
        
        # Add custom patterns
        patterns["my_custom_pattern"] = self._extract_custom_pattern()
        
        return patterns
```

### Custom Proposal Types

Extend `StrategyType` enum for new proposal types:

```python
class StrategyType(str, Enum):
    MODEL_ROUTING = "model_routing"
    BUDGET_ALLOCATION = "budget_allocation"
    MY_CUSTOM_TYPE = "my_custom_type"  # Add your own
```

### Integration with External Systems

```python
# Export archive for external analysis
with open("archive_export.jsonl", "w") as f:
    for trajectory in archive._trajectories.values():
        f.write(json.dumps(trajectory.to_dict()) + "\n")

# Import external data
for line in open("external_data.jsonl"):
    trajectory = ProjectTrajectory.from_dict(json.loads(line))
    archive.store(trajectory)
```

## Related Documentation

- [Adaptive Templates](./ADAPTIVE_TEMPLATES.md) — Template system details
- [Model Routing](./MODEL_ROUTING.md) — How model routing works
- [Budget System](./BUDGET_SYSTEM.md) — Budget allocation and tracking

## References

- **Hyperagents Paper:** arXiv:2603.19461 — Self-referential agents with metacognitive self-modification
- **Darwin Gödel Machine:** arXiv:2509.00083 — Open-ended self-improvement in coding
