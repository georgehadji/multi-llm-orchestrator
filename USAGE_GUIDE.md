# Multi-LLM Orchestrator — Usage Guide

**Version:** 2026.02 | **Quick Start for Common Tasks**

---

## Getting Started

### Installation

```bash
# Basic installation (includes aiosqlite for async cache)
pip install -e .

# With optional validators
pip install pytest ruff jsonschema

# From scratch (no editable install)
pip install openai anthropic google-genai aiosqlite pyyaml python-dotenv
```

### Environment Setup

```bash
# Create .env file with at least one provider key
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
echo 'GOOGLE_API_KEY=AIzaSy...' >> .env

# Load environment
source .env  # or export individually
```

---

## CLI Quick Start

### 1. Build a FastAPI Service (Simplest)

```bash
python -m orchestrator \
  --project "Build a FastAPI authentication service with JWT tokens" \
  --criteria "All endpoints tested, OpenAPI docs complete, requirements.txt present" \
  --budget 8.0 \
  --output-dir ./results
```

**What happens:**
- Automatically detects app type (fastapi)
- Decomposes into ~10–12 tasks (setup, models, auth routes, tests, docs, etc.)
- Routes tasks to cheapest suitable models
- Cross-provider review of critical code
- Saves all outputs to `./results/`

### 2. Build a Next.js App

```bash
python -m orchestrator \
  --project "Build a Next.js e-commerce storefront with product listing, cart, checkout" \
  --criteria "npm build succeeds, pages load, no console errors" \
  --output-dir ./storefront
```

**Auto-detected scaffolding:**
- Next.js 14 config + tailwind.config.ts
- TypeScript setup + tsconfig.json
- Framer Motion animations
- Tests with Jest

### 3. From Project YAML File

```bash
python -m orchestrator --file projects/example_full.yaml --output-dir ./results
```

**YAML structure:**
```yaml
project:
  title: "Rate limiter library"
  description: "Implement token bucket rate limiter"
  success_criteria: "pytest passes, ruff clean, README complete"

budget:
  max_usd: 5.0
  max_time_seconds: 3600

policy:
  enforcement_mode: hard
  allowed_providers: [openai, anthropic]
```

### 4. Resume Interrupted Run

```bash
python -m orchestrator --resume <project_id>
```

**State is checkpointed automatically after each task.**

### 5. List All Projects

```bash
python -m orchestrator --list-projects
```

---

## Python API Examples

### Example 1: Basic Usage

```python
import asyncio
from orchestrator import Orchestrator, Budget

async def main():
    budget = Budget(max_usd=5.0, max_time_seconds=3600)
    orch = Orchestrator(budget=budget)

    state = await orch.run_project(
        project_description="Implement a Python rate limiter using decorators",
        success_criteria="pytest suite passes, ruff linting clean",
    )

    print(f"Status: {state.status.value}")
    print(f"Spent: ${state.budget.spent_usd:.4f}")
    print(f"Tasks completed: {len([r for r in state.results.values() if r.score >= 0.85])}")

asyncio.run(main())
```

### Example 2: With Cost Prediction

```python
import asyncio
from orchestrator import Orchestrator, Budget, CostPredictor, CostForecaster

async def main():
    predictor = CostPredictor(alpha=0.1)  # EMA with alpha=0.1

    # Forecast before running
    tasks = [...]  # your task list
    budget = Budget(max_usd=10.0)
    report = CostForecaster.forecast(tasks, {}, predictor, budget)

    print(f"Estimated total: ${report.estimated_total_usd:.4f}")
    print(f"Risk level: {report.risk_level.value}")  # low/medium/high

    if report.risk_level.value != "high":
        orch = Orchestrator(budget=budget, cost_predictor=predictor)
        state = await orch.run_project(...)

asyncio.run(main())
```

### Example 3: Multi-Objective Optimization

```python
import asyncio
from orchestrator import Orchestrator, Budget
from orchestrator.optimization import ParetoBackend, WeightedSumBackend

async def main():
    orch = Orchestrator(budget=Budget(max_usd=10.0))

    # Option 1: Pareto-optimal routing
    orch.set_optimization_backend(ParetoBackend())

    # Option 2: Weighted sum (tune trade-off)
    orch.set_optimization_backend(WeightedSumBackend(
        w_quality=0.6,   # prioritize quality
        w_cost=0.2,
        w_trust=0.2,
    ))

    state = await orch.run_project(...)

asyncio.run(main())
```

### Example 4: Policy Governance

```python
from orchestrator import Orchestrator, Budget, Policy, PolicySet, EnforcementMode

# Define policies
policies = PolicySet(
    policies=[
        Policy(
            name="gdpr",
            allowed_regions=["eu", "global"],
            allow_training_on_output=False,
            enforcement_mode=EnforcementMode.HARD,
        ),
        Policy(
            name="cost_cap",
            max_cost_per_task_usd=0.50,
            max_latency_ms=5000.0,
            enforcement_mode=EnforcementMode.SOFT,
        ),
    ]
)

orch = Orchestrator(budget=Budget(max_usd=10.0), policy_set=policies)
state = asyncio.run(orch.run_project(...))
```

### Example 5: Event Hooks & Live Monitoring

```python
import asyncio
from orchestrator import Orchestrator, Budget
from orchestrator.hooks import EventType

async def main():
    orch = Orchestrator(budget=Budget(max_usd=10.0))

    # Subscribe to task completion
    orch.add_hook(EventType.TASK_COMPLETED, lambda task_id, result, **_:
        print(f"✓ {task_id}: score={result.score:.3f}, cost=${result.cost_usd:.4f}"))

    # Subscribe to budget warnings
    orch.add_hook(EventType.BUDGET_WARNING, lambda phase, ratio, **_:
        print(f"⚠ {phase}: {ratio:.0%} of budget used"))

    # Subscribe to validation failures
    orch.add_hook(EventType.VALIDATION_FAILED, lambda task_id, **_:
        print(f"✗ {task_id}: validator failed"))

    state = await orch.run_project(...)

asyncio.run(main())
```

### Example 6: Metrics Export

```python
import asyncio
from orchestrator import Orchestrator, Budget
from orchestrator.metrics import ConsoleExporter, JSONExporter, PrometheusExporter

async def main():
    orch = Orchestrator(budget=Budget(max_usd=10.0))

    # Export to console (ASCII table)
    orch.set_metrics_exporter(ConsoleExporter())
    state = await orch.run_project(...)
    orch.export_metrics()  # prints ASCII table

    # Export to JSON (for dashboards)
    orch.set_metrics_exporter(JSONExporter("/tmp/metrics.json"))
    orch.export_metrics()

    # Export to Prometheus format
    orch.set_metrics_exporter(PrometheusExporter("/var/lib/node_exporter/orch.prom"))
    orch.export_metrics()

asyncio.run(main())
```

### Example 7: Ensemble with AgentPool

```python
import asyncio
from orchestrator import Orchestrator, Budget, AgentPool
from orchestrator.optimization import ParetoBackend, GreedyBackend

async def main():
    pool = AgentPool()

    # Add multiple orchestrators with different strategies
    pool.add_agent("pareto", Orchestrator(
        budget=Budget(max_usd=5.0),
        optimization_backend=ParetoBackend(),
    ))
    pool.add_agent("greedy", Orchestrator(
        budget=Budget(max_usd=5.0),
        optimization_backend=GreedyBackend(),
    ))

    # Run both in parallel
    specs = {
        "pareto": "Build a rate limiter library",
        "greedy": "Build a rate limiter library",
    }
    results = await pool.run_parallel(specs)

    # Pick best result
    best = pool.best_result(results)
    print(f"Best strategy: {best}")

    # Merge telemetry across all agents
    merged = pool.merge_telemetry()

asyncio.run(main())
```

### Example 8: App Builder with Architecture Advisor

```python
import asyncio
from orchestrator import AppBuilder

async def main():
    builder = AppBuilder()

    # AppBuilder now uses ArchitectureAdvisor to decide architecture before generation
    result = await builder.build(
        description="Microservice-ready REST API with JWT auth, database persistence, WebSocket support",
        criteria="All endpoints tested, OpenAPI docs complete, async handlers for high throughput",
        output_dir="./my_api",
        app_type="fastapi",  # auto-detected if omitted
    )

    print(f"✓ Status: {result.success}")
    print(f"✓ App Type: {result.profile.app_type}")
    print(f"✓ Architecture Pattern: {result.profile.structural_pattern}")
    print(f"✓ Topology: {result.profile.topology}")
    print(f"✓ API Paradigm: {result.profile.api_paradigm}")
    print(f"✓ Data Storage: {result.profile.data_paradigm}")
    print(f"✓ Rationale: {result.profile.rationale}")
    print(f"✓ Files generated: {len(result.assembly.files_written)}")

asyncio.run(main())
```

**Output Example:**
```
✓ Status: True
✓ App Type: fastapi
✓ Architecture Pattern: hexagonal
✓ Topology: monolith
✓ API Paradigm: rest
✓ Data Storage: relational
✓ Rationale: Hexagonal architecture enables easy testing with port abstractions.
            Monolith avoids operational complexity at this scale. REST is
            standard for web services. PostgreSQL for ACID consistency.
✓ Files generated: 23
```

**How It Works:**
1. **ArchitectureAdvisor** analyzes your description and constraints
2. **DeepSeek Chat** (or Reasoner for complex specs) decides the best architecture
3. **Architecture decision is printed** to terminal (🏗 summary block)
4. **Decomposition prompt is enriched** with architectural constraints
5. **All generated code follows the chosen architecture**

### Example 9: Architecture Advisor (Standalone Usage)

```python
import asyncio
from orchestrator import ArchitectureAdvisor

async def main():
    advisor = ArchitectureAdvisor()

    # Get architectural recommendations without building
    decision = await advisor.analyze(
        description="Real-time collaborative document editor with conflict resolution",
        criteria="Low latency, supports 100+ concurrent users, auto-save",
    )

    print(f"Structural pattern: {decision.structural_pattern}")
    print(f"Topology: {decision.topology}")
    print(f"API paradigm: {decision.api_paradigm}")
    print(f"Data paradigm: {decision.data_paradigm}")
    print(f"Rationale: {decision.rationale}")

asyncio.run(main())
```

**Output:**
```
Structural pattern: event-driven
Topology: microservices
API paradigm: graphql
Data paradigm: document
Rationale: Event-driven architecture with eventual consistency enables
           low-latency conflict resolution across services. Document
           storage (MongoDB) naturally models collaborative state.
           GraphQL simplifies real-time subscriptions for clients.
```

### Example 11: Policy DSL (YAML)

```python
from orchestrator.policy_dsl import load_policy_file, PolicyAnalyzer

# Load from YAML
hierarchy = load_policy_file("policies.yaml")

# Or from JSON (always works)
hierarchy = load_policy_file("policies.json")

# Static analysis: detect contradictions
report = PolicyAnalyzer.analyze(hierarchy.policies_for(team="eng"))
if not report.is_clean():
    print("Policy errors:", report.errors)
    print("Policy warnings:", report.warnings)
```

**Example policies.yaml:**
```yaml
global:
  - name: gdpr
    allow_training_on_output: false
    enforcement_mode: hard
    max_cost_per_task_usd: 1.0

team:
  eng:
    - name: eu_only
      allowed_regions: [eu, global]
      cost_cap_usd: 0.50

job:
  job_001:
    - name: high_quality
      min_quality_score: 0.90
```

### Example 12: OTEL Tracing

```python
import asyncio
from orchestrator import Orchestrator, Budget, TracingConfig, configure_tracing

async def main():
    # Configure tracing
    tracing_config = TracingConfig(
        enabled=True,
        otlp_endpoint="http://localhost:4318",
        service_name="orchestrator",
        sample_rate=1.0,  # 100% sampling for development
    )
    configure_tracing(tracing_config)

    orch = Orchestrator(budget=Budget(max_usd=10.0))
    state = await orch.run_project(...)
    # Traces automatically sent to OTLP endpoint

asyncio.run(main())
```

---

## Advanced Recipes

### Recipe 1: Cost-Optimized Run (All Cheap Models)

```python
from orchestrator import Orchestrator, Budget
from orchestrator.optimization import WeightedSumBackend

orch = Orchestrator(
    budget=Budget(max_usd=2.0),  # strict budget
    optimization_backend=WeightedSumBackend(
        w_quality=0.2,   # lower quality tolerance
        w_cost=0.7,      # prioritize cost
        w_trust=0.1,
    ),
)
```

### Recipe 2: Quality-First Run (All Premium Models)

```python
from orchestrator import (
    Orchestrator, Budget, Policy, PolicySet, EnforcementMode
)

policies = PolicySet(
    policies=[
        Policy(
            name="quality_first",
            allowed_models=["claude-opus", "gpt-4o"],  # only top models
            min_quality_score=0.95,
            enforcement_mode=EnforcementMode.HARD,
        ),
    ]
)

orch = Orchestrator(
    budget=Budget(max_usd=50.0),  # generous budget
    policy_set=policies,
)
```

### Recipe 3: Multi-Run with Cumulative Budget

```python
from orchestrator import Orchestrator, Budget, BudgetHierarchy

# Set org-wide spending limits
hier = BudgetHierarchy(
    org_max_usd=100.0,
    team_budgets={"eng": 50.0, "research": 30.0},
    job_budgets={"job_001": 10.0, "job_002": 15.0},
)

# Each run checks against hierarchy
orch = Orchestrator(
    budget=Budget(max_usd=10.0),
    budget_hierarchy=hier,
)
# raises ValueError if any cap would be exceeded
```

### Recipe 4: Interactive Policy Refinement

```python
import asyncio
from orchestrator import Orchestrator, Budget, OrchestrationAgent

async def main():
    agent = OrchestrationAgent()

    # Generate draft from intent
    draft = await agent.draft(
        "Pipeline for code review, EU data only, max $2"
    )

    print("Draft job spec:", draft.job)
    print("Draft policy spec:", draft.policy)
    print("Rationale:", draft.rationale)

    # User provides feedback
    refined = await agent.refine(draft, "Increase budget to $5, allow US models")

    # Submit to control plane
    from orchestrator import ControlPlane
    control_plane = ControlPlane()
    state = await control_plane.submit(refined.job, refined.policy)

asyncio.run(main())
```

### Recipe 5: Monitor & Trace a Run

```python
import asyncio
from orchestrator import Orchestrator, Budget, ProjectEventBus

async def main():
    orch = Orchestrator(budget=Budget(max_usd=10.0))

    # Start run in background
    run_task = asyncio.create_task(
        orch.run_project(
            project_description="Build a rate limiter",
            success_criteria="pytest passes",
        )
    )

    # Simultaneously monitor events
    bus = orch.get_event_bus()
    monitor_task = asyncio.create_task(monitor_events(bus))

    state = await run_task
    await monitor_task

async def monitor_events(bus):
    async for event in bus.subscribe():
        if event.type == "TaskCompleted":
            print(f"✓ {event.data['task_id']}: {event.data['score']:.3f}")
        elif event.type == "BudgetWarning":
            print(f"⚠ {event.data['phase']}: {event.data['ratio']:.0%}")
        elif event.type == "ProjectCompleted":
            print(f"🏁 Project finished: {event.data['status']}")

asyncio.run(main())
```

---

## Troubleshooting

### Q: "Provider not available" error

**A:** Ensure environment variable is set correctly:
```bash
echo $OPENAI_API_KEY  # should print your key (not empty)
```

### Q: Budget exceeded but I set a limit

**A:** Budget is checked **before each task** and **mid-iteration**, not mid-API-call. Set budget 10–15% below true ceiling:
```python
# True ceiling: $10
budget = Budget(max_usd=8.5)  # safer buffer
```

### Q: Run is slow

**A:** Check concurrency and model routing:
```python
orch = Orchestrator(
    budget=budget,
    max_concurrency=6,  # increase from default 3
)
```

### Q: Resume doesn't work

**A:** Save project ID on initial run:
```python
state = await orch.run_project(
    ...,
    project_id="my-project-001",  # explicit ID
)

# Later:
state = await orch.run_project(..., project_id="my-project-001")  # resumes
```

### Q: Which models are cheapest?

**A:** Check this ranking (per 1M tokens input):
1. Kimi K2.5: $0.14
2. DeepSeek Chat: $0.27
3. Gemini Flash: $0.15
4. GPT-4o-mini: $0.15

For fast execution with reasonable cost: DeepSeek Chat or Kimi K2.5.

---

## Best Practices

### 1. Always Set Explicit Budget

```python
# Good
budget = Budget(max_usd=5.0, max_time_seconds=3600)

# Risky (uses defaults)
budget = Budget()  # $8.0, 90 min
```

### 2. Use Project Files for Complex Specs

```bash
# Good (reproducible)
python -m orchestrator --file projects/my_project.yaml

# Less ideal (long CLI string)
python -m orchestrator --project "very long description..."
```

### 3. Export Metrics for Analysis

```python
from orchestrator.metrics import JSONExporter

orch.set_metrics_exporter(JSONExporter("/tmp/metrics.json"))
orch.export_metrics()  # analyze later
```

### 4. Use Policy DSL for Governance

```python
# Good (externalized)
hierarchy = load_policy_file("policies.yaml")

# Less maintainable (hardcoded policies)
policies = PolicySet(policies=[...])
```

### 5. Add Event Hooks for Monitoring

```python
# Good (real-time feedback)
orch.add_hook(EventType.TASK_COMPLETED, my_callback)

# Silent (no feedback)
state = await orch.run_project(...)  # black box
```

### 6. Use ArchitectureAdvisor for Code Generation

```python
# Good: Let ArchitectureAdvisor decide (automatic in AppBuilder)
builder = AppBuilder()
result = await builder.build(
    description="REST API with high-throughput requirements",
    criteria="All endpoints tested",
    output_dir="./api",
)
# Inspect the chosen architecture
print(result.profile.structural_pattern)  # informed decision

# Less ideal: Hardcoded app_type without architectural consideration
result = await builder.build(
    description="...",
    criteria="...",
    app_type="fastapi",  # assumes you know the best architecture
)
```

### 7. Provide Rich Descriptions for Better Architecture Decisions

```python
# Good: Specific constraints help the LLM decide
description = """
Build a real-time chat application with:
- 10k concurrent users support
- Group messaging
- Message persistence
- Read receipts
- Typing indicators
"""

# Less ideal: Vague description
description = "Build a chat app"
```

---

## Next Steps

- See **CAPABILITIES.md** for feature deep-dive
- See **README.md** for architecture details
- Check `projects/` directory for example YAML specs
- Run tests: `pytest tests/`

