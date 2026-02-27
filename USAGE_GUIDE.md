# Multi-LLM Orchestrator — Usage Guide

**Version:** 2026.02 v5.1 | **Updated:** 2026-02-26 | **CLI & Python API Reference**

**New in v5.1:** Knowledge Management • Project Management • Product Management • Quality Control

**New in v5.0:** Performance Optimization • Dashboard v5.0 • Caching Layer • KPI Monitoring

---

## Getting Started

### Installation

```bash
# Basic installation (includes aiosqlite for async cache)
pip install -e .

# With optional validators
pip install pytest ruff jsonschema

# From scratch (no editable install)
pip install openai google-genai aiosqlite pyyaml python-dotenv
```

### Environment Setup

```bash
# Create .env file with at least one provider key
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'DEEPSEEK_API_KEY=sk-...' >> .env
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

### 6. Skip Project Enhancement (Use Original Spec)

By default, the orchestrator uses Project Enhancer to suggest improvements to your project description before decomposition. To skip this and run with your exact specification:

```bash
python -m orchestrator \
  --project "Build a FastAPI service" \
  --criteria "tests pass" \
  --no-enhance
```

**When to use `--no-enhance`:**
- Your spec is already well-defined and specific
- You want to reproduce exact results (determinism)
- You prefer not to see LLM-suggested improvements
- Time-sensitive execution

### 7. Bypass Auto-Resume Detection (Always Start Fresh)

By default, the orchestrator checks for incomplete projects with similar descriptions and offers to resume them. To start a completely fresh project and skip this check:

```bash
python -m orchestrator \
  --project "Build a React dashboard" \
  --criteria "npm build succeeds" \
  --new-project
```

Or use the short flag:

```bash
python -m orchestrator --project "..." --criteria "..." -N
```

**What happens with resume detection (default):**
1. Exact keyword match → Auto-resume with message
2. Single similar project → Prompts: "Resume it? [Y/n]"
3. Multiple similar projects → Shows ranked list with scores, user picks [1–N / n]

**When to use `--new-project`:**
- You want a fresh start (don't want resume prompts)
- Debugging different approaches to the same problem
- Explicit control over execution flow

### 8. Combine Flags

You can combine `--no-enhance` and `--new-project`:

```bash
python -m orchestrator \
  --project "Build a GraphQL API" \
  --criteria "schema complete, resolvers tested" \
  --no-enhance \
  --new-project \
  --budget 4.0
```

### 9. Launch Mission Control Dashboard

```bash
# Run optimized dashboard (v5.0)
python run_optimized_dashboard.py --port 8888

# With Redis caching
python run_optimized_dashboard.py \
  --redis-host localhost \
  --redis-port 6379 \
  --port 8888

# Allow external access
python run_optimized_dashboard.py --host 0.0.0.0 --port 8888
```

**Dashboard Features:**
- Real-time metrics visualization
- Sub-100ms load time (5x improvement)
- Gzip compression & ETag support
- HTTP polling (2s debounced updates)
- KPI monitoring with alerts

**Access:** http://localhost:8888

### 10. Run Quality Gate

```bash
# Python API - run quality checks
python -c "
import asyncio
from pathlib import Path
from orchestrator import get_quality_controller, TestLevel

async def check():
    qc = get_quality_controller()
    report = await qc.run_quality_gate(
        project_id='my_project',
        project_path=Path('.'),
        levels=[TestLevel.UNIT, TestLevel.PERFORMANCE],
    )
    print(f'Quality Score: {report.quality_score:.1f}/100')
    print(f'Passed: {report.passed}')

asyncio.run(check())
"
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

### Example 13: Management Systems (v5.1)

```python
import asyncio
from pathlib import Path
from orchestrator import (
    get_knowledge_base, get_project_manager,
    get_product_manager, get_quality_controller,
    KnowledgeType, RICEScore, FeaturePriority, TestLevel
)

async def management_systems_example():
    # ---- KNOWLEDGE MANAGEMENT ----
    kb = get_knowledge_base()
    
    # Add solution to knowledge base
    await kb.add_artifact(
        type=KnowledgeType.SOLUTION,
        title="Async Race Condition Fix",
        content="Use asyncio.Lock() to protect shared state...",
        tags=["async", "python", "concurrency"],
    )
    
    # Find similar solutions
    similar = await kb.find_similar("async race condition", top_k=3)
    for artifact in similar:
        print(f"Similar: {artifact.title} ({artifact.similarity_score:.1%})")
    
    # ---- PROJECT MANAGEMENT ----
    from orchestrator.project_manager import Resource, ResourceType
    from orchestrator.models import Task, TaskType
    
    pm = get_project_manager()
    
    resources = [
        Resource("gpt-4", ResourceType.MODEL, 100, 100, 0.03),
        Resource("claude", ResourceType.MODEL, 100, 100, 0.02),
    ]
    
    tasks = [
        Task("design", TaskType.ANALYSIS, priority=9),
        Task("implement", TaskType.CODE_GENERATION, priority=8),
        Task("test", TaskType.REFACTORING, priority=7),
    ]
    
    timeline = await pm.create_schedule(
        project_id="my_project",
        tasks=tasks,
        resources=resources,
        dependencies={"test": ["implement"]},
    )
    
    print(f"Critical path: {timeline.critical_path}")
    print(f"Duration: {timeline.total_duration}")
    
    # ---- PRODUCT MANAGEMENT ----
    pm_product = get_product_manager()
    
    # RICE Score = (Reach × Impact × Confidence) / Effort
    #            = (1000 × 3 × 0.85) / 3 = 850
    rice = RICEScore(reach=1000, impact=3, confidence=85, effort=3)
    
    feature = await pm_product.add_feature(
        name="AI Code Assistant",
        description="Context-aware code suggestions",
        rice_score=rice,
        priority=FeaturePriority.P0_CRITICAL,
        tags=["ai", "productivity"],
    )
    
    # Get prioritized backlog
    backlog = pm_product.get_prioritized_backlog(limit=5)
    for f in backlog:
        print(f"{f.name}: RICE={f.rice_score.score:.0f}")
    
    # ---- QUALITY CONTROL ----
    qc = get_quality_controller()
    
    report = await qc.run_quality_gate(
        project_id="my_project",
        project_path=Path("."),
        levels=[TestLevel.UNIT, TestLevel.PERFORMANCE],
    )
    
    print(f"Quality Score: {report.quality_score:.1f}/100")
    print(f"Test Coverage: {report.average_coverage:.1f}%")
    print(f"Passed: {report.passed}")
    
    # Check for regressions
    regressions = qc.detect_regression(report)
    if regressions:
        for reg in regressions:
            print(f"REGRESSION: {reg['message']}")

asyncio.run(management_systems_example())
```

### Example 14: Performance Optimization (v5.0)

```python
import asyncio
from orchestrator import cached, get_cache, ConnectionPool

# ---- CACHING ----

@cached(ttl=300)  # Cache for 5 minutes
async def get_expensive_data(query: str):
    # This will only execute once every 5 minutes for same query
    return await fetch_from_database(query)

# Direct cache access
cache = get_cache()
await cache.set("key", value, ttl=600)
value = await cache.get("key")

# ---- CONNECTION POOLING ----

async def create_db_connection():
    # Your connection factory
    return await asyncpg.connect(DATABASE_URL)

pool = ConnectionPool(
    create_db_connection,
    min_size=2,
    max_size=10,
)

async with pool.acquire() as conn:
    await conn.execute("SELECT * FROM table")

# ---- DASHBOARD ----

# Launch optimized dashboard
# python run_optimized_dashboard.py --port 8888

# Monitor at http://localhost:8888/api/metrics
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
            allowed_models=["deepseek-reasoner", "gpt-4o"],  # only premium models
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
from orchestrator.monitoring import KPIReporter

async def main():
    orch = Orchestrator(budget=Budget(max_usd=10.0))
    
    # Monitor with KPIs
    reporter = KPIReporter()
    
    state = await orch.run_project(
        project_description="Build a FastAPI service",
        success_criteria="tests pass",
    )
    
    # Get health score
    health = await reporter.get_health_score()
    print(f"Health Score: {health['overall']:.1f}/100")
    print(f"Status: {health['status']}")

asyncio.run(main())
```

### Recipe 6: Continuous Knowledge Capture

```python
import asyncio
from orchestrator import get_knowledge_base, KnowledgeType

async def learn_from_projects():
    kb = get_knowledge_base()
    
    # Automatically learn from completed project
    await kb.learn_from_project(
        project_id="project_123",
        artifacts_dir=Path("./results"),
        decisions=[
            {
                "title": "Chose PostgreSQL over MongoDB",
                "rationale": "ACID compliance required for financial data",
                "alternatives": ["MongoDB", "DynamoDB"],
                "tags": ["database", "architecture"],
            }
        ],
    )
    
    # Query for recommendations
    recs = await kb.get_recommendations("Build payment service")
    for rec in recs:
        print(f"Suggestion: {rec['title']} ({rec['relevance']})")

asyncio.run(learn_from_projects())
```

### Recipe 7: RICE-Based Product Planning

```python
import asyncio
from orchestrator import get_product_manager, RICEScore, FeaturePriority

async def plan_product():
    pm = get_product_manager()
    
    # Add features with RICE scores
    features = [
        ("AI Assistant", RICEScore(1000, 3, 90, 3)),    # Score: 900
        ("Dark Mode", RICEScore(800, 1, 95, 1)),        # Score: 760
        ("Export PDF", RICEScore(300, 2, 80, 2)),       # Score: 240
        ("Mobile App", RICEScore(2000, 3, 60, 12)),     # Score: 300
    ]
    
    for name, rice in features:
        await pm.add_feature(
            name=name,
            description=f"Implement {name}",
            rice_score=rice,
            priority=FeaturePriority.P1_HIGH if rice.score > 500 else FeaturePriority.P2_MEDIUM,
        )
    
    # Get auto-prioritized backlog
    backlog = pm.get_prioritized_backlog(limit=10)
    print("Prioritized Backlog:")
    for i, f in enumerate(backlog, 1):
        print(f"{i}. {f.name} (RICE: {f.rice_score.score:.0f})")
    
    # Plan release with top 3 features
    release = await pm.plan_release(
        name="Q1 2024 Launch",
        version="2.0.0",
        target_date=datetime(2024, 3, 31),
        capacity=3,
    )
    print(f"Release includes {len(release.features)} features")

asyncio.run(plan_product())
```

### Recipe 8: Project Scheduling with Critical Path

```python
import asyncio
from orchestrator import get_project_manager, Resource, ResourceType
from orchestrator.models import Task, TaskType

async def schedule_project():
    pm = get_project_manager()
    
    # Define resources
    resources = [
        Resource("gpt-4", ResourceType.MODEL, 100, 100, 0.03),
        Resource("deepseek", ResourceType.MODEL, 100, 100, 0.01),
    ]
    
    # Define tasks with dependencies
    tasks = [
        Task("requirements", TaskType.ANALYSIS, priority=9),
        Task("design", TaskType.ANALYSIS, priority=8),
        Task("implement_api", TaskType.CODE_GENERATION, priority=8),
        Task("implement_ui", TaskType.CODE_GENERATION, priority=7),
        Task("integrate", TaskType.REFACTORING, priority=8),
        Task("test", TaskType.REFACTORING, priority=9),
    ]
    
    # Create schedule
    timeline = await pm.create_schedule(
        project_id="web_app",
        tasks=tasks,
        resources=resources,
        dependencies={
            "design": ["requirements"],
            "implement_api": ["design"],
            "implement_ui": ["design"],
            "integrate": ["implement_api", "implement_ui"],
            "test": ["integrate"],
        },
    )
    
    print(f"Project duration: {timeline.total_duration}")
    print(f"Critical path: {' → '.join(timeline.critical_path)}")
    print(f"Risks identified: {len(timeline.risks)}")
    for risk in timeline.risks:
        print(f"  - {risk.description} (Score: {risk.risk_score:.2f})")

asyncio.run(schedule_project())
```

### Recipe 9: Comprehensive Quality Pipeline

```python
import asyncio
from pathlib import Path
from orchestrator import get_quality_controller, TestLevel, QualitySeverity

async def quality_pipeline():
    qc = get_quality_controller()
    
    # Run comprehensive quality gate
    report = await qc.run_quality_gate(
        project_id="production_app",
        project_path=Path("."),
        levels=[
            TestLevel.UNIT,
            TestLevel.INTEGRATION,
            TestLevel.PERFORMANCE,
            TestLevel.SECURITY,
        ],
    )
    
    # Check results
    print(f"\n=== Quality Report ===")
    print(f"Overall Score: {report.quality_score:.1f}/100")
    print(f"Test Coverage: {report.average_coverage:.1f}%")
    print(f"Tests: {sum(1 for t in report.test_results if t.passed)}/{len(report.test_results)} passed")
    
    # Issues by severity
    for severity in QualitySeverity:
        issues = report.get_issues_by_severity(severity)
        if issues:
            print(f"\n{severity.value.upper()} Issues ({len(issues)}):")
            for issue in issues[:3]:  # Show first 3
                print(f"  - {issue.description}")
    
    # Check for regressions
    regressions = qc.detect_regression(report)
    if regressions:
        print(f"\n⚠️ REGRESSIONS DETECTED:")
        for reg in regressions:
            print(f"  - {reg['message']}")
    
    # Generate badge
    if report.passed:
        print(f"\n✅ Quality Gate PASSED")
        print(f"Badge: {qc.generate_badge(report)}")
    else:
        print(f"\n❌ Quality Gate FAILED")

asyncio.run(quality_pipeline())
```

### Recipe 10: Monitor & Trace a Run

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
2. Gemini Flash: $0.15
3. GPT-4o-mini: $0.15
4. DeepSeek Coder: $0.27

For fast execution with reasonable cost: DeepSeek Coder or Kimi K2.5.

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

### 8. Post-Project Analysis

Automatically analyze completed projects and get improvement suggestions:

```python
import asyncio
from orchestrator import Orchestrator, Budget
from pathlib import Path

async def analyze_after_completion():
    orch = Orchestrator(budget=Budget(max_usd=5.0))
    
    # Run project with automatic analysis
    state = await orch.run_project(
        project_description="Build a Python CLI tool",
        success_criteria="pytest passes, CLI works",
        analyze_on_complete=True,
        output_dir=Path("./results/cli_tool")
    )
    
    # Analysis runs automatically and prints suggestions
    # Output includes:
    # - Quality score (0-100)
    # - Test coverage
    # - Improvement suggestions by priority
    # - Architecture insights

# Manual analysis
from orchestrator import ProjectAnalyzer

async def manual_analysis():
    analyzer = ProjectAnalyzer()
    report = await analyzer.analyze_project(
        project_path=Path("./results/cli_tool"),
        project_id="cli_tool_001"
    )
    
    print(f"Quality Score: {report.quality_score:.1f}/100")
    print(f"Test Coverage: {report.test_coverage:.1f}%")
    
    # Print top suggestions
    for suggestion in report.suggestions[:3]:
        print(f"[{suggestion.priority.value}] {suggestion.title}")
        print(f"  Effort: {suggestion.estimated_effort}")
        print(f"  {suggestion.description[:80]}...")

asyncio.run(analyze_after_completion())
```

---

## 🐛 Debugging & Troubleshooting

Having issues? Check these resources:

| Resource | Purpose |
|----------|---------|
| [DEBUGGING_OVERVIEW.md](./DEBUGGING_OVERVIEW.md) | Navigation to all debugging docs |
| [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md) | Comprehensive debugging manual |
| [TROUBLESHOOTING_CHEATSHEET.md](./TROUBLESHOOTING_CHEATSHEET.md) | Quick fixes for common errors |
| [PROJECT_DEBUGGING.md](./PROJECT_DEBUGGING.md) | Debug generated projects |

### Quick Diagnostic

```python
from orchestrator import SystemDiagnostic, print_diagnostic_report

async def check():
    diag = SystemDiagnostic()
    report = await diag.run_full_check()
    print_diagnostic_report(report)

import asyncio
asyncio.run(check())
```

### Common Commands

```bash
# Test environment
python -c "from orchestrator import Orchestrator; print('✓ OK')"

# Check API keys
echo $OPENAI_API_KEY

# View recent errors
tail -50 logs/orchestrator.log | grep ERROR

# Run full diagnostic
python -m orchestrator.diagnostics
```

---

## Next Steps

- See **CAPABILITIES.md** for feature deep-dive
- See **README.md** for architecture details
- Check `projects/` directory for example YAML specs
- Run tests: `pytest tests/`
- Read [docs/debugging/DEBUGGING_GUIDE.md](./docs/debugging/DEBUGGING_GUIDE.md) for troubleshooting

