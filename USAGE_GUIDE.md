# Multi-LLM Orchestrator — Usage Guide

**Version:** 2026.03 v6.2 | **Updated:** 2026-03-23 | **CLI & Python API Reference**

**New in v6.2:** ARA Pipeline (12 Advanced Reasoning Methods) • **Nexus Search Integration** • Intelligent Method Selection • Cost-Aware Routing

**New in v6.1:** Production Optimizations (-35% cost) • Command Center Dashboard • Tool Safety Validation

**New in v6.0:** Black Swan Resilience • Mission-Critical Monitoring • RBAC & Audit Logging

**New in v5.2:** Author Attribution • Smart Validator Filtering • Code Output Cleaning • Temperature Optimization

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

### 5. Code Generation Features

All generated code includes automatic documentation:

#### Author Attribution
Every code file automatically includes the author header:
```javascript
/**
 * Author: Georgios-Chrysovalantis Chatzivantsidis
 * Description: Performance manager for WebGL engine
 */
```

#### Thorough Comments
- Every function has JSDoc/docstring comments
- Complex logic blocks include inline explanations
- Classes include purpose and usage documentation

#### Smart Validation
- Python validators (`ruff`, `pytest`, `python_syntax`) auto-removed for HTML/CSS/JS
- Temperature optimized: 0.0 for code (deterministic), 0.2 for review
- Output cleaning removes markdown fences and placeholder comments

### 6. List All Projects

```bash
python -m orchestrator --list-projects
```

### 7. Skip Project Enhancement (Use Original Spec)

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

### 8. Bypass Auto-Resume Detection (Always Start Fresh)

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

### 9. Combine Flags

You can combine `--no-enhance` and `--new-project`:

```bash
python -m orchestrator \
  --project "Build a GraphQL API" \
  --criteria "schema complete, resolvers tested" \
  --no-enhance \
  --new-project \
  --budget 4.0
```

### 10. Launch Mission Control Dashboard

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

### 11. Run Quality Gate

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

### Example 10: Architecture Rules Engine with LLM Optimization

```python
import asyncio
from orchestrator import ArchitectureRulesEngine
from orchestrator.api_clients import UnifiedClient

async def main():
    # Create engine with LLM client for optimization
    client = UnifiedClient()
    engine = ArchitectureRulesEngine(client=client)
    
    # Generate rules with two-phase decision (rule-based + LLM optimization)
    rules = await engine.generate_rules(
        description="Build real-time analytics dashboard with live updates",
        criteria="High performance, scalable to 10k users, responsive UI",
        project_type="web_frontend"
    )
    
    # Check how the decision was made
    print(f"LLM Generated: {rules._llm_generated}")   # False (rule-based base)
    print(f"LLM Optimized: {rules._llm_optimized}")   # True (improved by LLM)
    
    # Print full summary
    print(engine.generate_summary(rules))
    
    # Save rules to output directory
    engine.save_rules(rules, Path("./output"))

asyncio.run(main())
```

**Output:**
```
🏗️ ARCHITECTURE DECISION
============================================================

Decided by: LLM (Rule-based → Optimized)

Style: Event Driven
Paradigm: Object Oriented
API: GraphQL
Database: Document

Technology Stack:
  Primary: typescript
  Frameworks: react, next.js
  Libraries: tailwindcss, zustand, recharts
  Databases: mongodb

Key Constraints:
  • All code must be type-annotated
  • Maximum cyclomatic complexity of 10 per function
  • Minimum 80% test coverage

Recommended Patterns:
  • Repository Pattern
  • Dependency Injection
  • Event Sourcing
  • Pub/Sub

Rules file: .orchestrator-rules.yml
============================================================
```

**How It Works:**
1. **Phase 1 — Rule-Based Detection:** Analyzes description for keywords (event, real-time, react)
2. **Phase 2 — LLM Optimization:** LLM reviews and suggests improvements if beneficial
3. **Decision Tracking:** Metadata shows how the decision was made (`_llm_optimized` flag)
4. **Constraint Enforcement:** Generated rules apply to all subsequent code generation

**Without LLM (Rule-based only):**
```python
# No client = no optimization
engine = ArchitectureRulesEngine()

rules = await engine.generate_rules(
    description="Build REST API",
    criteria="High performance"
)

print(engine.generate_summary(rules))
# Output: "Decided by: Rule-based"
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

### Example 15: Mission-Critical Command Center (v6.0)

```python
import asyncio
from orchestrator import Orchestrator
from orchestrator.command_center_integration import enable_command_center
from orchestrator.command_center_server import get_command_center_server, Severity

async def command_center_example():
    # Start orchestrator with command center integration
    orch = Orchestrator()
    cc = enable_command_center(orch)
    
    # Start WebSocket server (in production, run separately)
    server = get_command_center_server()
    # await server.start(host="0.0.0.0", port=8765)
    
    # Run project - dashboard auto-updates
    state = await orch.run_project(
        project_description="Build a REST API",
        success_criteria="All tests pass",
    )
    
    # Raise custom alerts from your code
    server.raise_alert(
        severity=Severity.WARNING,
        title="Custom Integration Alert",
        message="External service latency elevated",
        source="my_integration",
    )

# Access dashboard at: orchestrator/CommandCenter.html
# Or serve: python -m http.server 8080 --directory orchestrator
```

**Dashboard Features:**
- **Real-time metrics:** Model health, task queue, cost burn rate, quality scores
- **Alerting:** 5-level severity (Normal/Info/Warning/Critical/Failure)
- **Reliability:** WebSocket → SSE → polling graceful degradation
- **Security:** RBAC (viewer/operator/admin), immutable audit log

### Example 16: Production Optimizations (v6.1)

```python
import asyncio
from orchestrator import Orchestrator, Budget

async def optimizations_example():
    # All optimizations enabled by default in v6.1+
    orch = Orchestrator(budget=Budget(max_usd=5.0))
    
    # Check semantic cache statistics
    cache_stats = orch._semantic_cache.get_stats()
    print(f"Cache entries: {cache_stats['entries']}")
    print(f"Hot entries: {cache_stats['hot_entries']}")
    print(f"Avg quality: {cache_stats['avg_quality']:.2f}")
    
    # View tier escalation history
    print(f"Tier escalations: {orch._tier_escalation_count}")
    
    # Run project - optimizations apply automatically:
    # 1. Confidence-Based Early Exit (saves ~25% iterations)
    # 2. Tiered Model Selection (saves ~22% cost)
    # 3. Semantic Sub-Result Caching (saves ~15% cost)
    # 4. Fast Regression Detection (EMA α=0.2)
    state = await orch.run_project(
        project_description="Build a microservice",
        success_criteria="Docker build succeeds",
    )
    
    # Check cache after run
    updated_stats = orch._semantic_cache.get_stats()
    print(f"New cache entries: {updated_stats['entries'] - cache_stats['entries']}")

# Expected: 35% cost reduction vs v6.0
```

**Optimization Details:**

| Optimization | Mechanism | Impact |
|--------------|-----------|--------|
| **Confidence-Based Early Exit** | Exits when stable high performance (variance < 0.001) | -25% iterations |
| **Tiered Model Selection** | CHEAP→BALANCED→PREMIUM escalation | -22% cost |
| **Semantic Sub-Result Caching** | Pattern-based caching (not exact match) | -15% cost |
| **Fast Regression Detection** | EMA α=0.2 (was 0.1) | 2× faster response |

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

## Provider & Models Reference

### Model Pricing (per 1M tokens)

| Model | Input | Output | Provider | Best For |
|-------|-------|--------|----------|----------|
| **Gemini Flash Lite** | $0.075 | $0.30 | Google | Ultra-cheap tasks |
| **Gemini Flash** | $0.15 | $0.60 | Google | General purpose |
| **GPT-4o-mini** | $0.15 | $0.60 | OpenAI | Reliable cheap option |
| **DeepSeek Chat** | $0.28 | $0.42 | DeepSeek | **Best value code** |
| **DeepSeek Reasoner** | $0.28 | $0.42 | DeepSeek | **Best value reasoning** |
| **MiniMax-Text-01** | $0.50 | $1.50 | MiniMax | Frontier reasoning |
| **Claude 3 Haiku** | $0.25 | $1.25 | Anthropic | Fast & cheap |
| **o4-mini** | $1.50 | $6.00 | OpenAI | OpenAI reasoning |
| **Claude 3.5 Sonnet** | $3.00 | $15.00 | Anthropic | **Best coding** |
| **Gemini Pro** | $1.25 | $10.00 | Google | Premium quality |
| **GPT-4o** | $2.50 | $10.00 | OpenAI | Premium quality |

### Environment Variables

```bash
# Required (at least one)
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MINIMAX_API_KEY="..."

# Optional
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export ORCHESTRATOR_LOG_LEVEL="INFO"
```

### Model Capabilities

| Model | Context | Temperature | Vision | Tools | Reasoning |
|-------|---------|-------------|--------|-------|-----------|
| Gemini Flash Lite | 1M | 0-1.0 | ❌ | ✅ | ❌ |
| DeepSeek Chat | 64K | 0-2.0 | ❌ | ✅ | ❌ |
| DeepSeek Reasoner | 64K | Ignored | ❌ | ✅ | ✅ |
| Claude 3.5 Sonnet | 200K | 0-1.0 | ✅ | ✅ | ❌ |
| Claude 3 Haiku | 200K | 0-1.0 | ❌ | ❌ | ❌ |
| GPT-4o | 128K | 0-2.0 | ✅ | ✅ | ❌ |
| o4-mini | 128K | Ignored | ❌ | ✅ | ✅ |

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
1. Gemini Flash Lite: $0.075
2. Claude 3 Haiku: $0.25
3. DeepSeek Chat: $0.28
4. Gemini Flash: $0.15
5. GPT-4o-mini: $0.15

For fast execution with reasonable cost: DeepSeek Chat or Gemini Flash.

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

## 🧠 ARA Pipeline — Advanced Reasoning Methods

**New in v6.2:** 12 sophisticated reasoning methods from cognitive science and decision research.

### Overview

The ARA Pipeline (Advanced Reasoning & Analysis) provides 12 distinct reasoning strategies optimized for specific problem types:

| Method | Cost | Best For |
|--------|------|----------|
| **Multi-Perspective** | 4.0× | General problem analysis |
| **Iterative** | 2.0× | Optimization, design refinement |
| **Debate** | 2.5× | Strategic decisions, architecture |
| **Research** | 1.5× | Evidence-based, current events |
| **Jury** | 5.0× | High-stakes, critical code |
| **Scientific** | 2.0× | Technical decisions, algorithms |
| **Socratic** | 1.5× | Clarifying ambiguous requirements |
| **Pre-Mortem** ⭐ | 1.8× | Risk assessment, project planning |
| **Bayesian** | 2.2× | Decisions under uncertainty |
| **Dialectical** | 2.0× | Philosophical conflicts, policy |
| **Analogical** ⭐ | 1.9× | Innovation, cross-domain transfer |
| **Delphi** | 3.5× | Predictions, expert consensus |

⭐ **Recommended for most projects**

### Quick Start

```python
import asyncio
from orchestrator import Orchestrator, Budget
from orchestrator.ara_integration import create_ara_integration

async def use_ara_pipelines():
    orch = Orchestrator(budget=Budget(max_usd=20.0))
    
    # Create ARA integration with auto-select
    ara = create_ara_integration(
        client=orch.client,
        cache=orch.cache,
        telemetry=orch._telemetry,
        enabled=True,
        auto_select=True,  # Auto-select method per task
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
    print(f"Output: {result.output[:200]}...")

asyncio.run(use_ara_pipelines())
```

### Method Selection

#### Automatic Selection (Recommended)

```python
from orchestrator.method_selector import select_method_for_task

# Auto-select based on task characteristics
selection = select_method_for_task(
    task=task,
    complexity="high",      # low, medium, high, critical
    risk="high",            # low, medium, high, critical
    use_llm=True,           # Use LLM for optimization
    client=orch.client,
)

print(f"Recommended: {selection.method.value}")
print(f"Confidence: {selection.confidence}")
print(f"Rationale: {selection.rationale}")
print(f"Cost multiplier: {selection.estimated_cost_multiplier}×")
```

#### Manual Method Selection

```python
from orchestrator.ara_pipelines import ReasoningMethod, PipelineFactory

# Use Pre-Mortem for risk assessment
pipeline = PipelineFactory.create(
    method=ReasoningMethod.PRE_MORTEM,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Failure narrative: {result.metadata.get('failure_narrative', '')[:300]}")
print(f"Safeguards: {result.metadata.get('safeguards', [])}")
```

### Configuration

#### Environment Variables

```bash
# Enable ARA pipelines
export ORCHESTRATOR_ARA_ENABLED=true

# Auto-select method per task
export ORCHESTRATOR_ARA_AUTO_SELECT=true

# Default method (if auto-select disabled)
export ORCHESTRATOR_ARA_DEFAULT_METHOD=multi_perspective

# Enable LLM optimization
export ORCHESTRATOR_ARA_LLM_OPTIMIZATION=true

# Cost constraints
export ORCHESTRATOR_ARA_MAX_COST_MULTIPLIER=3.0
export ORCHESTRATOR_ARA_MAX_TIME_MULTIPLIER=1.5

# Method overrides for specific tasks
export ORCHESTRATOR_ARA_METHOD_OVERRIDES='{"auth_task": "jury", "risk_task": "pre_mortem"}'
```

#### Python Configuration

```python
from orchestrator.ara_integration import create_ara_integration

ara = create_ara_integration(
    client=orch.client,
    enabled=True,
    auto_select=True,
)

# Configure constraints and overrides
ara.configure(
    enabled=True,
    max_cost_multiplier=3.0,
    max_time_multiplier=1.5,
    method_overrides={
        "authentication": "jury",      # Highest quality for auth
        "payment_processing": "jury",  # Critical code
        "deployment_plan": "pre_mortem",  # Risk assessment
        "architecture_decision": "debate",  # Explore trade-offs
    },
)
```

### Method Selection Guide

| Task Type | Complexity | Risk | Recommended Method |
|-----------|------------|------|-------------------|
| Code Generation | Low/Medium | Low | Multi-Perspective, Iterative |
| Code Generation | High | Medium | Iterative |
| Code Generation | Critical | High | Jury, Pre-Mortem |
| Code Review | High | High | Jury, Pre-Mortem |
| Architecture | Medium | Medium | Debate, Dialectical |
| Risk Assessment | Any | High | Pre-Mortem ⭐ |
| Innovation | Medium | Low | Analogical ⭐ |
| Uncertainty | High | Medium | Bayesian |
| Predictions | High | High | Delphi |
| Requirements | Low | Low | Socratic |

### Examples

#### Example 1: High-Stakes Code Review with Jury

```python
from orchestrator.ara_pipelines import ReasoningMethod

task = Task(
    id="review_auth",
    type=TaskType.CODE_REVIEW,
    prompt="Review authentication module for security vulnerabilities",
    max_output_tokens=4000,
)

pipeline = PipelineFactory.create(
    method=ReasoningMethod.JURY,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Security issues: {result.output[:500]}")
```

#### Example 2: Architecture Decision with Debate

```python
task = Task(
    id="arch_decision",
    type=TaskType.REASONING,
    prompt="Should we use microservices or monolith for our startup?",
    max_output_tokens=4000,
)

pipeline = PipelineFactory.create(
    method=ReasoningMethod.DEBATE,
    client=orch.client,
)

result = await pipeline.execute(task)
print(f"Decision rationale: {result.output}")
```

#### Example 3: Risk Assessment with Pre-Mortem

```python
task = Task(
    id="deployment_risk",
    type=TaskType.REASONING,
    prompt="Deploy new payment system to production",
    max_output_tokens=4000,
)

pipeline = PipelineFactory.create(
    method=ReasoningMethod.PRE_MORTEM,
    client=orch.client,
)

result = await pipeline.execute(task)

# Access pre-mortem insights
print(f"Failure narrative: {result.metadata.get('failure_narrative', '')[:300]}")
print(f"Root cause: {result.metadata.get('root_cause', '')[:200]}")
print(f"Early signals: {result.metadata.get('early_signals', [])}")
print(f"Safeguards: {result.metadata.get('safeguards', [])}")
```

#### Example 4: Innovation with Analogical Transfer

```python
task = Task(
    id="ui_innovation",
    type=TaskType.WRITING,
    prompt="Design innovative UI for music creation app",
    max_output_tokens=4000,
)

pipeline = PipelineFactory.create(
    method=ReasoningMethod.ANALOGICAL,
    client=orch.client,
)

result = await pipeline.execute(task)

print(f"Source domains: {result.metadata.get('source_domains', [])}")
print(f"Best analogy: {result.metadata.get('best_source', '')}")
print(f"Solution: {result.output}")
```

#### Example 5: Full Project with Mixed Methods

```python
from orchestrator import Orchestrator, Budget
from orchestrator.ara_integration import create_ara_integration

orch = Orchestrator(budget=Budget(max_usd=50.0))
ara = create_ara_integration(client=orch.client, auto_select=True)

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

# Run project with ARA pipelines
state = await orch.run_project(
    project_description="Build e-commerce platform with payment processing",
    success_criteria="All tests pass, PCI compliant",
    analyze_on_complete=True,
)

# Check ARA statistics
stats = ara.get_stats()
print(f"Tasks executed: {stats['tasks_executed']}")
print(f"Method distribution: {ara.get_method_distribution()}")
print(f"Avg cost multiplier: {stats['avg_cost_multiplier']:.2f}×")
```

### Monitoring Statistics

```python
# Get execution statistics
stats = ara.get_stats()
print(f"Tasks executed: {stats['tasks_executed']}")
print(f"Methods used: {stats['methods_used']}")
print(f"Average cost multiplier: {stats['avg_cost_multiplier']:.2f}×")
print(f"Average time multiplier: {stats['avg_time_multiplier']:.2f}×")

# Get method distribution as percentages
distribution = ara.get_method_distribution()
for method, percentage in distribution.items():
    print(f"{method}: {percentage:.1f}%")
```

### Cost Planning

```python
from orchestrator.method_selector import METHOD_COST_MULTIPLIERS, ReasoningMethod

# Estimate cost for project with multiple methods
methods_needed = {
    ReasoningMethod.MULTI_PERSPECTIVE: 8,  # 8 general tasks
    ReasoningMethod.JURY: 2,               # 2 critical tasks
    ReasoningMethod.PRE_MORTEM: 1,         # 1 risk assessment
    ReasoningMethod.DEBATE: 1,             # 1 architecture decision
}

baseline_cost = 0.10  # Per task baseline
total_cost = sum(
    METHOD_COST_MULTIPLIERS[method] * count * baseline_cost
    for method, count in methods_needed.items()
)

print(f"Estimated total cost: ${total_cost:.2f}")
# Output: Estimated total cost: $0.98
```

### Troubleshooting

**Issue: Method always returns Multi-Perspective**

```python
# Enable auto-select and LLM optimization
ara.configure(
    auto_select=True,
    use_llm_for_selection=True,
)
```

**Issue: High costs**

```python
# Set cost constraints
ara.configure(
    max_cost_multiplier=2.0,  # Limit to 2× baseline
)

# Or use cheaper default method
ara.configure(
    default_method=ReasoningMethod.ITERATIVE,  # 2.0× vs 4.0×
)
```

**Issue: Pipeline execution fails**

ARA pipelines automatically fallback to standard execution on errors. Check logs:

```python
import logging
logging.getLogger("orchestrator").setLevel(logging.DEBUG)
```

### Additional Resources

- **[ARA_PIPELINE_GUIDE.md](./ARA_PIPELINE_GUIDE.md)** — Complete guide with all 12 methods
- **[orchestrator/ara_pipelines.py](./orchestrator/ara_pipelines.py)** — Pipeline implementations
- **[orchestrator/method_selector.py](./orchestrator/method_selector.py)** — Method selection logic
- **[orchestrator/ara_integration.py](./orchestrator/ara_integration.py)** — Integration layer

---

## 🔮 Nexus Search — Web Search Integration

**New in v6.2:** Private, self-hosted web search for AI Orchestrator.

Nexus Search provides intelligent web search capabilities powered by self-hosted search infrastructure. All searches are private, tracked-free, and integrated directly into the AI Orchestrator.

### Features

- 🔍 **Multi-Source Search** — Web, academic, tech, news, and code
- 🧠 **Query Classification** — Automatic optimal source selection
- 📚 **Deep Research** — Multi-step research with synthesis
- 🔒 **Privacy-First** — No tracking, no profiling
- 💰 **Zero Cost** — Self-hosted, no API fees
- ⚡ **Fast** — Local deployment, minimal latency

### Quick Start

#### 1. Start Nexus Search

```bash
# Using Docker Compose
docker-compose -f nexus-search.docker-compose.yml up -d

# Check status
docker ps | grep nexus-search
```

#### 2. Configure AI Orchestrator

```bash
# Add to .env
export NEXUS_SEARCH_ENABLED=true
export NEXUS_API_URL=http://localhost:8080
```

#### 3. Use in Code

```python
from orchestrator.nexus_search import search, research

# Simple search
results = await search("Python async best practices")
for result in results.top:
    print(f"{result.title}: {result.url}")

# Deep research
report = await research("Microservices architecture patterns 2026")
print(f"Found {report.source_count} sources")
print(f"Summary: {report.summary[:200]}...")
```

### CLI Commands

```bash
# Search the web
python -m orchestrator nexus search "Python async best practices"
python -m orchestrator nexus search "Microservices patterns" --sources tech,academic
python -m orchestrator nexus search "CVE 2026" --json

# Deep research
python -m orchestrator nexus research "AI architecture patterns 2026"
python -m orchestrator nexus research "Serverless best practices" --depth 5

# Check status
python -m orchestrator nexus status
python -m orchestrator nexus status --json

# Classify query
python -m orchestrator nexus classify "How to build FastAPI service"
python -m orchestrator nexus classify "Python async" --json
```

### Available Sources

| Source | Description | Examples |
|--------|-------------|----------|
| **Web** | General web search | Google, Bing, DuckDuckGo |
| **Academic** | Scholarly articles | Google Scholar, arXiv, PubMed |
| **Tech** | Technology content | HackerNews, tech blogs |
| **News** | News articles | Google News, Bing News |
| **Code** | Code repositories | GitHub, Stack Overflow |

### Integration Points

#### 1. Project Enhancer

Automatically searches latest best practices when enhancing projects.

```python
from orchestrator.enhancer import ProjectEnhancer

enhancer = ProjectEnhancer(nexus_enabled=True)
enhancements = await enhancer.analyze(
    description="Build FastAPI service",
    criteria="All endpoints tested",
    use_web_context=True,  # Uses Nexus Search
)
```

#### 2. Architecture Advisor

Searches latest architecture patterns.

```python
from orchestrator.architecture_advisor import ArchitectureAdvisor

advisor = ArchitectureAdvisor(nexus_enabled=True)
decision = await advisor.analyze(
    description="Real-time analytics dashboard",
    criteria="Low latency, 10k users",
    use_web_context=True,  # Uses Nexus Search
)
```

#### 3. ARA Research Pipeline

Uses Nexus for real web research (instead of LLM simulation).

```python
from orchestrator.ara_pipelines import PipelineFactory, ReasoningMethod

pipeline = PipelineFactory.create(
    method=ReasoningMethod.RESEARCH,
    nexus_enabled=True,  # Uses real web search
)
result = await pipeline.execute(task)
print(f"Sources found: {result.metadata.get('nexus_search', False)}")
```

#### 4. Project Analyzer

Checks for security vulnerabilities (CVEs) in dependencies.

```python
from orchestrator.project_analyzer import ProjectAnalyzer

analyzer = ProjectAnalyzer(nexus_enabled=True)
report = await analyzer.analyze_project(
    project_path=Path("./my_project"),
    project_id="proj_123",
)
# Includes CVE vulnerability suggestions
```

### Configuration

#### Environment Variables

```bash
# Enable/disable Nexus Search
export NEXUS_SEARCH_ENABLED=true

# Nexus API URL
export NEXUS_API_URL=http://localhost:8080

# Request timeout (seconds)
export NEXUS_TIMEOUT=30

# Maximum results per query
export NEXUS_MAX_RESULTS=20

# Rate limit (queries per minute)
export NEXUS_RATE_LIMIT=60

# Enable caching
export NEXUS_CACHE_ENABLED=true

# Cache TTL (seconds)
export NEXUS_CACHE_TTL=3600
```

#### Python Configuration

```python
from orchestrator.nexus_search import configure

configure(
    enabled=True,
    api_url="http://localhost:8080",
    max_results=10,
    cache_enabled=True,
)
```

### API Reference

#### Simple Search

```python
from orchestrator.nexus_search import search, SearchSource, OptimizationMode

results = await search(
    query="Python async",
    sources=[SearchSource.WEB, SearchSource.TECH],
    optimization=OptimizationMode.BALANCED,
    num_results=10,
)
```

#### Deep Research

```python
from orchestrator.nexus_search import research

report = await research(
    query="Microservices patterns",
    depth=3,  # Number of iterations
)

print(f"Findings: {len(report.findings)}")
print(f"Sources: {report.source_count}")
print(f"Summary: {report.summary}")
```

#### Query Classification

```python
from orchestrator.nexus_search import classify, QueryType

query_type = await classify("Python async best practices")
# Returns: QueryType.RESEARCH
```

### Troubleshooting

#### Nexus Search Not Available

```bash
# Check if container is running
docker ps | grep nexus-search

# Check logs
docker logs nexus-search

# Test health endpoint
curl http://localhost:8080/healthz
```

#### Slow Searches

```bash
# Increase timeout
export NEXUS_TIMEOUT=60

# Reduce results
export NEXUS_MAX_RESULTS=10
```

### Additional Resources

| Resource | Purpose |
|----------|---------|
| [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) | Complete Nexus Search documentation |
| [nexus-search.docker-compose.yml](./nexus-search.docker-compose.yml) | Docker setup |
| [nexus_config/settings.yml](./nexus_config/settings.yml) | Search configuration |
| [orchestrator/nexus_search/](./orchestrator/nexus_search/) | Source code |

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

