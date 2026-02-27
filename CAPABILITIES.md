# Multi-LLM Orchestrator — Capabilities Reference

**Version:** 2026.02 v5.1 | **Updated:** 2026-02-26

**Latest:**
- **v5.1:** Knowledge • Project • Product • Quality Management Systems
- **v5.0:** Performance Optimization (5x faster dashboard, dual-layer caching)
- **v4.x:** Cost-optimized routing with Minimax, Zhipu & DeepSeek

This document provides a comprehensive overview of all capabilities, features, and advanced functionality available in the multi-llm-orchestrator.

---

## Core Capabilities

### 1. Multi-Provider Model Routing

The orchestrator automatically routes tasks to the optimal AI model based on task type, cost, performance metrics, and availability.

**Supported Providers:**
- **OpenAI** — GPT-4o, GPT-4o-mini
- **Google** — Gemini 2.5 Pro, Gemini 2.5 Flash
- **Kimi (Moonshot)** — K2.5 (moonshot-v1, with variants: 8K, 32K, 128K context)
- **DeepSeek** — DeepSeek Coder, DeepSeek Reasoner (R1)
- **Minimax** — Minimax-3 (frontier reasoning, cost-effective)
- **Zhipu (GLM-4)** — GLM-4 (strong general purpose, competitive pricing)

**Task Types:** 7 core task types with optimized routing:
- `code_generation` — Generate code with fallback chains
- `code_review` — Cross-provider peer review
- `complex_reasoning` — Multi-step analytical tasks
- `creative_writing` — Long-form content generation
- `data_extraction` — Structured information retrieval
- `summarization` — Text condensing and abstraction
- `evaluation` — Quality scoring and assessment

---

### 2. Intelligent Cost Optimization

#### Automatic Model Selection by Cost

Models are pre-ranked by cost-effectiveness for each task type. The orchestrator uses:
- **Cost Table:** Per-model pricing (per 1M tokens: input/output)
- **Adaptive EMA:** Tracks actual observed costs across runs
- **Fallback Chains:** Cross-provider fallbacks when primary model fails

**Cost Reference (per 1M tokens):**

| Model | Input | Output | Provider | Tier |
|-------|-------|--------|----------|------|
| Kimi K2.5 | $0.14 | $0.56 | Kimi | Ultra-cheap |
| DeepSeek Coder | $0.27 | $1.10 | DeepSeek | Ultra-cheap |
| Gemini Flash | $0.15 | $0.60 | Google | Ultra-cheap |
| GPT-4o-mini | $0.15 | $0.60 | OpenAI | Ultra-cheap |
| Minimax-3 | $0.50 | $1.50 | Minimax | Budget-Efficient |
| GLM-4 | $0.50 | $2.00 | Zhipu | Budget-Efficient |
| DeepSeek Reasoner | $0.55 | $2.19 | DeepSeek | Standard |
| Gemini 2.5 Pro | $1.25 | $10.00 | Google | Standard |
| GPT-4o | $2.50 | $10.00 | OpenAI | Standard |

#### Budget Partitioning

Soft allocation of budget across execution phases:

| Phase | % of Budget | Example ($8 total) |
|-------|------------|-------------------|
| Decomposition | 5% | $0.40 |
| Generation | 45% | $3.60 |
| Cross-review | 25% | $2.00 |
| Evaluation | 15% | $1.20 |
| Reserve | 10% | $0.80 |

---

### 3. Quality Assurance & Validation

#### Deterministic Validators

Hard validators that gate task completion (score = 0.0 on failure):

| Validator | Purpose | Requirement |
|-----------|---------|-------------|
| `python_syntax` | Compile Python code | Built-in; always available |
| `pytest` | Run unit tests | Optional; skipped if pytest not in PATH |
| `ruff` | Python linting | Optional; skipped if ruff not installed |
| `json_schema` | Validate JSON | Optional; requires `jsonschema` package |
| `latex` | LaTeX compilation | Optional; requires `pdflatex` |
| `length` | Output size bounds | Built-in; enforces 10–50,000 characters |

#### Multi-Round Critique & Revision

Each task executes a generate → critique → revise loop:
1. **GENERATE** — Primary model produces output
2. **CRITIQUE** — Different provider provides feedback
3. **REVISE** — Primary model improves based on critique
4. **EVALUATE** — 2× independent LLM scoring with self-consistency check (Δ ≤ 0.05)

**Iteration Limits:**
- Code generation: 3 iterations
- Code review: 4 iterations (extra pass for context quality)
- Reasoning: 3 iterations
- Other tasks: 2 iterations

---

### 4. Policy Governance & Compliance

#### Policy-as-Code Framework

First-class `Policy` objects enable declarative compliance rules:

- **HARD mode** — Block any violation; raise exception
- **SOFT mode** — Log violation; allow execution
- **MONITOR mode** — Log only; never block

**Supported Constraints:**
- Allowed/blocked providers by region
- Training prohibition (`allow_training_on_output`)
- Cost caps per task
- Latency SLAs
- Quality thresholds
- Rate limits

#### Hierarchical Policy Enforcement

Policy inheritance from highest to lowest priority:

```
Org-level (global defaults)
  ↓
Team-level (override org)
  ↓
Job-level (override team)
  ↓
Node-level (specific task)
```

#### Policy DSL (YAML/JSON)

Externalize policies in declarative files:

```yaml
global:
  - name: gdpr
    allow_training_on_output: false
    enforcement_mode: hard

team:
  eng:
    - name: eu_only
      allowed_regions: [eu, global]
      cost_cap_usd: 0.50
```

**Static Analysis:** `PolicyAnalyzer` detects contradictions before execution.

---

### 5. Advanced Observability & Telemetry

#### OpenTelemetry Tracing

Full distributed tracing support with:
- Span instrumentation per task
- Trace context propagation
- OTLP exporter support

**Traced Operations:**
- `run_project` — Root span covering entire execution
- Task generation, critique, revision
- Policy checks and validations
- Model API calls

#### Telemetry Metrics Collection

Real-time telemetry tracking across all models:

**Metrics Per Model:**
- `calls_total` — Total API calls
- `success_rate` — Percentage of successful calls
- `latency_avg_ms` — EMA-smoothed average latency
- `latency_p95_ms` — Real p95 from sorted 50-sample buffer (not 2×avg approximation)
- `quality_score` — EMA-tracked quality metric
- `trust_factor` — Dynamic trust level (0–1, adjusts on success/failure)
- `cost_avg_usd` — EMA-tracked per-call cost
- `validator_failures_total` — Count of hard validator failures

**Real p95 Calculation:** Maintains sorted circular buffer of last 50 samples; computes true 95th percentile (not formula-based).

---

### 6. Architecture Advisor — LLM-Powered Architecture Decisions

**New in 2026-02:** Before generating any code, `ArchitectureAdvisor` makes intelligent architectural decisions:

**Input:** Project description + success criteria
**Output:** `ArchitectureDecision` specifying:
- **structural_pattern** (layered, hexagonal, CQRS, event-driven, MVC, script)
- **topology** (monolith, microservices, serverless, BFF, library)
- **api_paradigm** (REST, GraphQL, gRPC, CLI, none)
- **data_paradigm** (relational, document, time-series, key-value, none)
- **rationale** (2–3 sentences explaining the choices)

**Model Selection:**
- Descriptions >50 words → DeepSeek Reasoner (multi-dimensional reasoning)
- Descriptions ≤50 words → DeepSeek Coder (fast, cost-effective)
- Fallback: Minimax-3 (frontier reasoning) → GPT-4o

**Benefits:**
- Ensures all generated tasks follow a **consistent, coherent architecture**
- Reduces rework and architectural drift
- Injects constraints into decomposition prompt
- Automatic terminal summary (`🏗 Architecture Decision`)
- Full backward compatibility with `AppDetector`

**Usage:**
```python
advisor = ArchitectureAdvisor()
decision = await advisor.analyze(
    description="Build a FastAPI microservice",
    criteria="High throughput, auto-scaling"
)
print(decision.structural_pattern)  # "hexagonal" or chosen pattern
print(decision.topology)            # "microservices"
```

---

## Advanced Features

### 7. Multi-Objective Optimization Backends

Three pluggable routing strategies:
- **GreedyBackend** — Single-winner: maximize `quality × trust / cost`
- **WeightedSumBackend** — Tunable: `w_quality`, `w_trust`, `w_cost` weights
- **ParetoBackend** — Principled: non-dominated Pareto-optimal solutions

### 8. Economic Layer & Cost Prediction

- **AdaptiveCostPredictor:** EMA-tracked per-(model × task) costs
- **BudgetHierarchy:** Cross-run org/team/job spending caps
- **CostForecaster:** Pre-flight estimation with risk assessment

### 9. Multi-Agent Ensembles

- **AgentPool:** Run N orchestrators in parallel (A/B testing, ensemble voting)
- **TaskChannel:** Pub-sub messaging between tasks with non-destructive peek

### 10. Audit & Compliance

- **AuditLog:** Immutable JSONL structured audit trail
- **PolicyAnalyzer:** Static policy contradiction detection

### 11. App Builder & Scaffolding

- **AppBuilder:** Auto-generate complete web/backend applications
- **ArchitectureAdvisor:** LLM-powered architectural decision making
- **Supported types:** Next.js, React+Vite, HTML, FastAPI, GraphQL, microservices

### 12. Constraint Control Plane

- **JobSpecV2/PolicySpecV2:** Structured specs for hard constraint enforcement
- **ReferenceMonitor:** Synchronous, bypass-proof constraint checker
- **OrchestrationAgent:** Natural language intent → structured specs

### 13. Semantic Caching & Deduplication

- **SemanticCache:** Cache based on semantic similarity (not just hash)
- **DuplicationDetector:** Merge near-duplicate tasks automatically

### 14. Adaptive Routing

- **AdaptiveRouter:** Dynamically adjust model selection based on observed performance
- **ModelState:** Per-model telemetry tracking (latency, quality, trust)

### 15. Remediation Engine

- **RemediationEngine:** Auto-recovery strategies on task failure
- **RemediationPlan:** Retry same/fallback model, loosen constraints, escalate, or abort

### 16. Real-Time Progress & Visualization

- **ProgressRenderer:** Live terminal tree view of task progress
- **DagRenderer:** Directed acyclic graph visualization with costs
- **ProjectEventBus:** Async event streaming for monitoring

### 17. Project Enhancer

LLM-powered spec improvement before decomposition:
- **Enhancement:** Dataclass for suggested improvements (type: completeness|criteria|risk)
- **ProjectEnhancer.analyze():** Generates 3–7 LLM suggestions to improve project description and success criteria
- **_present_enhancements():** Interactive Y/n prompts for user to accept/reject each suggestion
- **_apply_enhancements():** Patches accepted suggestions into final description and criteria
- **--no-enhance flag:** Skip enhancement pass to run original spec directly
- **Model selection:** DeepSeek Reasoner (>50 words combined) vs Chat (≤50 words)
- **Budget cap:** $0.10 per enhancement request

**Improvement Categories:**
- **completeness** — Missing details about project scope/requirements
- **criteria** — Missing or unmeasurable success metrics
- **risk** — Unaddressed security, performance, or edge case concerns

### 18. Auto-Resume Detection

Intelligent resume suggestion for similar incomplete projects:
- **ResumeCandidate:** Project resume candidate with keyword matching and scoring
- **_extract_keywords():** Extracts 3+ character words, filters stopwords, returns sorted
- **_recency_factor():** Weights recent projects higher (1.0 = created today, 0.1 = 7+ days old)
- **_score_candidates():** Jaccard similarity on keywords + recency weighting (0.6×similarity + 0.4×recency)
- **_check_resume():** CLI gate with 200ms timeout for DB lookup
- **Resume workflows:**
  - **Exact match** → Auto-resume (prints message)
  - **Single fuzzy match** → Prompt [Y/n]
  - **Multiple matches** → Numbered list picker [1–N / n]
- **--new-project / -N flag:** Bypass resume detection, always start fresh

---

## v5.0 Performance Optimization

### Dashboard Performance Enhancements

**Mission Control Dashboard v5.0** delivers 5x faster load times:

| Optimization | Before | After | Benefit |
|--------------|--------|-------|---------|
| External CSS | 113KB inline | 35KB + 24h cache | 7.5x smaller initial load |
| Gzip Compression | - | Level 6 | 75% size reduction |
| ETag Support | - | 304 responses | Zero bandwidth repeat visits |
| Debounced Updates | 1s interval | 2s interval | 50% CPU reduction |
| Cache Hit Latency | N/A | <1ms | Instant cached responses |

**Performance Targets:**
- First Contentful Paint: <100ms (was ~450ms)
- Time to First Byte: <50ms (was ~200ms)
- P95 Response Time: <300ms
- Cache Hit Rate: >85%

### Dual-Layer Caching System

```python
# Redis (primary) → LRU Memory (fallback)
from orchestrator import cached, get_cache

@cached(ttl=300)  # Cache for 5 minutes
async def get_models():
    return await fetch_expensive_data()
```

**Features:**
- **Redis Integration:** Distributed caching with connection pooling
- **LRU Fallback:** Automatic failover to in-memory cache
- **TTL Support:** Per-key expiration with configurable defaults
- **Decorators:** Zero-code-change caching with `@cached()`

### Connection Pooling & Query Optimization

```python
from orchestrator import ConnectionPool, QueryOptimizer

# Bounded resource management
pool = ConnectionPool(create_conn, min_size=2, max_size=10)

# N+1 prevention with batch operations
results = await optimizer.batch_get(ids, fetch_func, batch_size=100)
```

---

## v5.1 Management Systems

### Knowledge Management

Central repository for organizational learning with semantic search:

**Key Features:**
- **Vector Search:** Embedding-based similarity (cosine similarity)
- **Knowledge Graph:** Relationship tracking between concepts
- **Pattern Recognition:** Auto-detect recurring patterns
- **Auto-Learning:** Extract knowledge from completed projects

```python
from orchestrator import get_knowledge_base, KnowledgeType

kb = get_knowledge_base()

# Add solution
await kb.add_artifact(
    type=KnowledgeType.SOLUTION,
    title="Race condition fix",
    content="Use asyncio.Lock()...",
    tags=["async", "python"],
)

# Find similar solutions
similar = await kb.find_similar("async race condition", top_k=5)
```

**Use Cases:**
- "Have we solved this bug before?"
- Auto-suggest solutions based on context
- Pattern library from historical projects

---

### Project Management

Advanced task scheduling with resource optimization:

**Key Features:**
- **Critical Path Analysis:** Identify bottlenecks with network analysis
- **Resource Scheduler:** Constraint-based optimal allocation
- **Risk Assessment:** ML-based delay prediction
- **Gantt Visualization:** Timeline charts with dependencies

```python
from orchestrator import get_project_manager, Resource, ResourceType

pm = get_project_manager()

# Define resources
resources = [
    Resource("gpt-4", ResourceType.MODEL, 100, 100, 0.03),
]

# Create schedule with dependencies
timeline = await pm.create_schedule(
    project_id="my_project",
    tasks=tasks,
    resources=resources,
    dependencies={"task_2": ["task_1"]},
)

print(f"Critical path: {timeline.critical_path}")
print(f"Duration: {timeline.total_duration}")
```

**Risk Detection:**
- Long critical paths (delay risk)
- Resource contention (bottlenecks)
- High dependency count (fragility)

---

### Product Management

Data-driven product development with RICE prioritization:

**RICE Scoring Framework:**
```
RICE = (Reach × Impact × Confidence) / Effort

Reach:      Users affected per quarter (1-1000)
Impact:     Impact magnitude (0.25=minimal, 3=massive)
Confidence: Certainty level (0-100%)
Effort:     Person-months required (1-12)
```

```python
from orchestrator import get_product_manager, RICEScore

pm = get_product_manager()

# Score = (500 × 3 × 0.8) / 2 = 600
rice = RICEScore(reach=500, impact=3, confidence=80, effort=2)

feature = await pm.add_feature(
    name="AI Assistant",
    rice_score=rice,
)

# Auto-prioritized backlog
backlog = pm.get_prioritized_backlog(limit=10)
```

**Additional Features:**
- **Feature Flags:** Gradual rollout with percentage control
- **Sentiment Analysis:** Auto-analyze user feedback
- **Release Planning:** Capacity-based release trains
- **Roadmap Generation:** Quarterly timeline visualization

---

### Quality Control

Automated quality assurance with multi-level testing:

**Testing Levels:**
1. **Unit** — pytest with coverage
2. **Integration** — Cross-component testing
3. **E2E** — End-to-end workflows
4. **Performance** — Benchmark regression
5. **Security** — Vulnerability scanning

**Static Analysis:**
- Cyclomatic complexity (<10 good, >20 critical)
- Documentation coverage (target >80%)
- Type hint coverage
- Code duplication detection
- Import best practices

```python
from orchestrator import get_quality_controller, TestLevel

qc = get_quality_controller()

report = await qc.run_quality_gate(
    project_id="my_project",
    project_path=Path("."),
    levels=[TestLevel.UNIT, TestLevel.PERFORMANCE],
)

if report.passed:
    print(f"Quality Score: {report.quality_score:.1f}/100")
else:
    print(f"Critical Issues: {len(report.get_issues_by_severity(QualitySeverity.CRITICAL))}")

# Regression detection
regressions = qc.detect_regression(report)
```

**Quality Gates:**
- Minimum 80% test coverage
- No critical issues
- Complexity threshold enforcement
- Documentation requirements

---

## Environment Variables

```bash
# At least one provider key required
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export KIMI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export MINIMAX_API_KEY="..."
export ZHIPUAI_API_KEY="...":

# Optional tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Optional
export ORCHESTRATOR_CACHE_DIR="~/.orchestrator_cache"
export ORCHESTRATOR_LOG_LEVEL="INFO"
```

---

## Feature Checklist

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-provider routing | ✅ | 6 providers, 9 models |
| Cost optimization | ✅ | EMA-tracked, adaptive |
| Deterministic validation | ✅ | 6 validator types |
| Cross-provider critique | ✅ | Different provider each review |
| Policy governance | ✅ | HARD/SOFT/MONITOR modes |
| OTEL tracing | ✅ | Full distributed tracing |
| Telemetry & metrics | ✅ | Real p95, trust factor EMA |
| Multi-objective optimization | ✅ | Greedy, Weighted, Pareto |
| Pre-flight cost forecasting | ✅ | Risk assessment |
| Architecture Advisor | ✅ | LLM architecture decisions |
| Project Enhancer | ✅ | LLM spec improvement before decomposition |
| Auto-Resume Detection | ✅ | Keyword matching + recency scoring |
| Ensemble/AgentPool | ✅ | Parallel orchestrators |
| Semantic caching | ✅ | Similarity-based dedup |
| App builder | ✅ | With ArchitectureAdvisor |
| Constraint control plane | ✅ | Hard guarantee enforcement |
| Orchestration agent | ✅ | Natural language → specs |
| Remediation engine | ✅ | Auto-recovery strategies |
| Real-time visualization | ✅ | Terminal + DAG rendering |
| **Performance Optimization v5.0** | ✅ | Caching, compression, monitoring |
| **Knowledge Management v5.1** | ✅ | Semantic search, pattern recognition |
| **Project Management v5.1** | ✅ | Critical path, resource scheduling |
| **Product Management v5.1** | ✅ | RICE scoring, feature flags |
| **Quality Control v5.1** | ✅ | Static analysis, compliance gates |
