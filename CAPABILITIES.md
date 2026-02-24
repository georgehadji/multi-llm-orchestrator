# Multi-LLM Orchestrator — Capabilities Reference

**Version:** 2026.02 | **Updated:** 2026-02-24

This document provides a comprehensive overview of all capabilities, features, and advanced functionality available in the multi-llm-orchestrator.

---

## Core Capabilities

### 1. Multi-Provider Model Routing

The orchestrator automatically routes tasks to the optimal AI model based on task type, cost, performance metrics, and availability.

**Supported Providers:**
- **OpenAI** — GPT-4o, GPT-4o-mini
- **Google** — Gemini 2.5 Pro, Gemini 2.5 Flash
- **Anthropic** — Claude 3.5 Opus, Claude 3.5 Sonnet, Claude 3.5 Haiku
- **Kimi (Moonshot)** — K2.5 (moonshot-v1, with variants: 8K, 32K, 128K context)
- **DeepSeek** — DeepSeek Chat (V3), DeepSeek Reasoner (R1)

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
| DeepSeek Chat | $0.27 | $1.10 | DeepSeek | Ultra-cheap |
| Kimi K2.5 | $0.14 | $0.56 | Kimi | Ultra-cheap |
| Gemini Flash | $0.15 | $0.60 | Google | Ultra-cheap |
| GPT-4o-mini | $0.15 | $0.60 | OpenAI | Ultra-cheap |
| Claude Haiku | $0.80 | $4.00 | Anthropic | Budget |
| DeepSeek Reasoner | $0.55 | $2.19 | DeepSeek | Standard |
| Gemini 2.5 Pro | $1.25 | $10.00 | Google | Standard |
| GPT-4o | $2.50 | $10.00 | OpenAI | Standard |
| Claude Sonnet | $3.00 | $15.00 | Anthropic | Premium |
| Claude Opus | $15.00 | $75.00 | Anthropic | Premium |

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

## Advanced Capabilities

### 6. Multi-Objective Optimization Backends

Three pluggable routing strategies:

- **GreedyBackend** — Single-winner: maximize `quality × trust / cost`
- **WeightedSumBackend** — Tunable: `w_quality`, `w_trust`, `w_cost` weights
- **ParetoBackend** — Principled: non-dominated Pareto-optimal solutions

### 7. Economic Layer & Cost Prediction

- **AdaptiveCostPredictor:** EMA-tracked per-(model × task) costs
- **BudgetHierarchy:** Cross-run org/team/job spending caps
- **CostForecaster:** Pre-flight estimation with risk assessment

### 8. Multi-Agent Ensembles

- **AgentPool:** Run N orchestrators in parallel (A/B testing, ensemble voting)
- **TaskChannel:** Pub-sub messaging between tasks with non-destructive peek

### 9. Audit & Compliance

- **AuditLog:** Immutable JSONL structured audit trail
- **PolicyAnalyzer:** Static policy contradiction detection

### 10. App Builder & Scaffolding

- **AppBuilder:** Auto-generate complete web/backend applications
- **AppDetector:** Infer app type from project description
- **Supported types:** Next.js, React+Vite, HTML, FastAPI, GraphQL, microservices

### 11. Constraint Control Plane

- **JobSpecV2/PolicySpecV2:** Structured specs for hard constraint enforcement
- **ReferenceMonitor:** Synchronous, bypass-proof constraint checker
- **OrchestrationAgent:** Natural language intent → structured specs

### 12. Semantic Caching & Deduplication

- **SemanticCache:** Cache based on semantic similarity (not just hash)
- **DuplicationDetector:** Merge near-duplicate tasks automatically

### 13. Adaptive Routing

- **AdaptiveRouter:** Dynamically adjust model selection based on observed performance
- **ModelState:** Per-model telemetry tracking (latency, quality, trust)

### 14. Remediation Engine

- **RemediationEngine:** Auto-recovery strategies on task failure
- **RemediationPlan:** Retry same/fallback model, loosen constraints, escalate, or abort

### 15. Real-Time Progress & Visualization

- **ProgressRenderer:** Live terminal tree view of task progress
- **DagRenderer:** Directed acyclic graph visualization with costs
- **ProjectEventBus:** Async event streaming for monitoring

---

## Environment Variables

```bash
# At least one provider key required
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export ANTHROPIC_API_KEY="sk-ant-..."
export KIMI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# Optional tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Optional
export ORCHESTRATOR_CACHE_DIR="~/.orchestrator_cache"
export ORCHESTRATOR_LOG_LEVEL="INFO"
```

---

## Quick Feature Checklist

| Feature | Status |
|---------|--------|
| Multi-provider routing | ✅ |
| Cost optimization | ✅ |
| Deterministic validation | ✅ |
| Cross-provider critique | ✅ |
| Policy governance | ✅ |
| OTEL tracing | ✅ |
| Telemetry & metrics | ✅ |
| Multi-objective optimization | ✅ |
| Pre-flight cost forecasting | ✅ |
| Ensemble/AgentPool | ✅ |
| Semantic caching | ✅ |
| App builder | ✅ |
| Constraint control plane | ✅ |
| Orchestration agent | ✅ |
| Remediation engine | ✅ |
| Real-time visualization | ✅ |
