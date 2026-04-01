# AI Orchestrator — Codebase Mind Map
> **Last updated:** 2026-04-01
> **Purpose:** Single-source truth for understanding how every module connects. Read this before making any architectural change.

---

## 1. Core Philosophy

**Multi-LLM Orchestrator** — coordinates OpenAI, Anthropic, Google, DeepSeek, etc. (all via OpenRouter) using a `generate → critique → revise → evaluate` pipeline. Key principles:

- **Hexagonal Architecture**: `engine.py` = Mediator. `models.py` = pure data. `api_clients.py` = provider adapter.
- **Policy-as-code**: Compliance rules are first-class `Policy` dataclasses — every routing decision is explainable.
- **Adaptive routing**: `TelemetryCollector` updates `ModelProfile` (latency, quality, trust) after every call; `ConstraintPlanner` re-routes automatically.
- **Cost pyramid**: 4 tiers of savings (80-90% → 40-60% → 30-50% → DevOps).

---

## 2. Entry Points

```
python -m orchestrator [args]
         ↓
orchestrator/__main__.py
         ↓
orchestrator/cli.py :: main()
```

### CLI Subcommands

| Command | Handler | Purpose |
|---------|---------|---------|
| *(default)* | `_async_new_project()` / `_async_file_project()` | Run a project (inline args or `--file foo.yaml`) |
| `--resume <id>` | `_async_resume()` | Resume from checkpoint |
| `--list-projects` | `_async_list_projects()` | Enumerate saved projects |
| `--dry-run` | `_async_dry_run()` | Pre-flight cost forecast |
| `--visualize` | `_async_visualize()` | Print task DAG (mermaid/ascii) |
| `analyze` | `_async_analyze()` | Codebase analysis report |
| `build` | AppBuilder pipeline | Full app generation |
| `agent` | Agent spec runner | NL intent → typed spec |
| `dashboard` | dashboard_core | Mission Control UI |
| `meta` | MetaOptimization | A/B testing, transfer learning |
| `cache-stats` | DiskCache | Show/clear cache |

### YAML Project File (`--file`)

Parsed by `project_file.py::load_project_file()` → `ProjectFileResult`

```yaml
project: "Build X"          # required
criteria: "Y passes"        # required
budget_usd: 8.0             # default 8.0
time_seconds: 3600          # default 5400
concurrency: 3              # default 3
tdd_first: true             # default true (TDD on by default)
tdd_quality: balanced       # budget | balanced | premium
output_dir: ./results       # optional
assemble: false             # optional: assemble into project tree
verify_cmd: pytest          # optional: run after assembly
policies:
  - name: no_openai
    blocked_providers: [openai]
  - name: eu_only
    allowed_regions: [eu]
```

---

## 3. The Engine (`orchestrator/engine.py`)

The single most important file. ~5000 lines. **Orchestrator** class is the Mediator.

### Construction — what gets wired at `__init__`

```
Orchestrator(budget, max_concurrency, tracing_cfg)
    ├── Budget                    # per-run spend + phase allocation
    ├── DiskCache                 # SQLite prompt dedup (default ~/.orchestrator_cache/cache.db)
    ├── StateManager              # async SQLite project checkpointing
    ├── UnifiedClient             # OpenRouter API client + retry
    ├── SemanticCache             # L3 quality-threshold cache (threshold=0.85)
    ├── CacheOptimizer (opt)      # Multi-level L1/L2/L3 cache
    ├── PolicyEngine              # Compliance validation
    ├── ConstraintPlanner         # Multi-objective model selection
    ├── TelemetryCollector        # Adaptive profiling (EMA)
    ├── AdaptiveRouter            # Circuit breaker + health tracking
    ├── TelemetryStore            # Cross-run learning persistence
    └── [optional, try/except]:
        A2AManager, AccountabilityTracker, AgentSafetyMonitor,
        TaskVerifier, RedTeamFramework, TokenOptimizer,
        PreflightValidator, SessionWatcher, PersonaManager,
        MemoryTierManager, HybridSearchPipeline,
        SessionLifecycleManager, MetaOptimization
```

### Execution Pipeline

```
run_project(description, criteria, project_id, ...)
    │
    ├── [Phase 0] _generate_architecture_rules()
    │       → calls LLM to decide: style, paradigm, stack, constraints
    │       → writes .orchestrator-rules.yml
    │
    ├── [Phase 1] _decompose()
    │       → calls LLM to split project into Task list (JSON)
    │       → retries with different model on JSON parse failure
    │       → StateManager.save_project() [first checkpoint]
    │
    ├── _topological_sort(tasks)
    │       → DAG resolution via Kahn's algorithm (collections.deque)
    │       → raises on cycle detection
    │
    └── [Phase 2-5] _execute_all(tasks, execution_order)
            │
            ├── [Per task, up to max_parallel_tasks concurrently]:
            │       │
            │       ├── [TDD path, if tdd_first=True and CODE_GEN]:
            │       │       TestFirstGenerator.generate_with_tests(task)
            │       │       → write tests first (claude-sonnet-4-6 balanced)
            │       │       → implement to pass tests (qwen-3-coder-next balanced)
            │       │       → run tests in sandbox
            │       │       → return early if tests pass
            │       │       → fallback to standard if TDD fails
            │       │
            │       ├── ConstraintPlanner.select_model(task_type, api_health, policies, budget)
            │       │       → filter: health, policy, budget
            │       │       → score: quality × trust / (cost + ε)
            │       │
            │       ├── [Optional] SpeculativeGenerator — early exit on cheap model
            │       │
            │       ├── UnifiedClient.call(model, prompt, system, max_tokens, temp, timeout)
            │       │       → DiskCache check
            │       │       → Semaphore (max_concurrency)
            │       │       → Retry loop (exponential backoff on 429)
            │       │       → OpenRouter POST
            │       │       → APIResponse(text, input_tokens, output_tokens, cost_usd, cached)
            │       │
            │       ├── _critique_cycle(task, gen_response)
            │       │       → cross-provider critique
            │       │       → revise up to max_iterations
            │       │       → plateau detection (stop if score stagnates)
            │       │
            │       ├── async_run_validators(output, task_type)
            │       │       → python_syntax, json_schema, ruff, pytest, latex
            │       │       → score=0 if any fail (deterministic override)
            │       │
            │       ├── Budget.commit_reservation(reserved, actual, phase)
            │       │
            │       ├── TelemetryCollector.record_call(model, latency, cost, success, quality)
            │       │       → EMA update for ModelProfile
            │       │       → trust_factor: ×0.95 on fail, ×1.001 on success
            │       │
            │       └── StateManager.save_project() [checkpoint per task]
            │
            └── _determine_final_status()
                    → SUCCESS / PARTIAL / DEGRADED / BUDGET_EXHAUSTED / TIMEOUT
```

### Key Method Signatures

```python
await orch.run_project(
    project_description: str,
    success_criteria: str,
    project_id: str | None = None,
    app_profile: str | None = None,
    analyze: bool = False,
    output_dir: Path | None = None,
) -> ProjectState

await orch.run_job(spec: JobSpec) -> ProjectState   # policy-driven, uses BudgetHierarchy

async for event in orch.run_project_streaming(...): # yields StreamingEvent
    ...

plan = await orch.dry_run(description, criteria)    # → ExecutionPlan (no API calls)
```

---

## 4. Core Data Models (`orchestrator/models.py`)

**Rule: Pure data only. No asyncio, no I/O, no behavior.**

### Key Enums

```python
TaskType  = CODE_GEN | CODE_REVIEW | REASONING | WRITING | DATA_EXTRACT | SUMMARIZE | EVALUATE
ProjectStatus = SUCCESS | PARTIAL_SUCCESS | COMPLETED_DEGRADED | BUDGET_EXHAUSTED | TIMEOUT | SYSTEM_FAILURE
TaskStatus    = PENDING | RUNNING | COMPLETED | FAILED | DEGRADED
```

### `Model` enum — 70+ values, all OpenRouter format `provider/model-name`

Selected examples:
```
openai/gpt-4o, openai/gpt-5.4-codex, openai/o4-mini
anthropic/claude-sonnet-4-6, anthropic/claude-opus-4-6
google/gemini-2.5-pro, google/gemini-2.5-flash
deepseek/deepseek-v3.2, deepseek/deepseek-r1
meta-llama/llama-4-maverick, meta-llama/llama-4-scout
xai/grok-4-20-beta, qwen/qwen-3-coder-next
moonshotai/kimi-k2.5, stepfun/step-3.5-flash
zhipu/glm-4.7-flash  ← cheapest ($0.06/1M)
xiaomi/mimo-vl-7b-rl ← cheapest SWE-bench (#1 $0.09/1M)
```

### ROUTING_TABLE — `dict[TaskType → list[Model]]` (priority order)

| TaskType | Primary choice | Rationale |
|----------|---------------|-----------|
| CODE_GEN | xiaomi/mimo-vl-7b-rl | #1 SWE-bench, $0.09/1M |
| CODE_REVIEW | xai/grok-4-20-beta | lowest hallucination |
| REASONING | stepfun/step-3.5-flash | $0.10/1M 196B MoE |
| WRITING | meta-llama/llama-4-maverick | $0.17/1M |
| DATA_EXTRACT | zhipu/glm-4.7-flash | $0.06/1M ultra-cheap |
| SUMMARIZE | zhipu/glm-4.7-flash | same |
| EVALUATE | xai/grok-4-20-beta | lowest hallucination |

### FALLBACK_CHAIN — `dict[Model → Model]` (always cross-provider)

```
gpt-4o         → claude-sonnet-4-6
o1             → claude-opus-4-6
deepseek-r1    → phi-4-reasoning
llama-4-mav    → llama-3.1-405b
...50+ chains
```

### COST_TABLE — `dict[Model → {input: float, output: float}]`
Range: $0.06–$60 per 1M tokens

### MAX_OUTPUT_TOKENS — `dict[TaskType → int]`
```python
CODE_GEN: 8192, CODE_REVIEW: 4096, REASONING: 4096,
WRITING: 4096, DATA_EXTRACT: 2048, SUMMARIZE: 1024, EVALUATE: 2048
```

### Key Dataclasses

```python
@dataclass
class Task:
    id: str                          # task_001, task_002, ...
    type: TaskType
    prompt: str
    context: str = ""
    dependencies: list[str] = []     # ids of tasks that must complete first
    hard_validators: list[str] = []  # e.g. ["python_syntax", "pytest"]
    max_output_tokens: int = 1500    # set from MAX_OUTPUT_TOKENS in __post_init__
    target_path: str = ""            # for app builder assembly
    module_name: str = ""
    tech_context: str = ""
    status: TaskStatus = PENDING

@dataclass
class TaskResult:
    task_id: str
    output: str
    score: float                     # 0.0 – 1.0
    model_used: Model
    reviewer_model: Model | None
    tokens_used: dict                # {input, output}
    iterations: int
    cost_usd: float
    status: TaskStatus
    critique: str = ""
    deterministic_check_passed: bool = True
    attempt_history: list[AttemptRecord] = []
    test_files: dict[str, str] = {}  # TDD artifacts
    tests_passed: int = 0
    tests_total: int = 0
    metadata: dict = {}

@dataclass
class ProjectState:
    project_description: str
    success_criteria: str
    tasks: list[Task]
    results: dict[str, TaskResult]
    status: ProjectStatus
    start_time: float
    end_time: float | None
    total_cost_usd: float
    budget: object                   # Budget instance (lazy type)
    project_id: str = ""
```

---

## 5. Budget & Cost (`orchestrator/budget.py`, `orchestrator/cost.py`)

### Budget (per-run, in `budget.py`)

```python
@dataclass
class Budget:
    max_usd: float = 8.0
    max_time_seconds: float = 5400.0
    spent_usd: float = 0.0
    phase_spent: dict = {}           # {decomposition, generation, cross_review, evaluation, reserve}
    _reserved_usd: float = 0.0      # atomic reservation
    _lock: asyncio.Lock              # race-free updates

# Key async methods:
await budget.reserve(amount)                          # → bool
await budget.commit_reservation(reserved, actual, phase)
await budget.release_reservation(amount)
await budget.charge(amount, phase)

# Properties:
budget.remaining_usd                                  # max_usd - spent_usd - _reserved_usd
budget.phase_budget(phase)                            # BUDGET_PARTITIONS[phase] * max_usd
budget.phase_remaining(phase)                         # phase_budget - phase_spent
```

### BUDGET_PARTITIONS (in `models.py`, used by `budget.py`)
```python
{"decomposition": 0.10, "generation": 0.60, "cross_review": 0.15,
 "evaluation": 0.10, "reserve": 0.05}
```

### BudgetHierarchy (cross-run, in `cost.py`)
```python
hier = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 30.0})
orch = Orchestrator(budget=Budget(max_usd=10.0), budget_hierarchy=hier)
# run_job() checks hierarchy before every task
```

### CostForecaster (pre-flight)
```python
report = CostForecaster.forecast(tasks, profiles, predictor, budget=Budget(...))
# report.risk_level: LOW | MEDIUM | HIGH
# report.estimated_total_usd
```

---

## 6. API Client (`orchestrator/api_clients.py`)

**All LLM calls go through here. No direct SDK calls anywhere else.**

```python
class UnifiedClient:
    async def call(
        model: Model,
        prompt: str,
        system: str = "",
        max_tokens: int = 1500,    # override this! default is too low for CODE_GEN
        temperature: float = 0.3,
        timeout: int = 60,
        retries: int = 2,
        bypass_cache: bool = False,
    ) -> APIResponse
```

**Internal flow:**
1. Hash(model + prompt + max_tokens) → DiskCache lookup
2. Semaphore.acquire(max_concurrency)
3. Retry loop: exponential backoff on 429 / rate-limit patterns
4. POST to `https://openrouter.ai/api/v1/chat/completions`
5. Special handling: reasoning models (o1, o3, deepseek-reasoner) → no system prompt
6. Normalize → `APIResponse(text, input_tokens, output_tokens, model, cost_usd, cached, latency_ms)`

**Rate limit detection patterns** (`_RATE_LIMIT_PATTERNS`):
`"rate_limit", "rate limit", "429", "too many requests", "resource_exhausted", "quota", "overloaded"`

---

## 7. Caching (`orchestrator/cache.py`)

```
~/.orchestrator_cache/cache.db   ← SQLite WAL mode
    TABLE cache:
        hash TEXT PRIMARY KEY    ← sha256(model + prompt + max_tokens)
        model TEXT
        response TEXT
        tokens_input INTEGER
        tokens_output INTEGER
        created_at REAL
```

- Connection TTL: 1 hour (prevents stale)
- Connection timeout: 10s (prevents hang)
- `DiskCache.get(model, prompt, max_tokens)` → cached dict or None
- `DiskCache.put(model, prompt, max_tokens, response, tokens_in, tokens_out)`

---

## 8. State / Checkpointing (`orchestrator/state.py`)

```
~/.orchestrator_cache/state.db   ← SQLite, JSON serialization
    TABLE projects:
        project_id TEXT PRIMARY KEY
        state_json TEXT           ← full ProjectState as JSON
        updated_at REAL
```

- `await state_mgr.save_project(project_id, state)` — called after each task
- `await state_mgr.load_project(project_id)` → ProjectState | None
- `await state_mgr.list_projects()` → list[dict]

**Resume logic:** `ResumeDetector` (resume_detector.py) checks `.orchestrator_cache/` for recent projects → suggests `--resume` if found within 24h.

---

## 9. Policy & Compliance (`orchestrator/policy.py`, `policy_engine.py`)

### Policy dataclass
```python
@dataclass
class Policy:
    name: str
    blocked_providers: list[str] | None = None   # e.g. ["openai", "azure"]
    allowed_providers: list[str] | None = None   # whitelist
    allowed_regions: list[str] | None = None     # e.g. ["eu"]
    blocked_models: list[Model] | None = None
    allow_training_on_output: bool = True
    pii_allowed: bool = True
    max_cost_per_task_usd: float | None = None
    max_latency_ms: float | None = None
```

### JobSpec (policy-driven job)
```python
@dataclass
class JobSpec:
    project_description: str
    success_criteria: str
    budget: Budget
    policy_set: PolicySet
    job_id: str = ""
    team: str = ""
    quality_mode: str = "standard"   # "standard" | "production"
```

### PolicyEngine.check()
```python
result = policy_engine.check(model, profile, policies)
# result.passed: bool
# result.violations: list[str]
# EnforcementMode.HARD → fail on any violation (default)
# EnforcementMode.SOFT → only fail on hard violations (provider/region/model/PII)
# EnforcementMode.MONITOR → always pass, log violations
```

---

## 10. Model Selection (`orchestrator/planner.py`)

```python
class ConstraintPlanner:
    def select_model(
        task_type: TaskType,
        api_health: dict[Model, bool],
        policies: PolicySet,
        budget_remaining: float,
    ) -> Model | None

# Selection algorithm:
# 1. Filter: api_health[m] == True
# 2. Filter: PolicyEngine.check(m, profile, policies).passed
# 3. Filter: task_type in profile.capable_task_types
# 4. Filter: estimated_cost(typical_tokens) <= budget_remaining
# 5. Score:  quality_score * trust_factor / (cost_per_typical_call + 1e-9)
# 6. Tiebreak: ROUTING_TABLE priority rank
```

---

## 11. Validation (`orchestrator/validators.py`)

All validators return `ValidationResult(passed: bool, message: str, validator_name: str)`

| Validator | Trigger | What it checks |
|-----------|---------|---------------|
| `validate_python_syntax` | CODE_GEN | compile() + truncation detection |
| `validate_json_schema` | DATA_EXTRACT | json.loads() + optional jsonschema |
| `validate_ruff` | CODE_GEN | subprocess ruff --select E,F,W |
| `validate_pytest` | CODE_GEN w/ hard_validators | subprocess pytest |
| `validate_latex` | WRITING w/ latex | subprocess pdflatex |

**Truncation detection** (in python_syntax validator):
- Last line ends with `:` (incomplete annotation)
- Last char not in `}`, `)`, `]`, `"`, `'`, `\`
- AND output has ≥ 50 lines
→ Returns failure: "Output appears truncated at token limit — retry with higher max_output_tokens"

`async_run_validators()` → runs subprocess validators in thread pool (non-blocking)

---

## 12. TDD System (`orchestrator/tdd_config.py`, `test_first_generator.py`)

```
TDD enabled by default (enable_tdd_first=True in OptimizationConfig)

Triggered when: task.type == CODE_GEN AND enable_tdd_first AND HAS_TDD

Flow:
    1. detect_testing_framework(task.prompt) → pytest / jest / vitest / go / cargo
    2. Generate tests:  model = tdd_config.test_generation (claude-sonnet-4-6 balanced)
    3. Generate impl:   model = tdd_config.implementation  (qwen-3-coder-next balanced)
    4. Run tests in sandbox
    5. Self-heal loop (max_iterations=3) if tests fail
    6. Return TDDResult → TaskResult (score=1.0 if all pass, 0.8 if partial)
    On exception: fall back to standard generation pipeline
```

### Quality Tiers

| Tier | Test Generation | Implementation | Test Review |
|------|----------------|----------------|-------------|
| budget | deepseek/deepseek-v3.2 | qwen/qwen-3-coder-next | deepseek/deepseek-v3.2 |
| **balanced** | **anthropic/claude-sonnet-4-6** | **qwen/qwen-3-coder-next** | **anthropic/claude-sonnet-4-6** |
| premium | openai/gpt-5.4-pro | openai/gpt-5.4-pro | openai/gpt-5.4-pro |

---

## 13. Telemetry & Adaptive Learning (`orchestrator/telemetry.py`)

```python
class TelemetryCollector:
    def record_call(model, latency_ms, cost_usd, success, quality_score)
    # Updates ModelProfile (lives in policy.py) in-place:
    #   latency_ema    → α=0.2 exponential moving average
    #   quality_ema    → α=0.2
    #   success_rate   → rolling window last 10 calls
    #   p95_latency    → sorted buffer last 50 samples
    #   cost_ema       → α=0.2
    #   trust_factor   → ×0.95 on fail, ×1.001 on success (capped at 1.0)
    # trust_factor directly feeds ConstraintPlanner scoring
```

---

## 14. OptimizationConfig (`orchestrator/cost_optimization/__init__.py`)

```python
@dataclass
class OptimizationConfig:
    enable_prompt_caching: bool = True
    enable_batch_api: bool = True
    enable_token_budget: bool = True
    enable_cascading: bool = False        # speculative cascading
    enable_speculative: bool = False
    enable_streaming_validation: bool = True
    enable_adaptive_temperature: bool = True
    enable_dependency_context: bool = True
    enable_auto_eval_dataset: bool = True
    enable_tdd_first: bool = True         # ON by default
    tdd_quality_tier: str = "balanced"
    tdd_max_iterations: int = 3
    tdd_min_test_coverage: float = 0.8
    enable_diff_revisions: bool = True    # 60% token savings on revisions

# Global singleton, updated by engine at startup:
config = get_optimization_config()
update_config(config)
```

---

## 15. Module Dependency Map

```
                    ┌─────────────────────────────────────┐
                    │           __init__.py               │
                    │     (lazy __getattr__ facade)       │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │            engine.py                │
                    │         (Orchestrator)              │
                    └──┬───┬───┬───┬───┬───┬───┬──────────┘
                       │   │   │   │   │   │   │
              ┌────────┘   │   │   │   │   │   └──────────────┐
              ▼            ▼   │   ▼   ▼   ▼                  ▼
         api_clients   budget  │ state planner telemetry  cost_optim/
         (UnifiedClient)       │  (SM) (CPlan) (Telm)    (Tiers 1-4)
              │                ▼                               │
              ▼             policy.py ──► policy_engine.py    │
           cache.py         (Policy,                          │
           (DiskCache)       PolicySet,                       │
                             JobSpec,                         │
                             ModelProfile)                    │
                                │                             │
                    ┌───────────┴──────────────┐             │
                    ▼                          ▼             │
                models.py              tdd_config.py ◄───────┘
           (Task, TaskResult,          test_first_generator.py
            ProjectState,
            ROUTING_TABLE,
            COST_TABLE,
            MAX_OUTPUT_TOKENS,
            FALLBACK_CHAIN)

        cli.py
          ├► engine.py
          ├► project_file.py ──► policy.py, budget.py, models.py
          ├► assembler.py
          ├► state.py
          ├► output_writer.py
          └► resume_detector.py
```

---

## 16. Known Fixes & Gotchas

| Issue | Fix location | Note |
|-------|-------------|------|
| Circular imports | `__init__.py` uses `__getattr__` lazy loading | `aiohttp` also lazy in `rate_limiter.py`, `a2a_protocol.py`, `provisioned_throughput.py`, `nexus_search/nexus_client.py` |
| `Task.model_used` doesn't exist | `test_first_generator.py` line ~351 | Fixed: uses `self._get_model("implementation")` |
| deepseek-v3.2 truncates at 1460 tokens | `engine.py` `_is_deepseek_chat` set, `models.py` MODEL_MAX_TOKENS | Now explicitly 8192 + 180s timeout |
| Budget reservation leak | `budget.py` reserve/commit/release pattern | Atomic asyncio.Lock |
| TDD fails immediately | `test_first_generator.py` `model or task.model_used` → AttributeError | Fixed to `self._get_model("implementation")` |
| Output truncated → infinite retry loop | validators.py truncation detection + insufficient max_tokens | `MAX_OUTPUT_TOKENS[CODE_GEN]=8192` |
| aiohttp hangs on import (Windows) | `TYPE_CHECKING` guard + lazy import in `_get_session()` | Affects 4 files |
| Rate limiter class-var mutation | `rate_limiter.py` `self.TIER_LIMITS = dict(self.__class__.TIER_LIMITS)` | Copy to instance in `__init__` |
| Tier downgrade bug | `rate_limiter.py` tier progression is one-way up only | `if new_tier > self.state.current_tier` |

---

## 17. File Locations Cheat Sheet

```
orchestrator/
├── __init__.py            ← lazy package facade
├── __main__.py            ← entry: calls cli.main()
├── cli.py                 ← all subcommands + arg parsing
├── engine.py              ← THE core loop (~5000 lines)
├── models.py              ← PURE DATA: enums, dataclasses, tables
├── budget.py              ← Budget (async methods, phase tracking)
├── api_clients.py         ← UnifiedClient → OpenRouter
├── cache.py               ← DiskCache (SQLite WAL)
├── state.py               ← StateManager (SQLite, JSON)
├── cost.py                ← BudgetHierarchy, CostForecaster
├── planner.py             ← ConstraintPlanner (model selection)
├── policy.py              ← Policy, PolicySet, JobSpec, ModelProfile
├── policy_engine.py       ← PolicyEngine.check()
├── validators.py          ← deterministic validators
├── tdd_config.py          ← TDDModelConfig, tier profiles
├── test_first_generator.py← TestFirstGenerator (TDD pipeline)
├── telemetry.py           ← TelemetryCollector (EMA adaptive)
├── tracing.py             ← OpenTelemetry integration
├── preflight.py           ← PreflightValidator (quality gates)
├── rate_limiter.py        ← GrokRateLimiter (TPM/RPM sliding window)
├── model_routing.py       ← ModelTier, TIER_ROUTING, select_model
├── semantic_cache.py      ← SemanticCache (quality-threshold L3)
├── streaming.py           ← StreamingPipeline, ProjectEventBus
├── project_file.py        ← load_project_file() YAML parser
├── resume_detector.py     ← ResumeDetector (recent project detection)
├── assembler.py           ← assemble_project() (write file tree)
├── output_writer.py       ← write_output_dir() (structured output)
├── progress.py            ← ProgressRenderer (terminal progress)
├── analyzer.py            ← CodebaseAnalyzer
├── events.py              ← EventBus, domain events
├── hooks.py               ← HookRegistry, EventType
├── unified_events.py      ← UnifiedEventBus (v6.0)
├── cost_optimization/     ← Tier 1-4 optimizations
│   ├── __init__.py        ← OptimizationConfig + exports
│   ├── prompt_cache.py
│   ├── batch_client.py
│   ├── model_cascading.py
│   ├── speculative_gen.py
│   ├── streaming_validator.py
│   ├── token_budget.py
│   ├── dependency_context.py
│   └── docker_sandbox.py
├── engine_core/           ← Decomposed engine sub-components
│   ├── core.py            ← OrchestratorCore facade
│   ├── task_executor.py
│   ├── critique_cycle.py
│   ├── fallback_handler.py
│   ├── budget_enforcer.py
│   └── dependency_resolver.py
└── nexus_search/          ← SearXNG-backed web search
    ├── nexus_client.py    ← aiohttp lazy-loaded
    └── ...
```

---

## 18. 3 Unbreakable Architecture Rules

1. **`engine.py` = Mediator** — New business logic goes into a NEW service module. Engine only wires services together.
2. **`models.py` = Pure data** — No `import asyncio`, no I/O, no behavior. Only dataclasses and enums.
3. **TDD without exceptions** — RED (failing test) → GREEN (implementation) → commit.
