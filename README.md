# multi-llm-orchestrator

Decomposes a project description into atomic tasks, routes each to the optimal provider (OpenAI / Anthropic / Google / Kimi), runs cross-provider generate → critique → revise cycles, 
and iterates until a quality threshold is met or a budget ceiling is hit.

State is checkpointed to SQLite after every task. Interrupted runs are resumable by project ID.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
│                                                      │
│  Decompose → Route → Generate → Critique → Revise    │
│       ↑                                    │         │
│       └──── Evaluate ← Deterministic Check ┘         │
│                                                      │
│  [Async Disk Cache]  [JSON State]  [Budget Control]  │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │              │
     ┌─────┴─────┐  ┌────┴────┐  ┌─────┴─────┐  ┌────┴────┐
     │  OpenAI   │  │ Gemini  │  │  Claude   │  │  Kimi   │
     └───────────┘  └─────────┘  └───────────┘  └─────────┘
```

---

## Requirements

- Python ≥ 3.10
- At least one provider key (others are skipped gracefully)

| Variable | Provider | Models |
|----------|----------|--------|
| `OPENAI_API_KEY` | OpenAI | GPT-4o, GPT-4o-mini |
| `ANTHROPIC_API_KEY` | Anthropic | Claude Opus, Sonnet, Haiku |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google | Gemini 2.5 Pro, Flash |
| `KIMI_API_KEY` or `MOONSHOT_API_KEY` | Kimi (moonshot.cn) | Kimi K2.5 (moonshot-v1) |

---

## Install

```bash
# Core (includes aiosqlite for async cache)
pip install -e .

# Optional validators
pip install pytest ruff jsonschema
```

Or without editable install:

```bash
pip install openai anthropic google-genai aiosqlite pyyaml python-dotenv
pip install pytest ruff jsonschema   # optional validators
```

> **Note:** The `openai` package is also used for Kimi K2.5 (OpenAI-compatible API). No extra dependency needed.

---

## CLI

```bash

# Αποθήκευση output σε φάκελο
python -m orchestrator \
  --project "Build a rate-limiter library" \
  --criteria "pytest passes, ruff clean" \
  --output-dir ./my_results

# New project (inline)
python -m orchestrator \
  --project  "Build a FastAPI auth service with JWT" \
  --criteria "All endpoints tested, OpenAPI spec complete" \
  --budget   8.0 \
  --time     5400

# From YAML project file
python -m orchestrator --file projects/example_full.yaml

# Save outputs to a directory
python -m orchestrator --file projects/example_full.yaml --output-dir ./results

# Resume interrupted run
python -m orchestrator --resume <project_id>

# List all saved projects
python -m orchestrator --list-projects
```

### CLI flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--project` / `-p` | str | — | Project description (**required** for new runs) |
| `--criteria` / `-c` | str | — | Acceptance criteria (**required** for new runs) |
| `--file` / `-f` | path | — | Load project spec from a YAML file |
| `--budget` / `-b` | float | 8.0 | Spend ceiling in USD |
| `--time` / `-t` | float | 5400 | Wall-clock limit in seconds |
| `--project-id` | str | auto | Explicit ID (auto-generated if blank) |
| `--resume` | str | — | Resume project by ID |
| `--list-projects` | flag | — | Print saved project IDs and statuses |
| `--concurrency` | int | 3 | Max simultaneous API calls |
| `--output-dir` / `-o` | path | — | Write structured output files to directory |
| `--verbose` / `-v` | flag | off | Enable DEBUG logging |

---

## Python API

```python
import asyncio
from orchestrator import Orchestrator, Budget

budget = Budget(max_usd=8.0, max_time_seconds=5400)
orch   = Orchestrator(budget=budget)

state = asyncio.run(orch.run_project(
    project_description="Build a rate-limiter library in Python",
    success_criteria="pytest suite passes, ruff clean, README present",
))

print(state.status.value)               # SUCCESS | PARTIAL_SUCCESS | ...
print(f"${state.budget.spent_usd:.4f}")

for tid, result in state.results.items():
    print(f"{tid}: score={result.score:.3f} model={result.model_used.value}")
```

### `Orchestrator(budget, cache, state_manager, max_concurrency)`

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `budget` | `Budget` | `Budget()` | Spend / time limits |
| `cache` | `DiskCache` | `DiskCache()` | `~/.orchestrator_cache/cache.db` |
| `state_manager` | `StateManager` | `StateManager()` | `~/.orchestrator_cache/state.db` |
| `max_concurrency` | `int` | `3` | Semaphore width across async tasks |

### `Orchestrator.run_project(project_description, success_criteria, project_id)`

Returns `ProjectState`.

### `ProjectState` fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | `ProjectStatus` | Final run outcome |
| `results` | `dict[str, TaskResult]` | Per-task outputs and scores |
| `budget` | `Budget` | Spend / time accounting |
| `tasks` | `dict[str, Task]` | Decomposed task graph |
| `execution_order` | `list[str]` | Topological sort order used |
| `api_health` | `dict[str, bool]` | Per-model availability at completion |

### `TaskResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Best output across all iterations |
| `score` | `float` | 0.0–1.0 LLM evaluator score |
| `status` | `TaskStatus` | `completed` / `degraded` / `failed` |
| `model_used` | `Model` | Primary model (may be a fallback) |
| `reviewer_model` | `Model \| None` | Cross-provider reviewer used |
| `iterations` | `int` | Revision cycles completed |
| `cost_usd` | `float` | Total spend for this task |
| `deterministic_check_passed` | `bool` | Hard validator gate result |
| `degraded_fallback_count` | `int` | Times primary failed and fallback ran |

### `Budget(max_usd, max_time_seconds)`

| Field / Method | Description |
|----------------|-------------|
| `spent_usd` | Cumulative spend |
| `remaining_usd` | `max_usd - spent_usd` |
| `elapsed_seconds` | Wall time since construction |
| `can_afford(cost)` | `remaining_usd >= cost` |
| `time_remaining()` | `elapsed_seconds < max_time_seconds` |
| `phase_spent` | Spend breakdown by phase |

---

## Control Loop

```
Phase 1 — Decompose
  cheapest_available_model → JSON task list (5–15 tasks)
  Kahn topological sort → execution_order

Phase 2–5 — Per-task (up to max_iterations per task type)
  GENERATE   primary_model(task + dependency_context)
  CRITIQUE   different_provider(output)        # skipped if no reviewer available
  REVISE     primary_model(output + critique)
  VALIDATE   deterministic_checks(revised)     # hard gate: fail → score = 0.0
  EVALUATE   eval_model(revised) × 2 runs      # self-consistency: Δ ≤ 0.05

  Stop iteration when:
    score ≥ threshold
    OR Δscore < 0.02 for 2 consecutive runs (plateau)
    OR budget / time exhausted
```

---

## Model Routing

| Task type | Priority order | Max tokens |
|-----------|---------------|------------|
| `code_generation` | **Kimi K2.5** → Claude Sonnet → GPT-4o → Gemini Pro | 4096 |
| `code_review` | **Kimi K2.5** → GPT-4o → Claude Opus → Gemini Pro | 2048 |
| `complex_reasoning` | **Kimi K2.5** → Claude Opus → GPT-4o → Gemini Pro | 2048 |
| `creative_writing` | Claude Opus → GPT-4o → Gemini Pro | 2048 |
| `data_extraction` | Gemini Flash → GPT-4o-mini → Claude Haiku | 1024 |
| `summarization` | Gemini Flash → Claude Haiku → GPT-4o-mini | 512 |
| `evaluation` | **Kimi K2.5** → Claude Opus → GPT-4o → Gemini Pro | 600 |

Kimi K2.5 is the **primary model** for code generation, code review, reasoning, and evaluation — it is the cheapest option ($0.14/$0.56 per 1M tokens) and falls back to Claude Sonnet if unavailable. Creative writing, data extraction, and summarization retain their original routing where Kimi offers no advantage.

The reviewer is always from a **different provider** than the generator (prevents shared-bias blind spots). Falls back to a different model tier, then any healthy model.

Max iterations per task: 3 (code, reasoning) / 2 (all others).

### Cost reference (per 1M tokens)

| Model | Input | Output | Provider |
|-------|-------|--------|----------|
| Claude Opus | $15.00 | $75.00 | Anthropic |
| Claude Sonnet | $3.00 | $15.00 | Anthropic |
| Claude Haiku | $0.80 | $4.00 | Anthropic |
| GPT-4o | $2.50 | $10.00 | OpenAI |
| GPT-4o-mini | $0.15 | $0.60 | OpenAI |
| Gemini 2.5 Pro | $1.25 | $10.00 | Google |
| Gemini 2.5 Flash | $0.15 | $0.60 | Google |
| Kimi K2.5 | $0.14 | $0.56 | Kimi |

> Kimi K2.5 is the most cost-effective option — cheaper than GPT-4o-mini and Gemini Flash.

---

## LLM Models Reference

### OpenAI Models

#### GPT-4o
- **Environment Variable:** `OPENAI_API_KEY`
- **Model ID:** `gpt-4o`
- **Context Window:** 128K tokens
- **Training Data Cutoff:** April 2024
- **Capabilities:** Vision, advanced reasoning, code generation, complex multi-step tasks
- **Cost:** $2.50 (input) / $10.00 (output) per 1M tokens
- **Best For:** Production code generation, architectural decisions, cross-domain reasoning
- **Specialized Domains:** Python, JavaScript, DevOps, System Design
- **API Endpoint:** `https://api.openai.com/v1/chat/completions`

#### GPT-4o-mini
- **Environment Variable:** `OPENAI_API_KEY`
- **Model ID:** `gpt-4o-mini`
- **Context Window:** 128K tokens
- **Training Data Cutoff:** April 2024
- **Capabilities:** Fast inference, cost-effective reasoning, vision support
- **Cost:** $0.15 (input) / $0.60 (output) per 1M tokens
- **Best For:** Data extraction, summarization, rapid iterations, cost-sensitive tasks
- **Specialized Domains:** Text processing, classification, quick fact retrieval
- **API Endpoint:** `https://api.openai.com/v1/chat/completions`
- **Note:** OpenAI's most cost-effective model with vision capabilities

### Google Gemini Models

#### Gemini 2.5 Pro
- **Environment Variable:** `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- **Model ID:** `gemini-2.5-pro`
- **Context Window:** 1M tokens
- **Training Data Cutoff:** October 2024
- **Capabilities:** Massive context, multimodal (text, image, video, audio), native file handling
- **Cost:** $1.25 (input) / $10.00 (output) per 1M tokens
- **Best For:** Large document processing, long conversation histories, code review of large files
- **Specialized Domains:** Full-stack web, data analysis, document comprehension
- **API Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent`
- **Note:** Largest context window enables processing of entire projects in a single call

#### Gemini 2.5 Flash
- **Environment Variable:** `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- **Model ID:** `gemini-2.5-flash`
- **Context Window:** 1M tokens
- **Training Data Cutoff:** October 2024
- **Capabilities:** Fast multimodal processing, optimized for speed, native file support
- **Cost:** $0.15 (input) / $0.60 (output) per 1M tokens
- **Best For:** Quick data extraction, image analysis, rapid prototyping, streaming responses
- **Specialized Domains:** Web scraping, image recognition, rapid content generation
- **API Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`
- **Note:** Fastest inference speed among major providers; ideal for interactive applications

### Anthropic Claude Models

#### Claude 3.5 Opus
- **Environment Variable:** `ANTHROPIC_API_KEY`
- **Model ID:** `claude-3-5-opus-20241022` (or latest)
- **Context Window:** 200K tokens
- **Training Data Cutoff:** April 2024
- **Capabilities:** Advanced reasoning, complex analysis, code generation, strategic thinking
- **Cost:** $15.00 (input) / $75.00 (output) per 1M tokens
- **Best For:** Architectural decisions, complex multi-domain problems, expert-level analysis
- **Specialized Domains:** System architecture, deep reasoning, large codebases
- **API Endpoint:** `https://api.anthropic.com/v1/messages`
- **Note:** Most capable Claude model; highest quality for complex reasoning

#### Claude 3.5 Sonnet
- **Environment Variable:** `ANTHROPIC_API_KEY`
- **Model ID:** `claude-3-5-sonnet-20241022` (or latest)
- **Context Window:** 200K tokens
- **Training Data Cutoff:** April 2024
- **Capabilities:** Balanced performance, strong code generation, detailed reasoning
- **Cost:** $3.00 (input) / $15.00 (output) per 1M tokens
- **Best For:** Default choice for code generation, documentation, balanced quality-cost
- **Specialized Domains:** Python, JavaScript, system design, technical writing
- **API Endpoint:** `https://api.anthropic.com/v1/messages`
- **Note:** Sweet spot for production use; excellent code quality at reasonable cost

#### Claude 3.5 Haiku
- **Environment Variable:** `ANTHROPIC_API_KEY`
- **Model ID:** `claude-3-5-haiku-20241022` (or latest)
- **Context Window:** 200K tokens
- **Training Data Cutoff:** April 2024
- **Capabilities:** Lightweight, fast responses, cost-effective reasoning
- **Cost:** $0.80 (input) / $4.00 (output) per 1M tokens
- **Best For:** Lightweight tasks, quick filtering, budget-constrained scenarios
- **Specialized Domains:** Classification, content filtering, simple transformations
- **API Endpoint:** `https://api.anthropic.com/v1/messages`
- **Note:** Most affordable Claude; suitable for high-volume operations

### DeepSeek Models

#### DeepSeek Chat
- **Environment Variable:** `DEEPSEEK_API_KEY`
- **Model ID:** `deepseek-chat`
- **Context Window:** 128K tokens
- **Training Data Cutoff:** 2024
- **Capabilities:** Fast inference, cost-efficient code generation, reasoning
- **Cost:** $0.14 (input) / $0.42 (output) per 1M tokens (approximate)
- **Best For:** Cost-sensitive code generation, rapid prototyping, scalable applications
- **Specialized Domains:** Python, JavaScript, system utilities
- **API Endpoint:** `https://api.deepseek.com/v1/chat/completions` (OpenAI-compatible)
- **Note:** Most cost-effective among fully-featured models

#### DeepSeek Reasoner
- **Environment Variable:** `DEEPSEEK_API_KEY`
- **Model ID:** `deepseek-reasoner`
- **Context Window:** 128K tokens
- **Training Data Cutoff:** 2024
- **Capabilities:** Deep reasoning, chain-of-thought inference, complex problem solving
- **Cost:** $0.55 (input) / $2.19 (output) per 1M tokens (approximate, with thinking tokens)
- **Best For:** Complex mathematical problems, algorithmic challenges, deep reasoning tasks
- **Specialized Domains:** Algorithm design, mathematical proofs, logic problems
- **API Endpoint:** `https://api.deepseek.com/v1/chat/completions` (OpenAI-compatible)
- **Note:** Specialized reasoning model; streaming extended thinking process

### Kimi (Moonshot) Models

#### Kimi K2.5
- **Environment Variable:** `KIMI_API_KEY` or `MOONSHOT_API_KEY`
- **Model ID:** `moonshot-v1` (default tier)
- **Available Variants:** `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`
- **Context Window:** Configurable (8K / 32K / 128K variants)
- **Training Data Cutoff:** 2024
- **Capabilities:** Optimized for Chinese and English, cost-effective, reliable
- **Cost:** $0.14 (input) / $0.56 (output) per 1M tokens
- **Best For:** Cost-optimized production use, bilingual applications, high-volume deployments
- **Specialized Domains:** Web development, general-purpose code, script generation
- **API Endpoint:** `https://api.moonshot.cn/v1/chat/completions` (OpenAI-compatible)
- **Note:** Cheapest fully-capable option; strong Chinese language support

---

## Model Selection Strategy

### Automatic Routing

The orchestrator automatically selects models based on task type, cost, and performance metrics:

**Decomposition Phase:**
- Uses cheapest available model (typically Gemini Flash or GPT-4o-mini)
- Fast execution critical as it determines overall task structure

**Code Generation (Primary Route):**
1. Kimi K2.5 — fastest, cheapest
2. Claude Sonnet — excellent code quality
3. GPT-4o — highest capability
4. Gemini Pro — multimodal support

**Code Review (Cross-Provider):**
1. Kimi K2.5 — cost-effective review
2. GPT-4o — comprehensive critique
3. Claude Opus — deep architectural insights
4. Gemini Pro — multimodal review

**Complex Reasoning:**
1. Kimi K2.5 — efficient reasoning
2. Claude Opus — expert-level analysis
3. GPT-4o — structured thinking
4. Gemini Pro — multimodal reasoning

**Data Extraction:**
1. Gemini Flash — fast multimodal extraction
2. GPT-4o-mini — cost-effective text extraction
3. Claude Haiku — lightweight extraction

**Evaluation/Scoring:**
1. Kimi K2.5 — consistent scoring, low cost
2. Claude Opus — highest quality evaluation
3. GPT-4o — reliable assessment
4. Gemini Pro — multimodal evaluation

### Manual Override

Specify preferred models via `ModelProfile` in policy configuration:

```yaml
global:
  - name: performance_focused
    allowed_providers: [anthropic, openai]
    allowed_models: [claude-opus, gpt-4o]

  - name: cost_focused
    allowed_providers: [kimi, deepseek]
    allowed_models: [moonshot-v1, deepseek-chat]
```

### Model Characteristics Matrix

| Model | Speed | Quality | Cost | Context | Multimodal | Best For |
|-------|-------|---------|------|---------|-----------|----------|
| Kimi K2.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | No | Production code |
| DeepSeek Chat | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | No | Fast generation |
| Gemini Flash | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1M | Yes | Data extraction |
| GPT-4o-mini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 128K | Yes | Quick tasks |
| Claude Haiku | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 200K | No | Filtering |
| Claude Sonnet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 200K | No | Default choice |
| Gemini Pro | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 1M | Yes | Large docs |
| GPT-4o | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 128K | Yes | Complex tasks |
| Claude Opus | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 200K | No | Expert analysis |
| DeepSeek Reasoner | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 128K | No | Deep reasoning |

---

## Budget Partitions

Soft caps — enforced per-phase, not hard-blocked:

| Phase | Allocation | Default ($8 total) |
|-------|-----------|---------------------|
| Decomposition | 5% | $0.40 |
| Generation | 45% | $3.60 |
| Cross-review | 25% | $2.00 |
| Evaluation | 15% | $1.20 |
| Reserve | 10% | $0.80 |

---

## Termination Outcomes

| `ProjectStatus` | Trigger |
|-----------------|---------|
| `SUCCESS` | All tasks ≥ threshold, deterministic checks pass, within budget |
| `PARTIAL_SUCCESS` | Some tasks degraded or below threshold |
| `BUDGET_EXHAUSTED` | `spent_usd ≥ max_usd` before all tasks complete |
| `TIMEOUT` | `elapsed_seconds ≥ max_time_seconds` |
| `SYSTEM_FAILURE` | Decomposition failed or zero models available |

---

## Validators

Specified per-task in the decomposed JSON under `"hard_validators"`. A failure forces `score = 0.0` regardless of LLM evaluation score.

| Name | Tool | Behaviour when tool absent |
|------|------|---------------------------|
| `python_syntax` | `compile()` | Always available |
| `pytest` | `pytest` subprocess | Returns fail if pytest not in PATH |
| `ruff` | `ruff` subprocess | Returns pass (skipped) if ruff not installed |
| `json_schema` | `json.loads` + `jsonschema` | Schema check skipped if `jsonschema` not installed |
| `latex` | `pdflatex` subprocess | Returns pass (skipped) if pdflatex not installed |
| `length` | Built-in | Always available; bounds: 10–50 000 chars |

---

## Disk Layout

```
~/.orchestrator_cache/
├── cache.db     # prompt hash → response (aiosqlite, WAL mode)
└── state.db     # project state + per-task checkpoints (WAL mode)

orchestrator/
├── __init__.py        # Exports: Orchestrator, Budget, Model, Task, TaskResult
├── __main__.py        # python -m orchestrator entry point
├── cli.py             # argparse CLI (--file, --output-dir, --resume, …)
├── models.py          # Enums, routing/cost tables, Budget, build_default_profiles()
├── api_clients.py     # UnifiedClient: OpenAI / Anthropic / Google / Kimi
├── engine.py          # Orchestrator: decompose → execute → checkpoint
├── validators.py      # Deterministic validator registry
├── cache.py           # DiskCache (aiosqlite, SHA-256 keyed, WAL)
├── state.py           # StateManager: JSON save/load/resume
├── project_file.py    # YAML project file loader (--file flag)
├── output_writer.py   # Structured output writer (--output-dir flag)
├── policy.py          # Policy, PolicySet, ModelProfile, JobSpec, PolicyHierarchy
├── policy_engine.py   # PolicyEngine: compliance checker (HARD/SOFT/MONITOR modes)
├── policy_dsl.py      # YAML/JSON policy loader + PolicyAnalyzer (contradiction checks)
├── planner.py         # ConstraintPlanner: multi-objective model selection
├── telemetry.py       # TelemetryCollector: EMA latency/quality/cost, real p95
├── hooks.py           # HookRegistry + EventType lifecycle events
├── metrics.py         # MetricsExporter: Console / JSON / Prometheus formats
├── optimization.py    # OptimizationBackend: Greedy / WeightedSum / Pareto
├── audit.py           # AuditLog + AuditRecord (JSONL structured audit trail)
├── cost.py            # BudgetHierarchy, CostPredictor (EMA), CostForecaster
└── agents.py          # AgentPool (multi-orchestrator) + TaskChannel (pub-sub)

projects/
├── example_simple.yaml    # Minimal 3-field project example
├── example_full.yaml      # All fields documented with comments
└── symplectic_engine.yaml # Full physics engine project spec
```

### Output directory structure (`--output-dir`)

```
results/
├── task_001_code_generation.py    # fence-stripped Python
├── task_002_code_review.md        # raw LLM prose
├── task_003_data_extraction.json  # pretty-printed JSON
├── summary.json                   # all scores, costs, raw outputs
└── README.md                      # human-readable results table
```

---

## Known Limitations

| Issue | Workaround |
|-------|-----------|
| Budget ceiling checked before each task **and** mid-iteration, but not mid-API-call | Set `--budget` 10–15% below true ceiling for safety |
| Resume iterates `execution_order` from saved state; verify order is correct for dependency chains | Prefer `--project-id` on initial run so resume is deterministic |
| `_ensure_schema` called on every cache operation (minor overhead) | Acceptable for current scale; add a connection-level init flag if profiling shows cost |
| Kimi K2.5 model name `moonshot-v1` maps to the default tier; for a specific snapshot append the date (e.g. `moonshot-v1-8k`) | Set the `KIMI_MODEL` env var or hardcode in `Model.KIMI_K2_5` if needed |

---

## Advanced Features

### Improvement 1 — Optimization Backends

Three pluggable model-selection strategies are available:

| Backend | Strategy | Best for |
|---------|----------|----------|
| `GreedyBackend` | Highest `quality × trust / cost` | Default; fast single-winner selection |
| `WeightedSumBackend` | Weighted sum of normalised metrics | Tunable trade-off between cost, quality, speed |
| `ParetoBackend` | Pareto-front filtering + scoring | Non-dominated solutions; principled multi-objective |

```python
from orchestrator import Orchestrator, Budget
from orchestrator.optimization import ParetoBackend, WeightedSumBackend

orch = Orchestrator(budget=Budget(max_usd=10.0))

# Switch to Pareto-optimal routing
orch.set_optimization_backend(ParetoBackend())

# Or tune weighted routing
orch.set_optimization_backend(WeightedSumBackend(
    w_quality=0.5, w_trust=0.3, w_cost=0.2
))
```

---

### Improvement 2 — Policy Governance

Policy constraints are **first-class artifacts** — every routing decision can be explained in terms of compliance tags, region constraints, cost caps, and latency SLAs.

```python
from orchestrator.policy import Policy, PolicySet, EnforcementMode, PolicyHierarchy

# 4-level hierarchy: Org → Team → Job → Node
hier = PolicyHierarchy(
    org=[Policy("gdpr", allow_training_on_output=False)],
    team={"eng": [Policy("eu_only", allowed_regions=["eu", "global"])]},
)
policies = hier.policies_for(team="eng", job_id="job_001")

# Audit log (structured JSONL)
from orchestrator.audit import AuditLog
log = AuditLog()
log.flush_jsonl("audit.jsonl")
```

`EnforcementMode` options: `HARD` (block on any violation), `SOFT` (allow soft violations), `MONITOR` (log only).

---

### Improvement 3 — Telemetry, Hooks & Metrics Export

#### Event Hooks

Subscribe to engine lifecycle events without modifying core logic:

```python
from orchestrator import Orchestrator, Budget
from orchestrator.hooks import EventType

orch = Orchestrator(budget=Budget(max_usd=10.0))

orch.add_hook(EventType.TASK_COMPLETED, lambda task_id, result, **_:
    print(f"[{task_id}] score={result.score:.3f}"))

orch.add_hook(EventType.BUDGET_WARNING, lambda phase, ratio, **_:
    print(f"WARNING: {phase} at {ratio:.0%} of budget"))

orch.add_hook(EventType.VALIDATION_FAILED, lambda task_id, model, **_:
    print(f"Validator failed for {task_id} using {model}"))
```

Available events: `TASK_STARTED`, `TASK_COMPLETED`, `VALIDATION_FAILED`, `BUDGET_WARNING`, `MODEL_SELECTED`.

#### Metrics Export

```python
from orchestrator.metrics import ConsoleExporter, JSONExporter, PrometheusExporter

# ASCII table to stdout
orch.set_metrics_exporter(ConsoleExporter())
orch.export_metrics()

# JSON file (for dashboards)
orch.set_metrics_exporter(JSONExporter("/tmp/orchestrator_metrics.json"))
orch.export_metrics()

# Prometheus text format (for node_exporter textfile collector)
orch.set_metrics_exporter(PrometheusExporter("/var/lib/node_exporter/orchestrator.prom"))
orch.export_metrics()
```

Exported metrics per model: `calls_total`, `success_rate`, `latency_avg_ms`, `latency_p95_ms`, `quality_score`, `trust_factor`, `cost_avg_usd`, `validator_failures_total`.

**Real p95 latency** is computed from a sorted rolling buffer of the last 50 samples — more accurate than the previous `2 × avg` approximation. **Cost EMA** tracks actual per-call USD spend, skipping cache hits and failures (which would corrupt the moving average toward zero).

---

### Improvement 4 — Policy DSL (YAML/JSON)

Externalize compliance rules from Python code into declarative files:

```yaml
# policies.yaml
global:
  - name: gdpr
    allow_training_on_output: false
    enforcement_mode: hard
team:
  eng:
    - name: eu_only
      allowed_regions: [eu, global]
job:
  job_001:
    - name: cost_cap
      max_cost_per_task_usd: 0.50
      max_latency_ms: 5000.0
```

```python
from orchestrator.policy_dsl import load_policy_file, PolicyAnalyzer

hierarchy = load_policy_file("policies.yaml")   # pyyaml required for .yml/.yaml
# JSON always works without extra dependencies:
hierarchy = load_policy_file("policies.json")

# Static contradiction analysis before running
from orchestrator.policy import Policy
report = PolicyAnalyzer.analyze(hierarchy.policies_for(team="eng"))
if not report.is_clean():
    print("Policy errors:", report.errors)
    print("Policy warnings:", report.warnings)
```

`PolicyAnalyzer` detects: allowed ∩ blocked provider overlap (error), empty `allowed_regions` (error), disjoint cross-policy `allowed_providers` (warning), missing cost cap / latency SLA (info).

> **Soft dependency:** `pyyaml` is only required for `.yaml`/`.yml` files. JSON loading uses stdlib `json` and always works.

---

### Improvement 5 — Advanced Agents (AgentPool + TaskChannel)

#### AgentPool — Multi-Orchestrator Meta-Controller

Run multiple Orchestrators in parallel (A/B testing, ensemble, load distribution):

```python
from orchestrator import Orchestrator, Budget, AgentPool
from orchestrator.optimization import ParetoBackend, GreedyBackend
from orchestrator.policy import JobSpec

pool = AgentPool()
pool.add_agent("pareto", Orchestrator(budget=Budget(max_usd=5.0)))
pool.add_agent("greedy", Orchestrator(budget=Budget(max_usd=5.0)))

spec = JobSpec(
    project_description="Build a FastAPI auth service",
    success_criteria="All tests pass",
    budget=Budget(max_usd=5.0),
)

results = asyncio.run(pool.run_parallel({"pareto": spec, "greedy": spec}))
best = pool.best_result(results)   # highest mean TaskResult.score

# Merge telemetry across all agents for global observability
merged_profiles = pool.merge_telemetry()
```

- One failing agent does not cancel other agents (`asyncio.gather(return_exceptions=True)`)
- `merge_telemetry()` averages EMA fields and sums counters across all agents

#### TaskChannel — Inter-Task Messaging

Share structured results between upstream and downstream tasks within a run:

```python
# In task handler / hook:
ch = orch.get_channel("artifacts")
await ch.put({"type": "schema", "content": schema_json})

# In a downstream hook (non-destructive read):
msgs = ch.peek_all()   # items remain in queue after peek
```

---

### Improvement 6 — Economic / Cost Layer

#### Hierarchical Budget Caps (Cross-Run)

```python
from orchestrator.cost import BudgetHierarchy, CostPredictor, CostForecaster, RiskLevel
from orchestrator import Budget

# Cross-run org/team/job caps (persist across multiple run_job() calls)
hier = BudgetHierarchy(
    org_max_usd=100.0,
    team_budgets={"eng": 30.0},
    job_budgets={"job_001": 10.0},
)
orch = Orchestrator(budget=Budget(max_usd=10.0), budget_hierarchy=hier)
# run_job() will raise ValueError if any cap would be exceeded
```

#### Adaptive Cost Prediction (EMA)

```python
predictor = CostPredictor(alpha=0.1)  # EMA; falls back to COST_TABLE for unknown pairs

# Record actual observed costs during execution
predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, actual_cost_usd=0.000032)

# Predict future cost (EMA or COST_TABLE fallback)
est = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)

# Find the cheapest model for a task type
cheapest = predictor.cheapest_model(TaskType.CODE_GEN, candidates=[...])
```

Wire the predictor into the engine for live adaptation:

```python
orch = Orchestrator(budget=Budget(max_usd=10.0), cost_predictor=predictor)
```

#### Pre-Flight Cost Forecasting

```python
from orchestrator.cost import CostForecaster, RiskLevel
from orchestrator.models import build_default_profiles

profiles = build_default_profiles()
report = CostForecaster.forecast(tasks, profiles, predictor, budget=Budget(max_usd=10.0))

print(f"Estimated total: ${report.estimated_total_usd:.4f}")
print(f"Risk level: {report.risk_level.value}")  # low / medium / high
print(f"Breakdown: {report.estimated_per_phase}")

if report.risk_level == RiskLevel.HIGH:
    print("WARNING: estimated cost exceeds 80% of budget!")
```

Risk thresholds: **LOW** < 50 %, **MEDIUM** 50–80 %, **HIGH** ≥ 80 % of `budget.max_usd`.

---

## Differentiation

| Capability | This Orchestrator | LangChain / AutoGen |
|------------|-------------------|---------------------|
| **Policy-as-code** | First-class `Policy` / `PolicySet` / `PolicyHierarchy` with static analysis | Ad-hoc guardrails, no formal policy model |
| **Pareto-optimal routing** | `ParetoBackend` selects non-dominated models across quality, trust, cost | No multi-objective routing |
| **Real p95 latency** | Sorted 50-sample buffer; true p95 | Not tracked |
| **Hierarchical budgets** | Org → Team → Job cross-run caps via `BudgetHierarchy` | Per-run only |
| **Adaptive cost EMA** | Per-(model × task_type) EMA; COST_TABLE fallback | No cost prediction |
| **Pre-flight forecasting** | `CostForecaster` estimates spend + risk before execution starts | No pre-flight estimation |
| **Policy DSL** | YAML/JSON externalized compliance rules + `PolicyAnalyzer` contradiction checks | No declarative policy files |
| **Event hooks** | `HookRegistry` with 5 lifecycle events; exception-isolated callbacks | Limited observability hooks |
| **Prometheus metrics** | Native Prometheus text format export; no `prometheus_client` dependency | Requires external integration |
| **Multi-agent ensemble** | `AgentPool` runs N orchestrators in parallel; `merge_telemetry()` aggregates | Single-agent by default |
| **Inter-task messaging** | `TaskChannel` (asyncio.Queue wrapper) with non-destructive `peek_all()` | No typed inter-task channels |
| **Audit log** | Structured JSONL with `AuditRecord` per decision | No built-in audit trail |
