# multi-llm-orchestrator

Decomposes a project description into atomic tasks, routes each to the optimal provider (OpenAI / Anthropic / Google / Kimi), runs cross-provider generate → critique → revise cycles, and iterates until a quality threshold is met or a budget ceiling is hit.

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
| `code_generation` | Claude Sonnet → GPT-4o → Kimi K2.5 → Gemini Pro | 4096 |
| `code_review` | GPT-4o → Claude Opus → Gemini Pro | 2048 |
| `complex_reasoning` | Claude Opus → GPT-4o → Gemini Pro → **Kimi K2.5** | 2048 |
| `creative_writing` | Claude Opus → GPT-4o → Gemini Pro | 2048 |
| `data_extraction` | Gemini Flash → GPT-4o-mini → Claude Haiku | 1024 |
| `summarization` | Gemini Flash → Claude Haiku → GPT-4o-mini | 512 |
| `evaluation` | Claude Opus → GPT-4o → Gemini Pro → Kimi K2.5 | 600 |

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
├── policy.py          # Policy, PolicySet, ModelProfile, JobSpec
├── policy_engine.py   # PolicyEngine: compliance checker
├── planner.py         # ConstraintPlanner: multi-objective model selection
└── telemetry.py       # TelemetryCollector: EMA latency/quality tracking

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
