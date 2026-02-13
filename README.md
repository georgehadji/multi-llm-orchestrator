# multi-llm-orchestrator

Decomposes a project description into atomic tasks, routes each to the optimal provider (OpenAI / Anthropic / Google), runs cross-provider generate → critique → revise cycles, and iterates until a quality threshold is met or a budget ceiling is hit.

State is checkpointed to SQLite after every task. Interrupted runs are resumable by project ID.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
│                                                      │
│  Decompose → Route → Generate → Critique → Revise   │
│       ↑                                    │         │
│       └──── Evaluate ← Deterministic Check ┘         │
│                                                      │
│  [Async Disk Cache]  [JSON State]  [Budget Control]  │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │
     ┌─────┴─────┐  ┌────┴────┐  ┌─────┴─────┐
     │  OpenAI   │  │ Gemini  │  │  Claude   │
     └───────────┘  └─────────┘  └───────────┘
```

---

## Requirements

- Python ≥ 3.10
- At least one provider key (others are skipped gracefully)

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI (GPT-4o, GPT-4o-mini) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude Opus, Sonnet, Haiku) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google (Gemini Pro, Flash) |

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
pip install openai anthropic google-genai aiosqlite
pip install pytest ruff jsonschema   # optional
```

---

## CLI

```bash
# New project
python -m orchestrator \
  --project  "Build a FastAPI auth service with JWT" \
  --criteria "All endpoints tested, OpenAPI spec complete" \
  --budget   8.0 \
  --time     5400

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
| `--budget` / `-b` | float | 8.0 | Spend ceiling in USD |
| `--time` / `-t` | float | 5400 | Wall-clock limit in seconds |
| `--project-id` | str | auto | Explicit ID (auto-generated if blank) |
| `--resume` | str | — | Resume project by ID |
| `--list-projects` | flag | — | Print saved project IDs and statuses |
| `--concurrency` | int | 3 | Max simultaneous API calls |
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

| Task type | Primary | Reviewer |
|-----------|---------|----------|
| `code_generation` | Claude Sonnet | GPT-4o |
| `code_review` | GPT-4o | Claude Opus |
| `complex_reasoning` | Claude Opus | GPT-4o |
| `creative_writing` | Claude Opus | GPT-4o |
| `data_extraction` | Gemini Flash | GPT-4o-mini |
| `summarization` | Gemini Flash | Claude Haiku |
| `evaluation` | Claude Opus | GPT-4o |

Reviewer is always from a **different provider** than the generator. Falls back to a different model tier, then any healthy model.

Max iterations per task: 3 (code, reasoning) / 2 (all others).

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
├── cli.py             # argparse CLI
├── models.py          # Enums, dataclasses, routing/cost tables, Budget
├── api_clients.py     # UnifiedClient: async, semaphore, retry, cache
├── engine.py          # Orchestrator: decompose → execute → checkpoint
├── validators.py      # Deterministic validator registry
├── cache.py           # DiskCache (aiosqlite, SHA-256 keyed, WAL)
├── state.py           # StateManager: JSON save/load/resume
└── pyproject.toml     # Package metadata, ruff config
```

---

## Known Limitations

| Issue | Workaround |
|-------|-----------|
| Budget ceiling checked before each task, not enforced mid-task | Set `--budget` 10–15% below true ceiling |
| `validate_ruff` writes a temp file without cleanup | Schedule periodic `tmp` cleanup or disable `ruff` validator |
| Google client uses `asyncio.get_event_loop()` (deprecated ≥ 3.12) | Pin Python ≤ 3.11 until upstream `google-genai` fixes async support |
| Resume iterates `execution_order` from saved state; verify order is correct for dependency chains | Prefer `--project-id` on initial run so resume is deterministic |
| `_ensure_schema` called on every cache operation (minor overhead) | Acceptable for current scale; add a connection-level init flag if profiling shows cost |
