# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Models](https://img.shields.io/badge/models-52-blue)]()
[![engine.py](https://img.shields.io/badge/engine.py-2.6k%20lines-green)]()
[![Import Contracts](https://img.shields.io/badge/import--linter-3%20contracts-brightgreen)]()

> **Autonomous Multi-Agent Software Development Platform**  
> Decomposes project specs into tasks, routes them to 52 LLMs across 15 providers, and executes generate→critique→revise→evaluate cycles with self-improving skill documents, 20+ ARA reasoning methods, and CI-quality gating.

---

## Quick Start

```bash
pip install -e ".[dev,security,tracing]"
cp .env.example .env          # add at least one API key
python -m orchestrator --project "Build a FastAPI todo app" --budget 2.0
```

### Other Commands

```bash
# Plan only (no execution)
python -m orchestrator --project "Build a REST API" --criteria "All tests pass" --dry-run

# Resume previous project
python -m orchestrator --resume <project_id>

# Dashboard with cross-run metrics
python -m orchestrator dashboard --days 30

# Interactive REPL
python -m orchestrator slash
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--budget 5.0` | 8.0 | Max spend in USD |
| `--concurrency 3` | 3 | Parallel API calls |
| `--tdd-first` | off | Test-Driven Development mode |
| `--dry-run` | off | Plan only, no execution |
| `--tracing` | off | OpenTelemetry tracing |
| `--output-dir ./my-app` | auto | Output directory |

---

## Key Capabilities

| Capability | What It Does |
|------------|-------------|
| **9 Specialized Agents** | Architect, Developer, Reviewer, Tester, DevOps, Researcher, PM, QA, User |
| **52 LLMs / 15 Providers** | GPT-5, Claude Sonnet 4.6, Gemini, DeepSeek, Qwen, Grok, and more |
| **SkillOpt** | Self-improving per-TaskType skill documents — optimizer LLM evolves prompts via trajectory feedback |
| **20+ ARA Reasoning Methods** | Debate, Jury, CoVE, SoT, ToT, Self-Discover, MAP-Elites |
| **Hallucination Defense** | 8-layer: PersuasionDefense + CoVE + cross-model review + self-consistency retry |
| **Dual Budget Enforcement** | Per-run `Budget` + cross-run `BudgetHierarchy` (org → team → job) |
| **Self-Correcting Agents** | 3-attempt retry loops with error feedback and dynamic replanning |
| **CI-Quality Gating** | Syntax validation, lint, security scan, hallucination detection |
| **Human-in-the-Loop** | Approval gates for architecture, security, and deployment |
| **Import Boundary Enforcement** | `import-linter` with 3 layer contracts (domain purity, app-no-infra) |

---

## Architecture

```
CommandCenter ──→ OrchestratorFacade ──→ GoalDecomposer (HTN)
                                                 │
                           ┌─────────────────────┴─────────┐
                           ▼                               ▼
                    ProductManager                  ArchitectAgent
                           │                               │
                           └─────────────┬─────────────────┘
                                         ▼
                              AgentOrchestrator (parallel)
                    ┌──────────┬─────────┬─────────┬───────────┐
                    ▼          ▼         ▼         ▼           ▼
              Developer   Reviewer   Tester    DevOps     Researcher
                                         │
                                         ▼
                                  ProjectWorkspace
                                  (shared blackboard)
```

### Core Pipeline (per task)

```
GenerateStage ──→ PersuasionDefenseStage ──→ CritiqueStage ──→ EvaluateStage
      │                                                                │
      └──── SelfConsistencyStage ──── PreflightStage ──── ValidateStage ◄──┘
```

Skill documents are injected into `GenerateStage` via `PipelineContext.skill_prefix` when SkillOpt has trained a skill for the current TaskType.

---

## Project Structure

```
orchestrator/
├── domain/              — Protocols (ports), models, config schemas
├── application/         — 15 extracted service classes
│   ├── skill_optimizer.py        — per-TaskType epoch loop
│   ├── skill_manager.py          — facade + epoch scheduling
│   ├── skill_store.py            — aiosqlite trajectory + skill persistence
│   ├── model_health_tracker.py   — circuit breaker record_success/record_failure
│   ├── resumption_service.py     — resume_project logic
│   ├── dashboard_bridge.py       — null-safe dashboard notifications
│   ├── git_bridge.py             — git commit dispatch
│   ├── project_runner.py         — run_project / run_job / dry_run
│   └── ...                       — evaluator, critique_cycle, decomposer, etc.
├── engine.py            — Mediator facade (2,643 lines, −50% from original)
├── engine_core/         — ServiceContainer, TaskPipeline, 7 pipeline stages
├── infrastructure/      — SQLite state, disk cache, LLM adapter
├── agents/              — 9 specialized agent implementations
└── events/              — Unified EventBus + SyncHookRegistry
```

---

## SkillOpt — Self-Improving Prompts

The orchestrator continuously improves its own per-TaskType system prompts through a text-space optimization loop — no fine-tuning required.

```
Task executes → Trajectory recorded (prompt, output, score, critique)
                       │
              Every epoch_size trajectories
                       │
                       ▼
          Optimizer LLM proposes structured patches
          (append / insert_after / replace / delete)
                       │
          Edit budget enforced (≤150 tokens total)
                       │
          Validation gate: held-out val split must improve
                       │
          Accepted → best_skill.md updated
          Rejected → negative-feedback buffer (informs next epoch)
                       │
          Every 5 epochs → slow update: rebuild ## Guidance block
```

**Off by default.** Activate with `ORCH_SKILL_OPTIMIZATION_ENABLED=true`.

---

## Pipeline Stages (7)

| Stage | What It Checks | Cost |
|-------|---------------|------|
| **Generate** | LLM code generation with optional skill injection | $0.01–0.05 |
| **PersuasionDefense** | Claim extraction + NLI verification | $0.05–0.10 |
| **Critique** | Cross-model review (different provider) | $0.02–0.08 |
| **Evaluate** | 2-pass self-consistency scoring | $0.02–0.06 |
| **Validate** | Syntax, bracket balance, ruff lint | $0.00 |
| **Preflight** | PASS/WARN/ENRICH/BLOCK quality gate | $0.00–0.03 |
| **SelfConsistency** | Score < 0.7 → retry with fallback model | $0.01–0.05 |

---

## 52 Models / 15 Providers

**Free:** owl-alpha, deepseek-v4-flash:free, nemotron-3-nano-omni, poolside/laguna-m.1, poolside/laguna-xs.2

**Budget ($0.01–0.50):** ling-2.6-flash, granite-4.1-8b, deepseek-v4-flash, qwen-3.5-flash, codestral-2508, gemini-2.5-flash

**Standard ($0.50–2.00):** deepseek-reasoner, qwen-3.6-plus, kimi-k2.6, gpt-5.4-nano, gemini-2.5-pro, grok-4.20

**Premium ($2.00+):** gpt-5, claude-sonnet-4.6, qwen-3.7-max, o3, sonar-pro

---

## 20+ ARA Reasoning Methods

| Method | Best For |
|--------|----------|
| Multi-Perspective | Architecture decisions, code review |
| Debate | Design trade-offs (two models argue, judge decides) |
| Jury | Critical quality (3 generators, 3 critics) |
| CoVE | Factual accuracy (draft → verify → revise) |
| SoT | Complex multi-part problems (parallel sub-solves) |
| ToT | Strategic choices (branch → evaluate → backtrack) |
| MAP-Elites | Code optimization (3×3 grid evolution) |
| PersuasionDefense | Hallucination prevention (claim extraction + NLI) |
| Brainstorming, PreMortem, Bayesian, Delphi, Socratic, Research, and more |

---

## Testing

```bash
pytest tests/ -v                    # All tests
pytest -m "not slow"                # Skip slow integration tests
pytest tests/contracts/ -v          # Protocol contract tests
pytest tests/unit/ -v               # Unit tests only
```

---

## Configuration

```bash
# Required (at least one provider key)
OPENAI_API_KEY=sk-...

# Optional tuning
ORCH_MAX_CONCURRENCY=3               # Parallel API calls
ORCH_DEFAULT_BUDGET_USD=10.0         # Default per-run budget
ORCH_CONTEXT_COMPRESSION=true        # Enable context compression
ORCH_SKILL_OPTIMIZATION_ENABLED=false  # SkillOpt self-improving prompts
LOG_FORMAT=json                      # Structured JSON logging
LOG_LEVEL=DEBUG                      # Debug logging
```

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| `docs/CODEBASE_MINDMAP.md` | Complete architecture reference |
| `docs/REFACTORING_PLAN_V7.md` | Phases 0–5 refactoring roadmap |
| `docs/DEPENDENCY_POLICY.md` | Import boundary rules |

---

## License

MIT License — see [LICENSE](LICENSE).

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** 6.2.0  
**Last updated:** 2026-05-28
