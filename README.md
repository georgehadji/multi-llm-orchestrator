# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Import Contracts](https://img.shields.io/badge/import--linter-5%20contracts-brightgreen)](.importlinter)
[![Tests](https://img.shields.io/badge/tests-91%20passing-brightgreen)](tests/)

> **Autonomous Multi-Agent Software Development Platform**  
> Decomposes project specifications into executable task graphs, routes them across 50+ LLMs from 15 providers, and executes generate → critique → revise → evaluate cycles with self-improving skill documents, structured verification gating, and CI-quality architectural enforcement.

---

## Quick Start

```bash
pip install -e ".[dev,security,tracing]"
cp .env.example .env          # add at least one API key
python -m orchestrator --project "Build a FastAPI todo app" --budget 2.0
```

### Essential Commands

```bash
# Generate a website (design-system-driven, any page type)
python -m orchestrator website -d "A modern SaaS landing page" --framework next.js

# Hunt for bugs, security issues, performance hotspots in an existing codebase
python -m orchestrator analyze --path "/path/to/codebase" --focus "quality,security,performance"

# Apply a smart modification/feature to an existing codebase safely with rollback
python -m orchestrator modify --repo "/path/to/codebase" --objective "Add login endpoint"

# Plan only (no execution)
python -m orchestrator --project "Build a REST API" --criteria "All tests pass" --dry-run

# Resume a previous project
python -m orchestrator --resume <project_id>

# Dashboard with cross-run metrics
python -m orchestrator dashboard --days 30

# Interactive REPL
python -m orchestrator slash
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--budget 5.0` | 8.0 | Maximum spend in USD per run |
| `--concurrency 3` | 3 | Parallel LLM API calls |
| `--tdd-first` | off | Test-Driven Development mode |
| `--dry-run` | off | Plan only, no execution |
| `--tracing` | off | OpenTelemetry distributed tracing |
| `--output-dir ./my-app` | auto | Artifact output directory |

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Task Graph Engine** | Decomposes projects into dependency-resolved task DAGs; parallel execution with worktree isolation |
| **50+ LLMs / 15 Providers** | GPT-5, Claude Sonnet 4.6, Gemini, DeepSeek, Qwen, Grok, and more — quality-aware routing with fallback chains |
| **Deterministic Verification Gate** | Hard floor beneath LLM scoring: syntax, lint, type, build, and security checks run before evaluation. Failed checks cap the score at 0.15 regardless of LLM opinion. |
| **SkillOpt (Self-Improving Prompts)** | Per-TaskType prompt evolution via trajectory feedback — held-out validation, edit budget enforcement, negative-feedback buffer |
| **20+ Reasoning Methods** | Debate, Jury, CoVE, SoT, ToT, Self-Discover, MAP-Elites, Brainstorming, PreMortem, Bayesian, Delphi, Socratic |
| **Hallucination Defense** | 8-layer: PersuasionDefense + CoVE + cross-model review + self-consistency retry |
| **Dual Budget Enforcement** | Per-run `Budget` + cross-run `BudgetHierarchy` (org → team → job) |
| **Self-Correcting Agents** | 3-attempt retry loops with error feedback and dynamic replanning |
| **Human-in-the-Loop** | Approval gates for architecture, security, and deployment decisions |
| **Import Boundary Enforcement** | `lint-imports` with 5 enforceable contracts (domain purity, application-no-infra, engine-core-no-infra, root-no-infra, application-no-engine) |
| **Website Generator** | Design-system-driven site builder with 20 curated themes, 3D/WebGL support, LLM-powered content |
| **Verification Receipts** | Structured execution receipts with timings, artifact hashes, and outcome classification (not_run/passed/failed/blocked) |
| **Codebase-Aware Agent** | Scans codebases using `CodebaseReader`, analyses patterns, hunts bugs, and applies safe topological modifications with automatic rollback on validation failure. |

---

## Verification Gate

The orchestrator enforces a mandatory deterministic floor before any LLM-based evaluation:

```
LLM-generated artifact
       │
       ▼
VerificationGate.run(artifact)
       │
       ├── Syntax check    (compile)     — REQUIRED
       ├── Security scan   (pattern)     — REQUIRED
       ├── Build check     (exec)        — RECOMMENDED
       ├── Lint check      (ruff)        — RECOMMENDED
       └── Type check      (mypy)        — RECOMMENDED
       │
       ▼
    All pass? ──yes──→ LLM scoring proceeds
       │
      no
       │
       ▼
    Score capped at 0.15 (FAIL_SCORE_FLOOR)
    Full receipts with artifact hash persisted through CritiqueReport
```

- Four outcome states: `not_run`, `passed`, `failed`, `blocked` (WBS-1)
- Policy-driven check configuration per task type
- SHA-256 artifact hashes for audit trail integrity
- Execution receipts with wall-clock timings and command labels
- CI guard (`scripts/check_subprocess_cleanup.py`) prevents subprocess lifecycle leaks

---

## Architecture

### Layered Model

```
┌──────────────────────────────────────────────────────────────┐
│                    Driving Adapters                           │
│          CLI · API · Dashboard · Webhook                     │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              Orchestrator Mediator (engine.py)                │
│                ServiceContainer (composition root)            │
├──────────────────────────────────────────────────────────────┤
│                    Engine Core                                │
│    TaskPipeline · PipelineExecutor · 7 Pipeline Stages       │
│    ProjectPlanner · StateCoordinator · Decomposer            │
├──────────────────────────────────────────────────────────────┤
│                 Application Layer                             │
│    EvaluatorService · VerificationGate · CritiqueCycle        │
│    ExecutorService · SkillOptimizer · SkillStore              │
│    ResumptionService · BudgetEnforcer · ModelHealthTracker    │
├──────────────────────────────────────────────────────────────┤
│                   Domain Layer                                │
│    Models · Ports (16 Protocols) · Exceptions                 │
│    Verification Types (CheckOutcome, ExecutionReceipt,        │
│      VerificationPolicy, DeterministicResult)                 │
├──────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                            │
│    LLM Client (OpenAI/Anthropic/Google/DeepSeek)             │
│    SQLite State · Disk Cache · Verification Check Adapters    │
│    Telemetry · Circuit Breaker · LSP Validator                │
└──────────────────────────────────────────────────────────────┘
```

### Core Pipeline (per task)

```
GenerateStage → PersuasionDefenseStage → CritiqueStage → EvaluateStage
      │                                                              │
      └── SelfConsistencyStage → PreflightStage → ValidateStage ◄───┘
```

### Import Boundaries

The architecture enforces five strict contracts (enforced by `lint-imports`):

| Contract | Rule |
|----------|------|
| **Domain purity** | Domain and models must not import from application, infrastructure, or engine |
| **Application no infra** | Application services must not import concrete infrastructure adapters |
| **Application no engine** | Application services must not import from engine (except documented shims) |
| **Engine core no infra** | Pipeline modules must not import from infrastructure (container.py is exempt) |
| **Root modules no infra** | Root-level `orchestrator/*.py` must not import from infrastructure |

---

## Design Quality Gates (taste-skill)

The orchestrator integrates **taste-skill**, a suite of anti-slop design rules that prevent AI-generated frontend UIs from defaulting to generic patterns. Activated by default for frontend tasks.

### Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `ORCH_TASTE_SKILL_ENABLED` | `true` | Inject anti-slop rules into frontend code generation |
| `ORCH_IMAGE_REFERENCE_PIPELINE` | `false` | Pre-flight visual-direction brief (experimental) |

### Tunable Dials (1–10)

| Dial | Default | What It Controls |
|------|---------|------------------|
| `ORCH_DESIGN_VARIANCE` | 5 | Layout experimentation |
| `ORCH_MOTION_INTENSITY` | 5 | Animation depth |
| `ORCH_VISUAL_DENSITY` | 5 | Information per viewport |

### Design Variants

| Variant | Use Case | Characteristic |
|---------|----------|----------------|
| `DEFAULT` | All frontend tasks | Core anti-slop rules |
| `SOFT` | Premium agency | Serifs, luxury spacing |
| `MINIMALIST` | Editorial | Typography-first, high whitespace |
| `BRUTALIST` | Swiss/industrial | Geometric, constrained palette |
| `REDESIGN` | "redesign" prompts | Structured audit critique |

### Pipeline

1. **Generator stage**: Prepends taste-skill rules to the LLM system prompt, parameterized by dials
2. **Critic stage**: If variant is REDESIGN, uses structured design audit (typography → color → layout → interactivity → content)
3. **Validator stage**: Anti-slop pattern detector (soft WARN-only)

---

## Project Structure

```
orchestrator/
├── domain/                  — Pure types, protocols, exceptions (zero I/O)
│   ├── verification.py      — CheckOutcome, ExecutionReceipt,
│   │                          VerificationPolicy, DeterministicResult
│   ├── ports.py             — 16 Protocol interfaces
│   └── model_registry.py    — Model capability database
├── application/             — 17+ service classes
│   ├── evaluator.py         — LLM scoring + deterministic gate integration
│   ├── verification_gate.py — Chain-of-responsibility check runner
│   ├── skill_optimizer.py   — Per-TaskType prompt evolution
│   └── ...
├── engine_core/             — Pipeline, stages, composition root
│   ├── container.py         — ServiceContainer (composition root)
│   ├── pipeline.py          — TaskPipeline
│   └── stages/              — 7 pipeline stages (Generate, Critique, ...)
├── infrastructure/          — Concrete adapters
│   ├── llm_client.py        — Unified LLM provider adapter
│   ├── verification_checks.py — 5 check adapters + factory
│   └── state.py             — SQLite-backed state persistence
├── agents/                  — Specialized agent implementations
├── scripts/                 — CI guards and tooling
│   ├── check_subprocess_cleanup.py   — AST-based cleanup verification
│   └── bandit_verification.py        — Targeted security scan
├── tests/
│   ├── unit/                — 47 unit tests (domain, gate, checks)
│   ├── regression/          — 44 regression tests (WBS-1 verification)
│   ├── integration/         — Integration tests
│   └── contracts/           — Port contract conformance tests
└── engine.py                — Mediator facade (wires services)
```

---

## SkillOpt — Self-Improving Prompts

The orchestrator continuously improves its per-TaskType system prompts through a text-space optimization loop with no fine-tuning required.

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

## 50+ Models / 15 Providers

| Tier | Cost | Examples |
|------|------|---------|
| **Free** | $0 | owl-alpha, deepseek-v4-flash:free, nemotron-3-nano-omni |
| **Budget** | $0.01–0.50 | deepseek-v4-flash, qwen-3.5-flash, gemini-2.5-flash |
| **Standard** | $0.50–2.00 | deepseek-reasoner, gpt-5.4-nano, gemini-2.5-pro, grok-4.20 |
| **Premium** | $2.00+ | gpt-5, claude-sonnet-4.6, qwen-3.7-max, o3 |

---

## Testing

```bash
# All verification tests (unit + regression)
python -c "import sys, os, importlib.util, asyncio; ..."  # see CI

# Full test suite
pytest tests/ -v

# Quick subset
pytest tests/ -m "not slow"
pytest tests/contracts/ -v
pytest tests/unit/ -v

# Import boundary enforcement
lint-imports

# Subprocess cleanup verification
python scripts/check_subprocess_cleanup.py

# Targeted security scan
python scripts/bandit_verification.py
```

---

## Configuration

```bash
# Required (at least one provider key)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...

# Verification gating
ORCH_MYPY_TIMEOUT=30             # Type check timeout in seconds (1-120)

# Execution
ORCH_MAX_CONCURRENCY=3           # Parallel API calls
ORCH_DEFAULT_BUDGET_USD=10.0     # Default per-run budget

# Skill optimization
ORCH_SKILL_OPTIMIZATION_ENABLED=false

# Design taste-skill
ORCH_TASTE_SKILL_ENABLED=true
ORCH_DESIGN_VARIANCE=5
ORCH_MOTION_INTENSITY=5
ORCH_VISUAL_DENSITY=5

# Observability
LOG_FORMAT=json                  # Structured JSON logging
LOG_LEVEL=INFO
```

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Architecture Mindmap | `architecture_mindmap_v7.md` | Forensic architecture reconstruction with dependency graph, data flows, risk register |
| Architecture Audit V2 | `architecture_audit_v2.md` | 7-phase architecture audit — score 9/10, refactoring roadmap |
| Architecture Score Plan | `architecture_score_improvement_plan.md` | 6-PR plan to raise score from 8/10 to 9/10+ |
| Implementation Audit | `implementation_audit_report.md` | WBS-1 verification subsystem audit |
| Evidence-Driven Plan | `implementation_plan.md` | Multi-release self-improvement strategy |
| REASONIX.md | `REASONIX.md` | Developer setup, commands, conventions |
| AGENTS.md | `AGENTS.md` | Architecture overview, CI pipeline, quick reference |

---

## License

MIT License — see [LICENSE](LICENSE).

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** 6.2.0
