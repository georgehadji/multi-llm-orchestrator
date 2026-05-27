# Multi-LLM Orchestrator — Complete Architecture Mindmap

> **Version:** v6.0.0 (2026-05-27)
> **Author:** Georgios-Chrysovalantis Chatzivantsidis
> **Codebase Size:** ~5036 lines (engine.py), ~150,000 lines total (orchestrator/)
> **Total Files:** 497 Python files (orchestrator/), 42 test files (tests/)

---

## System Overview (DDD Hexagonal Architecture)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Multi-LLM Orchestrator v6.0                        │
│          Autonomous Multi-Agent Software Development Platform          │
└──────────────────────────────────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────v──────┐         ┌──────v──────┐         ┌──────v──────┐
    │   DOMAIN    │         │ APPLICATION │         │INFRASTRUCTURE│
    │   Layer     │ ◄────── │    Layer    │ ◄────── │    Layer    │
    │ (pure logic)│         │ (use cases) │         │  (adapters) │
    └─────────────┘         └─────────────┘         └─────────────┘
           │                       │                       │
     models.py              engine.py              infrastructure/
     domain/                application/           api_clients.py
     exceptions.py          engine_core/           state.py
     ports.py               services/              budget.py
                            agents/                cache.py
```

**Layer Rules:**
- **Domain** — pure business logic, zero infrastructure imports, defines Protocols (ports)
- **Application** — orchestration facade, services, use cases; depends on domain ports only
- **Infrastructure** — concrete adapters (LLM clients, SQLite, caches); implements domain ports
- **Backward-compat** — `engine_core/` and `services/` are re-export shims to `application/`

---

## Current Execution Flow

```
User says: "Build a todo app"
       │
       ▼
CommandCenter ──→ OrchestratorFacade ──→ GoalDecomposer (HTN recursive)
       │                                        │
       │                                  ┌─────┴─────────┐
       │                                  ▼               ▼
       │                           ProductManager   ArchitectAgent
       │                           (generates        (designs system)
       │                            user stories)
       │                                  │               │
       │                                  ▼               ▼
       └──────────────────────────→  AgentOrchestrator dispatches
                                              │
                ┌────────────┬──────┬─────────┴──┬──────┬──────────┐
                ▼            ▼      ▼            ▼      ▼          ▼
          Developer    Reviewer   Tester      DevOps  Researcher  QCAgent
          (write)      (audit)    (test)      (build) (search)    (gate)
                │            │      │            │                    │
                └────────────┴──────┴────────────┴────────────────────┘
                                              │
                                              ▼
                                     ProjectWorkspace
                                     (shared blackboard)
                                              │
                                              ▼
                              ┌───────────────┼────────────────┐
                              ▼               ▼                ▼
                        Security Rules  UX Standards      OpenGraph Tags
                        (22 OWASP)      (20 WCAG)         (og:image 1200x630)
                              │               │                │
                              └───────────────┴────────────────┘
                                              │
                                              ▼
                                     Delivery: tested, secure, deployed
                                     GitIntegration (branch→commit→push→PR)
```

**New v6.0 entry points:**
| Entry | Command |
|-------|---------|
| CommandCenter REPL | `python -m orchestrator command_center` |
| Gateway multi-platform | `python -m orchestrator.gateway.run` |
| Kanban work queue | `python -m orchestrator.kanban.board` |
| IDE Backend | `python -m orchestrator.ide_backend.server` |

---

## 9 Specialized Agents

```
AgentOrchestrator (coordinator)
  │
  ├── UserAgent ──── Talks to the user: explains plans, asks questions
  ├── ProductManager ── Generates user stories, maps to modules, P0-P3 priorities
  ├── ArchitectAgent ── System design, framework selection, trade-off analysis
  ├── DeveloperAgent ── Code generation with 3-attempt self-correcting loop
  ├── ReviewerAgent ─── Security audit, cross-model code review
  ├── TesterAgent ───── Test generation with coverage tracking
  ├── DevOpsAgent ───── Docker, CI/CD, infra-as-code
  ├── QCAgent ───────── CI pipeline runner, quality reports, regression detection
  └── ResearcherAgent ── Web search, library research, best practices
```

**Agent Memory:**
```
AgentBase.memory: AgentMemory
  ├── successes[50]     # Pattern → score records
  ├── failures[50]      # Pattern → error records
  ├── success_rate()    # Historical accuracy per agent
  └── lesson()          # "I have 12 successes, avg score 0.89"
```

**Agent Rate Limiter** — Sliding window per-agent limits to prevent runaway costs.

---

## Pipeline Stages (7 Layers)

```
Every CODE_GEN task goes through this pipeline:

GenerateStage → PersuasionDefenseStage → CritiqueStage → EvaluateStage
    │                                                              │
    ▼                                                              ▼
ValidateStage ←────── PreflightStage ←────── SelfConsistencyStage ←──┘
    │
    ▼
Delivery or Retry (with fallback model)
```

| Stage | What It Checks | Cost |
|-------|---------------|------|
| Generate | Single LLM call for initial code | $0.01-0.05 |
| PersuasionDefense | Claim extraction → NLI verification → conflict detection | $0.05-0.10 |
| Critique | Cross-model review (different provider) | $0.02-0.08 |
| Evaluate | 2-pass self-consistency scoring | $0.02-0.06 |
| Validate | Syntax, bracket balance, ruff lint | $0.00 |
| Preflight | PASS/WARN/ENRICH/BLOCK quality gate | $0.00-0.03 |
| SelfConsistency | Score < 0.7 → retry with fallback model | $0.01-0.05 |

**Pipeline stages live in `engine_core/stages/`** (8 files: generate, persuasion_defense, critique, evaluate, validate, preflight, self_consistency, map_elites).

### Hallucination Defense (8 layers)

```
Layer 1: PersuasionDefense     (claim extraction + NLI)
Layer 2: CoVE                  (factored verification, cross-check)
Layer 3: Cross-Model Critique  (different provider)
Layer 4: Self-Consistency      (retry with fallback)
Layer 5: Preflight Gate        (PASS/WARN/ENRICH/BLOCK)
Layer 6: Syntax Validation     (ast.parse, bracket balance)
Layer 7: Developer Self-Correct (3-attempt retry)
Layer 8: Knowledge Graph       (historical confidence scoring)
```

---

## 20+ ARA Reasoning Methods

```
ARA METHODS (ara_pipelines.py)
═══════════════════════════════════════════════════════════════

STANDARD METHODS:
  Multi-Perspective  — 4 angles: constructive, destructive, systemic, minimalist
  Iterative          — Progressive refinement
  Debate             — 2 agents argue, meta-evaluator decides
  Research           — Web discovery + LLM synthesis
  Jury               — 4 generators, 3 critics, meta-evaluation
  Scientific         — Hypothesis: formulate → test → refine
  Socratic           — Probing questions uncover assumptions

SPECIALIZED METHODS:
  Pre-Mortem         — "Assume it failed — why?"
  Bayesian           — Prior → evidence → posterior
  Dialectical        — Thesis → antithesis → synthesis
  Analogical         — Cross-domain solution mapping
  Delphi             — 4 experts → aggregate → revise → converge
  Brainstorming      — Divergent idea generation → cluster → develop
  VerbalizedSampling — Probability calibration, uncertainty quantification
  PersuasionDefense  — Claim extraction + NLI verification

COGNITIVE METHODS (v2.0):
  CoVE (Chain-of-Verification)
    1. Draft → extract claims
    2. Verify: generate independent questions per claim
    3. Answer: factorial execution (parallel LLM calls)
    4. Cross-check: Factor+Revise inconsistency detection
    5. Revise: correct errors, add caveats

  SoT (Skeleton-of-Thought)
    1. Skeleton: decompose into 3-5 sub-problems
    2. Solve: parallel execution per sub-problem
    3. Assemble: synthesize into coherent answer

  ToT (Tree-of-Thoughts)
    1. Decompose into decision points
    2. Generate: N candidate actions per point
    3. Evaluate: score candidates
    4. Backtrack: decide to proceed/backtrack/terminate

  Self-Discover
    1. Select: choose 3-5 reasoning modules from inventory
    2. Adapt: convert modules to concrete instructions
    3. Implement: execute modules in sequence → synthesize

  MAP-Elites (ARA #21)
    1. Initialize: 9 variants in 3x3 grid (complexity × performance)
    2. Place: best variant per grid cell
    3. Select: 20% elite + 30% diverse + 50% exploratory
    4. Mutate: LLM-powered diff-based mutation
    5. Repeat: 3 generations → Pareto frontier
```

---

## 52 Models / 15 Providers

```
COST TIERS:
   FREE:       owl-alpha, deepseek-v4-flash:free, nemotron-3-nano-omni:free,
               poolside/laguna-m.1:free, poolside/laguna-xs.2:free
   ULTRA-LOW ($0.01-0.09):    ling-2.6-flash ($0.01), granite-4.1-8b ($0.05),
                               deepseek-v4-flash ($0.10/$0.20)
   BUDGET ($0.10-0.50):       qwen3.5-flash, qwen3-coder-next, codestral-2508,
                               qwen3.6-flash, gemini-2.5-flash ($0.075)
   STANDARD ($0.50-2.00):     deepseek-reasoner, qwen3.6-plus, kimi-k2.6,
                               gpt-5.4-nano, gemini-2.5-pro, grok-4.20
   PREMIUM ($2.00+):          gpt-5, claude-sonnet-4.6, qwen3.7-max, o3,
                               gpt-5.4, sonar-pro, grok-4
```

**Routing Architecture:**
```
ROUTING_TABLE (TaskType → Model Priority List)
═══════════════════════════════════════════════════════════════

CODE_GEN:
  1. DEEPSEEK_V4_FLASH      ($0.10/$0.20)
  2. CODESTRAL_2508         ($0.30/$0.90)
  3. QWEN_3_CODER_NEXT      ($0.11/$0.88)
  4. GPT_5                  ($1.25/$10.00)  * premium

CODE_REVIEW:
  1. QWEN_3_6_PLUS          ($0.33/$1.95)
  2. CLAUDE_SONNET_4_6      ($3.00/$15.00)
  3. GEMINI_2_5_PRO         ($1.25/$5.00)
  4. GROK_4_20              ($1.25/$2.50)

REASONING:
  1. DEEPSEEK_REASONER      ($0.70/$1.26)
  2. O3                     ($2.00/$8.00)
  3. QWEN_3_7_MAX           ($2.50/$7.50)

FALLBACK_CHAIN (for each model → cross-provider fallback):
  DEEPSEEK_V4_FLASH  → QWEN_3_6_FLASH
  CODESTRAL_2508      → QWEN_3_CODER_NEXT
  GPT_5              → CLAUDE_SONNET_4_6 → GEMINI_2_5_PRO
```

**Advanced routing components:**
- `adaptive_router.py` — Circuit breaker v2 with HEALTHY/DEGRADED/DISABLED states
- `outcome_router.py` — Outcome-weighted router using production feedback learning
- `escalation.py` — Automatic escalation to higher-capability models on quality failure
- `aggregator.py` — Cross-run profile aggregator for model performance stats

---

## Agent Model Assignments

```
Agent              Budget Model              Premium Model
──────             ────────────              ─────────────
Architect          DEEPSEEK_REASONER($0.70)  GPT_5 ($1.25)
Developer          CODESTRAL_2508  ($0.30)   GPT_5 ($1.25)
Reviewer           QWEN_3_6_PLUS   ($0.33)   CLAUDE_SONNET_4_6 ($3.00)
Tester             QWEN_3_CODER_NEXT($0.11)  GPT_5 ($1.25)
DevOps             DEEPSEEK_V4_FLASH($0.10)  GPT_5_2_CODEX ($1.75)
Researcher         SONAR_PRO       ($3.00)   SONAR_DEEP_RESEARCH ($2.00)
Product Manager    DEEPSEEK_REASONER($0.70)  GPT_5 ($1.25)
QA                 CODESTRAL_2508  ($0.30)   CLAUDE_SONNET_4_6 ($3.00)
User               OWL_ALPHA       (FREE)    CLAUDE_SONNET_4_6 ($3.00)
```

---

## 4 Pillars Management

```
┌─────────────────────────────────────────────────────────────┐
│                    FOUR PILLARS                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Project Mgmt     │ Product Mgmt     │ Knowledge Mgmt         │
│ SprintPlanner    │ ProductBacklog   │ KnowledgeBase          │
│ ProgressReporter │ UserStory        │ DocsGenerator          │
│ MilestoneTracker │ P0-P3 Priorities │ Searchable Decisions   │
├─────────────────┴─────────────────┼─────────────────────────┤
│ Quality Control                    │                         │
│ QualityReport (0-10 score)        │                         │
│ RegressionDetector                │                         │
│ PASS/REVISE/BLOCK recommendation  │                         │
└───────────────────────────────────┴─────────────────────────┘
```

---

## Security & UX

```
SECURITY (22 OWASP Rules)              UX STANDARDS (20 WCAG Rules)
══════════════════════════════════     ══════════════════════════════

Web:                                   Layout:
  CSP, HSTS, X-Frame-Options           Responsive, mobile-first
  X-Content-Type-Options               Spacing scale (4px/8px)
  Referrer-Policy                      Visual hierarchy

API:                                   Accessibility:
  SQL injection prevention              WCAG AA (4.5:1 contrast)
  Rate limiting (100/min)               Keyboard navigation
  Strict CORS (whitelist origins)       ARIA labels + landmarks
  HTTPS enforcement                     Touch targets (44×44px)

Auth:                                  Typography:
  bcrypt (cost 12+)                     Font pairing (2 max)
  HttpOnly + Secure + SameSite          Line length (60-80 chars)
  JWT RS256, short expiry (15m)         Type scale (1.25 ratio)
  CSRF tokens for state changes

Data:                                  Color + Interaction:
  Env vars (no .env in git)             CSS custom properties
  AES-256 at rest                       Dark mode support
  TLS 1.3 in transit                    Micro-interactions (150-300ms)
                                        Loading + error states

OPENTRAPH (Every generated page):
  og:title (60-70 chars)           twitter:card = summary_large_image
  og:description (150-160 chars)   JSON-LD structured data
  og:image  (1200×630 PNG)         All output HTML-escaped
```

---

## 5-Layer Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY LAYERS                            │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: ProjectWorkspace (in-memory, per-project)          │
│   File versions, architecture decisions, agent messages     │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: PersistentWorkspace (SQLite, crash recovery)       │
│   Auto-saves on every write_file() + record_decision()      │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: ExperienceBuffer (cross-task learning)             │
│   success_patterns[200], model_scores, method_effectiveness │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: KnowledgeGraph (relational, networkx)             │
│   Nodes: (TaskType, Model, Method, Score)                  │
│   Edges: used_with, produced_score, failed_on              │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: AgentCache (SHA-256 hash, TTL 1h)                 │
│   Cached LLM responses per (goal + context) hash           │
└─────────────────────────────────────────────────────────────┘
```

**Memory consolidation:** `memory/consolidation.py` and `learning/memory_compressor.py` summarize many patterns into compact lessons.

---

## v6.0 New Features

### A2A Protocol (Agent-to-Agent)
`a2a_protocol.py` (833 lines) — Invoke external agents (LangGraph, Vertex AI, Azure AI Foundry) using the A2A protocol. Adapter pattern for inter-agent communication.

### Saga Pattern
`sagas.py` (850 lines) — Distributed transaction management with compensation (rollback) on failure. Supports long-running multi-step workflows.

### Control Plane
`control_plane.py` (285 lines) — Full constraint-enforcement workflow:
1. `validate(job, policy)` — schema + static analysis
2. `monitor.check_global()` — hard constraints pre-run
3. `solve_constraints()` — routing plan, SLA fit

### Outcome-Weighted Router
`outcome_router.py` (580 lines) — Routes tasks to models based on proven production outcomes, not just cost/latency estimates. Creates a learning feedback loop.

### Adaptive Router (Circuit Breaker v2)
`adaptive_router.py` (175 lines) — Per-model states: HEALTHY → DEGRADED → DISABLED. Degraded models are skipped during cooldown; disabled models are permanently blocked.

### Escalation Engine
`escalation.py` (270 lines) — Automatic escalation to higher-capability models on quality threshold failure. Chain of Responsibility pattern.

### Health Checks
`health.py` (468 lines) — Kubernetes-style liveness/readiness probes for the orchestrator process.

### Operational Modes
`modes.py` (316 lines) — Per-request behavioral modes (Strict, Creative, etc.). Strategy pattern.

### Subagent Delegation
`delegation/subagent.py` and `delegation/batch_runner.py` — Spawn subagents and execute batch tasks in parallel.

### Pattern Learner
`pattern_learner/` (5 files) — Closed learning loop for reusable code patterns:
- `extractor.py` — Identify recurring patterns from task results
- `curator.py` — Quality-filter and score patterns
- `pattern_store.py` — Persist patterns for reuse
- `injector.py` — Inject learned patterns into prompts

### Kanban Board
`kanban/` (3 files) — Persistent multi-project work queue with dispatcher.

### Gateway
`gateway/` (3 files) — Multi-platform messaging interface with session management.

### Meta-Orchestrator
`meta_orchestrator.py`, `meta_config.py`, `meta_monitoring.py`, `meta_performance.py`, `meta_integration.py`, `meta_v2_integration.py` — Self-optimizing orchestration layer.

---

## Project Structure (DDD Layered)

```
orchestrator/                                    # 497 Python files
│
├── DOMAIN LAYER ──────────────────────────────────────────────
│   ├── domain/                               # Canonical domain (5 files)
│   │   ├── exceptions.py                     # Exception hierarchy
│   │   ├── model_registry.py                 # Model registry metadata
│   │   ├── ports.py                          # Protocol interfaces (CachePort, StatePort, EventPort)
│   │   └── task_factory.py                   # Task creation factory
│   │
│   ├── models.py                             # Core data models, enums, 52 models, routing tables, cost tables
│   ├── exceptions.py                         # Backward-compat exception re-export
│   ├── ports.py                              # Backward-compat ports re-export
│   ├── constants.py                          # Project-wide constants
│   ├── task_schemas.py                       # Task schema definitions
│   └── task_factory.py                       # Backward-compat task factory
│
├── APPLICATION LAYER ─────────────────────────────────────────
│   ├── engine.py                             # Main orchestrator (5,036 lines)
│   ├── orchestration_facade.py               # Safe accessors for optional subsystems (112 lines)
│   │
│   ├── application/                          # Canonical application services (11 files)
│   │   ├── executor.py                       # ExecutorService: task execution + timing + spans
│   │   ├── evaluator.py                      # EvaluatorService: 2-pass self-consistency scoring
│   │   ├── decomposer.py                     # DecomposerService: project decomposition
│   │   ├── observability.py                  # ObservabilityService: per-model metrics
│   │   ├── budget_enforcer.py                # BudgetEnforcer: mid-task budget checks
│   │   ├── critique_cycle.py                 # CritiqueCycle: multi-pass critique state machine
│   │   ├── fallback_handler.py               # FallbackHandler: cross-provider retry
│   │   ├── task_executor.py                  # ExecutionContext + TaskExecutor
│   │   ├── dependency_resolver.py            # DependencyResolver: task ordering
│   │   └── context_compressor.py             # ContextCompressor: prompt trimming
│   │
│   ├── engine_core/                          # Backward-compat shim → application/ (22 files)
│   │   ├── pipeline.py                       # TaskPipeline + PipelineContext
│   │   ├── protocols.py                      # 7 Protocol interfaces (async)
│   │   ├── container.py                      # ServiceContainer.build() factory
│   │   ├── utilities.py                      # _clean_code_output, _get_available_models
│   │   ├── validator.py                      # TaskValidator
│   │   ├── decomposer.py                     # Decomposer (re-export)
│   │   ├── architect.py                      # ArchitectureRules
│   │   └── stages/                           # 8 pluggable pipeline stages
│   │       ├── generate.py                   # LLM code generation
│   │       ├── persuasion_defense.py         # Hallucination verification (NLI)
│   │       ├── critique.py                   # Cross-model review
│   │       ├── evaluate.py                   # Quality scoring
│   │       ├── validate.py                   # Deterministic validation
│   │       ├── preflight.py                  # Quality gate
│   │       ├── self_consistency.py           # ARA retry with fallback
│   │       └── map_elites.py                 # MAP-Elites evolutionary optimization
│   │
│   ├── services/                             # Backward-compat shim → application/ (5 files)
│   │   ├── executor.py                       # → application/executor.py
│   │   ├── evaluator.py                      # → application/evaluator.py
│   │   ├── generator.py                      # → application/decomposer.py (alias: GeneratorService)
│   │   └── observability.py                  # → application/observability.py
│   │
│   ├── agents/                               # 9 specialized agents (12 files)
│   │   ├── base.py                           # AgentBase ABC, AgentRole, AgentTask
│   │   ├── coordinator.py                    # AgentOrchestrator (parallel DAG dispatch)
│   │   ├── developer.py                      # DeveloperAgent, ArchitectAgent, TesterAgent
│   │   ├── reviewer.py                       # ReviewerAgent
│   │   ├── devops.py                         # DevOpsAgent
│   │   ├── researcher.py                     # ResearcherAgent
│   │   ├── user.py                           # UserAgent (talks to user)
│   │   ├── product_manager.py                # ProductManagerAgent
│   │   ├── qc.py                             # QCAgent (CI pipeline)
│   │   ├── metrics.py                        # AgentMetrics, MetricsRegistry
│   │   └── rate_limiter.py                   # Sliding window per-agent limits
│   │
│   ├── delegation/                           # Subagent system (3 files)
│   │   ├── subagent.py                       # Subagent spawning
│   │   └── batch_runner.py                   # Batch parallel execution
│   │
│   ├── planning/                             # Goal decomposition (3 files)
│   │   ├── goal.py                           # Goal, SubGoal, Plan datatypes
│   │   └── decomposer.py                     # GoalDecomposer (HTN recursive)
│   │
│   ├── workspace/                            # Shared blackboard (5 files)
│   │   ├── workspace.py                      # ProjectWorkspace (file versions, decisions)
│   │   ├── message_bus.py                    # AgentMessageBus (publish/subscribe)
│   │   ├── persistent_workspace.py           # SQLite-backed workspace
│   │   └── audit.py                          # Append-only audit trail
│   │
│   ├── learning/                             # Memory & adaptation (6 files)
│   │   ├── agent_memory.py                   # Per-agent private memory
│   │   ├── agent_cache.py                    # SHA-256 hash + TTL caching
│   │   ├── experience_buffer.py              # Cross-task success/failure patterns
│   │   ├── knowledge_graph.py                # Relational pattern learning (networkx)
│   │   ├── memory_compressor.py              # Summarize patterns into lessons
│   │   └── prompt_enricher.py                # Unified memory query → prompt injection
│   │
│   ├── memory/                               # Memory consolidation (2 files)
│   │   ├── memory_manager.py                 # Multi-tier memory orchestration
│   │   └── consolidation.py                  # Pattern consolidation logic
│   │
│   ├── pattern_learner/                      # Closed learning loop (5 files)
│   │   ├── extractor.py                      # Pattern extraction from results
│   │   ├── curator.py                        # Quality filtering + scoring
│   │   ├── pattern_store.py                  # Persistence layer
│   │   └── injector.py                       # Prompt injection of learned patterns
│   │
│   ├── tools/                                # Agent tool layer (3 files)
│   │   ├── base.py                           # Tool ABC, ToolResult, ToolRegistry
│   │   └── shell_tool.py                     # ShellTool (subprocess)
│   │
│   ├── runtime/                              # Code execution (1 file)
│   │   └── sandbox.py                        # SandboxExecutor, TestRunner
│   │
│   ├── security/                             # OWASP security enforcement (2 files)
│   │   └── enhancer.py                       # SecurityEnhancer, OpenGraphGenerator
│   │
│   ├── ux/                                   # UX quality standards (1 file)
│   │   └── design_enhancer.py                # UXDesignReviewer, 20 WCAG standards
│   │
│   ├── ci/                                   # Continuous integration (1 file)
│   │   └── pipeline.py                       # CIPipeline, CIStep, LintStep
│   │
│   ├── hitl/                                 # Human-in-the-loop (1 file)
│   │   └── gate.py                           # DecisionGate, DecisionResult
│   │
│   ├── project/                              # Project management (3 files)
│   │   ├── sprint_planner.py                 # Sprint, Milestone, SprintPlanner
│   │   └── progress_reporter.py              # Formatted progress tables
│   │
│   ├── product/                              # Product management (2 files)
│   │   └── backlog.py                        # ProductBacklog, UserStory
│   │
│   ├── knowledge/                            # Knowledge management (2 files)
│   │   ├── knowledge_base.py                 # KnowledgeBase, KnowledgeEntry
│   │   └── docs_generator.py                 # ARCHITECTURE.md, DECISIONS.md
│   │
│   ├── quality/                              # Quality control (3 files)
│   │   ├── quality_report.py                 # Score 0-10, PASS/REVISE/BLOCK
│   │   └── regression.py                     # RegressionDetector
│   │
│   ├── kanban/                               # Multi-project work queue (3 files)
│   │   ├── board.py                          # Kanban board logic
│   │   └── dispatcher.py                     # Task dispatch engine
│   │
│   ├── gateway/                              # Multi-platform messaging (3 files)
│   │   ├── run.py                            # Gateway entry point
│   │   └── session.py                        # Session management
│   │
│   ├── scaffold/                             # Project templates (10 files)
│   │   ├── dynamic.py                        # DynamicScaffoldGenerator
│   │   └── templates/
│   │       ├── cli.py                        # CLI project template
│   │       ├── fastapi.py                    # FastAPI backend template
│   │       ├── nextjs.py                     # Next.js fullstack template
│   │       ├── react_vite.py                 # React+Vite SPA template
│   │       ├── html.py                       # Static HTML template
│   │       ├── library.py                    # Python library template
│   │       └── generic.py                    # Generic project template
│   │
│   ├── unified_events/                       # Event bus (2 files)
│   │   └── core.py                           # Decoupled event system
│   │
│   ├── dashboard_core/                       # Dashboard components (3 files)
│   │   ├── core.py                           # Dashboard logic
│   │   └── mission_control.py                # Mission control panel
│   │
│   ├── crosscutting/                         # Cross-cutting config (2 files)
│   │   └── config.py                         # Shared configuration
│   │
│   ├── plugins/                              # Plugin system (7 files)
│   │   ├── base.py                           # Plugin base class
│   │   ├── cost_optimization.py              # Cost optimization plugin
│   │   ├── nash_stability.py                 # Nash stability plugin
│   │   ├── context_provider.py               # Context provider plugin
│   │   ├── memory_provider.py                # Memory provider plugin
│   │   └── discovery.py                      # Plugin discovery
│   │
│   ├── ara_pipelines.py                      # 20+ ARA reasoning methods
│   ├── ara_execution_strategy.py             # ARA method selection + dispatch
│   ├── ara_integration.py                    # ARA integration layer
│   │
│   ├── codebase_reader.py                    # FileSystemWalker, ASTIndexer, DependencyGraph
│   ├── codebase_context.py                   # RelevanceRanker, QualityAnalyzer
│   ├── codebase_decomposer.py                # LLM-powered modification planner
│   ├── codebase_writer.py                    # Safe file ops + diff engine + safety gates
│   ├── codebase_analyzer.py                  # Multi-pass codebase analysis
│   ├── codebase_profile.py                   # Codebase profiling
│   ├── codebase_understanding.py             # Semantic understanding layer
│   │
│   ├── command_center.py                     # Interactive REPL (141 lines)
│   ├── command_center_integration.py         # REPL integration layer
│   ├── command_center_server.py              # REPL server backend
│   ├── command_registry.py                   # Command registration
│   │
│   ├── git_integration.py                    # Branch→commit→push→PR per milestone
│   ├── git_service.py                        # Git service abstraction
│   ├── git_sync.py                           # Git synchronization
│   ├── git_hooks.py                          # Git hook management
│   │
│   ├── control_plane.py                      # Constraint enforcement (285 lines)
│   ├── a2a_protocol.py                       # Agent-to-Agent protocol (833 lines)
│   ├── sagas.py                              # Distributed transactions (850 lines)
│   ├── escalation.py                         # Model escalation (270 lines)
│   ├── health.py                             # Health checks (468 lines)
│   ├── modes.py                              # Operational modes (316 lines)
│   │
│   ├── adaptive_router.py                    # Circuit breaker v2 (175 lines)
│   ├── outcome_router.py                     # Outcome-weighted routing (580 lines)
│   ├── aggregator.py                         # Cross-run profile aggregator (76 lines)
│   ├── model_selector.py                     # Intelligent model routing
│   ├── model_registry.py                     # Model registry and metadata
│   ├── model_routing.py                      # Model routing logic
│   │
│   ├── meta_orchestrator.py                  # Self-optimizing orchestration
│   ├── meta_config.py                        # Meta-configuration
│   ├── meta_monitoring.py                    # Meta-level monitoring
│   ├── meta_performance.py                   # Meta performance tracking
│   ├── meta_integration.py                   # Meta integration
│   ├── meta_v2_integration.py                # Meta v2 integration
│   │
│   ├── nash_stable_orchestrator.py           # Nash equilibrium orchestration
│   ├── nash_auto_tuning.py                   # Auto-tuning for Nash stability
│   ├── nash_events.py                        # Nash event system
│   ├── nash_infrastructure_v2.py             # Nash infrastructure v2
│   ├── nash_monitor.py                       # Nash stability monitor
│   │
│   ├── circuit_breaker.py                    # Circuit breaker pattern
│   ├── resilience.py                         # General resilience utilities
│   ├── retry_utils.py                        # Retry helper utilities
│   ├── error_handling.py                     # Error handling framework
│   ├── events_resilient.py                   # Resilient event handling
│   │
│   ├── sandbox.py                            # Sandbox execution
│   ├── sandbox_executor.py                   # Sandbox executor
│   ├── secure_execution.py                   # Secure execution wrapper
│   ├── policy_engine.py                      # Policy enforcement (HARD/SOFT/MONITOR)
│   ├── policy_dsl.py                         # Policy DSL
│   ├── policy.py                             # Policy definitions
│   │
│   ├── cost_optimization/                    # Cost optimization (12 files)
│   │   ├── batch_client.py                   # Batch LLM requests
│   │   ├── model_cascading.py                # Model cascade strategy
│   │   ├── prompt_cache.py                   # Prompt caching
│   │   ├── speculative_gen.py                # Speculative generation
│   │   ├── structured_output.py              # Structured output optimization
│   │   ├── streaming_validator.py            # Streaming validation
│   │   ├── token_budget.py                   # Token budget management
│   │   ├── dependency_context.py             # Dependency-aware context
│   │   ├── docker_sandbox.py                 # Docker-based sandbox
│   │   ├── github_push.py                    # GitHub push integration
│   │   └── tier3_quality.py                  # Tier 3 quality gates
│   │
│   ├── nexus_search/                         # Web search integration (21 files)
│   │   ├── core.py                           # Search orchestration
│   │   ├── models.py                         # Search models
│   │   ├── nexus_client.py                   # Nexus search client
│   │   ├── server_manager.py                 # Search server management
│   │   ├── config.py                         # Search configuration
│   │   ├── agents/
│   │   │   ├── classifier.py                 # Query classifier agent
│   │   │   └── researcher.py                 # Research agent
│   │   ├── providers/
│   │   │   ├── base.py                       # Search provider base
│   │   │   └── nexus.py                      # Nexus provider implementation
│   │   └── optimization/
│   │       ├── adaptive_depth.py             # Adaptive search depth
│   │       ├── circuit_breaker.py            # Search circuit breaker
│   │       ├── deduplication.py              # Result deduplication
│   │       ├── llm_classifier.py             # LLM-based classification
│   │       ├── parallel_search.py            # Parallel search execution
│   │       ├── query_cache.py                # Query cache
│   │       ├── query_expansion.py            # Query expansion
│   │       └── reranker.py                   # Result reranking
│   │
│   ├── ide_backend/                          # IDE integration (16 files)
│   │   ├── server.py                         # Main server
│   │   ├── ide_orchestrator_server.py        # Orchestrator bridge server
│   │   ├── websocket_manager.py              # WebSocket connection management
│   │   ├── session_manager.py                # Session state management
│   │   ├── launch.py                         # Launcher
│   │   ├── log_config.py                     # Log configuration
│   │   ├── api/
│   │   │   └── routes.py                     # API route definitions
│   │   ├── integration/
│   │   │   └── orchestrator_bridge.py        # Orchestrator integration bridge
│   │   └── websocket/
│   │       └── handlers.py                   # WebSocket event handlers
│   │
├── INFRASTRUCTURE LAYER ──────────────────────────────────────
│   ├── infrastructure/                       # Concrete adapters (3 files)
│   │   ├── cache.py                          # Disk-based caching
│   │   ├── llm_client.py                     # LLM client adapter
│   │   └── state.py                          # SQLite state persistence
│   │
│   ├── api_clients.py                        # Unified LLM client via OpenRouter
│   ├── state.py                              # SQLite state manager
│   ├── budget.py                             # Budget tracking + atomic reserve
│   ├── cache.py                              # Disk cache implementation
│   ├── semantic_cache.py                     # Semantic similarity caching
│   ├── secure_cache.py                       # Encrypted cache
│   ├── cache_optimizer.py                    # Cache optimization
│   ├── pricing_cache.py                      # Pricing data cache
│   ├── async_event_store.py                  # Async event persistence
│   ├── async_file_io.py                      # Async file I/O
│   ├── telemetry.py                          # Telemetry collection
│   ├── telemetry_store.py                    # Telemetry persistence
│   ├── tracing.py                            # OpenTelemetry tracing
│   ├── monitoring.py                         # Runtime monitoring
│   ├── streaming.py                          # Streaming LLM responses
│   ├── streaming_optimizer.py                # Streaming optimization
│   ├── streaming_resilient.py                # Resilient streaming
│   ├── provisioned_throughput.py             # Throughput provisioning
│   └── entity_rls.py                         # Entity-level row security
│
├── ENTRY POINTS ──────────────────────────────────────────────
│   ├── __init__.py                           # Lazy-loading entry point (v6.0.0)
│   ├── __main__.py                           # Module execution entry
│   ├── cli.py                                # Main CLI
│   ├── cli_dashboard.py                      # Dashboard CLI
│   ├── cli_nash.py                           # Nash stability CLI
│   ├── cli_website.py                        # Website generator CLI
│   ├── nexus_cli.py                          # Nexus search CLI
│   └── config.py                             # Centralized RuntimeConfig (188 lines)
│
├── GENERATORS & BUILDERS ─────────────────────────────────────
│   ├── fullstack_generator.py                # Full-stack app generator
│   ├── api_builder.py                        # API builder
│   ├── app_builder.py                        # App builder
│   ├── app_assembler.py                      # App assembler
│   ├── app_detector.py                       # App type detector
│   ├── app_verifier.py                       # App verifier
│   ├── database_generator.py                 # Database schema generator
│   ├── docker_generator.py                   # Docker config generator
│   ├── cicd_generator.py                     # CI/CD pipeline generator
│   ├── doc_generator.py                      # Documentation generator
│   ├── website_generator.py                  # Website generator
│   ├── website_validator.py                  # Website validator
│   ├── type_generator.py                     # Type definition generator
│   ├── component_library.py                  # Component library
│   ├── component_registry.py                 # Component registry
│   ├── design_system.py                      # Design system
│   ├── design_registry.py                    # Design registry
│   ├── design_to_code.py                     # Design-to-code converter
│   ├── diff_generator.py                     # Diff generation
│   ├── diff_view.py                          # Diff visualization
│   ├── multi_platform_generator.py           # Multi-platform generator
│   ├── copy_generator.py                     # Marketing copy generator
│   ├── image_generator.py                    # Image generation
│   ├── image_optimizer.py                    # Image optimization
│   ├── logging_generator.py                  # Logging setup generator
│   ├── opengraph_generator.py                # OpenGraph meta tag generator
│   ├── output_organizer.py                   # Output file organization
│   ├── output_writer.py                      # Output file writing
│   ├── output_writer_trimmed.py              # Trimmed output writing
│   ├── prompt_builder.py                     # Prompt construction
│   ├── prompt_compressor.py                  # Prompt compression
│   ├── prompt_enhancer.py                    # Prompt enhancement
│   ├── project_assembler.py                  # Project assembly
│   ├── project_copier.py                     # Project copying
│   ├── query_generator.py                    # Query generation
│   ├── query_expander.py                     # Query expansion
│   ├── release_manager.py                    # Release management
│   ├── version_manager.py                    # Version management
│   ├── secrets_generator.py                  # Secret generation
│   ├── secrets_manager.py                    # Secret management
│   ├── testing_templates.py                  # Test template generation
│   ├── test_first_generator.py               # TDD-first generator
│   └── team_templates.py                     # Team configuration templates
│
├── QUALITY & SECURITY ────────────────────────────────────────
│   ├── validators.py                         # Deterministic validation (syntax, tests)
│   ├── test_validator.py                     # Test validation
│   ├── code_validator.py                     # Code validation
│   ├── security_validator.py                 # Security validation
│   ├── preflight.py                          # Preflight checks
│   ├── assumption_gate.py                    # Assumption verification
│   ├── benchmark_suite.py                    # Performance benchmarks
│   ├── browser_testing.py                    # Browser testing
│   ├── security_review.py                    # Security review
│   ├── security_templates.py                 # Security templates
│   ├── frontend_rules.py                     # Frontend rules
│   ├── frontend_security.py                  # Frontend security
│   ├── red_team.py                           # Red team adversarial testing
│   ├── guardrails.py                         # Output guardrails
│   ├── tool_guardrails.py                    # Tool safety guardrails
│   ├── reference_monitor.py                  # Reference monitor
│   ├── input_validation.py                   # Input validation
│   ├── agent_safety.py                       # Agent safety controls
│   ├── autonomous_debugger.py                # Autonomous debugging
│   ├── auto_error_fix.py                     # Automatic error correction
│   ├── screenshot_diagnoser.py               # Screenshot diagnostics
│   ├── restoration_points.py                 # Snapshot restoration
│   └── dependency_scanner.py                 # Dependency vulnerability scanning
│
├── OPERATIONS ─────────────────────────────────────────────────
│   ├── deployment_service.py                 # Deployment orchestration
│   ├── deployment_feedback.py                # Deployment feedback loop
│   ├── feedback.py                           # Feedback collection
│   ├── feedback_loop.py                      # Feedback processing loop
│   ├── drift.py                              # Configuration drift detection
│   ├── diagnostics.py                        # System diagnostics
│   ├── system_diagnostics.py                 # Deep diagnostics
│   ├── issue_tracking.py                     # Issue tracking integration
│   ├── checkpoints.py                        # Execution checkpoints
│   ├── restore_points.py                     # Restore point management
│   ├── resume_detector.py                    # Auto-resume detection
│   ├── session_lifecycle.py                  # Session lifecycle management
│   ├── session_watcher.py                    # Session monitoring
│   ├── concurrency_controller.py             # Parallelism limits
│   ├── export_manager.py                     # Data export
│   ├── i18n.py                               # Internationalization
│   ├── automations.py                        # Workflow automation
│   ├── triggers.py                           # Event triggers
│   ├── hooks.py                              # Lifecycle hooks
│   ├── breakpoints.py                        # Execution breakpoints
│   ├── quick_actions.py                      # Quick action shortcuts
│   └── quick_self_test.py                    # Rapid self-test
│
├── INTEGRATIONS ───────────────────────────────────────────────
│   ├── api_server.py                         # API server
│   ├── mcp_server.py                         # MCP server
│   ├── slack_integration.py                  # Slack integration
│   ├── github_sync.py                        # GitHub sync
│   ├── openrouter_sync.py                    # OpenRouter sync
│   ├── openrouter_ab_testing.py              # OpenRouter A/B testing
│   ├── swiftstack_integration.py             # SwiftStack integration
│   ├── config_sync.py                        # Configuration sync
│   ├── multi_tenant_gateway.py               # Multi-tenant gateway
│   ├── tenancy.py                            # Tenancy management
│   └── project_file.py                       # Project file handling
│
├── ANALYTICS & VISUALIZATION ─────────────────────────────────
│   ├── dashboard.py                          # Dashboard
│   ├── dashboard_bridge.py                   # Dashboard bridge
│   ├── visualization.py                      # Data visualization
│   ├── cost.py                               # Cost calculation
│   ├── cost_tracker.py                       # Cost tracking
│   ├── cost_analytics.py                     # Cost analytics
│   ├── cost_optimization_integration.py      # Cost optimization integration
│   ├── leaderboard.py                        # Model leaderboard
│   ├── project_analyzer.py                   # Project analysis
│   ├── project_context.py                    # Project context
│   ├── performance.py                        # Performance tracking
│   ├── progressive_output.py                 # Progressive output rendering
│   ├── progress.py                           # Progress tracking
│   ├── progress_collector.py                 # Progress data collection
│   ├── progress_writer.py                    # Progress output
│   ├── metrics.py                            # Metrics collection
│   ├── projections.py                        # Cost/performance projections
│   └── pareto_frontier.py                    # Pareto frontier analysis
│
├── SCAFFOLD & CONFIG ─────────────────────────────────────────
│   ├── brain.py                              # Brain/central coordinator
│   ├── brainstorming.py                      # Brainstorming engine
│   ├── competitive.py                        # Competitive analysis
│   ├── config_as_code.py                     # Config-as-code
│   ├── connectors.py                         # External connectors
│   ├── data_sources.py                       # Data source abstractions
│   ├── dep_resolver.py                       # Dependency resolver
│   ├── dev_server.py                         # Development server
│   ├── dry_run.py                            # Dry run mode
│   ├── effort_estimation.py                  # Effort estimation (not yet tracked)
│   ├── file_scope.py                         # File scoping
│   ├── hierarchy.py                          # Agent hierarchy
│   ├── hitl_workflow.py                      # HITL workflow
│   ├── hybrid_search_pipeline.py             # Hybrid search
│   ├── improvement_suggester.py              # Improvement suggestions
│   ├── knowledge_base.py                     # Knowledge base (top-level)
│   ├── knowledge_graph.py                    # Knowledge graph (top-level)
│   ├── knowledge_sidebar.py                  # Knowledge sidebar UI
│   ├── learning_aggregator.py                # Learning aggregation
│   ├── log_config.py                         # Log configuration
│   ├── memory_bank.py                        # Memory bank
│   ├── memory_tier.py                        # Memory tiering
│   ├── method_selector.py                    # Method selection
│   ├── module_system.py                      # Module system
│   ├── multi_context.py                      # Multi-context management
│   ├── native_features.py                    # Native feature detection
│   ├── optimization.py                       # General optimization
│   ├── orchestration_agent.py                # Orchestration agent
│   ├── persona.py                            # Persona system
│   ├── persona_modes.py                      # Persona modes
│   ├── phase_aware_models.py                 # Phase-aware model selection
│   ├── pipeline_runner.py                    # Pipeline runner
│   ├── plan_reviewer.py                      # Plan review
│   ├── plan_review_data.py                   # Plan review data
│   ├── plan_then_build.py                    # Plan-then-build pattern
│   ├── planner.py                            # Project planner
│   ├── plugin_isolation.py                   # Plugin isolation
│   ├── plugin_isolation_secure.py            # Secure plugin isolation
│   ├── pre_submission_testing.py             # Pre-submission testing
│   ├── preview_server.py                     # Preview server
│   ├── product_manager.py                    # Product manager (top-level)
│   ├── project_manager.py                    # Project manager
│   ├── remediation.py                        # Remediation engine
│   ├── responsive_layouts.py                 # Responsive layout rules
│   ├── route_integration.py                  # Route integration
│   ├── run_tests.py                          # Test runner
│   ├── self_review.py                        # Self-review
│   ├── service_collection.py                 # Service collection
│   ├── site_manager.py                       # Site manager
│   ├── skills.py                             # Skills system
│   ├── skills_exporter.py                    # Skills export
│   ├── slash_commands.py                     # Slash command system
│   ├── slash_integrations.py                 # Slash command integrations
│   ├── specs.py                              # Specification system
│   ├── structured_outputs.py                 # Structured output formatting
│   ├── task_handlers.py                      # Task handler registry
│   ├── task_verifier.py                      # Task verification
│   ├── tdd_config.py                         # TDD configuration
│   ├── test_fixer.py                         # Test auto-fixer
│   ├── toml_validator.py                     # TOML validation
│   ├── token_budget.py                       # Token budget (top-level)
│   ├── token_optimizer.py                    # Token optimization
│   ├── transfer_learning.py                  # Transfer learning
│   ├── verification.py                       # Verification system
│   ├── wordpress_plugin_rules.py             # WordPress plugin rules
│   └── xai_search.py                         # Explainable AI search
│
└── __pycache__/                              # Python cache (gitignored)
```

### Test Suite (42 files)

```
tests/
├── conftest.py                               # Global fixtures
│
├── integration/                              # Integration tests (5 files)
│   ├── conftest.py                           # Shared fixtures (temp StateManager, mock tasks)
│   ├── test_full_run.py                      # End-to-end pipeline
│   ├── test_resume_after_crash.py            # Crash recovery
│   └── test_circuit_breaker_fail_fast.py     # Circuit breaker behavior
│
├── smoke/                                    # Smoke tests (3 files)
│   ├── test_cli.py                           # CLI contract tests
│   └── test_api_contracts.py                 # API contract tests
│
├── test_god_file_refactoring.py              # Engine core extraction (40 tests)
├── test_e2e_full_suite.py                    # End-to-end lifecycle (38 tests)
├── test_phase6_10_comprehensive.py           # Engine core, ARA, protocols (41 tests)
├── test_agentic_system.py                    # Agents, tools, workspace (22 tests)
├── test_optimizations.py                     # Rate limiter, cache, knowledge graph (22 tests)
├── test_capabilities_5_10.py                 # Message bus, learning, CI, HITL (21 tests)
├── test_pipeline.py                          # TaskPipeline + stages (14 tests)
├── test_validator.py                         # Validation pipeline (10 tests)
├── test_planning.py                          # Goal decomposition (9 tests)
├── test_decomposer.py                        # Decomposition (8 tests)
├── test_circuit_breaker.py                   # Circuit breaker unit tests
├── test_concurrency_controller.py            # Concurrency control
├── test_resilience.py                        # Resilience patterns
├── test_budget.py                            # Budget tracking
├── test_models.py                            # Model definitions
├── test_cost_tracker.py                      # Cost tracking
├── test_checkpoints.py                       # Checkpoint system
├── test_context_system.py                    # Context management
├── test_config_as_code.py                    # Config-as-code
├── test_state_validation.py                  # State validation
├── test_security_review.py                   # Security review
├── test_service_observability.py             # Service observability
├── test_evaluator_service.py                 # Evaluator service
├── test_executor_service.py                  # Executor service
├── test_generator_service.py                 # Generator/Decomposer service
├── test_integration.py                       # General integration
├── test_bug_fixes_v2.py                      # Bug regression
├── test_bug_regression.py                    # Bug regression
├── test_new_modules.py                       # New module validation
├── test_phase6_resilience.py                 # Phase 6 resilience
├── test_phase7_ports.py                      # Phase 7 port interfaces
├── test_phase8_mvos.py                       # Phase 8 MVOS audit
├── test_autonomy_config.py                   # Autonomy configuration
└── test_instructor_tenacity.py               # Instructor tenacity tests
```

---

## Key Metrics & Statistics

| Metric | Value |
|--------|-------|
| **Version** | v6.0.0 (2026-05-27) |
| **Core Code** | ~150,000 lines (orchestrator/) |
| **engine.py** | 5,036 lines |
| **Test Code** | 42 files (tests/) |
| **Total Files** | 497 Python files (orchestrator/) |
| **Agent Roles** | 9 (architect, developer, reviewer, tester, devops, researcher, user, pm, qa) |
| **ARA Methods** | 20+ (including CoVE, SoT, ToT, MAP-Elites) |
| **Pipeline Stages** | 7 core + MAP-Elites |
| **Models** | 52 from 15 providers |
| **Free Models** | 5 (owl, deepseek, nemotron, poolside m, poolside xs) |
| **Subpackages** | 33 (up from ~15 in previous version) |
| **Layers** | 3 (domain, application, infrastructure) per DDD |
| **New in v6.0** | A2A Protocol, SaGas, Control Plane, Outcome Router, Pattern Learner, Kanban, Gateway, Meta-Orchestrator, Subagent Delegation, Adaptive Router, Escalation, Health Checks, Operational Modes |

---

*Last Updated: 2026-05-27*
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*
*License: MIT*
