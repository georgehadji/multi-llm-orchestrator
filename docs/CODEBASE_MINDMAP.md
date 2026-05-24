# Multi-LLM Orchestrator — Complete Architecture Mindmap

> **Version:** v6.0.0 (2026-05-25)  
> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Codebase Size:** ~147,000 lines (orchestrator/) + ~3,200 lines (tests)  
> **Total Files:** ~351 Python files (orchestrator/), ~25 test files

---

## System Overview

```
+-----------------------------------------------------------------------------+
|                  Multi-LLM Orchestrator v6.0                                |
|           Autonomous Multi-Agent Software Development Platform              |
+-----------------------------------------------------------------------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
    +----v----+               +----v----+               +----v----+
    |  INPUT   |               | ENGINE  |               | OUTPUT  |
    | Layer    |<------------->|  Core   |<------------->| Layer   |
    +----|----+               +----|----+               +----|----+
         |                          |                          |
    Natural Language           Agent Coordination         Generated Code
    Project Specs              ARA Reasoning              Test Suite
    Codebase Paths             Pipeline Stages            Documentation
```

---

## Current Architecture (May 2026)

```
User says: "Build a todo app"
       │
       ▼
CommandCenter ──→ AgentOrchestrator ──→ GoalDecomposer (HTN recursive)
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

**Hallucination Defense (8 layers):**
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
                                     | 
Web:                                   Layout:
  CSP, HSTS, X-Frame-Options           Responsive, mobile-first
  X-Content-Type-Options               Spacing scale (4px/8px)
  Referrer-Policy                      Visual hierarchy
                                     |
API:                                   Accessibility:
  SQL injection prevention              WCAG AA (4.5:1 contrast)
  Rate limiting (100/min)               Keyboard navigation
  Strict CORS (whitelist origins)       ARIA labels + landmarks
  HTTPS enforcement                     Touch targets (44×44px)
                                     |
Auth:                                  Typography:
  bcrypt (cost 12+)                     Font pairing (2 max)
  HttpOnly + Secure + SameSite          Line length (60-80 chars)
  JWT RS256, short expiry (15m)         Type scale (1.25 ratio)
  CSRF tokens for state changes
                                     |  
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

---

## Project Structure

```
orchestrator/                         # Main package (~351 files)
├── agents/                           # 9 specialized agents
│   ├── base.py                       # AgentBase ABC, AgentRole, AgentTask
│   ├── coordinator.py                # AgentOrchestrator (parallel DAG dispatch)
│   ├── developer.py                  # DeveloperAgent, ArchitectAgent, TesterAgent
│   ├── reviewer.py                   # ReviewerAgent
│   ├── devops.py                     # DevOpsAgent
│   ├── researcher.py                 # ResearcherAgent
│   ├── user.py                       # UserAgent (talks to user)
│   ├── product_manager.py            # ProductManagerAgent
│   ├── qc.py                         # QCAgent (CI pipeline)
│   ├── metrics.py                    # AgentMetrics, MetricsRegistry
│   └── rate_limiter.py               # Sliding window per-agent limits
│
├── workspace/                        # Shared blackboard
│   ├── workspace.py                  # ProjectWorkspace (file versions, decisions)
│   ├── message_bus.py                # AgentMessageBus (publish/subscribe)
│   ├── persistent_workspace.py       # SQLite-backed workspace
│   └── audit.py                      # Append-only audit trail
│
├── planning/                         # Goal decomposition
│   ├── goal.py                       # Goal, SubGoal, Plan datatypes
│   └── decomposer.py                 # GoalDecomposer (HTN recursive)
│
├── tools/                            # Agent tool layer
│   ├── base.py                       # Tool ABC, ToolResult, ToolRegistry
│   └── shell_tool.py                 # ShellTool (subprocess)
│
├── runtime/                          # Code execution
│   └── sandbox.py                    # SandboxExecutor, TestRunner
│
├── learning/                         # Memory & adaptation
│   ├── agent_memory.py               # Per-agent private memory
│   ├── agent_cache.py                # SHA-256 hash + TTL caching
│   ├── experience_buffer.py           # Cross-task success/failure patterns
│   ├── knowledge_graph.py            # Relational pattern learning (networkx)
│   ├── memory_compressor.py          # Summarize many patterns into lessons
│   └── prompt_enricher.py            # Unified memory query → prompt injection
│
├── engine_core/                      # Core pipeline
│   ├── pipeline.py                   # TaskPipeline + PipelineContext
│   ├── protocols.py                  # 7 Protocol interfaces (ModelProvider, etc.)
│   ├── container.py                  # ServiceContainer.build() factory
│   ├── utilities.py                  # _clean_code_output, _get_available_models
│   ├── stages/                       # 7 pluggable pipeline stages
│   │   ├── generate.py, critique.py, evaluate.py
│   │   ├── validate.py, preflight.py
│   │   ├── self_consistency.py       # Enhanced with ARA retry
│   │   └── persuasion_defense.py     # Hallucination verification
│   ├── validator.py                  # TaskValidator
│   ├── decomposer.py                 # Decomposer
│   └── architect.py                  # ArchitectureRules
│
├── security/                         # OWASP security enforcement
│   └── enhancer.py                   # SecurityEnhancer, OpenGraphGenerator
│
├── ux/                               # UX quality standards
│   └── design_enhancer.py            # UXDesignReviewer, 20 standards
│
├── ci/                               # Continuous integration
│   └── pipeline.py                   # CIPipeline, CIStep, LintStep
│
├── hitl/                             # Human-in-the-loop
│   └── gate.py                       # DecisionGate, DecisionResult
│
├── project/                          # Project management
│   ├── sprint_planner.py             # Sprint, Milestone, SprintPlanner
│   └── progress_reporter.py          # Formatted progress tables
│
├── product/                          # Product management
│   └── backlog.py                    # ProductBacklog, UserStory
│
├── knowledge/                        # Knowledge management
│   ├── knowledge_base.py             # KnowledgeBase, KnowledgeEntry
│   └── docs_generator.py             # ARCHITECTURE.md, DECISIONS.md
│
├── quality/                          # Quality control
│   ├── quality_report.py             # Score 0-10, PASS/REVISE/BLOCK
│   └── regression.py                 # RegressionDetector
│
├── scaffold/                         # Project templates
│   └── dynamic.py                    # DynamicScaffoldGenerator
│
├── ara_pipelines.py                  # 20+ ARA reasoning methods
├── ara_execution_strategy.py         # ARA method selection + dispatch
├── codebase_reader.py                # FileSystemWalker, ASTIndexer, DependencyGraph
├── codebase_context.py               # RelevanceRanker, QualityAnalyzer
├── codebase_decomposer.py            # LLM-powered modification planner
├── codebase_writer.py                # Safe file ops + diff engine + safety gates
├── git_integration.py                # Branch→commit→push→PR per milestone
├── command_center.py                 # Interactive REPL
├── config.py                         # Centralized RuntimeConfig
├── engine.py                         # Main Orchestrator (~2,000 lines)
├── models.py                         # 52 models, routing, cost tables
├── engine_deps.py                    # All optional/try-except imports
└── __init__.py                       # Lazy-loading entry point (v6.0.0)

tests/
├── test_god_file_refactoring.py      # 40 tests: engine_core extraction
├── test_decomposer.py                # 8 tests: decomposition
├── test_validator.py                 # 10 tests: validation pipeline
├── test_pipeline.py                  # 14 tests: TaskPipeline + stages
├── test_agentic_system.py            # 22 tests: agents, tools, workspace
├── test_capabilities_5_10.py         # 21 tests: message bus, learning, CI, HITL
├── test_planning.py                  # 9 tests: goal decomposition
├── test_optimizations.py             # 22 tests: rate limiter, cache, knowledge graph
├── test_phase6_10_comprehensive.py   # 41 tests: engin_core, ARA, protocols
├── test_e2e_full_suite.py            # 38 tests: end-to-end lifecycle
```

---

## Key Metrics & Statistics

| Metric | Value |
|--------|-------|
| **Version** | v6.0.0 (2026-05-25) |
| **Core Code** | ~147,000 lines (orchestrator/) |
| **Test Code** | ~3,200 lines (tests/) |
| **Total Files** | ~351 Python files (orchestrator/) |
| **Test Count** | ~225 tests (72 core + 154 new) |
| **Agent Roles** | 9 (architect, developer, reviewer, tester, devops, researcher, user, pm, qa) |
| **ARA Methods** | 20+ (including CoVE, SoT, ToT, MAP-Elites) |
| **Pipeline Stages** | 7 (generate, persuasion, critique, evaluate, validate, preflight, consistency) |
| **Models** | 52 from 15 providers |
| **Free Models** | 5 (owl, deepseek, nemotron, poolside m, poolside xs) |
| **Providers** | 15 (openai, anthropic, google, deepseek, meta, xai, moonshot, minimax, zhipu, baidu, xiaomi, inclusionai, perplexity, nvidia, ibm, poolside) |

---

*Last Updated: 2026-05-25*  
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
