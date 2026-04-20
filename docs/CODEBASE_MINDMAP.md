# Multi-LLM Orchestrator — Complete Architecture Mindmap

> **Version:** v6.0.0 (2026-04-20)  
> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Codebase Size:** ~142,000 lines (orchestrator/) + ~1,700 lines (tests)  
> **Total Files:** ~350 Python files (orchestrator/), ~30 test files

---

## System Overview

```
+-----------------------------------------------------------------------------+
|                         Multi-LLM Orchestrator v6.0                          |
|                    Autonomous Software Development Platform                    |
+-----------------------------------------------------------------------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
    +----v----+               +----v----+               +----v----+
    |  INPUT   |               | ENGINE  |               | OUTPUT  |
    | Layer    |<------------->|  Core   |<------------->| Layer   |
    +----|----+               +----|----+               +----|----+
         |                          |                          |
    Project Desc              Pipeline Execution          Generated Code
    Success Criteria          Quality Assurance           Test Suite
    Budget Constraints        Cost Optimization           Documentation
```

---

## Architectural Audit V5 (Phases 0–9, 2026-04-20)

**Executive Summary:** Complete architectural refactor using Strategy B
(Strategic). Delivered all 7 MVOS invariants verified at runtime (141 tests,
100% passing).

```
Phase 0: Setup           mypy strict-mode, dead import cleanup (122 fixes)
Phase 1: Circuit Breaker AsyncCircuitBreaker + registry per-model isolation
Phase 2: State Safety    Pydantic validation on StateManager.load_project()
Phase 3: Service Extr.   ExecutorService, EvaluatorService, GeneratorService
Phase 4: Dead Code       ruff F401 auto-fix; 31 manual removals (safe)
Phase 5: Circular Import 6 cycles audited + TaskConcurrencyGuard added
Phase 6: Resilience      CircuitBreakerRegistry, ObservabilityService,
                        CascadePolicy, run_with_resilience (CB-aware)
Phase 7: Ports           CachePort, StatePort, EventPort + NullAdapters
Phase 8: MVOS Tests      24 tests for all 7 invariants (100% pass)
Phase 9: Audit Closure   Health grades updated, CHANGE_COST re-calculated
```

**Delivered Modules:**
- `circuit_breaker.py`: CircuitBreaker + CircuitBreakerRegistry
- `resilience.py`: CascadePolicy, ResiliencePolicy, run_with_resilience
- `ports.py`: CachePort, StatePort, EventPort, NullCache, NullState, NullEventBus
- `concurrency_controller.py`: TaskConcurrencyGuard
- `services/observability.py`: ObservabilityService with 20-call sliding window
- `services/executor.py`, `evaluator.py`, `generator.py`: Extracted services
- `tests/test_phase6_resilience.py`: 24 resilience + cascade tests
- `tests/test_phase7_ports.py`: 16 port protocol + DI tests
- `tests/test_phase8_mvos.py`: 24 MVOS invariant regression tests

**Key Metrics:**
- Total tests: 124 → 141 (+17)
- MVOS score: 0/7 → 7/7 (100%)
- Health grades: C+ → B (layering), F → B+ (resilience), F → B (observability)
- CHANGE_COST: 16.82 (under 17.73 estimate)

---

## Core Architecture

### 1. Main Pipeline Flow

```
Project Description
       |
       v
+-----------------+
| Auto-Resume     |<------------------+
| Detection       |                   |
+--------|--------+                   |
         |                            |
         v                            |
+-----------------+                   |
| Project         |                   |
| Enhancer        |                   |
+--------|--------+                   |
         |                            |
         v                            |
+-----------------+                   |
| Architecture    |                   |
| Advisor         |                   |
+--------|--------+                   |
         |                            |
         v                            |
+-----------------+     +-------------|-----+
| Decomposition   |---->| Checkpoint/Resume |
| (Atomic Tasks)  |     |     (SQLite)      |
+--------|--------+     +-------------------+
         |
         v
+-------------------------------------------------+
|              TASK EXECUTION LOOP                 |
|  +---------+  +----------+  +---------+        |
|  | Route   |->| Generate |->| Critique |        |
|  | Task    |  |   Code   |  | Output   |        |
|  +----|----+  +-----|----+  +----|----+        |
|       |            |             |              |
|       |      +-----v-----+       |              |
|       |      |  Revise   |<------+              |
|       |      |  (loop)   |                      |
|       |      +-----|-----+                      |
|       |            |                            |
|       v            v                            |
|  +---------+  +----------+                     |
|  |Evaluate |  |Deterministic                     |
|  | Quality |  |Validation|                     |
|  +----|----+  +----|-----+                     |
|       |            |                            |
|       +------------+                            |
|              |                                  |
|              v                                  |
|  +---------------------------------+           |
|  |   Cross-Provider Fallback Chain |           |
|  |   (if quality < threshold)      |           |
|  +---------------------------------+           |
+-------------------------------------------------+
         |
         v
+-----------------+
| Store Results   |
| + Telemetry     |
| + State         |
+-----------------+
```

### 2. Module Dependency Graph

```
                              +-----------------+
                              |     CLI / API    |
                              |  (Entry Points)  |
                              +--------|--------+
                                       |
                    +------------------|------------------+
                    |                  |                  |
                    v                  v                  v
           +-------------+    +-------------+    +-------------+
           |   Engine    |    |  Dashboard  |    |   Policy    |
           |   Core      |    |   System    |    |   Engine    |
           +------|------+    +------|------+    +------|------+
                  |                  |                  |
    +-------------|-------------+    |    +-------------|-------------+
    |             |             |    |    |             |             |
    v             v             v    |    v             v             v
+-------+   +---------+   +--------+||+---|----+  +------|-----+ +----|----+
|Models |   |  API    |   | State  ||| Events |  | Knowledge  | | Project |
|& Cost |   | Clients |   |Manager ||| System |  |   Base     | | Manager |
+---|---+   +----|----+   +---|----||+---|----+  +------------+ +---------+
    |            |            |     |    |
    |     +------|------+     |     |    |         +-------------+
    |     |             |     |     |    |         |   Product   |
    |     v             v     |     |    |         |   Manager   |
    |  +------+    +--------+ |     |    |         +-------------+
    +->|Routing|    |Semantic| |     |    |
       |Tables|    | Cache  | |     |    |         +-------------+
       +------+    +--------+ |     |    +-------->|   Quality   |
                              |     |              |   Control   |
                              |     |
                              |     |              +-------------+
                              |     +------------->|   Nexus     |
                              |                    |   Search    |
                              |                    +-------------+
                              |
                              |     +-------------------------------+
                              |     |      ARA Pipeline             |
                              +---->|  (12 Reasoning Methods)       |
                                    |  - Multi-Perspective          |
                                    |  - Iterative                  |
                                    |  - Debate                     |
                                    |  - Jury                       |
                                    |  - Scientific                 |
                                    +-------------------------------+
```

---

## Services Architecture (Phase 6)

The engine has been refactored to use a service-oriented architecture with
injected callbacks, observability, and resilience policies.

```
+------------------------------------------------------------------+
|                     APPLICATION SERVICES                          |
+------------------------------------------------------------------+

  +-------------------+  +-------------------+  +------------------+
  |  ExecutorService  |  | EvaluatorService  |  | GeneratorService |
  +-------------------+  +-------------------+  +------------------+
  | execute_fn        |  | client            |  | decompose_fn     |
  | task_guard        |  | budget            |  | tracer           |
  | tracer            |  | get_models_fn     |  |                  |
  | telemetry         |  | tracer            |  |                  |
  +-------------------+  | telemetry         |  +------------------+
                         +-------------------+

  Responsibilities:
  +-------------------+  +-------------------+  +------------------+
  | Timing metrics    |  | 2-pass eval       |  | Decomposition    |
  | Error normalizing |  | Score parsing     |  | timing           |
  | Tracer spans      |  | Quality gates     |  | Guard wrapping   |
  | Telemetry events  |  | Telemetry events  |  | Tracer spans     |
  +-------------------+  +-------------------+  +------------------+
```

### Service Wiring in Engine

```python
# orchestrator/engine.py
self._task_guard = _TaskGuard(name="tasks", max_concurrent=max_concurrency)
self._telemetry = TelemetryCollector(self._profiles)

# Services wired AFTER telemetry (prevents AttributeError)
self._executor = _ExecutorService(
    execute_fn=self._execute_task,
    guard=self._task_guard,
    tracer=_tracer,
    telemetry=self._telemetry,
)
self._evaluator = _EvaluatorService(
    client=self.client,
    budget=self.budget,
    get_models_fn=self._get_available_models,
    tracer=_tracer,
    telemetry=self._telemetry,
)
self._generator = _GeneratorService(
    decompose_fn=self._decompose,
    tracer=_tracer,
)
```

### Resilience Policy Forwarding

All service entry points accept an optional `_ResiliencePolicy`:

```python
async def execute(self, task: Task, policy: _ResiliencePolicy | None = None)
async def evaluate(self, task: Task, result: TaskResult, policy=None)
async def decompose(self, project: str, criteria: str, policy=None)
```

The `policy` is forwarded as a **keyword argument** to preserve backward
compatibility with existing callbacks that only accept positional args.

---

## Data Models & Types

### Core Data Structures

```python
# Task Types (ROUTING_TABLE keys)
TaskType
|-- CODE_GEN          # Code generation tasks
|-- CODE_REVIEW       # Code review tasks
|-- REASONING         # Complex reasoning tasks
|-- WRITING           # Creative writing
|-- DATA_EXTRACT      # Data extraction
|-- SUMMARIZE         # Summarization
|-- EVALUATE          # Evaluation tasks

# Project States
ProjectStatus
|-- SUCCESS
|-- PARTIAL_SUCCESS
|-- COMPLETED_DEGRADED
|-- BUDGET_EXHAUSTED
|-- TIMEOUT
|-- SYSTEM_FAILURE

# Task States
TaskStatus
|-- PENDING
|-- RUNNING
|-- COMPLETED
|-- FAILED
|-- DEGRADED
```

### Model Routing Architecture

```
ROUTING_TABLE (TaskType -> Model Priority List)
===================================================================

CODE_GEN:
  1. XIAOMI_MIMO_V2_FLASH  ($0.09/$0.29) * BEST
  2. QWEN_2_5_CODER_32B    ($0.66/$1.00)
  3. DEEPSEEK_V3_2         ($0.27/$1.10)
  4. MOONSHOT_KIMI_K2_5    ($0.42/$2.20)
  5. ZHIPU_GLM_4_7         ($0.39/$1.75)
  6. CLAUDE_SONNET_4_6     ($3/$15)

CODE_REVIEW:
  1. XAI_GROK_4_20         ($2/$6) * BEST
  2. DEEPSEEK_R1           ($0.55/$2.19)
  3. MOONSHOT_KIMI_K2_5    ($0.42/$2.20)

REASONING:
  1. STEPFUN_STEP_3_5_FLASH ($0.10/$0.30) * BEST VALUE
  2. DEEPSEEK_R1           ($0.55/$2.19)
  3. O3_MINI               ($1.10/$4.40)

EVALUATE:
  1. XAI_GROK_4_20         ($2/$6) * BEST
  2. DEEPSEEK_R1           ($0.55/$2.19)

COST_TABLE: Dict[Model, Dict["input"|"output", USD_per_1M_tokens]]
FALLBACK_CHAIN: Dict[Model, Model]  # Cross-provider resilience
MODEL_MAX_TOKENS: Dict[Model, int]  # Context window limits
```

---

## Core Pipeline Workflows

### 1. Generate -> Critique -> Revise -> Evaluate Loop

```
+------------------------------------------------------------------+
|                    ITERATION WORKFLOW                             |
|                    (max_iterations = 3)                          |
+------------------------------------------------------------------+

Phase 1: GENERATE
+----------------------------------------+
| 1. Select optimal model via ROUTING    |
| 2. Build context + dependencies        |
| 3. Call LLM API                        |
| 4. Parse response (code extraction)    |
| 5. Validate JSON structure             |
+--------------|-------------------------+
               |
               v
Phase 2: DETERMINISTIC VALIDATION
+----------------------------------------+
| * validate_python_syntax()             |
| * validate_pytest()                    |
| * validate_ruff()                      |
| * validate_json_schema()               |
| * Security checks (bandit)             |
+--------------|-------------------------+
               |
               v (if passed)
Phase 3: CRITIQUE
+----------------------------------------+
| 1. Cross-model review (different LLM)  |
| 2. Evaluate: correctness, style, tests |
| 3. Score: 0.0 - 1.0                    |
| 4. Generate critique feedback          |
+--------------|-------------------------+
               |
               v
Phase 4: EVALUATE
+----------------------------------------+
| Score >= threshold (0.85)?             |
| |-- YES -> Mark COMPLETED               |
| |-- NO  -> Continue to REVISE           |
+--------------|-------------------------+
               |
               v (if below threshold)
Phase 5: REVISE
+----------------------------------------+
| 1. Generate delta prompt               |
| 2. Include critique feedback           |
| 3. Loop back to GENERATE               |
| 4. Increment iteration counter         |
+----------------------------------------+

CIRCUIT BREAKER: Model marked unhealthy after 3 consecutive failures
```

### 2. Cross-Provider Fallback Chain

```
Task Failure / Low Quality
           |
           v
+--------------------------+
| 1. Try next model in     |
|    ROUTING_TABLE         |
+-----------|--------------+
            |
            v
+--------------------------+
| 2. Check FALLBACK_CHAIN  |
|    (cross-provider)      |
+-----------|--------------+
            |
            v
+--------------------------+
| 3. Escalate tier:        |
|    CHEAP -> BALANCED ->  |
|    PREMIUM               |
+-----------|--------------+
            |
            v
+--------------------------+
| 4. If all fail:          |
|    Mark DEGRADED         |
|    Record attempt history|
+--------------------------+
```

---

## ARA Pipeline (Advanced Reasoning & Analysis)

### 12 Reasoning Methods

```
+------------------------------------------------------------------+
|                    ARA PIPELINE v2.0                             |
|         12 Cognitive Science-Based Reasoning Methods            |
+------------------------------------------------------------------+

STANDARD METHODS:
+--------------------+-------------------------------------------+
| Method             | Strategy                                  |
+--------------------+-------------------------------------------+
| Multi-Perspective  | 4 perspectives: constructive, destructive,|
|                    | systemic, minimalist                      |
+--------------------+-------------------------------------------+
| Iterative          | Progressive refinement, step-by-step      |
+--------------------+-------------------------------------------+
| Debate             | 2+ agents argue, meta-evaluator decides   |
+--------------------+-------------------------------------------+
| Research           | Web discovery + LLM synthesis             |
+--------------------+-------------------------------------------+
| Jury               | 4 generators, 3 critics, meta-evaluation  |
+--------------------+-------------------------------------------+
| Scientific         | Hypothesis testing: formulate->test->refine|
+--------------------+-------------------------------------------+
| Socratic           | Probing questions uncover assumptions     |
+--------------------+-------------------------------------------+

SPECIALIZED METHODS:
+--------------------+-------------------------------------------+
| Method             | Strategy                                  |
+--------------------+-------------------------------------------+
| Pre-Mortem         | Imagine failure, work backward to prevent |
+--------------------+-------------------------------------------+
| Bayesian           | Prior->evidence->posterior probability     |
+--------------------+-------------------------------------------+
| Dialectical        | Thesis->antithesis->synthesis             |
+--------------------+-------------------------------------------+
| Analogical         | Map solutions from unrelated domains      |
+--------------------+-------------------------------------------+
| Delphi             | Iterative expert consensus with feedback  |
+--------------------+-------------------------------------------+

SELECTION LOGIC:
+-------------------------------------------------+
| Task complexity + Risk + Budget + Language      |
|                    |                            |
|                    v                            |
|            Route to Optimal                     |
|            Reasoning Method                     |
+-------------------------------------------------+
```

---

## Nexus Search Architecture

```
+------------------------------------------------------------------+
|                    NEXUS SEARCH SYSTEM                           |
|              Self-Hosted Web Search Integration                  |
+------------------------------------------------------------------+

ARCHITECTURE:
+-----------------+
| Search Query    |
+-------|---------+
        |
        v
+-----------------+     +-----------------+
| Query           |---->| Query Expansion |
| Classification  |     | (LLM-based)     |
+-----------------+     +-----------------+
        |
        v
+-----------------------------------------+
|      HYBRID SEARCH (BM25 + Vector)      |
|         Reciprocal Rank Fusion          |
|              (k=60)                     |
+-----------------------------------------+
        |
        v
+-----------------------------------------+
| Parallel Search Sources:                |
| * Web (General)                         |
| * Tech (Documentation)                  |
| * News (Current events)                 |
| * Social (Discussions)                  |
+-----------------------------------------+
        |
        v
+-----------------------------------------+
| Optimization Layer:                     |
| * Result Deduplication                  |
| * Query Caching (TTL-based)             |
| * Adaptive Search Depth                 |
| * Circuit Breaker                       |
+-----------------------------------------+
        |
        v
+-----------------------------------------+
| LLM Reranking                           |
| (Relevance Scoring)                     |
+-----------------------------------------+
        |
        v
+-----------------------------------------+
| Research Report Generation              |
+-----------------------------------------+

AGENT COMPONENTS:
+-----------------+  +-----------------+
| QueryClassifier |  | ResearchAgent   |
|                 |  |                 |
| * classify()    |  | * research()    |
| * get_recommended| | * synthesize()  |
|   _sources()    |  | * report()      |
+-----------------+  +-----------------+
```

---

## Management Systems

### 1. Knowledge Management System

```
+------------------------------------------------------------------+
|                    KNOWLEDGE BASE                                |
|              Centralized Learning Repository                     |
+------------------------------------------------------------------+

Data Model:
+-------------------+
| KnowledgeArtifact |
+-------------------+
| id: str           |
| type: KnowledgeType|
| title: str        |
| content: str      |
| context: dict     |
| tags: list[str]   |
| embedding: list   |
| similarity_score  |
| usage_count       |
+-------------------+

KnowledgeType Enum:
|-- CODE_SNIPPET
|-- SOLUTION
|-- BUGFIX
|-- PATTERN
|-- ARCHITECTURE
|-- DECISION
|-- LESSON_LEARNED

Features:
+-----------------------------------------+
| * Vector storage (sentence-transformers)|
| * Knowledge graph (concept relations)   |
| * Pattern matching                      |
| * Auto-generated documentation          |
| * LRU cache for queries                 |
| * Async indexing                        |
+-----------------------------------------+
```

### 2. Project Management System

```
+------------------------------------------------------------------+
|                    PROJECT MANAGER                               |
|          Task Scheduling & Resource Allocation                   |
+------------------------------------------------------------------+

Core Classes:
+-------------------+  +-------------------+  +-----------------+
|   TaskSchedule    |  |    Milestone      |  |      Risk       |
+-------------------+  +-------------------+  +-----------------+
| task_id: str      |  | id: str           |  | id: str         |
| start_time: datetime| name: str          |  | probability: float
| end_time: datetime |  | deadline: datetime| impact: float    |
| resources: list   |  | completed: bool   |  | risk_score()    |
| is_critical: bool |  | completion_date   |  | mitigation: str |
| slack: timedelta  |  +-------------------+  +-----------------+
+-------------------+

Algorithms:
+-----------------------------------------+
| CriticalPathAnalyzer                    |
| * Forward/backward pass algorithm       |
| * Earliest/latest start calculation     |
| * Float time computation                |
+-----------------------------------------+
| ResourceOptimizer                       |
| * Constraint-based allocation           |
| * Cost optimization                     |
| * Load balancing                        |
+-----------------------------------------+
```

### 3. Product Management System

```
+------------------------------------------------------------------+
|                    PRODUCT MANAGER                               |
|        Feature Prioritization & Release Planning                 |
+------------------------------------------------------------------+

Core Classes:
+-------------------+  +-------------------+  +-----------------+
|     Feature       |  |    RICEScore      |  |     Release     |
+-------------------+  +-------------------+  +-----------------+
| id: str           |  | reach: int        |  | id: str         |
| name: str         |  | impact: int       |  | version: str    |
| status: FeatureStatus| confidence: int   |  | target_date     |
| priority: P0-P3   |  | effort: int       |  | features: list  |
| rice_score        |  | score()           |  | status          |
| tags: list        |  +-------------------+  +-----------------+
+-------------------+

RICE Scoring:
  score = (Reach x Impact x Confidence) / Effort

FeatureStatus Lifecycle:
  IDEA -> RESEARCH -> PLANNED -> IN_PROGRESS -> BETA -> RELEASED -> DEPRECATED
```

### 4. Quality Control System

```
+------------------------------------------------------------------+
|                    QUALITY CONTROLLER                            |
|          Automated Testing & Quality Analysis                    |
+------------------------------------------------------------------+

Test Levels:
|-- UNIT           # Function/class tests
|-- INTEGRATION    # Multi-component tests
|-- E2E            # End-to-end tests
|-- PERFORMANCE    # Load/stress tests
|-- SECURITY       # Security scans

Quality Gates:
+-----------------------------------------+
| CodeMetrics                             |
| * Cyclomatic complexity                 |
| * Maintainability index                 |
| * Duplication percentage                |
| * Documentation coverage                |
| * Type hint coverage                    |
+-----------------------------------------+
| QualityIssue                            |
| * severity: CRITICAL/HIGH/MEDIUM/LOW    |
| * suggested_fix                         |
| * location (file:line:column)           |
+-----------------------------------------+
| QualityReport                           |
| * passed: bool                          |
| * average_coverage                      |
| * test_results                          |
| * issues                                |
+-----------------------------------------+
```

---

## iOS App Store Compliance Suite

```
+------------------------------------------------------------------+
|              iOS APP STORE COMPLIANCE SUITE (v6.0)               |
|                    6 Major Enhancements                          |
+------------------------------------------------------------------+

+-------------+ +-------------+ +-------------+ +-------------+
|Enhancement A| |Enhancement B| |Enhancement C| |Enhancement D|
| Multi-Platform| App Store   | | iOS HIG     | | App Store   |
| Generator   | | Validator   | | Prompts     | | Assets Gen  |
| (9 platforms)| (30 checks) | | (HIG comp)  | | (Auto)      |
+-------------+ +-------------+ +-------------+ +-------------+
       |               |               |               |
       +---------------+---------------+---------------+
                           |
       +-------------------|-------------------+
       v                   v                   v
+-------------+     +-------------+     +-------------+
|Enhancement E|     |Enhancement F|     |   TOTAL     |
| Native      |     | Pre-Submit  |     |             |
| Templates   |     | Testing     |     | 5,400+ lines|
| (10 types)  |     | (10 checks) |     | 135 tests   |
+-------------+     +-------------+     +-------------+

VALIDATION CHECKS:
+-----------------------------------------+
| Performance (IOS-2.1)                   |
| * Launch time < 20s                     |
| * Memory usage < 5x base                |
| * No memory leaks                       |
| * Efficient resource use                |
+-----------------------------------------+
| Metadata (IOS-2.5.2)                    |
| * Accurate descriptions                 |
| * Appropriate keywords                  |
| * Proper categorization                 |
+-----------------------------------------+
| Functionality (IOS-4.2)                 |
| * Beta testing compliance               |
| * Native iOS features (2+)              |
| * No placeholder content                |
| * Stable performance                    |
+-----------------------------------------+
| Legal (IOS-5.1)                         |
| * Info.plist presence                   |
| * Required declarations                 |
| * Privacy policy                        |
+-----------------------------------------+
| HIG Compliance                          |
| * iOS-standard controls                 |
| * Accessibility labels                  |
| * Dark mode support                     |
+-----------------------------------------+
| AI Transparency                         |
| * AI-generated content disclosed        |
| * No misleading claims                  |
+-----------------------------------------+
```

---

## Cost Optimization System

```
+------------------------------------------------------------------+
|                    COST OPTIMIZATION v6.0                        |
|                     35% Cost Reduction                           |
+------------------------------------------------------------------+

TIER 1: Provider-Level (80-90% input cost reduction)
+-----------------------------------------+
| * Prompt Caching (repeated patterns)    |
| * Batch API (parallel requests)         |
| * Token Budget Management               |
+-----------------------------------------+

TIER 2: Architectural (40-60% per-task reduction)
+-----------------------------------------+
| * Model Cascading (CHEAP->BALANCED->PREMIUM)|
| * Dependency Context Injection          |
| * Speculative Generation                |
| * Streaming Validation                  |
+-----------------------------------------+

TIER 3: Quality (30-50% fewer repair cycles)
+-----------------------------------------+
| * Tier-3 Quality Checks                 |
| * Confidence-Based Early Exit           |
| * Fast Regression Detection (EMA a=0.2) |
| * Semantic Sub-Result Caching           |
+-----------------------------------------+

TIER 4: DevOps (Security + DX)
+-----------------------------------------+
| * Docker Sandboxing                     |
| * GitHub Auto-Push                      |
| * Tool Safety Validation                |
+-----------------------------------------+

COST TRACKING:
+-----------------------------------------+
| EMA-based pricing (exponential moving)  |
| Budget hierarchy: org -> team -> job      |
| Real-time cost telemetry                |
| Budget alerts at 80% threshold          |
+-----------------------------------------+
```

---

## Resilience, Fault Tolerance & Observability

### Circuit Breaker

```python
# orchestrator/circuit_breaker.py
class CircuitBreaker:
    """Async per-key circuit breaker with CLOSED/OPEN/HALF_OPEN states."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        half_open_timeout: float = 30.0,
    )

# Usage:
async with cb.context("openai/gpt-4o"):
    response = await client.call(...)
```

| Property | Value |
|----------|-------|
| Failure threshold | 3-5 consecutive failures |
| Reset timeout | 60s (OPEN -> HALF_OPEN) |
| Success threshold | 2 consecutive successes to close |
| Coverage | 95% (121 statements, 6 misses) |

### Resilience Layer

```
+------------------------------------------------------------------+
|                    RESILIENCE MODULE                              |
|          (orchestrator/resilience.py)                            |
+------------------------------------------------------------------+

+-----------------------------------------+
| ResiliencePolicy                        |
|-----------------------------------------|
| retry_attempts: int = 3                 |
| base_delay: float = 1.0                 |
| max_delay: float = 60.0                 |
| timeout_seconds: float = 300.0          |
| fallback_enabled: bool = True           |
| circuit_breaker_enabled: bool = True    |
| jitter: JitterStrategy = FULL           |
+-----------------------------------------+

Components:
|-- ResilienceOrchestrator   # Coordinates retry + fallback + CB
|-- with_resilience()        # Decorator / context manager
|-- JitterStrategy           # NONE | EQUAL | FULL
```

### Service Observability

```
+------------------------------------------------------------------+
|                 OBSERVABILITY INJECTION                           |
|          (orchestrator/services/ + tracing.py)                   |
+------------------------------------------------------------------+

ExecutorService:
  +-- trace span: "executor.task"
  +-- attributes: task_id, task_type, wall_time_ms
  +-- error events: exception.message
  +-- telemetry: timing, result

EvaluatorService:
  +-- trace span: "evaluator.evaluate"
  +-- attributes: score, model_used, wall_time_ms
  +-- telemetry: evaluation results

GeneratorService:
  +-- trace span: "generator.decompose"
  +-- attributes: project_hash, wall_time_ms

TelemetryCollector:
  +-- events: budget, timing, quality
  +-- metrics: EMA-based tracking
```

### State Persistence & Resume

```
+-----------------------------------------+
| SQLite-backed (aiosqlite)               |
| * Project state                         |
| * Budget tracking                       |
| * Task results                          |
| * Resume capability                     |
| * JSON serialization (safe, readable)   |
+-----------------------------------------+

Budget Serialization (FIX-RESUME-001):
+-----------------------------------------+
| _budget_to_dict() includes:             |
|   max_usd, max_time_seconds,            |
|   spent_usd, start_time,                |
|   original_start_time                   |
+-----------------------------------------+
| _budget_from_dict() restores:           |
|   original_start_time for elapsed time  |
|   calculation when resuming             |
+-----------------------------------------+
```

---

## Event System Architecture

```
+------------------------------------------------------------------+
|                    UNIFIED EVENTS SYSTEM                         |
|              Event Sourcing & Telemetry                          |
+------------------------------------------------------------------+

Event Types:
+-----------------------------------------+
| Project Lifecycle                       |
| * PROJECT_STARTED                       |
| * PROJECT_COMPLETED                     |
| * PROJECT_FAILED                        |
+-----------------------------------------+
| Task Lifecycle                          |
| * TASK_CREATED                          |
| * TASK_STARTED                          |
| * TASK_PROGRESS                         |
| * TASK_COMPLETED                        |
| * TASK_FAILED                           |
| * TASK_RETRY                            |
+-----------------------------------------+
| Model/Routing                           |
| * MODEL_SELECTED                        |
| * MODEL_UNAVAILABLE                     |
| * FALLBACK_TRIGGERED                    |
| * CIRCUIT_BREAKER_OPEN                  |
+-----------------------------------------+
| Quality & Validation                    |
| * VALIDATION_PASSED                     |
| * VALIDATION_FAILED                     |
| * QUALITY_GATE_PASSED                   |
+-----------------------------------------+
| Budget & Cost                           |
| * BUDGET_WARNING                        |
| * BUDGET_EXHAUSTED                      |
| * COST_RECORDED                         |
+-----------------------------------------+
| Capability Usage                        |
| * CAPABILITY_USED                       |
| * CAPABILITY_COMPLETED                  |
| * CAPABILITY_FAILED                     |
+-----------------------------------------+

Architecture:
DomainEvent (base)
|-- ProjectStartedEvent
|-- ProjectCompletedEvent
|-- TaskStartedEvent
|-- TaskCompletedEvent
|-- ... (typed events)

Features:
+-----------------------------------------+
| * Immutable, serializable               |
| * Automatic projections (read models)   |
| * Event persistence (SQLite)            |
| * ContextVar for current event          |
| * Async event bus                       |
+-----------------------------------------+
```

---

## Dashboard & Monitoring

```
+------------------------------------------------------------------+
|                    LIVE DASHBOARD v4.0                           |
|              Real-time WebSocket Streaming                       |
+------------------------------------------------------------------+

Features:
+-----------------------------------------+
| * WebSocket (no polling!)               |
| * Toast notifications                   |
| * Gamification system                   |
| * Confetti celebrations                 |
| * Sound notifications                   |
| * Live task progress                    |
| * Test execution monitoring             |
+-----------------------------------------+

Gamification:
+-----------------------------------------+
| DashboardState                          |
| * level: int                            |
| * xp: int                               |
| * xp_to_next_level: int                 |
| * streak: int                           |
| * achievements: list                    |
+-----------------------------------------+
| Achievements                            |
| * first_task                            |
| * streak_7                              |
| * quality_master                        |
| * cost_optimizer                        |
| ...                                     |
+-----------------------------------------+

Dashboard Types:
|-- dashboard_live.py        # Gamified WebSocket
|-- dashboard_mission_control.py  # Professional telemetry
|-- dashboard_antd.py        # Ant Design UI
|-- cli_dashboard.py         # Terminal UI
```

---

## Validation Architecture

```
+------------------------------------------------------------------+
|                    DETERMINISTIC VALIDATORS                      |
|              Non-Negotiable Quality Gates                        |
+------------------------------------------------------------------+

Validators:
+-----------------------------------------+
| validate_python_syntax()                |
| * Compilation check                     |
| * Indentation error handling            |
| * Truncation detection                  |
+-----------------------------------------+
| validate_pytest()                       |
| * Test execution                        |
| * Coverage reporting                    |
| * Module availability check             |
+-----------------------------------------+
| validate_ruff()                         |
| * Linting                               |
| * Style checking                        |
| * Auto-fix support                      |
+-----------------------------------------+
| validate_json_schema()                  |
| * JSON validation                       |
| * Schema compliance                     |
+-----------------------------------------+
| Security Scans                          |
| * Bandit (Python security)              |
| * Safety (dependency vulnerabilities)   |
+-----------------------------------------+

Execution:
+-----------------------------------------+
| async_run_validators()                  |
| * Offloads to threads                   |
| * Non-blocking I/O                      |
| * Parallel execution                    |
+-----------------------------------------+

Rule: If deterministic check fails -> score = 0.0 (overrides LLM)
```

---

## Project Structure

```
Ai Orchestrator/
|-- orchestrator/                 # Main package (~349 files)
|   |-- __init__.py              # Lazy-loading entry point (v6.0.0)
|   |-- cli.py                   # CLI entry point
|   |-- engine.py                # Core orchestration (~2,000 lines)
|   |-- models.py                # Data models & routing (~222 lines)
|   |-- ara_pipelines.py         # 12 reasoning methods
|   |-- api_clients.py           # Unified LLM client
|   |-- budget.py                # Budget management
|   |-- state.py                 # SQLite persistence
|   |-- validators.py            # Deterministic validation
|   |-- cache.py                 # Disk-based caching
|   |-- semantic_cache.py        # Semantic similarity
|   |-- model_selector.py        # Intelligent routing
|   |-- circuit_breaker.py       # Async circuit breaker + registry (Phase 6)
|   |-- resilience.py            # Resilience policies, cascade, CB-aware flow (Phase 6)
|   |-- ports.py                 # CachePort, StatePort, EventPort + NullAdapters (Phase 7)
|   |-- concurrency_controller.py# TaskConcurrencyGuard (Phase 5)
|   |-- tracing.py               # OpenTelemetry tracing
|   |-- telemetry.py             # Telemetry collection
|   |
|   |-- services/                # Application services (Phase 6)
|   |   |-- __init__.py
|   |   |-- executor.py          # Task execution service
|   |   |-- evaluator.py         # LLM evaluation service
|   |   |-- generator.py         # Project decomposition service
|   |   |-- observability.py     # Per-model telemetry (Phase 6)
|   |
|   |-- knowledge_base.py        # Knowledge Management
|   |-- project_manager.py       # Project Management
|   |-- product_manager.py       # Product Management
|   |-- quality_control.py       # Quality Control
|   |
|   |-- nexus_search/            # Web Search (21 files)
|   |   |-- core.py              # Search orchestrator
|   |   |-- agents/              # Search agents
|   |   |-- optimization/        # Query expansion, caching
|   |   |-- providers/           # Search providers
|   |
|   |-- cost_optimization/       # Cost optimization (13 files)
|   |   |-- batch_client.py
|   |   |-- model_cascading.py
|   |   |-- prompt_cache.py
|   |   |-- ...
|   |
|   |-- engine_core/             # Core engine components
|   |   |-- core.py
|   |   |-- task_executor.py
|   |   |-- critique_cycle.py
|   |   |-- fallback_handler.py
|   |   |-- budget_enforcer.py
|   |   |-- dependency_resolver.py
|   |
|   |-- dashboard_core/          # Dashboard infrastructure
|   |-- unified_events/          # Event system
|   |-- scaffold/                # Project templates
|   |-- plugins/                 # Plugin system
|   |-- ide_backend/             # IDE integration
|
|-- tests/                       # Test suite (~30 files)
|   |-- conftest.py             # pytest configuration
|   |-- integration/            # Integration & E2E tests
|   |   |-- conftest.py
|   |   |-- test_full_run.py
|   |   |-- test_resume_after_crash.py
|   |   |-- test_circuit_breaker_fail_fast.py
|   |-- smoke/                  # Smoke tests
|   |   |-- test_api_contracts.py
|   |   |-- test_cli.py
|   |-- test_circuit_breaker.py
|   |-- test_concurrency_controller.py
|   |-- test_evaluator_service.py
|   |-- test_executor_service.py
|   |-- test_generator_service.py
|   |-- test_resilience.py
|   |-- test_service_observability.py
|   |-- test_state_validation.py
|
|-- docs/                        # Documentation
|   |-- CODEBASE_MINDMAP.md     # This file
|   |-- MVOS_CHECKLIST.md       # MVOS audit runbook
|-- scripts/                     # Utility scripts
|   |-- mvos_audit.py           # Automated MVOS checker
|-- pyproject.toml              # Package configuration
|-- requirements.txt            # Dependencies
```

---

## Provider Integration

```
+------------------------------------------------------------------+
|                    LLM PROVIDER SUPPORT                          |
|                    12+ Providers via OpenRouter                  |
+------------------------------------------------------------------+

UNIFIED API (OpenRouter):
+-----------------------------------------+
| api_clients.py                          |
| * UnifiedClient                         |
| * Async HTTP with aiohttp               |
| * Retry logic                           |
| * Rate limiting                         |
+-----------------------------------------+

Supported Providers:
+----------------+------------------------------------------+
| Provider       | Models                                   |
+----------------+------------------------------------------+
| OpenAI         | GPT-4o, GPT-5, o1, o3-mini, o4-mini      |
| Google         | Gemini Pro, Flash, Flash Lite            |
| Anthropic      | Claude 3.5 Sonnet, Opus, Haiku           |
| DeepSeek       | DeepSeek Chat, Reasoner (R1)             |
| Meta           | LLaMA 4 Maverick/Scout, LLaMA 3.3        |
| xAI            | Grok 4.20, Grok 4.1 Fast                 |
| Qwen           | Qwen 2.5 Coder                           |
| MiniMax        | MiniMax M2.7, M2.5                       |
| StepFun        | Step 3.5 Flash                           |
| Z.AI           | GLM 4.7 Flash                            |
| Moonshot       | Kimi K2.5                                |
| Xiaomi         | MiMo-V2 Flash/Pro/Omni                   |
+----------------+------------------------------------------+

Cost Optimization:
+-----------------------------------------+
| * EMA-based price tracking              |
| * Cost-performance profiling            |
| * Budget hierarchy enforcement          |
| * Cross-provider fallback               |
+-----------------------------------------+
```

---

## Testing Architecture

```
+------------------------------------------------------------------+
|                    TESTING FRAMEWORK                              |
|                    124 Tests, 11 Modules                          |
+------------------------------------------------------------------+

Test Organization:
+-----------------------------------------+
| tests/                                  |
| |-- conftest.py                         |
| |   * Markers: unit, integration, slow  |
| |   * Markers: requires_api, e2e        |
| |   * collect_ignore for standalone     |
| |                                         |
| |-- integration/                        |
| |   * conftest.py (mocked fixtures)     |
| |   * test_full_run.py (3 e2e tests)    |
| |   * test_resume_after_crash.py (2)    |
| |   * test_circuit_breaker_fail_fast.py |
| |                                         |
| |-- smoke/                              |
| |   * test_api_contracts.py (9 HTTP)    |
| |   * test_cli.py (6 subprocess)        |
| |                                         |
| |-- test_circuit_breaker.py (12)        |
| |-- test_state_validation.py (8)        |
| |-- test_executor_service.py (12)       |
| |-- test_evaluator_service.py (14)      |
| |-- test_generator_service.py (12)      |
| |-- test_concurrency_controller.py (8)  |
| |-- test_phase6_resilience.py (24)      |
| |   * Registry, Observability, Cascade  |
| |-- test_phase7_ports.py (16)           |
| |   * Protocol validation, NullAdapters |
| |-- test_phase8_mvos.py (24)            |
| |   * MVOS invariant verification       |
+-----------------------------------------+

Coverage by Module (tested modules):
+-------------------------------+----------+
| Module                        | Coverage |
+-------------------------------+----------+
| circuit_breaker.py            | 95.0%    |
| services/evaluator.py         | 94.5%    |
| services/executor.py          | 94.2%    |
| services/generator.py         | 100.0%   |
| resilience.py                 | 89.6%    |
| budget.py                     | 86.5%    |
| policy.py                     | 85.2%    |
| exceptions.py                 | 84.6%    |
| models.py                     | 96.4%    |
+-------------------------------+----------+
| Baseline (all orchestrator/)  | 12.2%    |
+-------------------------------+----------+

Test Markers:
+-----------------------------------------+
| @pytest.mark.unit                       |
| @pytest.mark.integration                |
| @pytest.mark.slow                       |
| @pytest.mark.requires_api               |
| @pytest.mark.e2e                        |
| @pytest.mark.load                       |
| @pytest.mark.stress                     |
| @pytest.mark.benchmark                  |
+-----------------------------------------+
```

---

## MVOS Audit System (Phase 8–9)

```
+------------------------------------------------------------------+
|        MINIMUM VIABLE OPERATIONAL STATE (MVOS) AUDIT             |
|                   tests/test_phase8_mvos.py                      |
|                  (24 automated unit tests)                        |
+------------------------------------------------------------------+

Purpose: Runtime verification that all critical architectural invariants
         hold after deployment or refactor. Phase 8 tests added 24
         regression guards covering all 7 invariants.

Invariants Verified (Phase 8–9 Status: ✅ 7/7 PASS):
+-------+------------------------------------------+-----+
| ID    | Invariant                              |Status|
+-------+------------------------------------------+-----+
| MVOS-1| Orchestrator instantiates cleanly      | ✅   |
|       | (NullAdapters, no SQLite)              |      |
| MVOS-2| UnifiedClient.circuit_breaker starts   | ✅   |
|       | CLOSED with correct thresholds         |      |
| MVOS-3| CircuitBreakerRegistry wired & per-    | ✅   |
|       | model isolated                         |      |
| MVOS-4| ObservabilityService accumulates       | ✅   |
|       | call metrics (latency, error rate,cost)|      |
| MVOS-5| All app-layer services present         | ✅   |
|       | (executor, evaluator, generator, guard)|      |
| MVOS-6| Concrete adapters satisfy Port         | ✅   |
|       | protocols (structural subtyping)       |      |
| MVOS-7| CascadePolicy builds cost-tier-sorted  | ✅   |
|       | ResiliencePolicy                       |      |
+-------+------------------------------------------+-----+

Health Grades (Pre→Post Refactor):
+-------+--------+----------+-------+
| Aspect| Before | After    | Delta |
+-------+--------+----------+-------+
| Layer | C+     | B        | ↑ 0.5 |
| Coupl | C      | B-       | ↑ 0.3 |
| Cmplx | D+     | C+       | ↑ 0.7 |
| State | C      | B-       | ↑ 0.3 |
| Error | B-     | B        | ↑ 0.2 |
| Types | B      | B        | →     |
| Resil | F      | B+       | ↑ 1.7 |
| Observ| F      | B        | ↑ 1.0 |
+-------+--------+----------+-------+

Usage (pytest):
    pytest tests/test_phase8_mvos.py -v
    # 24 tests verify all 7 invariants with full coverage

Legacy Script (manual verification):
    python scripts/mvos_audit.py [--verbose]

Exit Codes:
    0 -- All invariants pass (deployment ready)
    1 -- One or more invariants failed
```

---

## Workflow Summary

```
+------------------------------------------------------------------+
|                    COMPLETE WORKFLOW MAP                         |
+------------------------------------------------------------------+

Phase 1: Input Processing
+-----------------------------------------+
| 1. Parse CLI arguments                  |
| 2. Load environment (.env)              |
| 3. Check for resume candidates          |
| 4. Initialize budget                    |
| 5. Create project state                 |
+--------------|--------------------------+
               |
               v
Phase 2: Enhancement
+-----------------------------------------+
| 1. Project Enhancer (LLM)               |
| 2. Architecture Advisor                 |
| 3. Platform detection                   |
| 4. Template selection                   |
+--------------|--------------------------+
               |
               v
Phase 3: Decomposition
+-----------------------------------------+
| 1. Break into atomic tasks              |
| 2. Detect dependencies                  |
| 3. Build dependency graph               |
| 4. Topological sort                     |
+--------------|--------------------------+
               |
               v
Phase 4: Execution (per task)
+-----------------------------------------+
| While tasks remaining:                  |
|   1. Get ready tasks (deps satisfied)   |
|   2. Route to optimal model             |
|   3. Execute generate->critique->revise |
|   4. Run deterministic validators       |
|   5. Evaluate quality                   |
|   6. If degraded, trigger fallback      |
|   7. Store result                       |
|   8. Emit events                        |
+--------------|--------------------------+
               |
               v
Phase 5: Finalization
+-----------------------------------------+
| 1. Calculate final score                |
| 2. Generate quality report              |
| 3. Organize outputs                     |
| 4. Cleanup state                        |
| 5. Emit PROJECT_COMPLETED event         |
+-----------------------------------------+
```

---

## Key Metrics & Statistics

| Metric | Value |
|--------|-------|
| **Version** | v6.0.0 (2026-04-20) |
| **Core Code** | ~142,000 lines (orchestrator/) |
| **Test Code** | ~1,700 lines (tests/) |
| **Total Files** | ~350 Python files (orchestrator/) |
| **Test Files** | 11 active pytest modules |
| **Test Count** | 141 tests (all passing, +17 from Phases 6–8) |
| **Test Coverage** | 12.2% baseline (legacy), 90%+ for tested modules |
| **Cost Reduction** | 35% |
| **Providers** | 12+ (via OpenRouter) |
| **Reasoning Methods** | 12 (ARA Pipeline) |
| **Validation Checks** | 30+ (iOS compliance) |
| **MVOS Invariants** | 6 (all automated) |

---

## Key Connections & Dependencies

```
External Dependencies:
+-----------------------------------------+
| LLM APIs                                |
| * OpenRouter (unified)                  |
| * Provider SDKs (openai, anthropic,    |
|   google-genai)                         |
+-----------------------------------------+
| Database                                |
| * aiosqlite (async SQLite)              |
| * Redis (optional, caching)             |
+-----------------------------------------+
| Web Framework                           |
| * FastAPI (dashboard)                   |
| * WebSockets (real-time)                |
| * Uvicorn (ASGI server)                 |
+-----------------------------------------+
| Testing                                 |
| * pytest + pytest-asyncio               |
| * pytest-cov (coverage)                 |
| * pytest-xdist (parallel)               |
| * anyio                                 |
+-----------------------------------------+
| Code Quality                            |
| * ruff (linting)                        |
| * black (formatting)                    |
| * mypy (type checking)                  |
| * bandit (security)                     |
+-----------------------------------------+
| Observability                           |
| * OpenTelemetry (tracing)               |
| * Structured logging                    |
+-----------------------------------------+

Internal Dependencies:
+-----------------------------------------+
| Engine -> Models, API Clients, State    |
| ARA -> Models, API Clients, Cache       |
| Nexus -> Agents, Optimization, Providers|
| Management -> Performance, Log Config   |
| Dashboard -> Events, State              |
| Validators -> Models, Cache             |
| Services -> Engine callbacks, Tracing   |
+-----------------------------------------+
```

---

*Last Updated: 2026-04-20*  
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
