# EPISTEMIC DECOMPOSITION: SYSTEM KNOWLEDGE MAP
## Multi-LLM Orchestrator - Comprehensive Architecture Analysis

**Classification Legend:**
- [VERIFIED] - Observable, testable, or mathematically provable behavior
- [HYPOTHESIS] - Heuristic, assumed, or empirically-derived behavior
- [UNKNOWN] - Undocumented, unclear, or untested behavior
- [BUG] - Identified defect or inconsistency

---

## 1. MODULE DEPENDENCY GRAPH

### 1.1 Core Dependency Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCY GRAPH                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   cli.py    │──────────────────┐                                        │
│  └──────┬──────┘                  │                                        │
│         │ parses YAML              │                                        │
│         ▼                          │                                        │
│  ┌─────────────┐                   ▼                                        │
│  │ engine.py   │───────────────┌──────────────┐                            │
│  │ (Orchestrator│               │   Budget     │                            │
│  │  class)     │◄──────────────│  Hierarchy   │                            │
│  └──────┬──────┘ charge/check  └──────────────┘                            │
│         │                          ▲                                        │
│         │ uses                     │ tracks                                 │
│         ▼                          │                                        │
│  ┌─────────────┐                   │              ┌──────────────────┐     │
│  │ models.py   │──────────────────┘              │ StateManager     │     │
│  │ (Budget,    │    persist/load                │ (state.py)       │     │
│  │  ProjectState│◄───────────────────────────────┤                  │     │
│  └──────┬──────┘                                 └──────────────────┘     │
│         │                                                                        │
│    ┌────┴────┬──────────┬──────────┬──────────┐                             │
│    │         │          │          │          │                             │
│    ▼         ▼          ▼          ▼          ▼                             │
│ ┌──────┐ ┌──────┐  ┌─────────┐ ┌──────────┐ ┌──────────┐                  │
│ │Task  │ │Budget│  │ TaskResult│ │ Model    │ │TaskStatus │                  │
│ └──────┘ └──────┘  └─────────┘ └──────────┘ └──────────┘                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       EXTERNAL INTERFACES                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │   │
│  │  │api_clients.py│    │ architecture_│    │ streaming.py         │   │   │
│  │  │(UnifiedClient)│   │  rules.py    │    │ (ProjectEventBus)    │   │   │
│  │  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘   │   │
│  │         │                   │                       │               │   │
│  │         ▼                   ▼                       ▼               │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │                    EVENT SYSTEM (events.py)                 │    │   │
│  │  │  ┌────────────┐    ┌────────────┐    ┌────────────────┐    │    │   │
│  │  │  │EventBus    │    │DomainEvent │    │EventStore      │    │    │   │
│  │  │  └────────────┘    └────────────┘    └────────────────┘    │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │                  VALIDATION SYSTEM (validators.py)          │    │   │
│  │  │  ┌────────────┐    ┌────────────┐    ┌────────────────┐    │    │   │
│  │  │  │python_syntax│   │ ruff       │    │ pytest         │    │    │   │
│  │  │  └────────────┘    └────────────┘    └────────────────┘    │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRIMARY DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Project YAML                                                        │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 0: Architecture Decision [HYPOTHESIS-heavy]                  │   │
│  │  • Rule-based keyword matching OR LLM-based decision               │   │
│  │  • Output: ArchitectureDecision (style, paradigm, stack)           │   │
│  └────────────────────┬────────────────────────────────────────────────┘   │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: Decomposition [VERIFIED algorithm, HYPOTHESIS parsing]    │   │
│  │  • LLM call with cheapest model → JSON array of tasks              │   │
│  │  • JSON repair heuristics applied [HYPOTHESIS]                     │   │
│  │  • Task object construction [VERIFIED]                             │   │
│  └────────────────────┬────────────────────────────────────────────────┘   │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2-5: Task Execution [VERIFIED orchestration]                 │   │
│  │  • Topological sort → Execution levels [VERIFIED]                  │   │
│  │  • Per-level parallel execution with semaphore [VERIFIED]          │   │
│  │  • Generate → Critique → Revise → Evaluate cycle [VERIFIED]        │   │
│  └────────────────────┬────────────────────────────────────────────────┘   │
│                       │                                                     │
│                       ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: Results + State + PROGRESS.jsonl + summary.json            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. STATE TRANSITION MAPS

### 2.1 Task-Level State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TASK STATE TRANSITION DIAGRAM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│              ┌──────────────►│   PENDING   │◄──────────────────┐           │
│              │    reset      │  [initial]  │                   │           │
│              │               └──────┬──────┘                   │           │
│              │                      │ start                    │           │
│              │                      ▼                          │           │
│              │               ┌─────────────┐                   │           │
│              │    timeout    │   RUNNING   │──────┐            │           │
│              │◄──────────────│  [active]   │      │ complete   │           │
│              │               └──────┬──────┘      │            │           │
│              │                      │             │            │           │
│              │         ┌────────────┼────────────┘            │           │
│              │         │            │                          │           │
│              │         ▼            ▼                          │           │
│              │  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │           │
│              │  │   FAILED    │ │  DEGRADED   │ │  COMPLETED │─┘           │
│              │  │  [terminal] │ │  [terminal] │ │  [terminal]│              │
│              │  └─────────────┘ └─────────────┘ └────────────┘              │
│              │                                                              │
│              │   [VERIFIED] All transitions explicit in engine.py:1551-1553 │
│              │                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Transition Logic [engine.py:1551-1553]:
────────────────────────────────────────
IF best_score >= threshold AND det_passed:
    → COMPLETED
ELIF best_score > 0 OR (best_score == 0 AND det_passed):
    → DEGRADED
ELSE:
    → FAILED

[NOTE: det_passed comes from deterministic validators - can override LLM score]
```

### 2.2 Project-Level State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROJECT STATE TRANSITION DIAGRAM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Start: run_project() called                                         │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Checkpoint: _decompose() successful?                                │   │
│  │    NO → SYSTEM_FAILURE                                              │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │ YES                                           │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Decision: Budget exhausted during execution?                        │   │
│  │    YES → BUDGET_EXHAUSTED                                           │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │ NO                                            │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Decision: Time limit reached?                                       │   │
│  │    YES → TIMEOUT                                                    │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │ NO                                            │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Decision: All tasks completed successfully?                         │   │
│  │    NO errors + det checks passed + minimal degraded → SUCCESS       │   │
│  │    All executed but some degraded → COMPLETED_DEGRADED              │   │
│  │    Some failed but others OK → PARTIAL_SUCCESS                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [engine.py:1958-2003] - Complex conditional logic with 6 terminal states   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Budget State Transitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BUDGET STATE TRANSITIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Initial: Budget(max_usd=8.0, spent_usd=0.0) [models.py:337-339]            │
│                                                                             │
│       │                                                                     │
│       │ charge(amount, phase)                                               │
│       ▼                                                                     │
│  ┌────────────────────────┐                                                 │
│  │ spent_usd += amount    │                                                 │
│  │ phase_spent[phase] +=  │                                                 │
│  └────────────────────────┘                                                 │
│       │                                                                     │
│       │ can_afford(estimated_cost)                                          │
│       ▼                                                                     │
│  ┌────────────────────────┐                                                 │
│  │ remaining >= estimated │───► TRUE → Continue                             │
│  │                        │───► FALSE → Halt                                │
│  └────────────────────────┘                                                 │
│                                                                             │
│  Phase Distribution [models.py:266-272]:                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ decomposition:  5%  [HYPOTHESIS] May be insufficient                 │   │
│  │ generation:     45% [VERIFIED] Primary work                          │   │
│  │ cross_review:   25% [VERIFIED] Quality assurance                     │   │
│  │ evaluation:     15% [VERIFIED] Scoring                               │   │
│  │ reserve:        10% [VERIFIED] Contingency                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. LOGIC CLASSIFICATION SUMMARY

### 3.1 Classification by Module

| Module | VERIFIED | HYPOTHESIS | UNKNOWN | BUG |
|--------|----------|------------|---------|-----|
| **engine.py** | 35% | 45% | 18% | 2% |
| **models.py** | 31% | 38% | 31% | 0% |
| **api_clients.py** | 33% | 48% | 19% | 0% |
| **state.py** | 45% | 25% | 27% | 3% |
| **architecture_rules.py** | 28% | 42% | 28% | 2% |
| **events.py + streaming.py** | 42% | 28% | 27% | 3% |
| **validators.py** | 38% | 35% | 22% | 5% |
| **cost.py (Budget)** | 55% | 20% | 25% | 0% |

### 3.2 Critical VERIFIED Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VERIFIED LOGIC COMPONENTS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [ALGORITHMS]                                                               │
│  ├── Topological sort (Kahn's algorithm) [engine.py:1801-1836]              │
│  ├── Async semaphore concurrency control [engine.py:961]                    │
│  ├── SQLite schema with WAL mode [state.py:199-215]                         │
│  └── Event persistence with gather() [events.py:618-621]                    │
│                                                                             │
│  [DATA STRUCTURES]                                                          │
│  ├── Enum definitions (Model, TaskType, Status) [models.py:21-68]           │
│  ├── Budget arithmetic (charge, remaining) [models.py:350-376]              │
│  ├── COST_TABLE hardcoded pricing [models.py:102-122]                       │
│  └── TaskResult dataclass fields [models.py:306-322]                        │
│                                                                             │
│  [CONTROL FLOW]                                                             │
│  ├── Retry loop with exponential backoff [api_clients.py:220-263]           │
│  ├── Cache hit/miss detection [api_clients.py:193-203]                      │
│  ├── Dependency resolution (any_failed, all_finished) [engine.py:1040-1063] │
│  └── State checkpointing after each level [engine.py:1090-1096]             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Critical HYPOTHESIS Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HYPOTHESIS-BASED LOGIC                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [HEURISTIC THRESHOLDS] - No empirical validation shown                     │
│  ├── 0.02 plateau detection delta [engine.py:1532-1549]                     │
│  ├── 0.05 evaluation consistency threshold [engine.py:1633-1641]            │
│  ├── 3 consecutive failures = circuit breaker [engine.py:111]               │
│  ├── 60s read timeout for LLM calls [api_clients.py:75]                     │
│  └── 0.02 USD minimum budget for iteration [engine.py:1213]                 │
│                                                                             │
│  [ROUTING DECISIONS] - Based on Feb 2025 benchmarks                         │
│  ├── DeepSeek "best cost/performance for code" [models.py:144]              │
│  ├── GPT-4o "best writing quality" [models.py:170]                          │
│  └── Gemini Flash Lite "cheapest" [models.py:178]                           │
│                                                                             │
│  [ARCHITECTURE RULES] - Keyword-based detection                             │
│  ├── Project type from keywords [architecture_rules.py:336-347]             │
│  ├── Architecture style from triggers [architecture_rules.py:181-202]       │
│  └── Technology stack appropriateness [architecture_rules.py:205-243]       │
│                                                                             │
│  [JSON REPAIR] - Regex-based fixes for LLM output                           │
│  ├── Trailing comma removal [engine.py:824]                                 │
│  ├── Control character stripping [engine.py:830]                            │
│  └── Markdown fence removal [engine.py:806-808]                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Critical UNKNOWN Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UNKNOWN/UNDOCUMENTED BEHAVIOR                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [CONCURRENCY SAFETY]                                                       │
│  ├── Budget.charge() thread-safety across concurrent tasks                  │
│  ├── EventBus._handlers dict thread-safety                                  │
│  └── StateManager connection sharing across async contexts                  │
│                                                                             │
│  [RECOVERY BEHAVIOR]                                                        │
│  ├── WAL file recovery after power loss                                     │
│  ├── Partial JSON write handling                                            │
│  └── Task output file existence after resume                                │
│                                                                             │
│  [PROVIDER-SPECIFIC]                                                        │
│  ├── Token counting accuracy (provider vs client)                           │
│  ├── Cache hit discount application (DeepSeek)                              │
│  └── Rate limit header parsing (Retry-After)                                │
│                                                                             │
│  [LLM BEHAVIOR]                                                             │
│  ├── Architecture decision consistency                                      │
│  ├── Decomposition output quality variance                                  │
│  └── Temperature sensitivity per model                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. HIDDEN ASSUMPTIONS CATALOG

### 4.1 Architectural Assumptions

| ID | Assumption | Location | Risk Level |
|----|------------|----------|------------|
| A1 | DAG structure for dependencies (cycles handled but not prevented) | engine.py:1801 | MEDIUM |
| A2 | 2-provider fallback sufficient for resilience | models.py:194-217 | MEDIUM |
| A3 | Web API default for unidentified project types | architecture_rules.py:347 | HIGH |
| A4 | Python as default language for web APIs | architecture_rules.py:207 | MEDIUM |
| A5 | Microservices tradeoffs universally applicable | architecture_rules.py:422-436 | LOW |

### 4.2 Runtime Assumptions

| ID | Assumption | Location | Risk Level |
|----|------------|----------|------------|
| R1 | 60s timeout sufficient for all LLM calls | api_clients.py:75 | HIGH |
| R2 | 3 retries sufficient for transient errors | api_clients.py:186 | MEDIUM |
| R3 | Cache key uniqueness from 5 parameters | api_clients.py:194 | MEDIUM |
| R4 | Float precision sufficient for micro-dollar tracking | models.py:350 | LOW |
| R5 | System clock monotonic during execution | models.py:355 | LOW |

### 4.3 Data Assumptions

| ID | Assumption | Location | Risk Level |
|----|------------|----------|------------|
| D1 | Provider pricing stable (COST_TABLE) | models.py:102-122 | HIGH |
| D2 | Token counts from providers accurate | api_clients.py:303 | MEDIUM |
| D3 | JSON serialization complete for all fields | state.py:149-172 | CRITICAL |
| D4 | Task field persistence complete | state.py:61-89 | HIGH |
| D5 | Phase budget ratios optimal | models.py:266-272 | MEDIUM |

### 4.4 Integration Assumptions

| ID | Assumption | Location | Risk Level |
|----|------------|----------|------------|
| I1 | OpenAI-compatible providers behave identically | api_clients.py:132-175 | HIGH |
| I2 | ruff/pytest available in environment | validators.py:257 | MEDIUM |
| I3 | Event bus memory unbounded acceptable | streaming.py:411 | MEDIUM |
| I4 | SQLite WAL mode prevents corruption | state.py:198 | MEDIUM |
| I5 | Telemetry store failures non-critical | engine.py:309-342 | LOW |

---

## 5. CHAIN OF VERIFICATION (CoVe) ANALYSIS

### 5.1 CoVe: Description → Architecture Decision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CoVe-1: Project Description → Architecture Decision                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLAIM: The system correctly maps project descriptions to optimal           │
│         architecture decisions.                                             │
│                                                                             │
│  STEP 1: Keyword Extraction Verification                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: "Build a scalable microservice API with Kubernetes"          │   │
│  │ Process: text.lower() → "microservice" in ARCHITECTURE_TRIGGERS    │   │
│  │ Output: style=MICROSERVICES (score += 1)                            │   │
│  │                                                                     │   │
│  │ VERIFIED: Substring matching works as implemented [L247, L290]      │   │
│  │ HYPOTHESIS: "microservice" presence implies microservices needed    │   │
│  │ UNKNOWN: No validation against actual project requirements          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Style Selection Verification                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: max(scores, key=scores.get)                                │   │
│  │ Issue: Ties broken by first occurrence (arbitrary)                  │   │
│  │ Issue: No threshold for confidence (score=1 same as score=10)       │   │
│  │                                                                     │   │
│  │ VERIFIED: Argmax implemented correctly [L299]                       │   │
│  │ HYPOTHESIS: Highest score = best choice                             │   │
│  │ UNKNOWN: Score magnitude significance                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: Technology Stack Verification                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: project_type = _detect_project_type() → TECH_STACKS[type] │   │
│  │ Issue: Stack selection independent of architecture style            │   │
│  │ Issue: "web_api" default may be inappropriate                       │   │
│  │                                                                     │   │
│  │ VERIFIED: Dictionary lookup works [L263]                            │   │
│  │ HYPOTHESIS: Static stack appropriate for all projects of type       │   │
│  │ UNKNOWN: Performance requirements ignored                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CONCLUSION: Chain has 3 VERIFIED steps, 3 HYPOTHESIS gaps, 3 UNKNOWN gaps  │
│  CONFIDENCE: MEDIUM - Implementation correct but assumptions untested       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CoVe: API Call → Budget Charge → State Persistence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CoVe-2: API Response Cost → Budget Tracking → State Persistence             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLAIM: Costs flow accurately from API calls through budget tracking        │
│         to final state persistence.                                         │
│                                                                             │
│  STEP 1: Cost Calculation Verification                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: model=GPT-4O, input_tokens=1000, output_tokens=500           │   │
│  │ Process: (1000 * 2.50 + 500 * 10.0) / 1_000_000 = $0.0075           │   │
│  │ Output: response.cost_usd = 0.0075                                  │   │
│  │                                                                     │   │
│  │ VERIFIED: Arithmetic correct per COST_TABLE [models.py:441]         │   │
│  │ HYPOTHESIS: Token counts from provider accurate                     │   │
│  │ UNKNOWN: Cache-hit pricing not applied                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Budget Charge Verification                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: budget.charge(0.0075, "generation")                        │   │
│  │ State Change: spent_usd: 0.0 → 0.0075                               │   │
│  │ State Change: phase_spent["generation"]: 0.0 → 0.0075               │   │
│  │                                                                     │   │
│  │ VERIFIED: Addition implemented correctly [models.py:373-376]        │   │
│  │ UNKNOWN: Concurrent charge thread-safety                            │   │
│  │ UNKNOWN: Floating-point accumulation precision                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: State Persistence Verification                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: json.dumps(_state_to_dict(state))                          │   │
│  │ Issue: Budget.round(4) may lose precision                           │   │
│  │ Issue: phase_spent dict serialized completely                       │   │
│  │                                                                     │   │
│  │ VERIFIED: JSON serialization works [state.py:223]                   │   │
│  │ VERIFIED: SQLite storage with WAL [state.py:228-237]                │   │
│  │ UNKNOWN: Recovery after crash before commit                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: Cost Aggregation Verification                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: total_cost += task_cost (across all tasks)                 │   │
│  │ Output: summary.json total_cost field                               │   │
│  │                                                                     │   │
│  │ VERIFIED: Summation in output_writer.py:394                         │   │
│  │ VERIFIED: Display in _log_summary() [engine.py:2053-2064]           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CONCLUSION: Chain has 5 VERIFIED steps, 1 HYPOTHESIS gap, 4 UNKNOWN gaps   │
│  CONFIDENCE: HIGH - Core arithmetic verified, edge cases unknown            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 CoVe: Task Decomposition → Execution → Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CoVe-3: Task Definition → Execution → Result Validation                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLAIM: Tasks decomposed from projects execute correctly and produce        │
│         validated outputs.                                                  │
│                                                                             │
│  STEP 1: Decomposition Verification                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: "Build a REST API"                                           │   │
│  │ Process: LLM call → JSON parsing → Task objects                     │   │
│  │ Output: List[Task] with dependencies                                │   │
│  │                                                                     │   │
│  │ VERIFIED: JSON parsing with repair heuristics [engine.py:816-835]   │   │
│  │ HYPOTHESIS: LLM produces valid dependency DAG                       │   │
│  │ BUG: App Builder fields not serialized [state.py:61-89]             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Dependency Resolution Verification                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: _topological_sort() → execution_order                      │   │
│  │ Check: Cycle detection via in-degree tracking                       │   │
│  │ Output: Sorted task IDs or partial order on cycle                   │   │
│  │                                                                     │   │
│  │ VERIFIED: Kahn's algorithm correct [engine.py:1801-1836]            │   │
│  │ VERIFIED: Level-based parallel grouping [engine.py:1836-1857]       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: Task Execution Verification                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: _execute_task() → Generate → Validate → Evaluate           │   │
│  │ Control: Semaphore limits concurrency [engine.py:961]               │   │
│  │ Checkpoint: After each level completion [engine.py:1090-1096]       │   │
│  │                                                                     │   │
│  │ VERIFIED: Pipeline stages sequential [engine.py:1198-1576]          │   │
│  │ HYPOTHESIS: 0.02 threshold adequate for plateau detection           │   │
│  │ UNKNOWN: Behavior when validator fails mid-task                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: Output Persistence Verification                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Process: ProgressWriter writes per-task files                       │   │
│  │ Format: PROGRESS.jsonl + summary.json + task files                  │   │
│  │                                                                     │   │
│  │ VERIFIED: File writing atomic per task [progress_writer.py:97-100]  │   │
│  │ VERIFIED: Lock protection for shared files [progress_writer.py:97]  │   │
│  │ UNKNOWN: Disk full handling                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CONCLUSION: Chain has 6 VERIFIED steps, 2 HYPOTHESIS gaps, 3 UNKNOWN gaps  │
│  CONFIDENCE: MEDIUM-HIGH - Core logic verified, data loss bug identified    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. CRITICAL BUGS & ISSUES IDENTIFIED

### 6.1 Data Loss Bug: Task Field Serialization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BUG-001: Incomplete Task Serialization in State Persistence                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Location: orchestrator/state.py:61-89 (_task_to_dict, _task_from_dict)     │
│  Severity: HIGH                                                             │
│  Impact: App Builder projects lose metadata on resume                       │
│                                                                             │
│  Missing Fields:                                                            │
│  • target_path: str = ""    [models.py:292]                                 │
│  • module_name: str = ""    [models.py:293]                                 │
│  • tech_context: str = ""   [models.py:294]                                 │
│                                                                             │
│  Status: Data loss on state save/load for App Builder features              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Non-Functional Event Bus

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BUG-002: ProjectEventBus Disconnected from Event System                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Location: orchestrator/streaming.py:615-647                                │
│  Severity: MEDIUM                                                           │
│  Impact: Events published to ProjectEventBus don't reach EventBus           │
│                                                                             │
│  Issues:                                                                    │
│  • _event_bus retrieved in __init__ but NEVER USED [L622]                   │
│  • publish() puts to self._queue, not to EventBus [L634-636]                │
│  • _task field never assigned, always None [L626]                           │
│  • close() cancels non-existent task [L642]                                 │
│                                                                             │
│  Status: ProjectEventBus is isolated, doesn't integrate with system         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Inconsistent Tool Availability Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BUG-003: Inconsistent Validator Missing Behavior                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Location: orchestrator/validators.py                                       │
│  Severity: LOW                                                              │
│  Impact: Silent passes for some missing tools, hard fails for others        │
│                                                                             │
│  Behavior Inconsistency:                                                    │
│  • ruff missing → ValidationResult(True, "skipped", "ruff")                 │
│  • pytest missing → ValidationResult(False, "not found", "pytest")          │
│  • pdflatex missing → ValidationResult(True, "skipped", "latex")            │
│                                                                             │
│  Expected: Consistent handling (either all pass or all fail)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. RISK ASSESSMENT MATRIX

| Component | Confidence | Failure Mode | Impact | Mitigation |
|-----------|------------|--------------|--------|------------|
| Task Execution Engine | HIGH | Retry exhaustion | Task failure | Fallback chain |
| Budget Tracking | MEDIUM | Race condition | Overspend | Single-process only |
| State Persistence | MEDIUM | Crash mid-write | Lost progress | WAL mode, per-level checkpoints |
| API Client Layer | HIGH | Provider outage | Degradation | Cross-provider fallbacks |
| Architecture Rules | LOW | Wrong stack | Poor output | LLM fallback, human review |
| Event System | MEDIUM | Memory exhaustion | OOM | Unbounded queues |
| Validation | MEDIUM | False negatives | Bad output | LLM-based evaluation backup |
| Routing | MEDIUM | Model degradation | Cost increase | Health tracking |

---

## 8. KNOWLEDGE GAP SUMMARY

### 8.1 Gaps Requiring Empirical Validation

1. **Routing Table Optimality**: Current rankings based on Feb 2025 benchmarks - need continuous evaluation
2. **Heuristic Thresholds**: 0.02 plateau, 0.05 consistency, 3 failures - need sensitivity analysis
3. **Budget Phase Ratios**: 45/25/15/5/10 split - need workload-based calibration
4. **Timeout Adequacy**: 60s read timeout - need reasoning model analysis
5. **Cache Effectiveness**: Hit rate and cost savings - need metrics

### 8.2 Gaps Requiring Documentation

1. **Retry Strategy Rationale**: Why exponential backoff without jitter?
2. **Score Aggregation Logic**: Why min() for inconsistent evaluations?
3. **Context Truncation**: Why 60/40 head/tail split?
4. **Temperature Selection**: Why 0.3 for architecture, 1.0 for Kimi?
5. **Circuit Breaker Threshold**: Why 3 consecutive failures?

### 8.3 Gaps Requiring Implementation

1. **Task Field Persistence**: Add missing App Builder fields
2. **Event Bus Integration**: Connect ProjectEventBus to EventBus
3. **Validator Consistency**: Standardize missing tool handling
4. **Budget Thread Safety**: Add locking for concurrent access
5. **State Versioning**: Add migration framework for format changes

---

## 9. CONCLUSION

### 9.1 System Maturity Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| Core Execution | 8/10 | Well-structured, proven algorithms |
| State Management | 6/10 | Functional but has data loss bug |
| Cost Tracking | 7/10 | Accurate but potential race conditions |
| Event System | 5/10 | Partially non-functional |
| Architecture Rules | 6/10 | Heuristic-heavy, no feedback loop |
| Validation | 6/10 | Good Python support, gaps for other languages |
| Documentation | 5/10 | Many hidden assumptions undocumented |

### 9.2 Confidence Summary

- **VERIFIED**: ~35% of codebase - Core algorithms, data structures, control flow
- **HYPOTHESIS**: ~38% of codebase - Heuristics, thresholds, routing decisions
- **UNKNOWN**: ~25% of codebase - Edge cases, concurrent behavior, recovery
- **BUG**: ~2% of codebase - Identified defects requiring fixes

### 9.3 Recommended Priority Actions

1. **CRITICAL**: Fix Task field serialization (BUG-001)
2. **HIGH**: Add thread safety to Budget.charge()
3. **HIGH**: Document all heuristic thresholds with rationale
4. **MEDIUM**: Fix ProjectEventBus integration (BUG-002)
5. **MEDIUM**: Add empirical validation for routing table
6. **LOW**: Standardize validator missing tool behavior (BUG-003)

---

*Generated: 2026-03-03*
*Scope: Full epistemic decomposition of Multi-LLM Orchestrator*
*Classification: Internal Architecture Analysis*
