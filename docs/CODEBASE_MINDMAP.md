# Multi-LLM Orchestrator — Complete Architecture Mindmap

> **Version:** v6.3 (2026-04-04)  
> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Codebase Size:** ~43,000 lines (core) + ~5,400 lines (iOS Suite)  
> **Total Files:** 4,673 Python files

---

## 🎯 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Multi-LLM Orchestrator v6.3                          │
│                    Autonomous Software Development Platform                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    ┌────▼────┐               ┌────▼────┐               ┌────▼────┐
    │  INPUT   │               │ ENGINE  │               │ OUTPUT  │
    │ Layer    │◄─────────────►│  Core   │◄─────────────►│ Layer   │
    └────┬────┘               └────┬────┘               └────┬────┘
         │                          │                          │
    Project Desc              Pipeline Execution          Generated Code
    Success Criteria          Quality Assurance           Test Suite
    Budget Constraints        Cost Optimization           Documentation
```

---

## 🏗️ Core Architecture

### 1. Main Pipeline Flow

```
Project Description
       │
       ▼
┌─────────────────┐
│ Auto-Resume     │◄──────────────────┐
│ Detection       │                   │
└────────┬────────┘                   │
         │                            │
         ▼                            │
┌─────────────────┐                   │
│ Project         │                   │
│ Enhancer        │                   │
└────────┬────────┘                   │
         │                            │
         ▼                            │
┌─────────────────┐                   │
│ Architecture    │                   │
│ Advisor         │                   │
└────────┬────────┘                   │
         │                            │
         ▼                            │
┌─────────────────┐     ┌─────────────┴─────┐
│ Decomposition   │────►│ Checkpoint/Resume │
│ (Atomic Tasks)  │     │     (SQLite)      │
└────────┬────────┘     └───────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              TASK EXECUTION LOOP                 │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐        │
│  │ Route   │──► Generate │──► Critique │        │
│  │ Task    │  │   Code   │  │ Output   │        │
│  └────┬────┘  └────┬─────┘  └────┬────┘        │
│       │            │             │              │
│       │      ┌─────▼─────┐       │              │
│       │      │  Revise   │◄──────┘              │
│       │      │  (loop)   │                      │
│       │      └─────┬─────┘                      │
│       │            │                            │
│       ▼            ▼                            │
│  ┌─────────┐  ┌──────────┐                     │
│  │Evaluate │  │Deterministic                     │
│  │ Quality │  │Validation│                     │
│  └────┬────┘  └────┬─────┘                     │
│       │            │                            │
│       └────────────┘                            │
│              │                                  │
│              ▼                                  │
│  ┌─────────────────────────────────┐           │
│  │   Cross-Provider Fallback Chain │           │
│  │   (if quality < threshold)      │           │
│  └─────────────────────────────────┘           │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Store Results   │
│ + Telemetry     │
│ + State         │
└─────────────────┘
```

### 2. Module Dependency Graph

```
                              ┌─────────────────┐
                              │     CLI / API    │
                              │  (Entry Points)  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
           │   Engine    │    │  Dashboard  │    │   Policy    │
           │   Core      │    │   System    │    │   Engine    │
           └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                  │                  │                  │
    ┌─────────────┼─────────────┐    │    ┌─────────────┼─────────────┐
    │             │             │    │    │             │             │
    ▼             ▼             ▼    │    ▼             ▼             ▼
┌───────┐   ┌─────────┐   ┌────────┐│┌───┴────┐  ┌──────┴─────┐ ┌────┴────┐
│Models │   │  API    │   │ State  │││ Events │  │ Knowledge  │ │ Project │
│& Cost │   │ Clients │   │Manager │││ System │  │   Base     │ │ Manager │
└───┬───┘   └────┬────┘   └───┬────┘│└───┬────┘  └────────────┘ └─────────┘
    │            │            │     │    │
    │     ┌──────┴──────┐     │     │    │         ┌─────────────┐
    │     │             │     │     │    │         │   Product   │
    │     ▼             ▼     │     │    │         │   Manager   │
    │  ┌──────┐    ┌────────┐ │     │    │         └─────────────┘
    └──►Routing│    │Semantic│ │     │    │
       │Tables│    │ Cache  │ │     │    │         ┌─────────────┐
       └──────┘    └────────┘ │     │    └────────►│   Quality   │
                              │     │              │   Control   │
                              │     │              └─────────────┘
                              │     │
                              │     │              ┌─────────────┐
                              │     └─────────────►│   Nexus     │
                              │                    │   Search    │
                              │                    └─────────────┘
                              │
                              │     ┌───────────────────────────────┐
                              │     │      ARA Pipeline             │
                              │     │  (12 Reasoning Methods)       │
                              └────►│  - Multi-Perspective          │
                                    │  - Iterative                  │
                                    │  - Debate                     │
                                    │  - Jury                       │
                                    │  - Scientific                 │
                                    └───────────────────────────────┘
```

---

## 📊 Data Models & Types

### Core Data Structures

```python
# Task Types (ROUTING_TABLE keys)
TaskType
├── CODE_GEN          # Code generation tasks
├── CODE_REVIEW       # Code review tasks
├── REASONING         # Complex reasoning tasks
├── WRITING           # Creative writing
├── DATA_EXTRACT      # Data extraction
├── SUMMARIZE         # Summarization
└── EVALUATE          # Evaluation tasks

# Project States
ProjectStatus
├── SUCCESS
├── PARTIAL_SUCCESS
├── COMPLETED_DEGRADED
├── BUDGET_EXHAUSTED
├── TIMEOUT
└── SYSTEM_FAILURE

# Task States
TaskStatus
├── PENDING
├── RUNNING
├── COMPLETED
├── FAILED
└── DEGRADED
```

### Model Routing Architecture

```
ROUTING_TABLE (TaskType → Model Priority List)
═══════════════════════════════════════════════════════════════════

CODE_GEN:
  1. XIAOMI_MIMO_V2_FLASH  ($0.09/$0.29) ⭐ BEST
  2. QWEN_2_5_CODER_32B    ($0.66/$1.00)
  3. DEEPSEEK_V3_2         ($0.27/$1.10)
  4. MOONSHOT_KIMI_K2_5    ($0.42/$2.20)
  5. ZHIPU_GLM_4_7         ($0.39/$1.75)
  6. CLAUDE_SONNET_4_6     ($3/$15)

CODE_REVIEW:
  1. XAI_GROK_4_20         ($2/$6) ⭐ BEST
  2. DEEPSEEK_R1           ($0.55/$2.19)
  3. MOONSHOT_KIMI_K2_5    ($0.42/$2.20)

REASONING:
  1. STEPFUN_STEP_3_5_FLASH ($0.10/$0.30) ⭐ BEST VALUE
  2. DEEPSEEK_R1           ($0.55/$2.19)
  3. O3_MINI               ($1.10/$4.40)

EVALUATE:
  1. XAI_GROK_4_20         ($2/$6) ⭐ BEST
  2. DEEPSEEK_R1           ($0.55/$2.19)

COST_TABLE: Dict[Model, Dict["input"|"output", USD_per_1M_tokens]]
FALLBACK_CHAIN: Dict[Model, Model]  # Cross-provider resilience
MODEL_MAX_TOKENS: Dict[Model, int]  # Context window limits
```

---

## 🔄 Core Pipeline Workflows

### 1. Generate → Critique → Revise → Evaluate Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATION WORKFLOW                           │
│                    (max_iterations = 3)                         │
└─────────────────────────────────────────────────────────────────┘

Phase 1: GENERATE
┌────────────────────────────────────────┐
│ 1. Select optimal model via ROUTING    │
│ 2. Build context + dependencies        │
│ 3. Call LLM API                        │
│ 4. Parse response (code extraction)    │
│ 5. Validate JSON structure             │
└──────────────┬─────────────────────────┘
               │
               ▼
Phase 2: DETERMINISTIC VALIDATION
┌────────────────────────────────────────┐
│ • validate_python_syntax()             │
│ • validate_pytest()                    │
│ • validate_ruff()                      │
│ • validate_json_schema()               │
│ • Security checks (bandit)             │
└──────────────┬─────────────────────────┘
               │
               ▼ (if passed)
Phase 3: CRITIQUE
┌────────────────────────────────────────┐
│ 1. Cross-model review (different LLM)  │
│ 2. Evaluate: correctness, style, tests │
│ 3. Score: 0.0 - 1.0                    │
│ 4. Generate critique feedback          │
└──────────────┬─────────────────────────┘
               │
               ▼
Phase 4: EVALUATE
┌────────────────────────────────────────┐
│ Score >= threshold (0.85)?             │
│ ├── YES → Mark COMPLETED               │
│ └── NO  → Continue to REVISE           │
└──────────────┬─────────────────────────┘
               │
               ▼ (if below threshold)
Phase 5: REVISE
┌────────────────────────────────────────┐
│ 1. Generate delta prompt               │
│ 2. Include critique feedback           │
│ 3. Loop back to GENERATE               │
│ 4. Increment iteration counter         │
└────────────────────────────────────────┘

CIRCUIT BREAKER: Model marked unhealthy after 3 consecutive failures
```

### 2. Cross-Provider Fallback Chain

```
Task Failure / Low Quality
           │
           ▼
┌──────────────────────────┐
│ 1. Try next model in     │
│    ROUTING_TABLE         │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 2. Check FALLBACK_CHAIN  │
│    (cross-provider)      │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 3. Escalate tier:        │
│    CHEAP → BALANCED →    │
│    PREMIUM               │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 4. If all fail:          │
│    Mark DEGRADED         │
│    Record attempt history│
└──────────────────────────┘
```

---

## 🧠 ARA Pipeline (Advanced Reasoning & Analysis)

### 12 Reasoning Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARA PIPELINE v2.0                            │
│         12 Cognitive Science-Based Reasoning Methods            │
└─────────────────────────────────────────────────────────────────┘

STANDARD METHODS:
┌────────────────────┬───────────────────────────────────────────┐
│ Method             │ Strategy                                  │
├────────────────────┼───────────────────────────────────────────┤
│ Multi-Perspective  │ 4 perspectives: constructive, destructive,│
│                    │ systemic, minimalist                      │
├────────────────────┼───────────────────────────────────────────┤
│ Iterative          │ Progressive refinement, step-by-step      │
├────────────────────┼───────────────────────────────────────────┤
│ Debate             │ 2+ agents argue, meta-evaluator decides   │
├────────────────────┼───────────────────────────────────────────┤
│ Research           │ Web discovery + LLM synthesis             │
├────────────────────┼───────────────────────────────────────────┤
│ Jury               │ 4 generators, 3 critics, meta-evaluation  │
├────────────────────┼───────────────────────────────────────────┤
│ Scientific         │ Hypothesis testing: formulate→test→refine │
├────────────────────┼───────────────────────────────────────────┤
│ Socratic           │ Probing questions uncover assumptions     │
└────────────────────┴───────────────────────────────────────────┘

SPECIALIZED METHODS:
┌────────────────────┬───────────────────────────────────────────┐
│ Method             │ Strategy                                  │
├────────────────────┼───────────────────────────────────────────┤
│ Pre-Mortem         │ Imagine failure, work backward to prevent │
├────────────────────┼───────────────────────────────────────────┤
│ Bayesian           │ Prior→evidence→posterior probability      │
├────────────────────┼───────────────────────────────────────────┤
│ Dialectical        │ Thesis→antithesis→synthesis               │
├────────────────────┼───────────────────────────────────────────┤
│ Analogical         │ Map solutions from unrelated domains      │
├────────────────────┼───────────────────────────────────────────┤
│ Delphi             │ Iterative expert consensus with feedback  │
└────────────────────┴───────────────────────────────────────────┘

SELECTION LOGIC:
┌─────────────────────────────────────────────────┐
│ Task complexity + Risk + Budget + Language     │
│                    │                            │
│                    ▼                            │
│            Route to Optimal                     │
│            Reasoning Method                     │
└─────────────────────────────────────────────────┘
```

---

## 🔍 Nexus Search Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEXUS SEARCH SYSTEM                          │
│              Self-Hosted Web Search Integration                 │
└─────────────────────────────────────────────────────────────────┘

ARCHITECTURE:
┌─────────────────┐
│ Search Query    │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│ Query           │────►│ Query Expansion │
│ Classification  │     │ (LLM-based)     │
└─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│      HYBRID SEARCH (BM25 + Vector)      │
│         Reciprocal Rank Fusion          │
│              (k=60)                     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Parallel Search Sources:                │
│ • Web (General)                         │
│ • Tech (Documentation)                  │
│ • News (Current events)                 │
│ • Social (Discussions)                  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Optimization Layer:                     │
│ • Result Deduplication                  │
│ • Query Caching (TTL-based)             │
│ • Adaptive Search Depth                 │
│ • Circuit Breaker                       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ LLM Reranking                           │
│ (Relevance Scoring)                     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Research Report Generation              │
└─────────────────────────────────────────┘

AGENT COMPONENTS:
┌─────────────────┐  ┌─────────────────┐
│ QueryClassifier │  │ ResearchAgent   │
│                 │  │                 │
│ • classify()    │  │ • research()    │
│ • get_recommended│  │ • synthesize()  │
│   _sources()    │  │ • report()      │
└─────────────────┘  └─────────────────┘
```

---

## 🏢 Management Systems

### 1. Knowledge Management System

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE                               │
│              Centralized Learning Repository                    │
└─────────────────────────────────────────────────────────────────┘

Data Model:
┌───────────────────┐
│ KnowledgeArtifact │
├───────────────────┤
│ id: str           │
│ type: KnowledgeType│
│ title: str        │
│ content: str      │
│ context: dict     │
│ tags: list[str]   │
│ embedding: list   │
│ similarity_score  │
│ usage_count       │
└───────────────────┘

KnowledgeType Enum:
├── CODE_SNIPPET
├── SOLUTION
├── BUGFIX
├── PATTERN
├── ARCHITECTURE
├── DECISION
└── LESSON_LEARNED

Features:
┌─────────────────────────────────────────┐
│ • Vector storage (sentence-transformers)│
│ • Knowledge graph (concept relations)  │
│ • Pattern matching                      │
│ • Auto-generated documentation         │
│ • LRU cache for queries                │
│ • Async indexing                        │
└─────────────────────────────────────────┘
```

### 2. Project Management System

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT MANAGER                              │
│          Task Scheduling & Resource Allocation                  │
└─────────────────────────────────────────────────────────────────┘

Core Classes:
┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐
│   TaskSchedule    │  │    Milestone      │  │      Risk       │
├───────────────────┤  ├───────────────────┤  ├─────────────────┤
│ task_id: str      │  │ id: str           │  │ id: str         │
│ start_time: datetime│ name: str          │  │ probability: float
│ end_time: datetime │  │ deadline: datetime│ impact: float    │
│ resources: list   │  │ completed: bool   │  │ risk_score()    │
│ is_critical: bool │  │ completion_date   │  │ mitigation: str │
│ slack: timedelta  │  └───────────────────┘  └─────────────────┘
└───────────────────┘

Algorithms:
┌─────────────────────────────────────────┐
│ CriticalPathAnalyzer                    │
│ • Forward/backward pass algorithm       │
│ • Earliest/latest start calculation     │
│ • Float time computation                │
├─────────────────────────────────────────┤
│ ResourceOptimizer                       │
│ • Constraint-based allocation           │
│ • Cost optimization                     │
│ • Load balancing                        │
└─────────────────────────────────────────┘
```

### 3. Product Management System

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCT MANAGER                              │
│        Feature Prioritization & Release Planning                │
└─────────────────────────────────────────────────────────────────┘

Core Classes:
┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐
│     Feature       │  │    RICEScore      │  │     Release     │
├───────────────────┤  ├───────────────────┤  ├─────────────────┤
│ id: str           │  │ reach: int        │  │ id: str         │
│ name: str         │  │ impact: int       │  │ version: str    │
│ status: FeatureStatus│ confidence: int   │  │ target_date     │
│ priority: P0-P3   │  │ effort: int       │  │ features: list  │
│ rice_score        │  │ score()           │  │ status          │
│ tags: list        │  └───────────────────┘  └─────────────────┘
└───────────────────┘

RICE Scoring:
  score = (Reach × Impact × Confidence) / Effort

FeatureStatus Lifecycle:
  IDEA → RESEARCH → PLANNED → IN_PROGRESS → BETA → RELEASED → DEPRECATED
```

### 4. Quality Control System

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY CONTROLLER                           │
│          Automated Testing & Quality Analysis                   │
└─────────────────────────────────────────────────────────────────┘

Test Levels:
├── UNIT           # Function/class tests
├── INTEGRATION    # Multi-component tests
├── E2E            # End-to-end tests
├── PERFORMANCE    # Load/stress tests
└── SECURITY       # Security scans

Quality Gates:
┌─────────────────────────────────────────┐
│ CodeMetrics                             │
│ • Cyclomatic complexity                 │
│ • Maintainability index                 │
│ • Duplication percentage                │
│ • Documentation coverage                │
│ • Type hint coverage                    │
├─────────────────────────────────────────┤
│ QualityIssue                            │
│ • severity: CRITICAL/HIGH/MEDIUM/LOW   │
│ • suggested_fix                         │
│ • location (file:line:column)          │
├─────────────────────────────────────────┤
│ QualityReport                           │
│ • passed: bool                          │
│ • average_coverage                      │
│ • test_results                          │
│ • issues                                │
└─────────────────────────────────────────┘
```

---

## 🍎 iOS App Store Compliance Suite

```
┌─────────────────────────────────────────────────────────────────┐
│              iOS APP STORE COMPLIANCE SUITE (v6.3)              │
│                    6 Major Enhancements                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Enhancement A│ │Enhancement B│ │Enhancement C│ │Enhancement D│
│ Multi-Platform│ App Store   │ │ iOS HIG     │ │ App Store   │
│ Generator   │ │ Validator   │ │ Prompts     │ │ Assets Gen  │
│ (9 platforms)│ (30 checks) │ │ (HIG comp)  │ │ (Auto)      │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │               │               │               │
       └───────────────┴───────────────┴───────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│Enhancement E│     │Enhancement F│     │   TOTAL     │
│ Native      │     │ Pre-Submit  │     │             │
│ Templates   │     │ Testing     │     │ 5,400+ lines│
│ (10 types)  │     │ (10 checks) │     │ 135 tests   │
└─────────────┘     └─────────────┘     └─────────────┘

VALIDATION CHECKS:
┌─────────────────────────────────────────┐
│ Performance (IOS-2.1)                   │
│ • Launch time < 20s                     │
│ • Memory usage < 5x base                │
│ • No memory leaks                       │
│ • Efficient resource use                │
├─────────────────────────────────────────┤
│ Metadata (IOS-2.5.2)                    │
│ • Accurate descriptions                 │
│ • Appropriate keywords                  │
│ • Proper categorization                 │
├─────────────────────────────────────────┤
│ Functionality (IOS-4.2)                 │
│ • Beta testing compliance               │
│ • Native iOS features (2+)              │
│ • No placeholder content                │
│ • Stable performance                    │
├─────────────────────────────────────────┤
│ Legal (IOS-5.1)                         │
│ • Info.plist presence                   │
│ • Required declarations                 │
│ • Privacy policy                        │
├─────────────────────────────────────────┤
│ HIG Compliance                          │
│ • iOS-standard controls                 │
│ • Accessibility labels                  │
│ • Dark mode support                     │
├─────────────────────────────────────────┤
│ AI Transparency                         │
│ • AI-generated content disclosed        │
│ • No misleading claims                  │
└─────────────────────────────────────────┘
```

---

## 💰 Cost Optimization System

```
┌─────────────────────────────────────────────────────────────────┐
│                    COST OPTIMIZATION v6.1                       │
│                     35% Cost Reduction                          │
└─────────────────────────────────────────────────────────────────┘

TIER 1: Provider-Level (80-90% input cost reduction)
┌─────────────────────────────────────────┐
│ • Prompt Caching (repeated patterns)    │
│ • Batch API (parallel requests)         │
│ • Token Budget Management               │
└─────────────────────────────────────────┘

TIER 2: Architectural (40-60% per-task reduction)
┌─────────────────────────────────────────┐
│ • Model Cascading (CHEAP→BALANCED→PREMIUM)│
│ • Dependency Context Injection          │
│ • Speculative Generation                │
│ • Streaming Validation                  │
└─────────────────────────────────────────┘

TIER 3: Quality (30-50% fewer repair cycles)
┌─────────────────────────────────────────┐
│ • Tier-3 Quality Checks                 │
│ • Confidence-Based Early Exit           │
│ • Fast Regression Detection (EMA α=0.2) │
│ • Semantic Sub-Result Caching           │
└─────────────────────────────────────────┘

TIER 4: DevOps (Security + DX)
┌─────────────────────────────────────────┐
│ • Docker Sandboxing                     │
│ • GitHub Auto-Push                      │
│ • Tool Safety Validation                │
└─────────────────────────────────────────┘

COST TRACKING:
┌─────────────────────────────────────────┐
│ EMA-based pricing (exponential moving)  │
│ Budget hierarchy: org → team → job      │
│ Real-time cost telemetry                │
│ Budget alerts at 80% threshold          │
└─────────────────────────────────────────┘
```

---

## 🛡️ Resilience & Fault Tolerance

```
┌─────────────────────────────────────────────────────────────────┐
│              BLACK SWAN RESILIENCE v6.0                         │
│                    99.85% Risk Reduction                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ Resilient Event     │ │ Secure Plugin       │ │ Streaming           │
│ Store               │ │ Runtime             │ │ Backpressure        │
│                     │ │                     │ │                     │
│ • WAL + Replication │ │ • seccomp           │ │ • Bounded queues    │
│ • Checksums         │ │ • Landlock          │ │ • Circuit breaker   │
│ • Async aiosqlite   │ │ • Capabilities      │ │ • Rate limiting     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   CIRCUIT BREAKER       │
                    │   (3-strike rule)       │
                    │                         │
                    │ Model marked unhealthy  │
                    │ after 3 failures        │
                    │ Auto-recovery           │
                    └─────────────────────────┘

STATE PERSISTENCE:
┌─────────────────────────────────────────┐
│ SQLite-backed (aiosqlite)              │
│ • Project state                         │
│ • Budget tracking                       │
│ • Task results                          │
│ • Resume capability                     │
│ JSON serialization (safe, readable)    │
└─────────────────────────────────────────┘
```

---

## 📡 Event System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED EVENTS SYSTEM                        │
│              Event Sourcing & Telemetry                         │
└─────────────────────────────────────────────────────────────────┘

Event Types:
┌─────────────────────────────────────────┐
│ Project Lifecycle                       │
│ • PROJECT_STARTED                       │
│ • PROJECT_COMPLETED                     │
│ • PROJECT_FAILED                        │
├─────────────────────────────────────────┤
│ Task Lifecycle                          │
│ • TASK_CREATED                          │
│ • TASK_STARTED                          │
│ • TASK_PROGRESS                         │
│ • TASK_COMPLETED                        │
│ • TASK_FAILED                           │
│ • TASK_RETRY                            │
├─────────────────────────────────────────┤
│ Model/Routing                           │
│ • MODEL_SELECTED                        │
│ • MODEL_UNAVAILABLE                     │
│ • FALLBACK_TRIGGERED                    │
│ • CIRCUIT_BREAKER_OPEN                  │
├─────────────────────────────────────────┤
│ Quality & Validation                    │
│ • VALIDATION_PASSED                     │
│ • VALIDATION_FAILED                     │
│ • QUALITY_GATE_PASSED                   │
├─────────────────────────────────────────┤
│ Budget & Cost                           │
│ • BUDGET_WARNING                        │
│ • BUDGET_EXHAUSTED                      │
│ • COST_RECORDED                         │
├─────────────────────────────────────────┤
│ Capability Usage                        │
│ • CAPABILITY_USED                       │
│ • CAPABILITY_COMPLETED                  │
│ • CAPABILITY_FAILED                     │
└─────────────────────────────────────────┘

Architecture:
DomainEvent (base)
├── ProjectStartedEvent
├── ProjectCompletedEvent
├── TaskStartedEvent
├── TaskCompletedEvent
└── ... (typed events)

Features:
┌─────────────────────────────────────────┐
│ • Immutable, serializable              │
│ • Automatic projections (read models)  │
│ • Event persistence (SQLite)           │
│ • ContextVar for current event         │
│ • Async event bus                      │
└─────────────────────────────────────────┘
```

---

## 🎮 Dashboard & Monitoring

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE DASHBOARD v4.0                          │
│              Real-time WebSocket Streaming                      │
└─────────────────────────────────────────────────────────────────┘

Features:
┌─────────────────────────────────────────┐
│ • WebSocket (no polling!)              │
│ • Toast notifications                  │
│ • Gamification system                  │
│ • Confetti celebrations                │
│ • Sound notifications                  │
│ • Live task progress                   │
│ • Test execution monitoring            │
└─────────────────────────────────────────┘

Gamification:
┌─────────────────────────────────────────┐
│ DashboardState                          │
│ • level: int                            │
│ • xp: int                               │
│ • xp_to_next_level: int                 │
│ • streak: int                           │
│ • achievements: list                    │
├─────────────────────────────────────────┤
│ Achievements                            │
│ • first_task                            │
│ • streak_7                              │
│ • quality_master                        │
│ • cost_optimizer                        │
│ ...                                     │
└─────────────────────────────────────────┘

Dashboard Types:
├── dashboard_live.py        # Gamified WebSocket
├── dashboard_mission_control.py  # Professional telemetry
├── dashboard_antd.py        # Ant Design UI
└── cli_dashboard.py         # Terminal UI
```

---

## ✅ Validation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC VALIDATORS                     │
│              Non-Negotiable Quality Gates                       │
└─────────────────────────────────────────────────────────────────┘

Validators:
┌─────────────────────────────────────────┐
│ validate_python_syntax()                │
│ • Compilation check                     │
│ • Indentation error handling            │
│ • Truncation detection                  │
├─────────────────────────────────────────┤
│ validate_pytest()                       │
│ • Test execution                        │
│ • Coverage reporting                    │
│ • Module availability check             │
├─────────────────────────────────────────┤
│ validate_ruff()                         │
│ • Linting                               │
│ • Style checking                        │
│ • Auto-fix support                      │
├─────────────────────────────────────────┤
│ validate_json_schema()                  │
│ • JSON validation                       │
│ • Schema compliance                     │
├─────────────────────────────────────────┤
│ Security Scans                          │
│ • Bandit (Python security)              │
│ • Safety (dependency vulnerabilities)   │
└─────────────────────────────────────────┘

Execution:
┌─────────────────────────────────────────┐
│ async_run_validators()                  │
│ • Offloads to threads                   │
│ • Non-blocking I/O                      │
│ • Parallel execution                    │
└─────────────────────────────────────────┘

Rule: If deterministic check fails → score = 0.0 (overrides LLM)
```

---

## 📦 Project Structure

```
Ai Orchestrator/
├── orchestrator/                 # Main package (~276 files)
│   ├── __init__.py              # Lazy-loading entry point
│   ├── cli.py                   # CLI entry point
│   ├── engine.py                # Core orchestration (4,732 lines)
│   ├── models.py                # Data models & routing (719 lines)
│   ├── ara_pipelines.py         # 12 reasoning methods (2,974 lines)
│   ├── api_clients.py           # Unified LLM client
│   ├── budget.py                # Budget management
│   ├── state.py                 # SQLite persistence
│   ├── validators.py            # Deterministic validation
│   ├── cache.py                 # Disk-based caching
│   ├── semantic_cache.py        # Semantic similarity
│   ├── model_selector.py        # Intelligent routing
│   │
│   ├── knowledge_base.py        # Knowledge Management
│   ├── project_manager.py       # Project Management
│   ├── product_manager.py       # Product Management
│   ├── quality_control.py       # Quality Control
│   │
│   ├── nexus_search/            # Web Search (21 files)
│   │   ├── core.py              # Search orchestrator
│   │   ├── agents/              # Search agents
│   │   ├── optimization/        # Query expansion, caching
│   │   └── providers/           # Search providers
│   │
│   ├── cost_optimization/       # Cost optimization (13 files)
│   │   ├── batch_client.py
│   │   ├── model_cascading.py
│   │   ├── prompt_cache.py
│   │   └── ...
│   │
│   ├── engine_core/             # Core engine components
│   │   ├── core.py
│   │   ├── task_executor.py
│   │   ├── critique_cycle.py
│   │   └── fallback_handler.py
│   │
│   ├── dashboard_core/          # Dashboard infrastructure
│   ├── unified_events/          # Event system
│   ├── scaffold/                # Project templates
│   ├── plugins/                 # Plugin system
│   └── ide_backend/             # IDE integration
│
├── tests/                       # Test suite (232 files)
│   ├── conftest.py             # pytest configuration
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── docs/                        # Documentation (75+ files)
├── scripts/                     # Utility scripts
├── pyproject.toml              # Package configuration
└── requirements.txt            # Dependencies
```

---

## 🔌 Provider Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM PROVIDER SUPPORT                         │
│                    6+ Providers via OpenRouter                  │
└─────────────────────────────────────────────────────────────────┘

UNIFIED API (OpenRouter):
┌─────────────────────────────────────────┐
│ api_clients.py                          │
│ • UnifiedClient                         │
│ • Async HTTP with aiohttp               │
│ • Retry logic                           │
│ • Rate limiting                         │
└─────────────────────────────────────────┘

Supported Providers:
┌────────────────┬──────────────────────────────────────────┐
│ Provider       │ Models                                   │
├────────────────┼──────────────────────────────────────────┤
│ OpenAI         │ GPT-4o, GPT-5, o1, o3-mini, o4-mini      │
│ Google         │ Gemini Pro, Flash, Flash Lite            │
│ Anthropic      │ Claude 3.5 Sonnet, Opus, Haiku           │
│ DeepSeek       │ DeepSeek Chat, Reasoner (R1)             │
│ Meta           │ LLaMA 4 Maverick/Scout, LLaMA 3.3        │
│ xAI            │ Grok 4.20, Grok 4.1 Fast                 │
│ Qwen           │ Qwen 2.5 Coder                           │
│ MiniMax        │ MiniMax M2.7, M2.5                       │
│ StepFun        │ Step 3.5 Flash                           │
│ Z.AI           │ GLM 4.7 Flash                            │
│ Moonshot       │ Kimi K2.5                                │
│ Xiaomi         │ MiMo-V2 Flash/Pro/Omni                   │
└────────────────┴──────────────────────────────────────────┘

Cost Optimization:
┌─────────────────────────────────────────┐
│ • EMA-based price tracking              │
│ • Cost-performance profiling            │
│ • Budget hierarchy enforcement          │
│ • Cross-provider fallback               │
└─────────────────────────────────────────┘
```

---

## 🧪 Testing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TESTING FRAMEWORK                            │
│                    159+ Tests, ~93% Coverage (iOS)              │
└─────────────────────────────────────────────────────────────────┘

Test Organization:
┌─────────────────────────────────────────┐
│ tests/                                  │
│ ├── conftest.py                        │
│ │   • Markers: unit, integration, slow │
│ │   • Markers: requires_api, e2e       │
│ │   • Parallel execution config        │
│ │                                        │
│ ├── unit/                              │
│ │   • Component isolation              │
│ │   • Mocked dependencies              │
│ │                                        │
│ ├── integration/                       │
│ │   • Multi-component tests            │
│ │   • Database interactions            │
│ │                                        │
│ └── orchestrator/                      │
│     • Module-specific tests            │
└─────────────────────────────────────────┘

Test Markers:
┌─────────────────────────────────────────┐
│ @pytest.mark.unit                      │
│ @pytest.mark.integration               │
│ @pytest.mark.slow                      │
│ @pytest.mark.requires_api              │
│ @pytest.mark.e2e                       │
│ @pytest.mark.load                      │
│ @pytest.mark.stress                    │
│ @pytest.mark.benchmark                 │
└─────────────────────────────────────────┘

Coverage:
┌─────────────────────────────────────────┐
│ Core: ~13% baseline                     │
│ iOS Suite: ~93%                         │
│ Target: Ratcheted (never decrease)     │
└─────────────────────────────────────────┘
```

---

## 🔄 Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW MAP                        │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Input Processing
┌─────────────────────────────────────────┐
│ 1. Parse CLI arguments                  │
│ 2. Load environment (.env)              │
│ 3. Check for resume candidates          │
│ 4. Initialize budget                    │
│ 5. Create project state                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
Phase 2: Enhancement
┌─────────────────────────────────────────┐
│ 1. Project Enhancer (LLM)               │
│ 2. Architecture Advisor                 │
│ 3. Platform detection                   │
│ 4. Template selection                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
Phase 3: Decomposition
┌─────────────────────────────────────────┐
│ 1. Break into atomic tasks              │
│ 2. Detect dependencies                  │
│ 3. Build dependency graph               │
│ 4. Topological sort                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
Phase 4: Execution (per task)
┌─────────────────────────────────────────┐
│ While tasks remaining:                  │
│   1. Get ready tasks (deps satisfied)  │
│   2. Route to optimal model            │
│   3. Execute generate→critique→revise  │
│   4. Run deterministic validators      │
│   5. Evaluate quality                  │
│   6. If degraded, trigger fallback     │
│   7. Store result                      │
│   8. Emit events                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
Phase 5: Finalization
┌─────────────────────────────────────────┐
│ 1. Calculate final score                │
│ 2. Generate quality report              │
│ 3. Organize outputs                     │
│ 4. Cleanup state                        │
│ 5. Emit PROJECT_COMPLETED event        │
└─────────────────────────────────────────┘
```

---

## 📊 Key Metrics & Statistics

| Metric | Value |
|--------|-------|
| **Version** | v6.3 (2026-03-25) |
| **Core Code** | ~43,000 lines |
| **iOS Suite** | ~5,400 lines |
| **Total Files** | 4,673 Python files |
| **Test Files** | 159+ |
| **Test Coverage** | ~93% (iOS), 13% (core) |
| **Cost Reduction** | 35% |
| **Providers** | 6+ (via OpenRouter) |
| **Reasoning Methods** | 12 (ARA Pipeline) |
| **Validation Checks** | 30+ (iOS compliance) |
| **Risk Reduction** | 99.85% |

---

## 🔗 Key Connections & Dependencies

```
External Dependencies:
┌─────────────────────────────────────────┐
│ LLM APIs                                │
│ • OpenRouter (unified)                  │
│ • Provider SDKs (openai, anthropic,    │
│   google-genai)                         │
├─────────────────────────────────────────┤
│ Database                                │
│ • aiosqlite (async SQLite)              │
│ • Redis (optional, caching)             │
├─────────────────────────────────────────┤
│ Web Framework                           │
│ • FastAPI (dashboard)                   │
│ • WebSockets (real-time)                │
│ • Uvicorn (ASGI server)                 │
├─────────────────────────────────────────┤
│ Testing                                 │
│ • pytest + pytest-asyncio               │
│ • pytest-cov (coverage)                 │
│ • pytest-xdist (parallel)               │
├─────────────────────────────────────────┤
│ Code Quality                            │
│ • ruff (linting)                        │
│ • black (formatting)                    │
│ • mypy (type checking)                  │
│ • bandit (security)                     │
├─────────────────────────────────────────┤
│ Observability                           │
│ • OpenTelemetry (tracing)               │
│ • Structured logging                    │
└─────────────────────────────────────────┘

Internal Dependencies:
┌─────────────────────────────────────────┐
│ Engine → Models, API Clients, State     │
│ ARA → Models, API Clients, Cache        │
│ Nexus → Agents, Optimization, Providers │
│ Management → Performance, Log Config    │
│ Dashboard → Events, State               │
│ Validators → Models, Cache              │
└─────────────────────────────────────────┘
```

---

*Last Updated: 2026-04-04*  
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
