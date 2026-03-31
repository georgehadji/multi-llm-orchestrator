# Architecture Overview

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Complete architecture documentation** for the AI Orchestrator system.

---

## System Overview

The AI Orchestrator is a multi-layered system that decomposes project specifications into atomic tasks, routes them to optimal LLM providers, and executes generate→critique→revise cycles until quality thresholds are met.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI ORCHESTRATOR                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   CLI        │    │   Dashboard  │    │   MCP Server │               │
│  │   Interface  │    │   Web UI     │    │   (External) │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             │                                            │
│                    ┌────────▼────────┐                                   │
│                    │  Control Plane  │                                   │
│                    │  (Orchestrator) │                                   │
│                    └────────┬────────┘                                   │
│                             │                                            │
│         ┌───────────────────┼───────────────────┐                        │
│         │                   │                   │                        │
│  ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐                 │
│  │  Decompose   │   │   Route      │   │   Execute    │                 │
│  │  Project     │   │   Tasks      │   │   Tasks      │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             │                                            │
│                    ┌────────▼────────┐                                   │
│                    │  Infrastructure │                                   │
│                    │  Layer          │                                   │
│                    └────────┬────────┘                                   │
│                             │                                            │
│         ┌───────────────────┼───────────────────┐                        │
│         │                   │                   │                        │
│  ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐                 │
│  │  LLM         │   │   Event      │   │   Storage    │                 │
│  │  Providers   │   │   Store      │   │   (SQLite)   │                 │
│  └──────────────┘   └──────────────┘   └──────────────┘                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Layer Architecture](#1-layer-architecture)
2. [Component Details](#2-component-details)
3. [Data Flow](#3-data-flow)
4. [Deployment Architecture](#4-deployment-architecture)
5. [Security Architecture](#5-security-architecture)

---

## 1. Layer Architecture

### L1: Interface Layer

The interface layer provides multiple entry points for users and external systems.

| Component | Description | Location |
|-----------|-------------|----------|
| **CLI** | Command-line interface | `orchestrator/cli.py` |
| **Dashboard** | Web UI (unified) | `orchestrator/dashboard_core/` |
| **MCP Server** | Model Context Protocol | `orchestrator/mcp_server.py` |
| **API Server** | REST API | `orchestrator/api_server.py` |

### L2: Control Plane

The control plane orchestrates all operations.

| Component | Description | Location |
|-----------|-------------|----------|
| **Orchestrator** | Main orchestration engine | `orchestrator/engine.py` |
| **Project Enhancer** | Spec improvement | `orchestrator/enhancer.py` |
| **Architecture Advisor** | Architecture selection | `orchestrator/architecture_advisor.py` |
| **Decomposer** | Task decomposition | `orchestrator/engine.py` |
| **Router** | Task routing | `orchestrator/adaptive_router.py` |

### L3: Execution Layer

The execution layer handles task execution.

| Component | Description | Location |
|-----------|-------------|----------|
| **Task Executor** | Execute individual tasks | `orchestrator/engine.py` |
| **Validator** | Run validators | `orchestrator/validators.py` |
| **ARA Pipeline** | Advanced reasoning | `orchestrator/ara_pipelines.py` |
| **Code Generator** | Generate code | `orchestrator/assembler.py` |

### L4: Infrastructure Layer

The infrastructure layer provides supporting services.

| Component | Description | Location |
|-----------|-------------|----------|
| **LLM Clients** | Provider APIs | `orchestrator/api_clients.py` |
| **Event Store** | Event persistence | `orchestrator/events.py` |
| **Telemetry** | Metrics collection | `orchestrator/telemetry.py` |
| **Cache** | Response caching | `orchestrator/caching.py` |
| **Database** | SQLite storage | `orchestrator/state.py` |

---

## 2. Component Details

### Orchestrator Engine

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator Engine                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Budget    │  │   Policy    │  │   Profiles  │          │
│  │   Manager   │  │   Enforcer  │  │   (LLMs)    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Task      │  │   Quality   │  │   Telemetry │          │
│  │   Queue     │  │   Evaluator │  │   Collector │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────────────────────────────────────────┐         │
│  │          Integrations                            │         │
│  │  • Nexus Search  • A2A Protocol  • MCP Server   │         │
│  │  • BM25 Search   • Reranker      • LangGraph    │         │
│  └─────────────────────────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### ARA Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    ARA Pipeline v2.0                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 0: Classification → Phase 1: Decomposition           │
│                              ↓                               │
│                    Context Vetting (RAG)                    │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Method Selection (Auto/Manual)            │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Standard Methods:                                   │    │
│  │  • Multi-Perspective  • Iterative     • Debate      │    │
│  │  • Research           • Jury          • Scientific  │    │
│  │  • Socratic                                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Specialized Methods:                                │    │
│  │  • Pre-Mortem  ⭐     • Bayesian      • Dialectical │    │
│  │  • Analogical         • Delphi                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                              ↓                               │
│                    Phase 5: Synthesis                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified Dashboard Core                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   KPI       │  │   Live      │  │   Mission   │          │
│  │   View      │  │   Metrics   │  │   Control   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Analytics │  │   Task      │  │   Model     │          │
│  │   View      │  │   Progress  │  │   Health    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              WebSocket Server                        │    │
│  │         (Real-time Updates via /ws)                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### Project Execution Flow

```
1. User Input (CLI/Dashboard/API)
         │
         ▼
2. Project Enhancer (improve spec)
         │
         ▼
3. Architecture Advisor (select architecture)
         │
         ▼
4. Decomposer (break into tasks)
         │
         ▼
5. Router (assign models)
         │
         ▼
6. Task Executor (generate→critique→revise)
         │
         ├──▶ Validator (syntax, tests, lint)
         │
         ▼
7. Quality Evaluator (score ≥ threshold?)
         │
         ├── No ──▶ Revise
         │
         ▼ Yes
8. Store Results (SQLite + Event Store)
         │
         ▼
9. Update Dashboard (WebSocket)
         │
         ▼
10. Complete Project
```

### Event Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Component  │────▶│  Event Bus   │────▶│  Handlers    │
│              │     │              │     │              │
│  Engine      │     │  Unified     │     │  Telemetry   │
│  Dashboard   │     │  Event       │     │  Store       │
│  Validators  │     │  Store       │     │  Dashboard   │
└──────────────┘     └──────────────┘     └──────────────┘

Event Types:
• ProjectStarted, ProjectCompleted
• TaskStarted, TaskCompleted, TaskFailed
• CapabilityUsed, CapabilityCompleted
• BudgetWarning, ModelSelected
• FallbackTriggered
```

---

## 4. Deployment Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      Single Node                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AI Orchestrator                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │   CLI   │  │Dashboard│  │   API   │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────┐         │    │
│  │  │         SQLite Database                  │         │    │
│  │  │  • Projects  • Tasks  • Events  • Cache │         │    │
│  │  └─────────────────────────────────────────┘         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Nexus Search (Docker)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Node Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
│                    (nginx/HAProxy)                           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Node 1      │   │   Node 2      │   │   Node 3      │
│               │   │               │   │               │
│  Orchestrator │   │  Orchestrator │   │  Orchestrator │
│  Dashboard    │   │  Dashboard    │   │  Dashboard    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Shared Storage │
                  │  (PostgreSQL)   │
                  └─────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │     Redis       │
                  │  (Cache/Queue)  │
                  └─────────────────┘
```

---

## 5. Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: Perimeter                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • API Authentication (JWT/OAuth)                   │    │
│  │  • Rate Limiting (TPM/RPM)                          │    │
│  │  • Input Validation                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  L2: Application                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Policy Enforcement (HARD/SOFT/MONITOR)           │    │
│  │  • Preflight Validation                             │    │
│  │  • Code Sandboxing                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  L3: Execution                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Process Isolation                                │    │
│  │  • seccomp-bpf Filtering                            │    │
│  │  • Landlock LSM                                     │    │
│  │  • Capability Restrictions                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  L4: Data                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Encryption at Rest                               │    │
│  │  • Secure Secret Management                         │    │
│  │  • Audit Logging                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Plugin Security

```
┌─────────────────────────────────────────────────────────────┐
│               Secure Plugin Runtime                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Plugin    │────▶│  Sandbox    │────▶│  Verified   │   │
│  │   Upload    │     │  Check      │     │  Execution  │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│                           │                     │            │
│                           ▼                     ▼            │
│                    ┌─────────────┐     ┌─────────────┐       │
│                    │  SHA256     │     │  seccomp    │       │
│                    │  Verify     │     │  Filter     │       │
│                    └─────────────┘     └─────────────┘       │
│                                                              │
│  Security Measures:                                          │
│  • seccomp-bpf: Block ptrace, execve, fork                  │
│  • Landlock: Restrict filesystem access                     │
│  • Capabilities: Drop CAP_SYS_ADMIN                         │
│  • Resource Limits: CPU, memory, file descriptors           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
E:\Documents\Vibe-Coding\Ai Orchestrator\
├── orchestrator/                    # Main package
│   ├── __init__.py                 # Package init, exports
│   ├── __main__.py                 # CLI entry point
│   ├── engine.py                   # Main orchestration engine
│   ├── cli.py                      # CLI interface
│   │
│   ├── domain/                     # Business logic
│   │   ├── models.py               # Data models
│   │   ├── budget.py               # Budget management
│   │   ├── policy.py               # Policy enforcement
│   │   └── events.py               # Event definitions
│   │
│   ├── infrastructure/             # External services
│   │   ├── api_clients.py          # LLM API clients
│   │   ├── telemetry.py            # Metrics collection
│   │   ├── caching.py              # Response caching
│   │   └── state.py                # State persistence
│   │
│   ├── capabilities/               # High-level features
│   │   ├── enhancer.py             # Project enhancement
│   │   ├── architecture_advisor.py # Architecture selection
│   │   ├── ara_pipelines.py        # Advanced reasoning
│   │   └── validators.py           # Code validation
│   │
│   ├── integrations/               # External integrations
│   │   ├── nexus_search/           # Web search
│   │   ├── mcp_server.py           # MCP protocol
│   │   ├── a2a_protocol.py         # Agent-to-agent
│   │   ├── bm25_search.py          # Full-text search
│   │   └── reranker.py             # LLM reranking
│   │
│   ├── dashboard_core/             # Unified dashboard
│   │   ├── core.py                 # Dashboard core
│   │   └── mission_control.py      # Mission Control view
│   │
│   └── management/                 # Management systems
│       ├── knowledge_base.py       # Knowledge management
│       ├── project_manager.py      # Project management
│       ├── product_manager.py      # Product management
│       └── quality_controller.py   # Quality control
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── e2e/                        # End-to-end tests
│
├── docs/                           # Documentation
│   ├── debugging/                  # Debug guides
│   └── performance/                # Performance guides
│
├── scripts/                        # Utility scripts
│   ├── batch/                      # Batch operations
│   ├── git/                        # Git utilities
│   ├── setup/                      # Setup scripts
│   └── utils/                      # General utilities
│
└── [Documentation Files]
    ├── README.md
    ├── USAGE_GUIDE.md
    ├── CAPABILITIES.md
    ├── METHODS.md
    ├── NEXUS_SEARCH_README.md
    ├── A2A_PROTOCOL_GUIDE.md
    ├── PREFLIGHT_SESSION_GUIDE.md
    ├── DASHBOARD_MIGRATION_GUIDE.md
    ├── MANAGEMENT_SYSTEMS_GUIDE.md
    ├── TOKEN_OPTIMIZER_GUIDE.md
    ├── INTEGRATIONS_COMPLETE.md
    └── ARCHITECTURE_OVERVIEW.md (this file)
```

---

## Related Documentation

- [README.md](./README.md) — Installation and quick start
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — CLI & Python API reference
- [CAPABILITIES.md](./CAPABILITIES.md) — Feature documentation
- [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md) — External integrations

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
