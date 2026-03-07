# SYSTEM KNOWLEDGE MAP

**Project**: AI Orchestrator v6.0  
**Date**: 2026-03-07  
**Classification**: VERIFIED / INFERRED / UNKNOWN with confidence scores (0-1)

---

## LEGEND

| Classification | Meaning | Confidence | Verification Method |
|----------------|---------|------------|---------------------|
| **[VERIFIED]** | Direct code trace | 0.90 - 1.0 | File read, import test, syntax check |
| **[INFERRED]** | Pattern-derived | 0.60 - 0.89 | Code structure analysis, naming conventions |
| **[UNKNOWN]** | Insufficient visibility | 0.0 - 0.59 | No direct evidence, speculation only |

---

## MODULE INVENTORY

### Core Orchestration (Critical Path)

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `engine.py` | 3,077 | Task execution loop | Tasks, Budget, Policy | TaskResult, ProjectState | results: dict, _background_tasks: set | UnifiedClient, StateManager, DiskCache | [VERIFIED] | 0.95 |
| `state.py` | 470 | State persistence | ProjectState | JSON → SQLite | SQLite connection | aiosqlite | [VERIFIED] | 0.95 |
| `models.py` | 851 | Data structures | N/A | Task, Budget, Model, etc. | Enums, dataclasses | pydantic | [VERIFIED] | 0.98 |
| `api_clients.py` | ~1,200 | LLM API unification | Prompt, model, params | APIResponse | Retry state, rate limits | openai, google-genai, anthropic | [VERIFIED] | 0.90 |
| `cache.py` | ~400 | Disk caching | Key, value | Cached value | SQLite cache DB | aiosqlite | [VERIFIED] | 0.85 |
| `validators.py` | 519 | Deterministic validation | Output, rules | ValidationResult | N/A | ast, subprocess | [VERIFIED] | 0.90 |

---

### Memory & Search

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `memory_tier.py` | 561 | HOT/WARM/COLD hierarchy | MemoryEntry | Retrieved memories | Tier files (JSONL) | bm25_search | [VERIFIED] | 0.90 |
| `bm25_search.py` | 445 | FTS5 full-text search | Query, project_id | SearchResult list | SQLite FTS5 index | sqlite3 | [VERIFIED] | 0.90 |
| `reranker.py` | ~300 | LLM re-ranking | Query, results | Ranked results | N/A | LLM client | [VERIFIED] | 0.85 |
| `semantic_cache.py` | ~500 | Semantic similarity cache | Query, content | Cache hit/miss | Similarity index | N/A | [VERIFIED] | 0.80 |
| `knowledge_graph.py` | ~800 | Performance knowledge graph | Telemetry data | Paths, similarities | Graph DB (SQLite) | sqlite3 | [INFERRED] | 0.70 |
| `knowledge_base.py` | ~400 | Artifact storage | KnowledgeArtifact | Retrieved artifacts | JSON files | pathlib | [INFERRED] | 0.65 |

---

### Policy & Control

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `policy.py` | ~600 | Policy definitions | Config | ModelProfile, PolicySet | N/A | pydantic | [VERIFIED] | 0.90 |
| `policy_engine.py` | ~500 | Policy enforcement | Policy, context | Validation result | N/A | policy.py | [INFERRED] | 0.75 |
| `planner.py` | ~400 | Constraint planning | Constraints | Plan | Optimization backend | optimization.py | [INFERRED] | 0.70 |
| `control_plane.py` | ~1,000 | Workflow enforcement | JobSpec, SLAs | RoutingPlan, MonitorResult | Decision state | policy_engine, monitoring | [INFERRED] | 0.65 |
| `optimization.py` | ~300 | Optimization backends | Objectives, weights | Optimal selection | N/A | N/A | [VERIFIED] | 0.80 |

---

### Agent Communication

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `a2a_protocol.py` | 560 | Agent-to-agent protocol | TaskSendRequest, A2AMessage | TaskResult | _agents, _tasks, _message_queues, _pending_responses, _response_timeouts | asyncio.Queue | [VERIFIED] | 0.95 |
| `agent_safety.py` | ~500 | Cross-agent safety | SafetyEvent | SafetyDecision | _profiles, _interaction_log | accountability.py | [INFERRED] | 0.70 |
| `agents.py` | ~200 | Task channels | Messages | Queued messages | TaskChannel queues | asyncio.Queue | [VERIFIED] | 0.85 |

---

### Security & Accountability

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `accountability.py` | 493 | Action attribution | Actor, action, target | ActionID, ImpactRecord | _actions, _impacts, _delegations | uuid, json | [VERIFIED] | 0.85 |
| `task_verifier.py` | 376 | Completion verification | Expected, actual | VerificationResult | _expected_outcomes | N/A | [VERIFIED] | 0.85 |
| `red_team.py` | ~400 | Attack simulation | AttackScenario | VulnerabilityReport | _scenarios, _results | N/A | [INFERRED] | 0.65 |
| `plugin_isolation.py` | ~300 | Plugin sandboxing | Plugin code | Execution result | Process handles | multiprocessing, seccomp | [INFERRED] | 0.60 |
| `secrets_manager.py` | ~200 | Secret handling | Secret requests | Decrypted secrets | Encrypted store | cryptography | [UNKNOWN] | 0.45 |

---

### Observability

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `telemetry_store.py` | 750 | Cross-run learning | ModelProfile, events | Historical profiles | SQLite tables | aiosqlite | [VERIFIED] | 0.85 |
| `telemetry.py` | 219 | Per-model telemetry | Model, metrics | TelemetryRecord | In-memory buffers | N/A | [VERIFIED] | 0.85 |
| `unified_events.py` | ~700 | Event bus | DomainEvent | Handler results | Event store (SQLite) | aiosqlite | [VERIFIED] | 0.90 |
| `audit.py` | ~300 | Audit logging | AuditRecord | Immutable log | Audit file | json, pathlib | [VERIFIED] | 0.85 |
| `metrics.py` | ~200 | Metrics export | Metrics | Prometheus/JSON/Console | N/A | N/A | [INFERRED] | 0.70 |
| `tracing.py` | 146 | Distributed tracing | Spans | OTLP export | Tracer context | opentelemetry | [INFERRED] | 0.65 |
| `hooks.py` | ~200 | Event hooks | EventType, callback | Callback results | Hook registry | N/A | [VERIFIED] | 0.85 |

---

### Optimization & Routing

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `adaptive_router.py` | ~400 | Circuit breaker routing | Model health, task | Selected model | _models: dict[ModelState] | telemetry_store | [VERIFIED] | 0.90 |
| `cache_optimizer.py` | ~700 | L1/L2/L3 cache | Key, value | Cached data | L1: dict, L2: SQLite, L3: semantic | cache.py, semantic_cache.py | [VERIFIED] | 0.85 |
| `pareto_frontier.py` | ~500 | Cost-quality optimization | Historical data | ParetoPoints | Frontier cache | knowledge_graph | [INFERRED] | 0.65 |
| `outcome_router.py` | ~200 | Outcome-based routing | Task type, context | Model selection | Routing rules | N/A | [INFERRED] | 0.60 |

---

### Advanced Features (NASH Stability)

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `nash_stable_orchestrator.py` | ~1,500 | Integrated stability features | All orchestrator inputs | Enhanced results | All component states | All modules | [INFERRED] | 0.70 |
| `nash_events.py` | ~400 | Stability event bus | StabilityEvent | Handler results | Event queue | unified_events | [INFERRED] | 0.65 |
| `nash_backup.py` | ~350 | Backup/restore | BackupManifest | RestoreResult | Backup files | pathlib, json | [INFERRED] | 0.60 |
| `nash_auto_tuning.py` | ~400 | Parameter auto-tuning | Performance metrics | TuningResult | Parameter history | N/A | [INFERRED] | 0.55 |
| `federated_learning.py` | ~500 | Cross-org learning | Local insights | Global baseline | Model weights | cryptography | [INFERRED] | 0.55 |
| `adaptive_templates.py` | ~800 | Prompt template adaptation | Context, task | TemplateVariant | Template registry | N/A | [INFERRED] | 0.65 |

---

### Dashboard & UI

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `dashboard_core.py` | ~800 | Unified dashboard | Project state, telemetry | HTML, WebSocket | View state | FastAPI, websockets | [VERIFIED] | 0.85 |
| `dashboard_mission_control.py` | ~1,300 | Mission control (legacy) | Telemetry | HTML dashboard | WebSocket clients | FastAPI | [INFERRED] | 0.50 |
| `dashboard_live.py` | ~500 | Live dashboard (legacy) | Events | Real-time HTML | Event subscribers | FastAPI | [INFERRED] | 0.50 |
| `dashboard_antd.py` | ~600 | Ant Design dashboard (legacy) | State | React-like HTML | Component state | FastAPI | [INFERRED] | 0.50 |
| `dashboard_enhanced.py` | ~500 | Enhanced dashboard (legacy) | Metrics | HTML | Cache | FastAPI | [INFERRED] | 0.50 |
| `dashboard_real.py` | ~400 | Real-time dashboard (legacy) | Stream | HTML | Stream buffer | FastAPI | [INFERRED] | 0.50 |
| `dashboard_optimized.py` | ~500 | Optimized dashboard (legacy) | Aggregated | HTML | Aggregation cache | FastAPI | [INFERRED] | 0.50 |
| `dashboard.py` | ~400 | Original dashboard (legacy) | Basic state | Simple HTML | Minimal | FastAPI | [INFERRED] | 0.50 |

---

### Project Management Extensions

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `project_manager.py` | ~600 | Project scheduling | Tasks, resources | TaskSchedule | Schedule state | N/A | [INFERRED] | 0.60 |
| `product_manager.py` | ~500 | Product management | Features, feedback | ReleasePlan | Feature backlog | N/A | [INFERRED] | 0.55 |
| `quality_control.py` | ~400 | Quality assurance | Code, tests | QualityReport | Issue database | ast, pytest | [INFERRED] | 0.60 |

---

### Application Building

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `app_builder.py` | ~500 | App generation | Spec | AppBuildResult | Build context | N/A | [INFERRED] | 0.65 |
| `app_detector.py` | ~300 | App type detection | Codebase | AppProfile | Detection rules | pathlib, ast | [INFERRED] | 0.60 |
| `app_assembler.py` | ~400 | App assembly | Components | AssembledApp | Assembly state | N/A | [INFERRED] | 0.55 |
| `app_verifier.py` | ~250 | App verification | AssembledApp | VerificationResult | Test results | subprocess | [INFERRED] | 0.60 |

---

### Codebase Analysis

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `codebase_reader.py` | ~400 | Code reading | Path | CodeContent | Read cache | pathlib | [INFERRED] | 0.65 |
| `codebase_analyzer.py` | ~500 | Static analysis | Code | AnalysisResult | Analysis cache | ast | [INFERRED] | 0.65 |
| `codebase_profile.py` | ~300 | Profiling | Codebase | Profile | Profile data | N/A | [INFERRED] | 0.60 |
| `codebase_understanding.py` | ~600 | Semantic understanding | Codebase | UnderstandingGraph | Knowledge graph | N/A | [INFERRED] | 0.55 |

---

### Integration & Extensions

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `mcp_server.py` | 566 | MCP protocol server | MCP requests | Tool results | Component instances | mcp SDK | [VERIFIED] | 0.85 |
| `git_integration.py` | ~400 | Git operations | Repo path | GitResult | Repo state | gitpython | [INFERRED] | 0.65 |
| `slack_integration.py` | ~1,500 | Slack bot | Slack events | Responses | Message state | slack_sdk | [INFERRED] | 0.60 |
| `plugins.py` | ~300 | Plugin registry | Plugin specs | Registered plugins | Plugin registry | N/A | [INFERRED] | 0.65 |

---

### Utility & Support

| Module | LOC | Responsibility | Inputs | Outputs | State | External Deps | Classification | Confidence |
|--------|-----|----------------|--------|---------|-------|---------------|----------------|------------|
| `config.py` | ~200 | Configuration | Env vars, files | Config object | Cached config | pydantic-settings | [VERIFIED] | 0.90 |
| `log_config.py` | ~150 | Logging setup | Config | Logger instances | Logger state | logging | [VERIFIED] | 0.95 |
| `exceptions.py` | ~200 | Exception hierarchy | N/A | Exception classes | N/A | N/A | [VERIFIED] | 0.95 |
| `compat.py` | ~300 | Backward compatibility | Legacy calls | Modern equivalents | Migration state | N/A | [VERIFIED] | 0.85 |
| `health.py` | ~150 | Health checks | N/A | HealthStatus | Health cache | N/A | [INFERRED] | 0.70 |
| `diagnostics.py` | ~200 | Diagnostics | N/A | DiagnosticReport | Diagnostic state | N/A | [INFERRED] | 0.65 |

---

## STATE TRANSITION DIAGRAMS

### Task Lifecycle (engine.py)

```
[VERIFIED] Confidence: 0.95

PENDING
   │
   ├─(budget check fail)──→ FAILED
   │
   ├─(time limit reached)─→ FAILED
   │
   └─(execute)────────────→ WORKING
                              │
                              ├─(validation fail)─→ [revision loop]
                              │
                              ├─(success)─────────→ COMPLETED
                              │
                              └─(error)──────────→ FAILED
```

### A2A Task State (a2a_protocol.py)

```
[VERIFIED] Confidence: 0.95

PENDING
   │
   └─(submit)─────────────→ SUBMITTED
                              │
                              ├─(handler found)──→ WORKING
                              │                     │
                              │                     ├─(success)──→ COMPLETED
                              │                     │
                              │                     ├─(timeout)──→ FAILED
                              │                     │
                              │                     └─(error)────→ FAILED
                              │
                              └─(no handler)─────→ COMPLETED (passthrough)
```

### Adaptive Router ModelState (adaptive_router.py)

```
[VERIFIED] Confidence: 0.90

HEALTHY
   │
   ├─(consecutive failures ≥ 3)─→ DEGRADED
   │                               │
   │                               ├─(success)────────→ HEALTHY
   │                               │
   │                               └─(continued fails)─→ DISABLED
   │
   └─(manual disable)───────────→ DISABLED
                                   │
                                   └─(manual enable)──→ HEALTHY
```

### Memory Tier Migration (memory_tier.py)

```
[INFERRED] Confidence: 0.70

HOT (days 1-3)
   │
   └─(age > 3 days)─────────→ WARM (days 4-30)
                               │
                               └─(age > 30 days)─→ COLD (day 30+)
```

---

## DEPENDENCY GRAPH

### Core Dependencies (Verified)

```
orchestrator.engine
├── orchestrator.models [VERIFIED 0.98]
├── orchestrator.api_clients [VERIFIED 0.90]
├── orchestrator.cache [VERIFIED 0.85]
├── orchestrator.state [VERIFIED 0.95]
├── orchestrator.validators [VERIFIED 0.90]
├── orchestrator.policy [VERIFIED 0.90]
├── orchestrator.policy_engine [INFERRED 0.75]
├── orchestrator.planner [INFERRED 0.70]
├── orchestrator.telemetry [VERIFIED 0.85]
├── orchestrator.audit [VERIFIED 0.85]
├── orchestrator.hooks [VERIFIED 0.85]
├── orchestrator.cost [INFERRED 0.70]
├── orchestrator.agents [VERIFIED 0.85]
└── External SDKs
    ├── openai [VERIFIED 0.95]
    ├── google-genai [VERIFIED 0.95]
    └── aiosqlite [VERIFIED 0.95]
```

### New Integration Dependencies (v6.0)

```
orchestrator.engine (enhanced)
├── orchestrator.task_verifier [VERIFIED 0.85]
├── orchestrator.accountability [VERIFIED 0.85]
├── orchestrator.agent_safety [INFERRED 0.70]
├── orchestrator.red_team [INFERRED 0.65]
├── orchestrator.token_optimizer [VERIFIED 0.85]
├── orchestrator.preflight [VERIFIED 0.85]
├── orchestrator.session_watcher [VERIFIED 0.85]
├── orchestrator.persona [VERIFIED 0.85]
├── orchestrator.memory_tier [VERIFIED 0.90]
│   └── orchestrator.bm25_search [VERIFIED 0.90]
│       └── orchestrator.reranker [VERIFIED 0.85]
└── orchestrator.a2a_protocol [VERIFIED 0.95]
```

---

## CONFIDENCE DISTRIBUTION

| Classification | Module Count | Percentage |
|----------------|--------------|------------|
| **[VERIFIED]** (0.90-1.0) | 35 | 25% |
| **[INFERRED]** (0.60-0.89) | 75 | 54% |
| **[UNKNOWN]** (0.0-0.59) | 30 | 21% |
| **Total** | **140** | **100%** |

### Confidence by Layer

| Layer | Avg Confidence | Notes |
|-------|----------------|-------|
| Core Orchestration | 0.92 | Well-tested, direct traces |
| Memory & Search | 0.80 | Good visibility, some inference |
| Policy & Control | 0.70 | Complex interactions, partial inference |
| Agent Communication | 0.85 | Recently fixed, well-documented |
| Security & Accountability | 0.70 | New modules, limited testing |
| Observability | 0.80 | Good code visibility |
| Optimization & Routing | 0.75 | Mix of verified and inferred |
| NASH Stability | 0.60 | Complex, less tested |
| Dashboard & UI | 0.55 | Frontend not analyzed |
| Extensions | 0.60 | External dependencies |

---

## HIGH-RISK MODULES (Low Confidence + High Impact)

| Module | Confidence | Risk Reason | Recommendation |
|--------|------------|-------------|----------------|
| `secrets_manager.py` | 0.45 | Security-critical, unknown implementation | Security audit required |
| `federated_learning.py` | 0.55 | Cryptographic operations, unverified | Cryptography review |
| `nash_auto_tuning.py` | 0.55 | Autonomous parameter changes | Add safeguards, testing |
| `control_plane.py` | 0.65 | Central routing logic | Integration testing |
| `plugin_isolation.py` | 0.60 | Security boundary | Penetration testing |

---

**Map End**

*This knowledge map should be updated as analysis progresses. Modules marked [UNKNOWN] require additional investigation before reliability claims can be made.*
