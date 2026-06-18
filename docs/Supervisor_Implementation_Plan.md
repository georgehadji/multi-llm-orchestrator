# Supervisor Implementation Plan

> **Status:** Draft v1.0 · **Date:** 2026-06-18 · **Owner:** Platform/Orchestration
> **Scope:** A persistent, learning **Supervisor** layer above the existing orchestration engine — a single agent the user converses with, that external agents can direct, and that captures and re-applies lessons from failures so mistakes are not repeated.
> **Target branch:** `feat/supervisor-spine`

---

## 1. Executive Summary

### 1.1 Problem

Today the orchestrator is **run-and-exit**. Each invocation (`python -m orchestrator …`, the `website` generator, the `chat` flow, `--file` jobs) spins up an `Orchestrator`, runs `run_project()` / `run_project_streaming()`, and terminates. There is:

- **No persistent identity** — nothing the user "talks to" that remembers prior sessions.
- **No inbound agent channel** — external agents cannot send directives; only humans via CLI.
- **No closed learning loop** — `learning/prompt_enricher.py::AgentPromptEnricher` exists but is **never called at runtime** (confirmed: zero call sites). Failures are logged and discarded; the next run repeats them.

The `projects/hyperagent_website_factory.yaml` "Hyperagent" describes this vision in prose, but it is a *build prompt fed to the generic engine*, not a runtime topology — **no `HyperAgent`/agent classes exist** in `orchestrator/`.

### 1.2 Solution

Introduce a new, isolated **`orchestrator/supervisor/`** module that sits **above** `engine.py` and treats the engine as its single worker. It provides three capabilities behind one identity:

1. **Human channel** — a conversational, stateful interface (persistent REPL / WebSocket) the user directs.
2. **Agent channel** — an authenticated inbound API where external agents submit directives. *Internally identical to the human channel* — both normalize to one `Directive` → one intake.
3. **Continuous learning** — every failure/bug/degradation is captured as a structured **lesson**; relevant lessons are re-injected into future planning/generation. Three mechanisms, sequenced: (a) lessons retrieval, (b) routing reinforcement, (c) codified guardrails.

### 1.3 Guiding principles (honor existing architecture)

| Rule (from `CLAUDE.md`) | Consequence for Supervisor |
|---|---|
| `engine.py` = **Mediator** | Supervisor is a **new service module**; it *wires/delegates*, never adds business logic to `engine.py`. **Zero edits to `engine.py` in Phase 1.** |
| `models.py` = **pure data** | New dataclasses live in `supervisor/models.py` (pure); behavior lives in services. |
| **Dependencies point inward** | `supervisor` depends on application/engine ports, not vice-versa. Adapters (CLI, HTTP) are the outermost ring. |
| **TDD without exceptions** | Every component starts with a failing test. Fake `Orchestrator`/event stream for unit isolation. |

### 1.4 Phased delivery

| Phase | Theme | Outcome | Est. |
|---|---|---|---|
| **0** | Foundations | Module scaffold, data models, persistent store | 2 d |
| **1** | **Spine** | Persistent supervisor + human CLI + delegate to engine + capture & re-inject lessons (loop closed) | 5 d |
| **2** | Agent channel | Authenticated REST directive endpoint + policy enforcement + audit | 4 d |
| **3** | Learning maturation | Wire enricher + knowledge graph + semantic retrieval; reinforcement routing; guardrails | 6 d |
| **4** | Observability & hardening | Telemetry, dashboard panel, structured audit, RBAC, rate limits | 4 d |
| **5** | (Optional) Multi-worker & A2A | Swap single worker for specialist workers; A2A/MCP protocol | 8 d |

**Phases 0–1 prove the entire concept** (talk → delegate → fail → remember → improve) with ~4 new files and zero engine edits.

---

## 2. Supervisor Capability Inventory

| # | Capability | User-facing statement | Mechanism | Phase |
|---|---|---|---|---|
| C1 | Conversational control | "An agent I can talk to and give directions" | Persistent session + `ConversationAgent` reuse | 1 |
| C2 | Inbound agent directives | "External agents talk to him and direct projects" | `Directive(source="agent")` → same intake; REST on `APIServer` | 2 |
| C3 | Unified intent intake | One path regardless of source | `supervisor/intake.py` normalizer | 1 |
| C4 | Delegated execution | Supervisor runs work via the engine | `Orchestrator.run_project_streaming()` | 1 |
| C5 | Failure capture | Record every bug/fail/degradation | Subscribe to `ProjectEventBus` (`TaskFailed`, `VALIDATION_RESULT`, `ERROR`) | 1 |
| C6 | Lesson re-injection | "Do not repeat mistakes" | Prepend relevant lessons to project description; later via `AgentPromptEnricher` | 1→3 |
| C7 | Persistent memory & identity | Remembers across restarts | `SupervisorStore` (SQLite under `~/.orchestrator_cache/`) | 1 |
| C8 | Policy/permission enforcement | Constraints cannot be bypassed by prompt | `ControlPlane` + `ReferenceMonitor` | 2 |
| C9 | Audit trail | Immutable record of directives & decisions | `AuditLog` | 2 |
| C10 | Routing reinforcement | Learn which models/routes succeed | `meta_orchestrator` + telemetry feedback | 3 |
| C11 | Codified guardrails | Turn recurring bugs into blocking checks | `validators.py` / `preflight` rules generated from lessons | 3 |
| C12 | Semantic lesson retrieval | Find *relevant* lessons, not just recent | Vector store (`learning/knowledge_graph.py`, embeddings) | 3 |
| C13 | Observability | See sessions, lessons, decisions live | Dashboard panel + telemetry | 4 |
| C14 | Multi-worker orchestration | Specialist workers under one supervisor | Worker registry behind intake | 5 |
| C15 | A2A/MCP interop | Standard agent-to-agent handshake | Protocol adapter | 5 |

---

## 3. Architecture Integration Map

### 3.1 Hexagonal placement

```
                         ┌──────────────── DRIVING ADAPTERS (outermost) ────────────────┐
   Human  ──────────────►│  supervisor CLI REPL        Dashboard WS (chat_view)         │
   External agents ──────►│  POST /supervisor/directive (APIServer)                      │
                         └───────────────────────────────┬──────────────────────────────┘
                                                          │ Directive (one shape)
                                            ┌─────────────▼──────────────┐
                                            │   supervisor/intake.py      │   normalize
                                            └─────────────┬──────────────┘
                                            ┌─────────────▼──────────────┐
                                            │  supervisor/service.py      │  Supervisor (APP CORE)
                                            │  - session lifecycle        │
                                            │  - lesson injection         │
                                            │  - delegate + monitor       │
                                            │  - post-run post-mortem     │
                                            └───┬───────────┬─────────┬───┘
                          enforce (C8/C9) ┌─────▼───┐   ┌───▼────┐ ┌──▼─────────────┐
                                          │Control- │   │ engine │ │ SupervisorStore │
                                          │Plane +  │   │  .run_ │ │ sessions/lessons│
                                          │Audit    │   │project │ │ (~/.orch_cache) │
                                          └─────────┘   └───┬────┘ └────────▲────────┘
                                                            │ events         │ record lessons
                                                   ┌────────▼────────┐      │ (C5)
                                                   │ ProjectEventBus │──────┘
                                                   │ TaskFailed /    │
                                                   │ VALIDATION_RES /│
                                                   │ ERROR / Complete│
                                                   └─────────────────┘
   DRIVEN ADAPTERS: LLM providers (UnifiedClient) · cache · telemetry · learning/*
```

### 3.2 Reuse map — what already exists vs. what is new

| Concern | Existing component (reuse) | New work |
|---|---|---|
| Worker execution | `engine.py::Orchestrator.run_project_streaming` | — (no edits) |
| Event stream | `streaming.py::ProjectEventBus`, `PipelineEventType` | Subscriber that maps events→lessons |
| NL intake | `application/conversation_agent.py::ConversationAgent`, `application/chat_cli.py` | Session-bound wrapper |
| Policy enforcement | `control_plane.py::ControlPlane`, `reference_monitor.py` | Directive→`JobSpecV2`/`PolicySpecV2` mapping |
| Audit | `AuditLog` (control_plane `.audit`) | Directive + decision events |
| Inbound HTTP | `api_server.py::APIServer` (rate limit, API keys, CORS, size middleware) | `/supervisor/directive`, `/supervisor/sessions/*` routes |
| Persistence pattern | `~/.orchestrator_cache/*.db` (e.g. `skill_store.py` trajectories/skills) | `supervisor.db` |
| Learning seed | `learning/prompt_enricher.py` (dead), `knowledge_graph.py`, `memory_compressor.py`, `learning_aggregator.py`, `transfer_learning.py` | Wire into loop |
| Meta-optimization | `meta_orchestrator.py` | Feed lesson signals into routing |
| Guardrails | `validators.py`, `preflight` | Lesson→rule generation |

---

## 4. Component-by-Component Implementation Plan

### 4.1 `supervisor/models.py` — pure data (Phase 0)
Dataclasses only, no I/O (honors models-are-pure rule):
- `Directive(source: Literal["human","agent"], text, project_id?, criteria?, budget?, metadata)`
- `Lesson(id, session_id, project_id, task_type, kind, signal, detail, created_at)` where `kind ∈ {task_failed, validation_failed, degraded, error, budget}`
- `SupervisorSession(id, created_at, updated_at, status, summary, directive_count)`
- `SupervisorResult(session_id, project_status, lessons_recorded, output_dir?)`

### 4.2 `supervisor/store.py` — persistence (Phase 0→1)
`SupervisorStore` (aiosqlite), DB at `~/.orchestrator_cache/supervisor.db`:
- Tables `sessions`, `lessons` (+ indexes on `task_type`, `created_at`).
- Async methods: `create_session`, `touch_session`, `set_summary`, `record_lesson`, `recent_lessons(task_type=None, limit=N)`, `search_lessons(text)` (FTS in Phase 3).
- WAL mode; same connection-pool conventions as existing stores.

### 4.3 `supervisor/intake.py` — unified intake (Phase 1)
- `normalize(directive) -> JobArgs` — turns either source into `{project, criteria, budget, project_id}`.
- For free-text directives, optionally route through `ConversationAgent` to enrich into a spec (Phase 1: pass-through; Phase 3: clarify-loop).

### 4.4 `supervisor/service.py` — the Supervisor (Phase 1, core)
`Supervisor.handle(directive) -> SupervisorResult`:
1. `create_session`/lookup; `audit` the directive (Phase 2).
2. `lessons = store.recent_lessons(task_type)`; **prepend formatted lessons to `project_description`** (C6, simplest injection).
3. (Phase 2) `ControlPlane.submit` for policy gate; else direct.
4. `async for event in orch.run_project_streaming(...)`: forward progress to caller; on failure events → `store.record_lesson`.
5. On completion → `store.set_summary`, return `SupervisorResult`.
- **Decision points:** on repeated failure → retry / replan / escalate-to-human (Phase 1: escalate; Phase 3: auto-replan).

### 4.5 `supervisor/failure_tap.py` — event→lesson mapping (Phase 1)
Maps `ProjectEventBus` events to `Lesson`s:
- `TaskFailedEvent` → `kind=task_failed`
- `TaskCompletedEvent(status in {failed,degraded})` → `kind=degraded`
- `PipelineEventType.VALIDATION_RESULT` (failed) → `kind=validation_failed`
- `PipelineEventType.ERROR` → `kind=error`
- `BudgetWarning` → `kind=budget`

### 4.6 Adapters
- **CLI** `supervisor/cli_adapter.py` + `supervisor` subcommand (Phase 1): persistent REPL → `Supervisor.handle`.
- **HTTP** routes on `APIServer` (Phase 2): `POST /supervisor/directive`, `GET /supervisor/sessions`, `GET /supervisor/sessions/{id}/lessons`.
- **Dashboard** panel (Phase 4): sessions + lessons + decision feed.

### 4.7 Learning maturation (Phase 3)
- Wire `AgentPromptEnricher` (currently dead) so `Supervisor` passes `store` as its `memories`/`graph` backend; replace naive prepend with `enrich()`.
- Embeddings + `knowledge_graph.py` for **semantic** lesson retrieval (C12).
- `memory_compressor.py` to summarize old lessons (avoid prompt bloat).
- `meta_orchestrator` consumes lesson signals to bias routing (C10).
- Lesson→`validators`/`preflight` rule generation for recurring classes (C11).

---

## 5. Interfaces and APIs

### 5.1 Internal (Python)
```python
class Supervisor:
    def __init__(self, store: SupervisorStore, orchestrator_factory: Callable[[], Orchestrator],
                 control_plane: ControlPlane | None = None, audit: AuditLog | None = None): ...
    async def handle(self, directive: Directive) -> SupervisorResult: ...
    async def stream(self, directive: Directive) -> AsyncIterator[StreamEvent]: ...  # progress passthrough
```

### 5.2 External (HTTP, Phase 2) — mounted on `APIServer`
| Method | Path | Auth | Body / Result |
|---|---|---|---|
| POST | `/supervisor/directive` | API key | `{text, project_id?, criteria?, budget?}` → `{session_id, status}` |
| GET | `/supervisor/sessions` | API key | list of `SupervisorSession` |
| GET | `/supervisor/sessions/{id}` | API key | session + status |
| GET | `/supervisor/sessions/{id}/lessons` | API key | list of `Lesson` |
| WS | `/supervisor/stream/{id}` | API key | live progress events |

Reuses existing `APIServer` middleware: `TokenBucketRateLimiter`, API-key verification, CORS, request-size limits.

### 5.3 Directive contract (both channels)
Identical JSON shape; `source` is set by the adapter, never trusted from the body.

---

## 6. Data Models and State Management

### 6.1 Schema (`~/.orchestrator_cache/supervisor.db`)
```sql
CREATE TABLE sessions (
  id TEXT PRIMARY KEY, created_at REAL, updated_at REAL,
  status TEXT, summary TEXT, directive_count INTEGER DEFAULT 0
);
CREATE TABLE lessons (
  id TEXT PRIMARY KEY, session_id TEXT, project_id TEXT,
  task_type TEXT, kind TEXT, signal TEXT, detail TEXT, created_at REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX idx_lessons_tasktype ON lessons(task_type);
CREATE INDEX idx_lessons_created ON lessons(created_at);
-- Phase 3: CREATE VIRTUAL TABLE lessons_fts USING fts5(detail, signal, content='lessons');
```

### 6.2 State management rules
- Single writer per session; WAL for concurrent reads.
- Lessons are **append-only** (immutability principle); corrections add new rows.
- Retention: keep raw lessons 90 days; compress older into summarized rows (Phase 3).
- No secrets in `detail` — sanitize before persisting (see §10).

---

## 7. Workflow Orchestration Logic

### 7.1 Supervisor state machine
```
IDLE ─(directive)→ INTAKE ─→ ENRICH(lessons) ─→ [POLICY GATE] ─→ EXECUTING
EXECUTING ─(event:fail)→ RECORD_LESSON ─→ DECIDE
DECIDE ─(retryable & budget ok)→ EXECUTING (replan)         [Phase 3]
DECIDE ─(blocked / exhausted)→ ESCALATE(human)              [Phase 1 default]
EXECUTING ─(ProjectCompleted)→ POST_MORTEM ─→ SUMMARIZE ─→ IDLE
```

### 7.2 Decision points
| Point | Inputs | Phase-1 behavior | Mature behavior |
|---|---|---|---|
| After intake | directive completeness | pass-through | clarify via `ConversationAgent` |
| Pre-exec | policy result | run | block + audit (C8) |
| On failure | event kind, retry count, budget | record + continue/escalate | auto-replan with lesson context |
| On completion | tasks_failed count | summarize | extract guardrail candidates (C11) |

### 7.3 Lesson injection format (C6)
Prepended block, capped (e.g. top 5 by relevance), e.g.:
```
## Known pitfalls (learned from prior runs) — avoid repeating:
- [code_generation] Output truncated when section >12k chars → request 2× tokens / split.
- [validation] HTML components emitted as full documents → emit body fragments only.
```

---

## 8. Error Handling Strategy

| Failure surface | Detection | Handling | Lesson? |
|---|---|---|---|
| Worker task failure | `TaskFailedEvent` | record; continue or escalate | yes |
| Validation/quality fail | `VALIDATION_RESULT` failed | record; surface to caller | yes |
| Degraded completion | `TaskCompletedEvent(status=degraded)` | record | yes |
| Budget breach | `BudgetWarning` | pause/escalate, never silently overspend | yes (budget) |
| Engine exception | exception from `run_project_streaming` | catch, mark session `failed`, audit, return error | yes (error) |
| Store unavailable | aiosqlite error | degrade gracefully — **run without learning, never block execution** | n/a |
| Adapter/transport error | HTTP/WS error | standard API error envelope; retry-after on rate limit | n/a |

Principles: **never swallow errors silently** (per coding-style rules); learning is best-effort and must never block the primary workflow.

---

## 9. Monitoring, Logging, and Observability

- **Structured logs** under `orchestrator.supervisor.*` logger namespace; one line per state transition.
- **Telemetry**: emit counters — `directives_total{source}`, `lessons_recorded{kind}`, `sessions_active`, `escalations_total`, `injection_hit_rate` (lessons injected that prevented a repeat — measured via before/after failure rates).
- **Dashboard panel** (Phase 4) on existing dashboard (`dashboard_core/`): live sessions, recent lessons, decision feed, learning-effectiveness chart.
- **Audit log** (Phase 2): every directive + policy decision (immutable).
- **Health**: extend `/health` with supervisor liveness + store reachability.

**Key metric (success proof):** repeat-failure rate for a given `(task_type, signal)` should trend **down** after lessons accumulate.

---

## 10. Security Controls

| Control | Mechanism | Phase |
|---|---|---|
| Authn (agent channel) | Existing `APIServer` API-key verification | 2 |
| Authz / source trust | `source` set by adapter, never from body; per-key scopes | 2 |
| Policy enforcement | `ControlPlane` + `ReferenceMonitor` — constraints not bypassable by prompt content | 2 |
| Rate limiting | `TokenBucketRateLimiter` (existing) | 2 |
| Request size limits | existing size middleware | 2 |
| Secret hygiene | sanitize lesson `detail`/`signal` before persist; reuse website sanitizer patterns; never store API keys/tokens | 1 |
| Prompt-injection containment | lessons are **internal-authored summaries**, not raw external text; external directive text is treated as untrusted and policy-gated | 2 |
| Audit immutability | append-only `AuditLog` | 2 |
| RBAC (multi-tenant) | `multi_tenant_gateway.py` integration | 4 |

---

## 11. Testing Strategy (TDD)

| Layer | Tests | Tooling |
|---|---|---|
| Unit — store | session CRUD, lesson append/retrieve, task-type filter, retention | pytest + temp sqlite |
| Unit — failure_tap | each event type → correct `Lesson` kind | fixtures of `ProjectEventBus` events |
| Unit — service | failure event → lesson recorded; 2nd run → lessons present in description passed to a **fake Orchestrator**; escalation on exhausted retries | fake orchestrator/event stream |
| Unit — intake | human vs agent directive → identical `JobArgs` | parametrized |
| Integration | CLI adapter round-trip; HTTP directive → session created → lessons queryable | aiohttp test client |
| Security | unauth request rejected; oversized body rejected; policy violation blocked + audited; secret redaction in lessons | pytest |
| Learning efficacy | seed lesson DB → assert injected; simulate repeat scenario → assert mitigation applied | pytest |
| Regression | existing suites stay green; **zero** behavioral change to `engine.py` | full `pytest -m unit` |

Coverage target ≥ 80% on `supervisor/` (project standard). Mark `unit`/`integration`.

---

## 12. Deployment Strategy

- **Phase 1**: ships as a CLI subcommand — no service to deploy. Persistent only while REPL runs; state survives in `supervisor.db`.
- **Phase 2+**: long-running service via existing `APIServer` (`start_server`) — run as a managed process (the repo already has `start_dashboard.py` / `.bat` patterns and PM2 skill). Single instance initially (SQLite single-writer).
- **Config**: env-driven (`ORCH_CACHE_HOME`, API keys, budgets) consistent with `crosscutting/config.py`.
- **Backwards compatibility**: additive only; all existing entrypoints unchanged.

---

## 13. Migration and Rollout Plan

1. **Shadow (Phase 1)**: Supervisor used opt-in via `supervisor` subcommand; existing flows untouched.
2. **Internal dogfood**: route your own real tasks (e.g. the website generator) through the Supervisor; verify lessons accumulate and injection fires.
3. **Agent channel canary (Phase 2)**: enable `/supervisor/directive` behind a single API key; monitor audit + rate limits.
4. **Learning ramp (Phase 3)**: enable enrichment for one `task_type` first (e.g. `code_generation`), measure repeat-failure delta, then expand.
5. **GA**: enable dashboard panel, broaden API keys, document.
- **Rollback**: feature-flag each phase; disabling reverts to current run-and-exit behavior with no data loss (lessons remain in DB).

---

## 14. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Lesson injection bloats prompts / raises cost | Med | Med | Cap top-N by relevance; compress old (Phase 3); track token delta |
| R2 | Irrelevant/noisy lessons degrade output quality | Med | High | Relevance filter (task_type → semantic); A/B injection on/off; allow lesson curation |
| R3 | Prompt injection via external directives | Med | High | Treat external text as untrusted; policy gate; lessons are internal-authored only |
| R4 | SQLite single-writer limits concurrency | Low | Med | Single service instance first; migrate to Postgres if needed |
| R5 | Scope creep into multi-agent swarm too early | High | High | Phases 0–1 use **one** worker; swarm deferred to Phase 5 |
| R6 | Coupling creep into `engine.py` | Med | High | Hard rule: zero engine edits in P1; enforce via review + import-direction test |
| R7 | Learning loop never proven effective | Med | High | Define repeat-failure metric up front; gate Phase 3 expansion on measured delta |
| R8 | Secret leakage into lessons DB | Low | High | Sanitize before persist; secret-scan test |
| R9 | Long-running service stability (memory/leaks) | Med | Med | Bounded caches; periodic compaction; health checks |
| R10 | Event-type drift in `streaming.py` breaks tap | Low | Med | Map via enum constants; contract test against event classes |

---

## 15. Implementation Roadmap

### 15.1 Timeline (solo dev + AI assist; ~1 phase/week)

```
Week 1   [P0] scaffold, models, store ───────────► [P1] spine: intake, service, failure_tap, CLI
Week 2   [P1] tests + dogfood + close loop ──────► verify repeat-failure capture/injection
Week 3   [P2] HTTP directive + auth + ControlPlane + AuditLog
Week 4   [P3] wire enricher + semantic retrieval + routing reinforcement
Week 5   [P3] guardrail generation + [P4] telemetry/dashboard/RBAC
Week 6+  [P5] (optional) multi-worker + A2A/MCP
```

### 15.2 Task breakdown

| Task | Phase | Component | Depends on | Owner role | Effort | Acceptance criteria |
|---|---|---|---|---|---|---|
| T0.1 Module scaffold + import-direction test | 0 | supervisor/ | — | Platform | 0.5 d | `supervisor` imports no outer ring; test green |
| T0.2 `models.py` dataclasses | 0 | models | T0.1 | Backend | 0.5 d | Frozen/pure; unit constructs each |
| T0.3 `SupervisorStore` + schema | 0 | store | T0.2 | Backend | 1 d | CRUD + filter tests green; WAL on |
| T1.1 `intake.normalize` | 1 | intake | T0.2 | Backend | 0.5 d | human==agent → identical JobArgs |
| T1.2 `failure_tap` event→lesson | 1 | failure_tap | T0.2 | Backend | 1 d | each event kind mapped (tests) |
| T1.3 `Supervisor.handle` + lesson inject | 1 | service | T0.3,T1.1,T1.2 | Backend | 1.5 d | fake-orch test: fail→lesson; rerun→injected |
| T1.4 `supervisor` CLI subcommand | 1 | cli_adapter | T1.3 | Platform | 1 d | persistent REPL; survives restart |
| T1.5 Dogfood + metric baseline | 1 | — | T1.4 | QA | 0.5 d | website task run via supervisor; lessons in DB |
| T2.1 HTTP routes on APIServer | 2 | http_adapter | T1.3 | Backend | 1 d | endpoints per §5.2 |
| T2.2 API-key auth + source trust | 2 | security | T2.1 | Security | 0.5 d | unauth rejected; source not body-trusted |
| T2.3 ControlPlane policy gate | 2 | service | T2.1 | Backend | 1 d | violation blocked + audited |
| T2.4 AuditLog wiring | 2 | audit | T2.1 | Security | 0.5 d | directive+decision recorded immutably |
| T2.5 Secret sanitization | 1/2 | store | T0.3 | Security | 0.5 d | secret-scan test passes |
| T3.1 Wire `AgentPromptEnricher` | 3 | learning | T1.3 | Backend | 1 d | enrich() called; replaces naive prepend |
| T3.2 Semantic retrieval (embeddings/FTS) | 3 | store/learning | T3.1 | Backend | 1.5 d | relevance > recency in eval |
| T3.3 Memory compression | 3 | learning | T3.1 | Backend | 1 d | old lessons summarized; token delta bounded |
| T3.4 Routing reinforcement | 3 | meta | T1.2 | Backend | 1.5 d | model success bias measurable |
| T3.5 Guardrail generation | 3 | validators | T1.2 | Backend | 1 d | recurring class → blocking check |
| T4.1 Telemetry counters | 4 | observability | T1.3 | Platform | 0.5 d | metrics emitted |
| T4.2 Dashboard panel | 4 | dashboard | T4.1 | Frontend | 1.5 d | sessions+lessons+efficacy chart |
| T4.3 RBAC / multi-tenant | 4 | security | T2.2 | Security | 1 d | per-key scopes enforced |
| T4.4 Health + stability | 4 | observability | T2.1 | Platform | 1 d | health checks; soak test |
| T5.1 Worker registry | 5 | service | T1.3 | Backend | 3 d | ≥2 worker types behind intake |
| T5.2 A2A/MCP adapter | 5 | adapter | T2.1 | Backend | 3 d | external standard handshake |

---

## 16. Traceability Matrix

| Capability | Architecture component(s) | Implementation task(s) | Test(s) |
|---|---|---|---|
| C1 Conversational control | `conversation_agent`, CLI | T1.4 | integration: CLI round-trip |
| C2 Agent directives | `APIServer` | T2.1, T2.2 | integration: HTTP directive |
| C3 Unified intake | `supervisor/intake` | T1.1 | unit: human==agent |
| C4 Delegated execution | `engine.run_project_streaming` | T1.3 | unit: fake-orch invoked |
| C5 Failure capture | `ProjectEventBus`, `failure_tap` | T1.2 | unit: event→lesson |
| C6 Lesson re-injection | `service`, (`AgentPromptEnricher`) | T1.3, T3.1 | unit: injected on rerun |
| C7 Persistent memory | `SupervisorStore` | T0.3 | unit: CRUD/retention |
| C8 Policy enforcement | `ControlPlane`, `ReferenceMonitor` | T2.3 | security: blocked+audited |
| C9 Audit | `AuditLog` | T2.4 | security: immutability |
| C10 Routing reinforcement | `meta_orchestrator`, telemetry | T3.4 | unit: bias applied |
| C11 Guardrails | `validators`/`preflight` | T3.5 | unit: rule blocks |
| C12 Semantic retrieval | `knowledge_graph`, embeddings/FTS | T3.2 | eval: relevance |
| C13 Observability | `dashboard_core`, telemetry | T4.1, T4.2 | manual + smoke |
| C14 Multi-worker | worker registry | T5.1 | integration |
| C15 A2A/MCP | protocol adapter | T5.2 | integration |

---

## 17. Checklists

### 17.1 Phase 0 — Foundations
- [ ] `orchestrator/supervisor/` package created with `__init__.py`
- [ ] Import-direction test (no dependency on outer ring) green
- [ ] `models.py` pure dataclasses, no I/O
- [ ] `SupervisorStore` schema + WAL + indexes
- [ ] Store unit tests ≥ 80% coverage

### 17.2 Phase 1 — Spine (loop closed)
- [ ] `intake.normalize` — human/agent parity test
- [ ] `failure_tap` maps every event kind
- [ ] `Supervisor.handle` delegates to engine (no engine edits)
- [ ] Lessons recorded on failure; injected on next run (test-proven)
- [ ] `supervisor` CLI subcommand; state survives restart
- [ ] Secret sanitization before persist
- [ ] Dogfood: real task run through supervisor; baseline metric captured

### 17.3 Phase 2 — Agent channel
- [ ] `/supervisor/*` routes on `APIServer`
- [ ] API-key auth; `source` not trusted from body
- [ ] `ControlPlane` gate blocks violations + audits
- [ ] Rate limit + size limits active
- [ ] `AuditLog` records directives + decisions

### 17.4 Phase 3 — Learning maturation
- [ ] `AgentPromptEnricher` wired and called
- [ ] Semantic/relevance retrieval beats recency in eval
- [ ] Memory compression bounds prompt growth
- [ ] Routing reinforcement measurable
- [ ] Guardrail generated from a real recurring lesson

### 17.5 Phase 4 — Observability & hardening
- [ ] Telemetry counters emitted
- [ ] Dashboard panel live
- [ ] Repeat-failure rate trends down (efficacy proven)
- [ ] RBAC / multi-tenant scopes
- [ ] Soak/health stability

---

## 18. Decision Log

| ID | Decision | Rationale | Alternatives rejected |
|---|---|---|---|
| D1 | Supervisor is a **new module**, zero edits to `engine.py` in P1 | Honors Mediator rule; isolates risk | Extending engine (violates architecture) |
| D2 | **One intake**, two adapters (human/agent identical internally) | Collapses two requirements into one path | Separate human/agent pipelines (duplication) |
| D3 | Start learning with **(a) lessons retrieval**, then (b) routing, (c) guardrails | Cheapest, most literal "don't repeat mistakes"; highest ROI | Reinforcement-first (slow), guardrails-first (manual) |
| D4 | **Single existing engine as the only worker** initially | Proves loop without swarm complexity | 9-agent swarm now (premature, high risk) |
| D5 | Persist to **SQLite under `~/.orchestrator_cache/`** | Matches existing store conventions; zero new infra | Postgres/Redis now (overkill at this scale) |
| D6 | **Plain REST** for agent channel first | Simplest; reuses `APIServer` auth/rate-limit | A2A/MCP first (unknown callers, more surface) |
| D7 | Lessons are **append-only, internal-authored** | Immutability + prompt-injection containment | Mutable rows / raw external text (unsafe) |
| D8 | Learning is **best-effort, never blocks execution** | Reliability > learning; store outage can't halt work | Hard dependency on store (fragile) |
| D9 | Lesson injection **capped + relevance-filtered** | Controls cost (R1) and noise (R2) | Inject all lessons (bloat/noise) |
| D10 | **Feature-flag per phase** | Safe incremental rollout + instant rollback | Big-bang cutover |

---

## 19. Appendix — Definition of Done (Phase 1 / concept proof)

The concept is proven when, in a single persistent session:
1. You (or an agent) issue a directive in natural language.
2. The Supervisor runs it via the existing engine and streams progress.
3. A failure during the run is captured as a structured lesson in `supervisor.db`.
4. A **subsequent** related run shows the prior lesson **injected** into the plan (verified in the description passed to the engine and via the repeat-failure metric).
5. All of the above with **zero changes to `engine.py`** and ≥ 80% test coverage on `supervisor/`.
