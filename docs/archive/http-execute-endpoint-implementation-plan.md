# Implementation Plan — HTTP `/execute` Endpoint (api_server.py)

**Version:** 1.0
**Date:** 2026-07-11
**Owner:** Orchestrator core
**Source:** `orchestrator/api_server.py` lines 290–339 (currently stubbed)
**Status:** Proposed

---

## 1. Current State

`APIServer.execute_task()` at line 290 returns fake results:

```python
# Lines 322-336
# In a real implementation, we would call the orchestrator here
# For now, we'll simulate the execution
task_id = hashlib.sha256(...).hexdigest()[:16]
result = {
    "task_id": task_id,
    "status": "completed",
    "result": f"Simulated execution of: {task_description}",
    "cost": 0.05,
    "tokens_used": 150,
    "execution_time": 2.5,
}
```

No actual `Orchestrator` is created, no `StateManager` is used, and the returned `task_id` is a one-way hash with no persistence — status polling returns garbage.

### What already works (pattern to follow)

The **Supervisor endpoint** (`supervisor_directive`, line 499) successfully wires a real service via DI:

```python
def __init__(self, ..., supervisor: Supervisor | None = None):  # line 122
    self.supervisor = supervisor                               # line 142

async def supervisor_directive(self, request):                  # line 499
    result = await self.supervisor.handle(directive)            # line 549
```

The Orchestrator endpoint should follow the **exact same pattern** — inject an `Orchestrator` instance, call its public methods, persist state, and return structured results.

---

## 2. Architecture Decisions

### 2.1 Orchestrator Lifecycle

The `Orchestrator` class supports async context manager (`__aenter__` / `__aexit__`). In the APIServer, the orchestrator is long-lived — it should be started once at server boot and reused for all requests. This respects the existing `self.supervisor` pattern (supervisor is also long-lived).

**Decision:** Inject `orchestrator: Orchestrator | None = None` into `APIServer.__init__`, same as `supervisor`. The orchestrator must be `await`-entered before the first request and is shared across requests.

### 2.2 Request/Response Model

The current single-request format (`task`, `criteria`, `budget`, `model`) conflates a one-shot LLM call with a full multi-task project. We need two clear modes:

| Mode | Endpoint | Input | Output |
|:---|:---|:---|:---|
| **Project** (multi-task with decompose) | `POST /execute/project` | `project_description`, `success_criteria`, `budget`, `output_dir` | `project_id`, async status URL |
| **Tasks** (pre-composed, no decompose) | `POST /execute/tasks` | `tasks: [{id, type, prompt, target_path, ...}]`, `budget` | `project_id`, async status URL |
| **Spec-Kit ingest** | `POST /execute/from-speckit` | `spec_dir: "/path/to/specs/001/"`, `budget` | `project_id`, task count |

All three return immediately with a `project_id`. The caller polls or streams.

### 2.3 Streaming

The `Orchestrator.run_project_streaming()` yields `PipelineEvent` objects (14 event types: `PROJECT_START`, `TASK_START`, `TASK_COMPLETE`, `BUDGET_UPDATE`, `PROJECT_COMPLETE`, etc. — see `orchestrator/streaming.py:68-84`). These are ideal for SSE.

**Decision:** Add `GET /projects/{project_id}/stream` — SSE endpoint that subscribes to the `ProjectEventBus` for the running project and pushes `PipelineEvent.to_dict()` as JSON lines.

### 2.4 Concurrency

The `Orchestrator` already handles concurrent execution via `asyncio.Lock` (`_task_guard`, `_results_lock`) and `max_concurrency`. Multiple HTTP requests can run concurrently — each spawns a background `asyncio.Task` that calls `orchestrator.run_project_streaming()`. The APIServer tracks active runs in a dict `{project_id: asyncio.Task}`.

### 2.5 Layer Discipline

`api_server.py` lives at the **interfaces** layer (driving adapter in hexagonal terms). It may import `Orchestrator` and domain types (`ProjectConstitution`, `Task`) but must NOT import infrastructure internals. The pattern is already established — the supervisor endpoint imports `from orchestrator.supervisor.models import Directive` without touching infrastructure layers.

| Layer | Import allowed? | Examples used |
|:---|:---:|:---|
| `orchestrator.engine.Orchestrator` | ✅ | Public API |
| `orchestrator.budget.Budget` | ✅ | Domain value |
| `orchestrator.models.Task, ProjectState` | ✅ | Domain data |
| `orchestrator.ingest.SpecKitAdapter` | ✅ | Application adapter |
| `orchestrator.infrastructure.*` | ❌ | Not needed — Orchestrator abstracts this |

### 2.6 Import-Linter Compliance

`api_server.py` is at `orchestrator/api_server.py` — a **root-level module**. Contract 5 (`root-modules-no-infra`) prohibits importing from `orchestrator.infrastructure`. The implementation will only import from `orchestrator.engine`, `orchestrator.models`, `orchestrator.budget`, `orchestrator.ingest` — all of which are allowed (non-infrastructure).

---

## 3. API Contract

### 3.1 `POST /execute/project`

**Request:**
```json
{
  "project_description": "Build a FastAPI todo app with PostgreSQL",
  "success_criteria": "All endpoints tested, coverage >= 80%",
  "budget": 5.0,
  "max_time_seconds": 3600,
  "concurrency": 3,
  "output_dir": "/tmp/orchestrator-out/",
  "agent_profile": "standard"
}
```

**Response (202 Accepted):**
```json
{
  "project_id": "a1b2c3d4e5f6",
  "status": "accepted",
  "status_url": "/projects/a1b2c3d4e5f6",
  "stream_url": "/projects/a1b2c3d4e5f6/stream",
  "estimated_tasks": null
}
```

### 3.2 `POST /execute/tasks`

**Request:**
```json
{
  "project_description": "User auth feature",
  "success_criteria": "All tasks complete with passing tests",
  "budget": 3.0,
  "tasks": [
    {
      "id": "T001",
      "type": "code_generation",
      "prompt": "Create User model in src/models/user.py",
      "target_path": "src/models/user.py",
      "hard_validators": ["python_syntax", "ruff"]
    },
    {
      "id": "T002",
      "type": "code_generation",
      "prompt": "Implement AuthService in src/services/auth.py",
      "target_path": "src/services/auth.py",
      "dependencies": ["T001"]
    }
  ]
}
```

**Response (202 Accepted):** same shape as `/execute/project`.

### 3.3 `POST /execute/from-speckit`

**Request:**
```json
{
  "spec_dir": "/path/to/specs/001-auth/",
  "budget": 10.0,
  "max_concurrency": 3,
  "output_dir": "/tmp/out/"
}
```

**Response (202 Accepted):**
```json
{
  "project_id": "b2c3d4e5f6a1",
  "status": "accepted",
  "task_count": 12,
  "status_url": "/projects/b2c3d4e5f6a1",
  "stream_url": "/projects/b2c3d4e5f6a1/stream"
}
```

### 3.4 `GET /projects/{project_id}`

**Response (200 OK):**
```json
{
  "project_id": "a1b2c3d4e5f6",
  "status": "running",
  "tasks_total": 12,
  "tasks_completed": 5,
  "tasks_failed": 0,
  "cost_spent_usd": 1.23,
  "elapsed_seconds": 42.0,
  "results": {
    "T001": {"status": "completed", "score": 0.92, "cost_usd": 0.05, "output_preview": "class User..."},
    "T002": {"status": "running", "cost_usd": 0.0}
  }
}
```

### 3.5 `GET /projects/{project_id}/stream`

SSE endpoint. Returns `text/event-stream` with `PipelineEvent.to_dict()` as JSON data.

```
event: task_start
data: {"type":"TASK_START","project_id":"abc","timestamp":"...","data":{"task_id":"T001","task_type":"code_generation"}}

event: task_complete
data: {"type":"TASK_COMPLETE","project_id":"abc","timestamp":"...","data":{"task_id":"T001","score":0.92,"cost_usd":0.05}}

event: project_complete
data: {"type":"PROJECT_COMPLETE","project_id":"abc","timestamp":"...","data":{"status":"completed","total_cost":2.45}}
```

### 3.6 `GET /models` — make it real

Currently returns hardcoded JSON (line 383). Wire it to `orchestrator.model_selector.ModelSelector` or `orchestrator.models.ROUTING_TABLE` to return the actual configured model list.

---

## 4. Implementation Tasks (WBS)

### Phase 1 — Core Wiring (3 days)

| # | Task | File(s) | Effort |
|:---|:---|:---|:---:|
| 4.1.1 | Add `orchestrator: Orchestrator | None` parameter to `APIServer.__init__`, follow supervisor pattern | `api_server.py` | 0.5d |
| 4.1.2 | Add `_active_projects: dict[str, asyncio.Task]` and `_active_streams: dict[str, ProjectEventBus]` for state tracking | `api_server.py` | 0.5d |
| 4.1.3 | Implement `_execute_project_async()` — creates `Orchestrator`, calls `run_project()`, persists via `StateManager`, stores result | `api_server.py` | 1d |
| 4.1.4 | Replace stubbed `execute_task` → route to `POST /execute/project` with 202 response | `api_server.py` | 0.5d |
| 4.1.5 | Add `POST /execute/tasks` — accept `Task` list, create `dict[str, Task]`, call `run_project_with_tasks()` | `api_server.py` | 0.5d |

### Phase 2 — Spec-Kit & Streaming (2 days)

| # | Task | File(s) | Effort |
|:---|:---|:---|:---:|
| 4.2.1 | Add `POST /execute/from-speckit` — use `SpecKitAdapter` + `FileReader`, call `run_project_with_tasks()` | `api_server.py` | 0.5d |
| 4.2.2 | Add `GET /projects/{project_id}` — query `StateManager.load_project()` and return structured status | `api_server.py` | 0.5d |
| 4.2.3 | Add `GET /projects/{project_id}/stream` — SSE endpoint consuming `ProjectEventBus.subscribe()` | `api_server.py` | 0.5d |
| 4.2.4 | Add `GET /projects/{project_id}/output` — return output directory contents listing | `api_server.py` | 0.5d |

### Phase 3 — Observability & Hardening (1 day)

| # | Task | File(s) | Effort |
|:---|:---|:---|:---:|
| 4.3.1 | De-stub `list_models` — wire to real `ModelSelector` or `ROUTING_TABLE` | `api_server.py` | 0.25d |
| 4.3.2 | Add `DELETE /projects/{project_id}` — cancel a running project | `api_server.py` | 0.25d |
| 4.3.3 | Add request validation (reject tasks with no type, invalid budget, negative values) | `api_server.py` | 0.25d |
| 4.3.4 | Add `POST /execute` backward-compat wrapper — auto-detect old format and redirect to `/project` | `api_server.py` | 0.25d |

---

## 5. Design Decisions & Tradeoffs

### 5.1 Long-running Orchestrator vs per-request Orchestrator

| Approach | Pros | Cons |
|:---|:---|:---|
| **Long-running** (one `Orchestrator` per server) | StateManager/cache reused; circuit-breaker persists across requests; budget tracking accumulates | Concurrency must be managed carefully; one project failure shouldn't crash others |
| **Per-request** (new `Orchestrator` per call) | Isolated; no shared state bugs | No shared circuit-breaker or cost tracking; startup overhead per request |

**Decision:** Long-running. The `Orchestrator` already uses `_job_lock` for serializing state mutations and `self._active_policies` for per-job policy isolation. Background `asyncio.Task` per project is the established pattern (see `engine.py:947` where `run_project_streaming` spawns a background task). Each project gets its own `ProjectEventBus` instance for streaming isolation.

### 5.2 Fire-and-forget vs Request-response

The `POST /execute/*` endpoints accept the job and return immediately (202). This is deliberate:

- Projects can take minutes with multi-model generate→critique→revise→evaluate cycles
- HTTP timeouts would kill the connection before completion
- The caller uses polling (`GET /projects/{id}`) or SSE (`/stream`) for progress

### 5.3 Task persistence per HTTP request

Each project run persists its `ProjectState` to `StateManager` (SQLite) after each task completes — this is already the default behavior in `Orchestrator` (see `engine.py` — `save_project` called after task execution). The `/status` endpoint simply reads back from the same SQLite store.

### 5.4 Security

The existing auth and rate-limiting infrastructure (`_require_auth`, `TokenBucketRateLimiter`, Bearer token validation) is reused without changes. The execute endpoints are gated behind the same `self.auth_required` check as all other endpoints.

---

## 6. Testing Strategy

### Unit Tests (`tests/test_api_server.py`)

- Mock `Orchestrator` with `AsyncMock`, verify correct method is called per endpoint
- Verify 202 response shape on `/execute/project` with valid input
- Verify 400 on missing fields
- Verify 401 on missing auth header
- Verify 404 on unknown `project_id` in `/status`
- Verify SSE `Content-Type` header on stream endpoint

### Integration Tests (`tests/integration/test_api_server_integration.py`)

- Start a real `APIServer` with a real `Orchestrator` (mocked LLM client)
- POST `/execute/project` → get 202 + `project_id`
- Poll `GET /projects/{project_id}` until status is `completed` or `failed`
- Verify task results in the final state
- Test Spec-Kit ingest with an in-memory `FileReaderPort` (same `_InMemoryReader` from `test_speckit_adapter.py`)

### Contract Tests

- Assert response schema for all new endpoints against a pydantic model

---

## 7. Rollback & Feature Flags

| Feature | Flag | Default | Rollback |
|:---|:---|:---:|:---|
| Real execution (de-stub) | None needed — replacing stubs | — | Revert to `git checkout` previous `api_server.py` |
| `/execute/from-speckit` | `ORCHESTRATOR_HTTP_INGEST_ENABLED` (env var) | `true` | Set to `false` to return 501 |
| SSE streaming | `ORCHESTRATOR_HTTP_STREAM_ENABLED` | `true` | Set to `false`, `/stream` returns 501 |

---

## 8. Acceptance Criteria

- [ ] `POST /execute/project` with `project_description` + `success_criteria` returns 202 with valid `project_id`
- [ ] `GET /projects/{project_id}` returns current `ProjectState` from SQLite
- [ ] `GET /projects/{project_id}/stream` returns SSE with `PipelineEvent` JSON
- [ ] `POST /execute/tasks` accepts pre-composed tasks and skips decomposition (spy-asset)
- [ ] `POST /execute/from-speckit` accepts `spec_dir`, parses with `SpecKitAdapter`, executes tasks
- [ ] `GET /models` returns real model list (not hardcoded JSON)
- [ ] Multiple concurrent projects run without cross-contamination
- [ ] `project_id` is stable (same ID across status polls and stream)
- [ ] Auth token required when `auth_required=True` (existing behavior preserved)
- [ ] All 5 import-linter contracts KEEP
- [ ] No new imports from `orchestrator.infrastructure` in `api_server.py`
- [ ] Coverage ratchet raised: 7 → 8
- [ ] Default `POST /execute` path returns 410 Gone with migration message to `/execute/project`

---

## 9. File Change Summary

| File | Change | Lines |
|:---|:---|:---:|
| `orchestrator/api_server.py` | Replace stubs, add 5 new endpoints, add orchestrator DI, add SSE streaming | ~300 added, ~50 removed |
| `tests/test_api_server.py` | **NEW** — unit tests with mocked orchestrator | ~200 |
| `tests/integration/test_api_server_integration.py` | **NEW** — real orchestrator + mocked LLM | ~150 |
| `pyproject.toml` | Raise `fail_under` 7→8 | 1 line |
