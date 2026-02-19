# Web UI Design — Multi-LLM Orchestrator Dashboard
**Date:** 2026-02-20
**Status:** Approved

---

## Overview

A Next.js 15 + FastAPI dashboard for the Multi-LLM Orchestrator. Users can start new jobs (via form or YAML upload), monitor live progress with task cards and a log stream, browse history, and inspect per-model telemetry.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           Next.js 15 (App Router)           │
│  /          → Dashboard (active jobs)        │
│  /new       → New job form + YAML upload     │
│  /jobs/[id] → Live job view (tasks + logs)   │
│  /history   → Past projects list             │
│  /models    → Model telemetry charts         │
└──────────────┬──────────────────────────────┘
               │ REST + SSE (fetch / EventSource)
┌──────────────▼──────────────────────────────┐
│           FastAPI (Python)                   │
│  POST   /api/jobs              → start job   │
│  GET    /api/jobs              → list all    │
│  GET    /api/jobs/{id}         → job state   │
│  GET    /api/jobs/{id}/stream  → SSE stream  │
│  POST   /api/jobs/{id}/resume  → resume job  │
│  DELETE /api/jobs/{id}         → delete job  │
│  GET    /api/models            → telemetry   │
└──────────────┬──────────────────────────────┘
               │ imports existing modules
┌──────────────▼──────────────────────────────┐
│     Existing Orchestrator (Python)           │
│  engine.py, state.py, telemetry.py, etc.    │
└─────────────────────────────────────────────┘
```

---

## Pages

### `/` — Dashboard
- Active jobs cards (status badge, budget bar, task progress)
- "New Job" button → `/new`
- Sidebar: compact model stats (trust factor, success rate, cost)

### `/new` — New Job
- **Tab A: Form** — project description, success criteria, budget (USD), time limit (s), concurrency
- **Tab B: YAML Upload** — drag & drop, parse & preview fields before submit
- Submit → POST `/api/jobs` → redirect to `/jobs/[id]`

### `/jobs/[id]` — Live Job View
- Header: project title, status badge, budget bar, elapsed time counter
- Top half: Task cards grid — task id, type, status, model used, score, cost, iterations
- Bottom half: Live log stream via SSE — scrollable terminal-style with color coding (INFO/WARN/ERROR)
- Actions: Cancel (if running), Resume (if stopped/failed)

### `/history` — Past Projects
- Table: project id, description, status, total cost, date
- Row actions: View (`/jobs/[id]`), Resume, Delete

### `/models` — Model Telemetry
- Per-model cards: success rate bar, trust factor gauge, avg/p95 latency, avg cost, call count
- Recharts: latency trend, cost distribution per provider

---

## FastAPI Backend

### Module layout

```
orchestrator/api/
  __init__.py
  main.py           # FastAPI app, CORS, router mounts
  routes/
    jobs.py         # CRUD + SSE stream
    models.py       # telemetry endpoint
  job_runner.py     # async task: runs Orchestrator, captures logs
  sse_handler.py    # logging.Handler that feeds SSE queue
```

### SSE event schema

```json
{ "type": "log",  "level": "INFO", "msg": "Starting task T1...", "ts": 1234567890 }
{ "type": "task", "task_id": "T1", "status": "completed", "score": 0.92, "model": "kimi-k2.5", "cost": 0.003 }
{ "type": "done", "status": "SUCCESS", "total_cost": 0.021 }
```

### Job lifecycle
1. `POST /api/jobs` → creates job record in memory + DB, spawns background asyncio task
2. `GET /api/jobs/{id}/stream` → SSE connection, reads from per-job asyncio Queue
3. Background task feeds Queue with log + task events
4. Task completes → sends `done` event → SSE closes

### Error handling
- HTTP 404 for unknown job IDs
- HTTP 409 if job already running on resume attempt
- SSE auto-reconnects with exponential backoff (native EventSource behavior)

---

## Next.js Stack

| Concern | Choice |
|---------|--------|
| Framework | Next.js 15 (App Router, TypeScript) |
| Styling | Tailwind CSS |
| Components | shadcn/ui (cards, badges, buttons, tabs) |
| Charts | Recharts |
| Live data | Native `EventSource` API |
| HTTP client | Native `fetch` |

### Directory structure

```
ui/
  app/
    page.tsx                # Dashboard
    new/page.tsx            # New job form + YAML upload
    jobs/[id]/page.tsx      # Live job view
    history/page.tsx        # Past projects
    models/page.tsx         # Model telemetry
    layout.tsx              # Sidebar + nav shell
  components/
    JobCard.tsx             # Active job summary card
    TaskCard.tsx            # Individual task status card
    LogStream.tsx           # SSE-fed terminal log viewer
    ModelSidebar.tsx        # Compact model stats widget
    YamlUpload.tsx          # Drag & drop YAML parser
    BudgetBar.tsx           # Progress bar for USD spend
  lib/
    api.ts                  # fetch wrappers for all endpoints
    sse.ts                  # useSSE React hook (EventSource)
  package.json
```

---

## Out of Scope (YAGNI)

- Authentication / multi-user support
- Docker / production deployment config
- Mobile-optimized layout
- Dark mode toggle (Tailwind `dark:` classes available but no switcher)

---

## Implementation Order

1. FastAPI backend (`orchestrator/api/`) + SSE handler
2. Next.js project scaffold (`ui/`) + Tailwind + shadcn/ui
3. `/new` page (form + YAML upload)
4. `/jobs/[id]` live view (task cards + log stream)
5. `/` dashboard + `/history` page
6. `/models` telemetry page + Recharts
7. ModelSidebar widget
