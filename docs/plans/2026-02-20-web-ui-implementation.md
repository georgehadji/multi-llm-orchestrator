# Web UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Next.js 15 dashboard + FastAPI backend so users can start, monitor, browse, and manage Multi-LLM Orchestrator jobs from a browser.

**Architecture:** FastAPI wraps the existing `Orchestrator` engine and exposes REST + SSE endpoints. Next.js App Router consumes those endpoints with `fetch` and native `EventSource`. The Python backend lives at `orchestrator/api/`; the frontend lives at `ui/`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, Next.js 15, TypeScript, Tailwind CSS, shadcn/ui, Recharts, js-yaml (YAML parsing in browser).

---

## Task 1: FastAPI app skeleton + CORS

**Files:**
- Create: `orchestrator/api/__init__.py`
- Create: `orchestrator/api/main.py`

**Step 1: Create `orchestrator/api/__init__.py`** (empty)

```python
```

**Step 2: Create `orchestrator/api/main.py`**

```python
"""
FastAPI application entry point.
Run with: uvicorn orchestrator.api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Orchestrator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health():
    return {"status": "ok"}
```

**Step 3: Install FastAPI + uvicorn**

```bash
pip install fastapi uvicorn[standard] python-multipart pyyaml
```

**Step 4: Smoke test**

```bash
uvicorn orchestrator.api.main:app --port 8000
curl http://localhost:8000/healthz
```
Expected: `{"status":"ok"}`

**Step 5: Commit**

```bash
git add orchestrator/api/
git commit -m "feat: FastAPI app skeleton with CORS"
```

---

## Task 2: SSE handler — capture orchestrator logs into a queue

**Files:**
- Create: `orchestrator/api/sse_handler.py`
- Create: `tests/test_sse_handler.py`

**Step 1: Write failing test**

```python
# tests/test_sse_handler.py
import asyncio
import logging
from orchestrator.api.sse_handler import SSELogHandler, JobEventQueue

def test_log_handler_puts_event_in_queue():
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    handler = SSELogHandler(queue, loop)
    logger = logging.getLogger("test.sse")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("hello world")

    event = loop.run_until_complete(queue.get())
    assert event["type"] == "log"
    assert event["level"] == "INFO"
    assert "hello world" in event["msg"]
    loop.close()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_sse_handler.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement `orchestrator/api/sse_handler.py`**

```python
"""
SSELogHandler — bridges Python logging to an asyncio Queue for SSE streaming.

Each log record emitted by the orchestrator logger is converted to a JSON-
serialisable dict and placed on a per-job asyncio.Queue. The SSE route reads
from that queue and forwards events to the connected browser.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any


class SSELogHandler(logging.Handler):
    """
    A logging.Handler that converts LogRecords to SSE event dicts and
    puts them on an asyncio.Queue in a thread-safe way.

    Parameters
    ----------
    queue : asyncio.Queue
        The per-job queue shared with the SSE route.
    loop : asyncio.AbstractEventLoop
        The running event loop (needed for call_soon_threadsafe).
    """

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = queue
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        event = {
            "type": "log",
            "level": record.levelname,
            "msg": self.format(record),
            "ts": int(time.time()),
        }
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


@dataclass
class JobEventQueue:
    """Holds the asyncio.Queue and metadata for a running job."""
    job_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
```

**Step 4: Run test**

```bash
pytest tests/test_sse_handler.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/api/sse_handler.py tests/test_sse_handler.py
git commit -m "feat: SSELogHandler bridges Python logging to asyncio Queue"
```

---

## Task 3: Job runner — wraps Orchestrator in a background asyncio task

**Files:**
- Create: `orchestrator/api/job_runner.py`
- Create: `tests/test_job_runner.py`

**Step 1: Write failing test**

```python
# tests/test_job_runner.py
import asyncio
from orchestrator.api.job_runner import JobRecord, JobStatus

def test_job_record_initial_state():
    rec = JobRecord(
        job_id="test-001",
        project_description="Test project",
        success_criteria="All tasks pass",
        budget_usd=1.0,
        time_limit_s=60,
        concurrency=2,
    )
    assert rec.status == JobStatus.PENDING
    assert rec.job_id == "test-001"
    assert rec.total_cost == 0.0
    assert rec.tasks == {}
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_job_runner.py -v
```

**Step 3: Implement `orchestrator/api/job_runner.py`**

```python
"""
JobRunner — manages in-memory job records and background orchestration tasks.

Each call to start_job() spawns an asyncio background task that:
  1. Attaches an SSELogHandler to the root orchestrator logger
  2. Runs Orchestrator.run_project()
  3. Feeds task-completion events into the job's SSE queue
  4. Sends a final "done" event when finished
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from orchestrator.models import Budget, ProjectState
from orchestrator.engine import Orchestrator
from orchestrator.api.sse_handler import SSELogHandler, JobEventQueue


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    project_description: str
    success_criteria: str
    budget_usd: float
    time_limit_s: float
    concurrency: int
    status: JobStatus = JobStatus.PENDING
    total_cost: float = 0.0
    tasks: dict[str, Any] = field(default_factory=dict)
    final_status: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _bg_task: Optional[asyncio.Task] = field(default=None, repr=False)


# In-memory store: job_id → JobRecord
_jobs: dict[str, JobRecord] = {}


def get_job(job_id: str) -> Optional[JobRecord]:
    return _jobs.get(job_id)


def list_jobs() -> list[JobRecord]:
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


async def start_job(record: JobRecord) -> None:
    """Register and launch a job as a background asyncio task."""
    _jobs[record.job_id] = record
    record._bg_task = asyncio.create_task(_run(record))


async def delete_job(job_id: str) -> bool:
    record = _jobs.pop(job_id, None)
    if record is None:
        return False
    if record._bg_task and not record._bg_task.done():
        record._bg_task.cancel()
    return True


async def _run(record: JobRecord) -> None:
    loop = asyncio.get_running_loop()
    handler = SSELogHandler(record.event_queue, loop)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    orch_logger = logging.getLogger("orchestrator")
    orch_logger.addHandler(handler)

    record.status = JobStatus.RUNNING
    record.updated_at = time.time()

    try:
        budget = Budget(
            max_usd=record.budget_usd,
            max_time_seconds=record.time_limit_s,
        )
        orch = Orchestrator(budget=budget, max_concurrency=record.concurrency)

        state: ProjectState = await orch.run_project(
            project_description=record.project_description,
            success_criteria=record.success_criteria,
            project_id=record.job_id,
        )

        # Push task-completion events
        for tid, result in state.results.items():
            record.tasks[tid] = {
                "task_id": tid,
                "status": result.status.value,
                "score": result.score,
                "model": result.model_used.value,
                "cost": result.cost_usd,
                "iterations": result.iterations,
            }
            await record.event_queue.put({
                "type": "task",
                **record.tasks[tid],
            })

        record.total_cost = state.budget.spent_usd
        record.final_status = state.status.value
        record.status = JobStatus.DONE

    except Exception as exc:
        record.status = JobStatus.FAILED
        record.final_status = f"error: {exc}"
        await record.event_queue.put({
            "type": "log",
            "level": "ERROR",
            "msg": f"Job failed: {exc}",
            "ts": int(time.time()),
        })
    finally:
        orch_logger.removeHandler(handler)
        record.updated_at = time.time()
        await record.event_queue.put({
            "type": "done",
            "status": record.final_status,
            "total_cost": record.total_cost,
        })
```

**Step 4: Run test**

```bash
pytest tests/test_job_runner.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/api/job_runner.py tests/test_job_runner.py
git commit -m "feat: JobRunner with in-memory registry and background asyncio task"
```

---

## Task 4: Jobs API routes (CRUD + SSE stream)

**Files:**
- Create: `orchestrator/api/routes/__init__.py`
- Create: `orchestrator/api/routes/jobs.py`
- Modify: `orchestrator/api/main.py`

**Step 1: Create `orchestrator/api/routes/__init__.py`** (empty)

**Step 2: Create `orchestrator/api/routes/jobs.py`**

```python
"""
Jobs router — REST CRUD + SSE stream endpoint.

POST   /api/jobs              → create & start a new job
GET    /api/jobs              → list all jobs
GET    /api/jobs/{id}         → single job state
GET    /api/jobs/{id}/stream  → SSE live event stream
POST   /api/jobs/{id}/resume  → resume a stopped job
DELETE /api/jobs/{id}         → cancel & delete a job
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orchestrator.api.job_runner import (
    JobRecord, JobStatus, get_job, list_jobs, start_job, delete_job,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class JobCreate(BaseModel):
    project_description: str
    success_criteria: str
    budget_usd: float = Field(default=8.0, gt=0)
    time_limit_s: float = Field(default=5400, gt=0)
    concurrency: int = Field(default=3, ge=1, le=10)


class JobSummary(BaseModel):
    job_id: str
    project_description: str
    status: str
    total_cost: float
    final_status: str
    created_at: float
    updated_at: float
    task_count: int


def _to_summary(r: JobRecord) -> JobSummary:
    return JobSummary(
        job_id=r.job_id,
        project_description=r.project_description,
        status=r.status.value,
        total_cost=r.total_cost,
        final_status=r.final_status,
        created_at=r.created_at,
        updated_at=r.updated_at,
        task_count=len(r.tasks),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_job(body: JobCreate) -> dict[str, str]:
    job_id = str(uuid.uuid4())[:8]
    record = JobRecord(
        job_id=job_id,
        project_description=body.project_description,
        success_criteria=body.success_criteria,
        budget_usd=body.budget_usd,
        time_limit_s=body.time_limit_s,
        concurrency=body.concurrency,
    )
    await start_job(record)
    return {"job_id": job_id}


@router.get("")
async def list_all_jobs() -> list[JobSummary]:
    return [_to_summary(r) for r in list_jobs()]


@router.get("/{job_id}")
async def get_job_detail(job_id: str) -> dict[str, Any]:
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        **_to_summary(record).model_dump(),
        "tasks": record.tasks,
        "success_criteria": record.success_criteria,
        "budget_usd": record.budget_usd,
    }


@router.get("/{job_id}/stream")
async def stream_job(job_id: str):
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(record.event_queue.get(), timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"  # SSE comment to keep connection alive

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/resume", status_code=202)
async def resume_job(job_id: str) -> dict[str, str]:
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Job already running")
    # Reset queue and relaunch
    record.event_queue = asyncio.Queue()
    record.status = JobStatus.PENDING
    await start_job(record)
    return {"job_id": job_id, "status": "resuming"}


@router.delete("/{job_id}", status_code=204)
async def remove_job(job_id: str):
    deleted = await delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
```

**Step 3: Mount router in `main.py`**

```python
# Add after existing imports in orchestrator/api/main.py:
from orchestrator.api.routes.jobs import router as jobs_router
app.include_router(jobs_router)
```

**Step 4: Smoke test**

```bash
uvicorn orchestrator.api.main:app --port 8000 --reload
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"project_description":"test","success_criteria":"pass","budget_usd":0.01}'
```
Expected: `{"job_id": "xxxxxxxx"}`

**Step 5: Commit**

```bash
git add orchestrator/api/routes/ orchestrator/api/main.py
git commit -m "feat: jobs API router with CRUD + SSE stream"
```

---

## Task 5: Models telemetry endpoint

**Files:**
- Create: `orchestrator/api/routes/models.py`
- Modify: `orchestrator/api/main.py`

**Step 1: Create `orchestrator/api/routes/models.py`**

```python
"""
Models router — exposes live telemetry for all registered models.

GET /api/models → list of model stats (success_rate, trust, latency, cost, etc.)
"""
from __future__ import annotations

from fastapi import APIRouter
from orchestrator.models import build_default_profiles

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def get_model_stats() -> list[dict]:
    profiles = build_default_profiles()
    result = []
    for model, profile in profiles.items():
        result.append({
            "model": model.value,
            "provider": profile.provider,
            "success_rate": profile.success_rate,
            "trust_factor": profile.trust_factor,
            "avg_latency_ms": profile.avg_latency_ms,
            "latency_p95_ms": profile.latency_p95_ms,
            "avg_cost_usd": profile.avg_cost_usd,
            "call_count": profile.call_count,
            "failure_count": profile.failure_count,
            "cost_per_1m_input": profile.cost_per_1m_input,
            "cost_per_1m_output": profile.cost_per_1m_output,
        })
    return sorted(result, key=lambda x: x["provider"])
```

**Step 2: Mount in `main.py`**

```python
from orchestrator.api.routes.models import router as models_router
app.include_router(models_router)
```

**Step 3: Test**

```bash
curl http://localhost:8000/api/models | python -m json.tool | head -40
```
Expected: JSON array of model objects

**Step 4: Commit**

```bash
git add orchestrator/api/routes/models.py orchestrator/api/main.py
git commit -m "feat: models telemetry endpoint"
```

---

## Task 6: Next.js project scaffold

**Files:**
- Create: `ui/` directory (via `create-next-app`)

**Step 1: Scaffold Next.js app**

```bash
cd "E:\Documents\Vibe-Coding\Ai Orchestrator\.claude\worktrees\wonderful-bardeen"
npx create-next-app@latest ui \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --no-src-dir \
  --import-alias "@/*"
```

**Step 2: Install additional dependencies**

```bash
cd ui
npm install recharts js-yaml
npm install -D @types/js-yaml
npx shadcn@latest init
```
When prompted by shadcn: choose "Default" style, "Slate" base color, CSS variables yes.

**Step 3: Install shadcn components we'll use**

```bash
npx shadcn@latest add card badge button tabs progress table
```

**Step 4: Set API base URL in env**

Create `ui/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Step 5: Verify dev server**

```bash
npm run dev
```
Open http://localhost:3000 — should show default Next.js page.

**Step 6: Commit**

```bash
cd ..
git add ui/
git commit -m "feat: scaffold Next.js 15 app with Tailwind + shadcn/ui"
```

---

## Task 7: App layout — sidebar + navigation shell

**Files:**
- Modify: `ui/app/layout.tsx`
- Create: `ui/components/ModelSidebar.tsx`
- Create: `ui/lib/api.ts`

**Step 1: Create `ui/lib/api.ts`** — fetch wrappers

```typescript
// ui/lib/api.ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  jobs: {
    list: () => apiFetch<JobSummary[]>("/api/jobs"),
    get: (id: string) => apiFetch<JobDetail>(`/api/jobs/${id}`),
    create: (body: JobCreate) => apiFetch<{ job_id: string }>("/api/jobs", {
      method: "POST", body: JSON.stringify(body),
    }),
    resume: (id: string) => apiFetch(`/api/jobs/${id}/resume`, { method: "POST" }),
    delete: (id: string) => apiFetch(`/api/jobs/${id}`, { method: "DELETE" }),
  },
  models: {
    list: () => apiFetch<ModelStat[]>("/api/models"),
  },
};

// ── Types ────────────────────────────────────────────────────────────────────
export interface JobSummary {
  job_id: string;
  project_description: string;
  status: string;
  total_cost: number;
  final_status: string;
  created_at: number;
  updated_at: number;
  task_count: number;
}

export interface JobDetail extends JobSummary {
  tasks: Record<string, TaskInfo>;
  success_criteria: string;
  budget_usd: number;
}

export interface TaskInfo {
  task_id: string;
  status: string;
  score: number;
  model: string;
  cost: number;
  iterations: number;
}

export interface JobCreate {
  project_description: string;
  success_criteria: string;
  budget_usd: number;
  time_limit_s: number;
  concurrency: number;
}

export interface ModelStat {
  model: string;
  provider: string;
  success_rate: number;
  trust_factor: number;
  avg_latency_ms: number;
  latency_p95_ms: number;
  avg_cost_usd: number;
  call_count: number;
  failure_count: number;
  cost_per_1m_input: number;
  cost_per_1m_output: number;
}
```

**Step 2: Create `ui/components/ModelSidebar.tsx`**

```typescript
// ui/components/ModelSidebar.tsx
"use client";

import { useEffect, useState } from "react";
import { api, ModelStat } from "@/lib/api";
import Link from "next/link";

export function ModelSidebar() {
  const [models, setModels] = useState<ModelStat[]>([]);

  useEffect(() => {
    api.models.list().then(setModels).catch(console.error);
    const interval = setInterval(() => {
      api.models.list().then(setModels).catch(console.error);
    }, 15_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="w-56 shrink-0 border-r bg-muted/30 p-4 flex flex-col gap-4">
      <nav className="flex flex-col gap-1 text-sm font-medium">
        <Link href="/" className="hover:underline">Dashboard</Link>
        <Link href="/new" className="hover:underline">New Job</Link>
        <Link href="/history" className="hover:underline">History</Link>
        <Link href="/models" className="hover:underline">Models</Link>
      </nav>
      <hr />
      <div>
        <p className="text-xs uppercase text-muted-foreground mb-2">Models</p>
        {models.map((m) => (
          <div key={m.model} className="mb-2">
            <p className="text-xs font-semibold truncate">{m.model}</p>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>SR {(m.success_rate * 100).toFixed(0)}%</span>
              <span>T {m.trust_factor.toFixed(2)}</span>
              <span>${m.cost_per_1m_input}/1M</span>
            </div>
          </div>
        ))}
        <Link href="/models" className="text-xs text-blue-500 hover:underline">View all →</Link>
      </div>
    </aside>
  );
}
```

**Step 3: Update `ui/app/layout.tsx`**

```typescript
// ui/app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ModelSidebar } from "@/components/ModelSidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Orchestrator Dashboard",
  description: "Multi-LLM Orchestrator",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} flex h-screen overflow-hidden`}>
        <ModelSidebar />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </body>
    </html>
  );
}
```

**Step 4: Verify layout**

```bash
npm run dev
```
Open http://localhost:3000 — sidebar with nav links should appear.

**Step 5: Commit**

```bash
git add ui/app/layout.tsx ui/components/ModelSidebar.tsx ui/lib/api.ts
git commit -m "feat: app layout with sidebar nav and model stats widget"
```

---

## Task 8: `/new` page — form + YAML upload

**Files:**
- Create: `ui/app/new/page.tsx`
- Create: `ui/components/YamlUpload.tsx`
- Create: `ui/components/BudgetBar.tsx`

**Step 1: Create `ui/components/BudgetBar.tsx`**

```typescript
// ui/components/BudgetBar.tsx
import { Progress } from "@/components/ui/progress";

interface BudgetBarProps {
  spent: number;
  max: number;
  className?: string;
}

export function BudgetBar({ spent, max, className }: BudgetBarProps) {
  const pct = max > 0 ? Math.min((spent / max) * 100, 100) : 0;
  return (
    <div className={className}>
      <div className="flex justify-between text-xs text-muted-foreground mb-1">
        <span>${spent.toFixed(4)} spent</span>
        <span>${max.toFixed(2)} max</span>
      </div>
      <Progress value={pct} className="h-2" />
    </div>
  );
}
```

**Step 2: Create `ui/components/YamlUpload.tsx`**

```typescript
// ui/components/YamlUpload.tsx
"use client";

import { useRef, useState } from "react";
import yaml from "js-yaml";
import { Button } from "@/components/ui/button";
import { JobCreate } from "@/lib/api";

interface YamlUploadProps {
  onParsed: (data: Partial<JobCreate>) => void;
}

export function YamlUpload({ onParsed }: YamlUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string>("");
  const [error, setError] = useState<string>("");

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      try {
        const parsed = yaml.load(text) as Record<string, unknown>;
        setPreview(JSON.stringify(parsed, null, 2));
        setError("");
        // Map YAML fields to JobCreate fields
        onParsed({
          project_description: (parsed.project_description as string) ?? "",
          success_criteria: (parsed.success_criteria as string) ?? "",
          budget_usd: (parsed.budget_usd as number) ?? 8.0,
          time_limit_s: (parsed.time_limit_s as number) ?? 5400,
          concurrency: (parsed.concurrency as number) ?? 3,
        });
      } catch (err) {
        setError(`YAML parse error: ${err}`);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div
      className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".yaml,.yml"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
      <p className="text-muted-foreground">Drag & drop a YAML file or click to browse</p>
      {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
      {preview && (
        <pre className="mt-4 text-left text-xs bg-muted p-3 rounded overflow-auto max-h-48">
          {preview}
        </pre>
      )}
    </div>
  );
}
```

**Step 3: Create `ui/app/new/page.tsx`**

```typescript
// ui/app/new/page.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { YamlUpload } from "@/components/YamlUpload";
import { api, JobCreate } from "@/lib/api";

const defaults: JobCreate = {
  project_description: "",
  success_criteria: "",
  budget_usd: 8,
  time_limit_s: 5400,
  concurrency: 3,
};

export default function NewJobPage() {
  const router = useRouter();
  const [form, setForm] = useState<JobCreate>(defaults);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const set = (key: keyof JobCreate) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const submit = async () => {
    setLoading(true);
    setError("");
    try {
      const { job_id } = await api.jobs.create(form);
      router.push(`/jobs/${job_id}`);
    } catch (err) {
      setError(String(err));
      setLoading(false);
    }
  };

  const fieldClass = "w-full border rounded px-3 py-2 text-sm bg-background";

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">New Job</h1>
      <Tabs defaultValue="form">
        <TabsList className="mb-4">
          <TabsTrigger value="form">Form</TabsTrigger>
          <TabsTrigger value="yaml">YAML Upload</TabsTrigger>
        </TabsList>

        <TabsContent value="form">
          <Card>
            <CardHeader><CardTitle>Project Details</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Project description</label>
                <textarea className={`${fieldClass} mt-1`} rows={3} value={form.project_description} onChange={set("project_description")} />
              </div>
              <div>
                <label className="text-sm font-medium">Success criteria</label>
                <textarea className={`${fieldClass} mt-1`} rows={2} value={form.success_criteria} onChange={set("success_criteria")} />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-sm font-medium">Budget (USD)</label>
                  <input type="number" className={`${fieldClass} mt-1`} value={form.budget_usd} onChange={set("budget_usd")} step="0.5" min="0.1" />
                </div>
                <div>
                  <label className="text-sm font-medium">Time limit (s)</label>
                  <input type="number" className={`${fieldClass} mt-1`} value={form.time_limit_s} onChange={set("time_limit_s")} step="60" min="60" />
                </div>
                <div>
                  <label className="text-sm font-medium">Concurrency</label>
                  <input type="number" className={`${fieldClass} mt-1`} value={form.concurrency} onChange={set("concurrency")} min="1" max="10" />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="yaml">
          <Card>
            <CardHeader><CardTitle>Upload YAML</CardTitle></CardHeader>
            <CardContent>
              <YamlUpload onParsed={(data) => setForm((f) => ({ ...f, ...data }))} />
              {form.project_description && (
                <p className="mt-3 text-sm text-green-600">✓ Parsed: {form.project_description.slice(0, 60)}…</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {error && <p className="text-red-500 text-sm mt-3">{error}</p>}
      <Button className="mt-4 w-full" onClick={submit} disabled={loading || !form.project_description}>
        {loading ? "Starting…" : "Start Job"}
      </Button>
    </div>
  );
}
```

**Step 4: Verify**

```bash
npm run dev
```
Navigate to http://localhost:3000/new — form and YAML tabs should render.

**Step 5: Commit**

```bash
git add ui/app/new/ ui/components/YamlUpload.tsx ui/components/BudgetBar.tsx
git commit -m "feat: /new page with form and YAML upload tabs"
```

---

## Task 9: `/jobs/[id]` — live job view with SSE

**Files:**
- Create: `ui/app/jobs/[id]/page.tsx`
- Create: `ui/components/TaskCard.tsx`
- Create: `ui/components/LogStream.tsx`
- Create: `ui/lib/sse.ts`

**Step 1: Create `ui/lib/sse.ts`**

```typescript
// ui/lib/sse.ts
"use client";

import { useEffect, useRef, useState } from "react";

export interface SSEEvent {
  type: "log" | "task" | "done";
  [key: string]: unknown;
}

export function useSSE(url: string, enabled: boolean = true) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [done, setDone] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e) => {
      const event: SSEEvent = JSON.parse(e.data);
      setEvents((prev) => [...prev, event]);
      if (event.type === "done") {
        setDone(true);
        es.close();
      }
    };

    es.onerror = () => {
      es.close();
    };

    return () => { es.close(); };
  }, [url, enabled]);

  return { events, done };
}
```

**Step 2: Create `ui/components/TaskCard.tsx`**

```typescript
// ui/components/TaskCard.tsx
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { TaskInfo } from "@/lib/api";

const statusColor: Record<string, string> = {
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  running: "bg-blue-100 text-blue-800",
  pending: "bg-gray-100 text-gray-800",
  degraded: "bg-yellow-100 text-yellow-800",
};

interface TaskCardProps {
  task: TaskInfo;
}

export function TaskCard({ task }: TaskCardProps) {
  return (
    <Card className="text-sm">
      <CardContent className="pt-4 space-y-1">
        <div className="flex items-center justify-between">
          <span className="font-mono font-semibold">{task.task_id}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor[task.status] ?? "bg-gray-100"}`}>
            {task.status}
          </span>
        </div>
        <div className="text-muted-foreground text-xs">{task.model}</div>
        <div className="flex gap-3 text-xs">
          <span>Score: <strong>{task.score?.toFixed(3)}</strong></span>
          <span>Cost: <strong>${task.cost?.toFixed(4)}</strong></span>
          <span>Iters: <strong>{task.iterations}</strong></span>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Step 3: Create `ui/components/LogStream.tsx`**

```typescript
// ui/components/LogStream.tsx
"use client";

import { useEffect, useRef } from "react";
import { SSEEvent } from "@/lib/sse";

const levelColor: Record<string, string> = {
  ERROR: "text-red-400",
  WARNING: "text-yellow-400",
  INFO: "text-green-300",
  DEBUG: "text-gray-400",
};

interface LogStreamProps {
  events: SSEEvent[];
}

export function LogStream({ events }: LogStreamProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const logs = events.filter((e) => e.type === "log");

  return (
    <div className="bg-gray-950 rounded-lg p-4 font-mono text-xs overflow-y-auto h-64">
      {logs.length === 0 && (
        <p className="text-gray-500">Waiting for logs…</p>
      )}
      {logs.map((e, i) => (
        <div key={i} className="flex gap-2">
          <span className={levelColor[e.level as string] ?? "text-gray-400"}>
            [{e.level as string}]
          </span>
          <span className="text-gray-200">{e.msg as string}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

**Step 4: Create `ui/app/jobs/[id]/page.tsx`**

```typescript
// ui/app/jobs/[id]/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BudgetBar } from "@/components/BudgetBar";
import { TaskCard } from "@/components/TaskCard";
import { LogStream } from "@/components/LogStream";
import { useSSE } from "@/lib/sse";
import { api, JobDetail, TaskInfo } from "@/lib/api";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const statusVariant: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  running: "default",
  done: "secondary",
  failed: "destructive",
  pending: "outline",
};

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [job, setJob] = useState<JobDetail | null>(null);
  const [tasks, setTasks] = useState<Record<string, TaskInfo>>({});
  const [totalCost, setTotalCost] = useState(0);

  const { events, done } = useSSE(`${BASE}/api/jobs/${id}/stream`, !!id);

  // Poll job state initially + after SSE ends
  useEffect(() => {
    api.jobs.get(id).then((j) => { setJob(j); setTasks(j.tasks); }).catch(console.error);
  }, [id, done]);

  // Apply SSE task events live
  useEffect(() => {
    for (const e of events) {
      if (e.type === "task") {
        const t = e as unknown as TaskInfo & { type: string };
        setTasks((prev) => ({ ...prev, [t.task_id]: t }));
      }
      if (e.type === "done") {
        setTotalCost(e.total_cost as number);
      }
    }
  }, [events]);

  const handleDelete = async () => {
    if (!confirm("Delete this job?")) return;
    await api.jobs.delete(id);
    router.push("/history");
  };

  const handleResume = async () => {
    await api.jobs.resume(id);
    window.location.reload();
  };

  if (!job) return <p className="text-muted-foreground">Loading…</p>;

  const taskList = Object.values(tasks);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold">{job.project_description.slice(0, 80)}</h1>
          <p className="text-sm text-muted-foreground">{job.job_id}</p>
        </div>
        <Badge variant={statusVariant[job.status] ?? "outline"}>{job.status}</Badge>
      </div>

      <BudgetBar spent={totalCost || job.total_cost} max={job.budget_usd} />

      {/* Actions */}
      <div className="flex gap-2">
        {job.status !== "running" && (
          <Button variant="outline" size="sm" onClick={handleResume}>Resume</Button>
        )}
        <Button variant="destructive" size="sm" onClick={handleDelete}>Delete</Button>
      </div>

      {/* Task grid */}
      <div>
        <h2 className="font-semibold mb-3">Tasks ({taskList.length})</h2>
        {taskList.length === 0 ? (
          <p className="text-muted-foreground text-sm">No tasks yet…</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {taskList.map((t) => <TaskCard key={t.task_id} task={t} />)}
          </div>
        )}
      </div>

      {/* Log stream */}
      <div>
        <h2 className="font-semibold mb-2">Live Logs</h2>
        <LogStream events={events} />
      </div>
    </div>
  );
}
```

**Step 5: Verify**

```bash
npm run dev
```
Navigate to http://localhost:3000/jobs/any-id — should render job page (error state if job not found is OK).

**Step 6: Commit**

```bash
git add ui/app/jobs/ ui/components/TaskCard.tsx ui/components/LogStream.tsx ui/lib/sse.ts
git commit -m "feat: /jobs/[id] live view with SSE task cards and log stream"
```

---

## Task 10: Dashboard (`/`) + History (`/history`)

**Files:**
- Modify: `ui/app/page.tsx`
- Create: `ui/components/JobCard.tsx`
- Create: `ui/app/history/page.tsx`

**Step 1: Create `ui/components/JobCard.tsx`**

```typescript
// ui/components/JobCard.tsx
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BudgetBar } from "@/components/BudgetBar";
import { JobSummary } from "@/lib/api";

const statusVariant: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  running: "default",
  done: "secondary",
  failed: "destructive",
  pending: "outline",
};

interface JobCardProps {
  job: JobSummary;
}

export function JobCard({ job }: JobCardProps) {
  return (
    <Link href={`/jobs/${job.job_id}`}>
      <Card className="hover:border-primary transition-colors cursor-pointer">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm truncate">{job.project_description.slice(0, 60)}</CardTitle>
            <Badge variant={statusVariant[job.status] ?? "outline"}>{job.status}</Badge>
          </div>
          <p className="text-xs text-muted-foreground font-mono">{job.job_id}</p>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{job.task_count} tasks</span>
            <span>${job.total_cost.toFixed(4)}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
```

**Step 2: Update `ui/app/page.tsx`**

```typescript
// ui/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { JobCard } from "@/components/JobCard";
import { api, JobSummary } from "@/lib/api";

export default function DashboardPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);

  useEffect(() => {
    const load = () => api.jobs.list().then(setJobs).catch(console.error);
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  const active = jobs.filter((j) => j.status === "running" || j.status === "pending");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <Button asChild><Link href="/new">+ New Job</Link></Button>
      </div>

      <div>
        <h2 className="font-semibold mb-3">Active Jobs ({active.length})</h2>
        {active.length === 0 ? (
          <p className="text-muted-foreground text-sm">No active jobs. <Link href="/new" className="underline">Start one →</Link></p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {active.map((j) => <JobCard key={j.job_id} job={j} />)}
          </div>
        )}
      </div>
    </div>
  );
}
```

**Step 3: Create `ui/app/history/page.tsx`**

```typescript
// ui/app/history/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { api, JobSummary } from "@/lib/api";

export default function HistoryPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobSummary[]>([]);

  const load = () => api.jobs.list().then(setJobs).catch(console.error);

  useEffect(() => { load(); }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this job?")) return;
    await api.jobs.delete(id);
    load();
  };

  const handleResume = async (id: string) => {
    await api.jobs.resume(id);
    router.push(`/jobs/${id}`);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">History</h1>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Description</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Cost</TableHead>
            <TableHead>Date</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {jobs.map((j) => (
            <TableRow key={j.job_id} className="cursor-pointer" onClick={() => router.push(`/jobs/${j.job_id}`)}>
              <TableCell className="font-mono text-xs">{j.job_id}</TableCell>
              <TableCell className="max-w-xs truncate">{j.project_description.slice(0, 60)}</TableCell>
              <TableCell><Badge variant="outline">{j.status}</Badge></TableCell>
              <TableCell>${j.total_cost.toFixed(4)}</TableCell>
              <TableCell className="text-xs">{new Date(j.created_at * 1000).toLocaleDateString()}</TableCell>
              <TableCell onClick={(e) => e.stopPropagation()}>
                <div className="flex gap-1">
                  <Button size="sm" variant="outline" onClick={() => handleResume(j.job_id)}>Resume</Button>
                  <Button size="sm" variant="destructive" onClick={() => handleDelete(j.job_id)}>Delete</Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {jobs.length === 0 && <p className="text-muted-foreground text-sm">No jobs yet.</p>}
    </div>
  );
}
```

**Step 4: Verify**

```bash
npm run dev
```
Dashboard at http://localhost:3000, history at http://localhost:3000/history.

**Step 5: Commit**

```bash
git add ui/app/page.tsx ui/app/history/ ui/components/JobCard.tsx
git commit -m "feat: dashboard and history pages"
```

---

## Task 11: `/models` — telemetry page with Recharts

**Files:**
- Create: `ui/app/models/page.tsx`

**Step 1: Create `ui/app/models/page.tsx`**

```typescript
// ui/app/models/page.tsx
"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { api, ModelStat } from "@/lib/api";

const PROVIDER_COLORS: Record<string, string> = {
  anthropic: "#d97706",
  openai: "#10b981",
  google: "#3b82f6",
  kimi: "#8b5cf6",
  deepseek: "#ef4444",
};

export default function ModelsPage() {
  const [models, setModels] = useState<ModelStat[]>([]);

  useEffect(() => {
    api.models.list().then(setModels).catch(console.error);
  }, []);

  const costData = models.map((m) => ({
    name: m.model.split("-").slice(0, 2).join("-"),
    input: m.cost_per_1m_input,
    provider: m.provider,
  }));

  const latencyData = models.map((m) => ({
    name: m.model.split("-").slice(0, 2).join("-"),
    avg: m.avg_latency_ms,
    p95: m.latency_p95_ms,
    provider: m.provider,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Model Telemetry</h1>

      {/* Per-model cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {models.map((m) => (
          <Card key={m.model}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-mono">{m.model}</CardTitle>
              <p className="text-xs text-muted-foreground capitalize">{m.provider}</p>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>Success rate</span>
                  <span>{(m.success_rate * 100).toFixed(0)}%</span>
                </div>
                <Progress value={m.success_rate * 100} className="h-1.5" />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>Trust factor</span>
                  <span>{m.trust_factor.toFixed(3)}</span>
                </div>
                <Progress value={m.trust_factor * 100} className="h-1.5" />
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-muted-foreground">Avg latency</p>
                  <p className="font-semibold">{m.avg_latency_ms.toFixed(0)}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">p95 latency</p>
                  <p className="font-semibold">{m.latency_p95_ms.toFixed(0)}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Calls</p>
                  <p className="font-semibold">{m.call_count}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">$/1M input</p>
                  <p className="font-semibold">${m.cost_per_1m_input}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle className="text-sm">Cost per 1M input tokens (USD)</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={costData} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 9 }} width={90} />
                <Tooltip />
                <Bar dataKey="input">
                  {costData.map((d, i) => (
                    <Cell key={i} fill={PROVIDER_COLORS[d.provider] ?? "#6b7280"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-sm">Avg latency (ms)</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={latencyData} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 9 }} width={90} />
                <Tooltip />
                <Bar dataKey="avg">
                  {latencyData.map((d, i) => (
                    <Cell key={i} fill={PROVIDER_COLORS[d.provider] ?? "#6b7280"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

**Step 2: Verify**

```bash
npm run dev
```
Navigate to http://localhost:3000/models — cards and charts should render.

**Step 3: Commit**

```bash
git add ui/app/models/
git commit -m "feat: /models telemetry page with per-model cards and Recharts"
```

---

## Task 12: Add FastAPI + uvicorn to pyproject.toml and final integration test

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies**

```toml
# In pyproject.toml, add to [project] dependencies:
"fastapi>=0.111",
"uvicorn[standard]>=0.30",
"python-multipart>=0.0.9",
"pyyaml>=6.0",
```

**Step 2: Full integration smoke test**

Terminal 1 — start API:
```bash
uvicorn orchestrator.api.main:app --port 8000 --reload
```

Terminal 2 — start UI:
```bash
cd ui && npm run dev
```

Open http://localhost:3000, click "New Job", fill form with:
- Description: `Write a hello world Python script`
- Criteria: `Script prints Hello World`
- Budget: `0.50`

Submit → should redirect to `/jobs/[id]` and show live log stream.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add FastAPI/uvicorn to project dependencies"
```

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | FastAPI skeleton + CORS |
| 2 | SSELogHandler + queue |
| 3 | JobRunner + in-memory registry |
| 4 | Jobs router (CRUD + SSE) |
| 5 | Models telemetry endpoint |
| 6 | Next.js scaffold + shadcn/ui |
| 7 | Layout + sidebar + api.ts |
| 8 | `/new` page (form + YAML) |
| 9 | `/jobs/[id]` live view |
| 10 | Dashboard + History |
| 11 | `/models` telemetry + Recharts |
| 12 | pyproject.toml + integration test |
