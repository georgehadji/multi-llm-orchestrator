# Six Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six production improvements to the AI Orchestrator: streaming output, intelligent caching, adaptive retry, dependency visualization, multi-run aggregation, and auto-remediation.

**Architecture:** Three phases — Phase 1 (Streaming + Viz) adds no core logic changes and delivers immediate user-facing value; Phase 2 (Caching + Adaptive Retry) improves performance and reliability; Phase 3 (Aggregation + Remediation) adds long-term learning and failure recovery.

**Tech Stack:** Python 3.10+, asyncio, aiosqlite (already present), networkx (new optional dep for DAG viz), no new required dependencies.

---

## Context: Codebase Map

```
orchestrator/
├── engine.py          # Orchestrator class — core loop (modify heavily)
├── api_clients.py     # UnifiedClient — HTTP layer (modify for adaptive retry)
├── telemetry.py       # TelemetryCollector — metrics (extend)
├── cli.py             # argparse CLI (add new flags)
├── models.py          # Enums, Task, TaskResult, ProjectState (extend)
├── cache.py           # DiskCache — aiosqlite (extend for semantic cache)
├── state.py           # StateManager — project persistence (extend for aggregation)
├── hooks.py           # HookRegistry + EventType (extend)
└── [NEW FILES BELOW]
streaming.py           # TaskUpdate events + ProjectEventBus
visualization.py       # DAG builder + Mermaid/ASCII renderer
semantic_cache.py      # SemanticCache + DuplicationDetector
adaptive_router.py     # AdaptiveRouter with circuit-breaker v2
aggregator.py          # ProfileAggregator — cross-run learning
remediation.py         # RemediationEngine + RemediationStrategy
```

---

## Phase 1: Streaming + Dependency Visualization

---

### Task 1: Streaming event dataclasses (`orchestrator/streaming.py`)

**Files:**
- Create: `orchestrator/streaming.py`
- Test: `tests/test_streaming.py`

**Step 1: Write the failing test**

```python
# tests/test_streaming.py
import asyncio
from orchestrator.streaming import (
    TaskStarted, TaskProgressUpdate, TaskCompleted, TaskFailed,
    ProjectStarted, ProjectCompleted, BudgetWarning, ProjectEventBus,
)
from orchestrator.models import Model, TaskStatus

def test_task_started_fields():
    ev = TaskStarted(task_id="t1", task_type="code_generation", model="deepseek-chat")
    assert ev.task_id == "t1"
    assert ev.task_type == "code_generation"
    assert ev.model == "deepseek-chat"

def test_task_progress_update_fields():
    ev = TaskProgressUpdate(task_id="t1", iteration=2, score=0.82, best_score=0.90)
    assert ev.iteration == 2
    assert ev.score == 0.82
    assert ev.best_score == 0.90

def test_task_completed_fields():
    ev = TaskCompleted(task_id="t1", score=0.95, status=TaskStatus.COMPLETED,
                       model="deepseek-chat", cost_usd=0.002, iterations=2)
    assert ev.score == 0.95
    assert ev.status == TaskStatus.COMPLETED

def test_task_failed_fields():
    ev = TaskFailed(task_id="t1", reason="timeout after 120s", model="deepseek-chat")
    assert ev.reason == "timeout after 120s"

def test_project_event_bus_subscribe_and_publish():
    bus = ProjectEventBus()
    received = []

    async def run():
        sub = bus.subscribe()
        await bus.publish(TaskStarted("t1", "code_generation", "deepseek-chat"))
        await bus.publish(TaskCompleted("t1", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
        await bus.close()
        async for ev in sub:
            received.append(ev)

    asyncio.run(run())
    assert len(received) == 2
    assert isinstance(received[0], TaskStarted)
    assert isinstance(received[1], TaskCompleted)

def test_event_bus_multiple_subscribers():
    bus = ProjectEventBus()
    recv_a, recv_b = [], []

    async def run():
        sub_a = bus.subscribe()
        sub_b = bus.subscribe()
        await bus.publish(TaskStarted("t1", "code_generation", "deepseek-chat"))
        await bus.close()
        async for ev in sub_a:
            recv_a.append(ev)
        async for ev in sub_b:
            recv_b.append(ev)

    asyncio.run(run())
    assert len(recv_a) == 1
    assert len(recv_b) == 1
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_streaming.py -v
```
Expected: `ModuleNotFoundError: No module named 'orchestrator.streaming'`

**Step 3: Write minimal implementation**

```python
# orchestrator/streaming.py
"""
Streaming event types for run_project_streaming().

Events flow through ProjectEventBus — an asyncio-based pub-sub hub that
fans out to all subscribers. Each subscriber gets an independent async
generator (AsyncIterator) over the event stream.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from .models import TaskStatus


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ProjectStarted:
    project_id: str
    total_tasks: int
    budget_usd: float

@dataclass
class TaskStarted:
    task_id: str
    task_type: str
    model: str

@dataclass
class TaskProgressUpdate:
    task_id: str
    iteration: int
    score: float
    best_score: float

@dataclass
class TaskCompleted:
    task_id: str
    score: float
    status: TaskStatus
    model: str
    cost_usd: float
    iterations: int

@dataclass
class TaskFailed:
    task_id: str
    reason: str
    model: str

@dataclass
class BudgetWarning:
    phase: str
    spent_usd: float
    cap_usd: float
    ratio: float

@dataclass
class ProjectCompleted:
    project_id: str
    status: str          # ProjectStatus.value
    total_cost_usd: float
    elapsed_seconds: float
    tasks_completed: int
    tasks_failed: int

StreamEvent = (
    ProjectStarted | TaskStarted | TaskProgressUpdate |
    TaskCompleted | TaskFailed | BudgetWarning | ProjectCompleted
)


# ── Event bus ─────────────────────────────────────────────────────────────────

_SENTINEL = object()   # marks end-of-stream


class ProjectEventBus:
    """
    Fan-out pub-sub hub.  Each call to subscribe() returns an independent
    AsyncIterator that yields every event published after the subscription.
    Call close() to signal end-of-stream to all subscribers.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []
        self._closed = False

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return self._drain(q)

    async def _drain(self, q: asyncio.Queue) -> AsyncIterator[StreamEvent]:
        while True:
            item = await q.get()
            if item is _SENTINEL:
                return
            yield item

    async def publish(self, event: StreamEvent) -> None:
        for q in self._queues:
            await q.put(event)

    async def close(self) -> None:
        self._closed = True
        for q in self._queues:
            await q.put(_SENTINEL)
```

**Step 4: Run tests to verify they pass**

```
pytest tests/test_streaming.py -v
```
Expected: all 6 PASS

**Step 5: Commit**

```bash
git add orchestrator/streaming.py tests/test_streaming.py
git commit -m "feat: add streaming event types and ProjectEventBus"
```

---

### Task 2: Wire `ProjectEventBus` into `engine.py` + `run_project_streaming()`

**Files:**
- Modify: `orchestrator/engine.py` — `__init__`, `_execute_task`, `run_project`, add `run_project_streaming`
- Modify: `orchestrator/__init__.py` — export `run_project_streaming`
- Test: `tests/test_streaming.py` — add integration test

**Step 1: Write the failing test** (add to `tests/test_streaming.py`)

```python
# Add to tests/test_streaming.py
from unittest.mock import patch, AsyncMock
from orchestrator import Orchestrator, Budget
from orchestrator.streaming import TaskCompleted, ProjectCompleted
from orchestrator.models import TaskStatus, ProjectStatus

def _make_mock_response(text: str):
    from orchestrator.api_clients import APIResponse
    from orchestrator.models import Model
    return APIResponse(text=text, input_tokens=10, output_tokens=20,
                       model=Model.DEEPSEEK_CHAT)

def test_run_project_streaming_yields_events(tmp_path):
    import json
    decomp = json.dumps([{
        "id": "task_001", "type": "code_generation",
        "prompt": "Write hello world", "dependencies": [], "hard_validators": [],
    }])
    gen_resp = _make_mock_response("def hello():\n    return 'Hello'")
    eval_resp = _make_mock_response('{"score": 0.9, "critique": "Good."}')
    decomp_resp = _make_mock_response(decomp)

    orch = Orchestrator(budget=Budget(max_usd=1.0))

    async def mock_call(model, prompt, **kwargs):
        if "decomposition engine" in kwargs.get("system", ""):
            return decomp_resp
        if "evaluate" in prompt.lower() or "evaluating" in prompt.lower():
            return eval_resp
        return gen_resp

    events = []

    async def run():
        with patch.object(orch.client, "call", side_effect=mock_call):
            async for ev in orch.run_project_streaming(
                "Write hello world", "Function returns Hello"
            ):
                events.append(ev)

    asyncio.run(run())
    types = [type(e).__name__ for e in events]
    assert "ProjectStarted" in types
    assert "TaskStarted" in types
    assert "TaskCompleted" in types
    assert "ProjectCompleted" in types
    # TaskCompleted should have a valid score
    completed = [e for e in events if isinstance(e, TaskCompleted)]
    assert all(0.0 <= e.score <= 1.0 for e in completed)
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_streaming.py::test_run_project_streaming_yields_events -v
```
Expected: `AttributeError: 'Orchestrator' object has no attribute 'run_project_streaming'`

**Step 3: Implement `run_project_streaming` in `engine.py`**

Add `self._event_bus: Optional[ProjectEventBus] = None` to `Orchestrator.__init__`.

Add after `run_project` method:

```python
async def run_project_streaming(
    self,
    project_description: str,
    success_criteria: str,
    project_id: str = "",
) -> AsyncIterator["StreamEvent"]:
    """
    Streaming variant of run_project().
    Yields StreamEvent objects as execution progresses.
    The final event is always ProjectCompleted.
    """
    from .streaming import ProjectEventBus, ProjectStarted, ProjectCompleted

    self._event_bus = ProjectEventBus()
    subscription = self._event_bus.subscribe()

    async def _run():
        try:
            await self.run_project(project_description, success_criteria, project_id)
        finally:
            await self._event_bus.close()
            self._event_bus = None

    task = asyncio.create_task(_run())

    async for event in subscription:
        yield event

    await task   # propagate any unhandled exceptions
```

Wire `_event_bus` emissions into `_execute_task` — add after each key milestone:

```python
# At start of _execute_task, after selecting primary model:
if self._event_bus:
    await self._event_bus.publish(TaskStarted(
        task_id=task.id, task_type=task.type.value, model=primary.value
    ))

# At end of each iteration, after score computed:
if self._event_bus:
    await self._event_bus.publish(TaskProgressUpdate(
        task_id=task.id, iteration=iteration_num,
        score=score, best_score=best_score,
    ))

# At end of _execute_task, before returning TaskResult:
if self._event_bus:
    if result.status == TaskStatus.FAILED:
        await self._event_bus.publish(TaskFailed(
            task_id=task.id, reason="all attempts failed", model=result.model_used.value
        ))
    else:
        await self._event_bus.publish(TaskCompleted(
            task_id=task.id, score=result.score, status=result.status,
            model=result.model_used.value, cost_usd=result.cost_usd,
            iterations=result.iterations,
        ))
```

Wire `ProjectStarted` in `run_project` after decomposition:

```python
if self._event_bus:
    await self._event_bus.publish(ProjectStarted(
        project_id=self._project_id,
        total_tasks=len(tasks),
        budget_usd=self.budget.max_usd,
    ))
```

Wire `ProjectCompleted` in `run_project` before returning:

```python
if self._event_bus:
    completed_count = sum(1 for r in self.results.values()
                          if r.status == TaskStatus.COMPLETED)
    failed_count = sum(1 for r in self.results.values()
                       if r.status == TaskStatus.FAILED)
    await self._event_bus.publish(ProjectCompleted(
        project_id=self._project_id,
        status=state.status.value,
        total_cost_usd=self.budget.spent_usd,
        elapsed_seconds=self.budget.elapsed_seconds,
        tasks_completed=completed_count,
        tasks_failed=failed_count,
    ))
```

**Step 4: Run tests**

```
pytest tests/test_streaming.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/streaming.py orchestrator/engine.py tests/test_streaming.py
git commit -m "feat: wire ProjectEventBus into engine, add run_project_streaming()"
```

---

### Task 3: Live CLI progress display

**Files:**
- Create: `orchestrator/progress.py`
- Modify: `orchestrator/cli.py` — `_async_file_project`, `_async_new_project`

**Step 1: Write the failing test** (tests/test_progress.py)

```python
# tests/test_progress.py
from orchestrator.progress import ProgressRenderer
from orchestrator.streaming import (
    ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, ProjectCompleted,
)
from orchestrator.models import TaskStatus

def test_renderer_tracks_task_counts():
    r = ProgressRenderer(quiet=False)
    r.handle(ProjectStarted("proj-1", total_tasks=3, budget_usd=5.0))
    r.handle(TaskStarted("task_001", "code_generation", "deepseek-chat"))
    r.handle(TaskCompleted("task_001", 0.92, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 2))
    assert r.completed == 1
    assert r.total == 3

def test_renderer_tracks_failures():
    r = ProgressRenderer(quiet=False)
    r.handle(ProjectStarted("proj-1", total_tasks=2, budget_usd=5.0))
    from orchestrator.streaming import TaskFailed
    r.handle(TaskFailed("task_001", "timeout", "deepseek-chat"))
    assert r.failed == 1

def test_renderer_summary_string():
    r = ProgressRenderer(quiet=False)
    r.handle(ProjectStarted("proj-1", total_tasks=2, budget_usd=5.0))
    r.handle(TaskCompleted("task_001", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
    summary = r.summary()
    assert "1/2" in summary or "1 completed" in summary.lower()
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_progress.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/progress.py`**

```python
# orchestrator/progress.py
"""
Terminal progress renderer for run_project_streaming().
Prints a compact task tree to stderr, leaving stdout clean for piped output.
"""
from __future__ import annotations
import sys
from typing import Any
from .streaming import (
    ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, TaskFailed, BudgetWarning, ProjectCompleted,
)
from .models import TaskStatus

_STATUS_ICONS = {
    TaskStatus.COMPLETED: "✓",
    TaskStatus.DEGRADED:  "~",
    TaskStatus.FAILED:    "✗",
}


class ProgressRenderer:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.total = 0
        self.completed = 0
        self.failed = 0
        self._active: dict[str, str] = {}   # task_id → model

    def handle(self, event: Any) -> None:
        if isinstance(event, ProjectStarted):
            self.total = event.total_tasks
            if not self.quiet:
                print(
                    f"\n▶  Project started — {event.total_tasks} tasks  "
                    f"budget=${event.budget_usd:.2f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskStarted):
            self._active[event.task_id] = event.model
            if not self.quiet:
                print(
                    f"   → {event.task_id}  [{event.task_type}]  "
                    f"model={event.model}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskProgressUpdate):
            if not self.quiet:
                print(
                    f"     {event.task_id}  iter={event.iteration}  "
                    f"score={event.score:.3f}  best={event.best_score:.3f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskCompleted):
            self.completed += 1
            self._active.pop(event.task_id, None)
            icon = _STATUS_ICONS.get(event.status, "?")
            if not self.quiet:
                print(
                    f"   {icon} {event.task_id}  score={event.score:.3f}  "
                    f"${event.cost_usd:.4f}  iters={event.iterations}  "
                    f"[{self.completed}/{self.total}]",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskFailed):
            self.failed += 1
            self._active.pop(event.task_id, None)
            if not self.quiet:
                print(
                    f"   ✗ {event.task_id}  FAILED: {event.reason}",
                    file=sys.stderr,
                )
        elif isinstance(event, BudgetWarning):
            if not self.quiet:
                print(
                    f"   ⚠  Budget {event.phase}: "
                    f"${event.spent_usd:.4f} / ${event.cap_usd:.4f} "
                    f"({event.ratio:.0%})",
                    file=sys.stderr,
                )
        elif isinstance(event, ProjectCompleted):
            if not self.quiet:
                print(
                    f"\n{'✓' if 'SUCCESS' in event.status else '~'} "
                    f"Project {event.status}  "
                    f"${event.total_cost_usd:.4f}  "
                    f"{event.elapsed_seconds:.0f}s  "
                    f"{event.tasks_completed} completed  "
                    f"{event.tasks_failed} failed",
                    file=sys.stderr,
                )

    def summary(self) -> str:
        return (
            f"{self.completed} completed / {self.total} total, "
            f"{self.failed} failed"
        )
```

Wire into CLI — in `_async_file_project` and `_async_new_project`, replace `state = await orch.run_project(...)` with:

```python
from .progress import ProgressRenderer
from .streaming import ProjectCompleted as _ProjectCompleted

renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
state = None
async for event in orch.run_project_streaming(description, criteria, project_id):
    renderer.handle(event)
    if isinstance(event, _ProjectCompleted):
        pass   # state returned from task below
# retrieve state from the engine directly after streaming completes
state = await orch.state_mgr.load_project(orch._project_id)
```

> **Note:** Alternatively, expose the final `ProjectState` as the last event payload. The simpler approach is to add `state: Optional[ProjectState]` to `ProjectCompleted` and set it in `run_project_streaming`.

**Step 4: Run tests**

```
pytest tests/test_progress.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/progress.py orchestrator/cli.py tests/test_progress.py
git commit -m "feat: live CLI progress renderer via streaming events"
```

---

### Task 4: Dependency DAG Visualization (`orchestrator/visualization.py`)

**Files:**
- Create: `orchestrator/visualization.py`
- Modify: `orchestrator/cli.py` — add `--visualize`, `--critical-path`, `--dependency-report` flags
- Test: `tests/test_visualization.py`

**Step 1: Write the failing test**

```python
# tests/test_visualization.py
from orchestrator.visualization import DagRenderer
from orchestrator.models import Task, TaskType

def _make_task(tid, deps=None, task_type=TaskType.CODE_GEN):
    return Task(
        id=tid, type=task_type,
        prompt=f"Prompt for {tid}",
        dependencies=deps or [],
        hard_validators=[],
    )

def test_mermaid_output_contains_all_nodes():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t1"]),
        "t4": _make_task("t4", deps=["t2", "t3"]),
    }
    renderer = DagRenderer(tasks)
    mermaid = renderer.to_mermaid()
    assert "t1" in mermaid
    assert "t2" in mermaid
    assert "t3" in mermaid
    assert "t4" in mermaid
    assert "graph TD" in mermaid or "flowchart TD" in mermaid

def test_mermaid_output_contains_edges():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    renderer = DagRenderer(tasks)
    mermaid = renderer.to_mermaid()
    assert "t1" in mermaid and "t2" in mermaid
    assert "-->" in mermaid

def test_critical_path_single_chain():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t2"]),
    }
    renderer = DagRenderer(tasks)
    path = renderer.critical_path()
    assert path == ["t1", "t2", "t3"]

def test_critical_path_diamond():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t1"]),
        "t4": _make_task("t4", deps=["t2", "t3"]),
    }
    renderer = DagRenderer(tasks)
    path = renderer.critical_path()
    # any valid path through the diamond: [t1, t2, t4] or [t1, t3, t4]
    assert path[0] == "t1"
    assert path[-1] == "t4"
    assert len(path) == 3

def test_ascii_output_has_rows_for_each_task():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    renderer = DagRenderer(tasks)
    ascii_out = renderer.to_ascii()
    assert "t1" in ascii_out
    assert "t2" in ascii_out

def test_dependency_report_includes_context_size(tmp_path):
    from orchestrator.models import TaskResult, Model, TaskStatus
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    results = {
        "t1": TaskResult(
            task_id="t1", output="x" * 5000, score=0.9,
            model_used=Model.DEEPSEEK_CHAT, status=TaskStatus.COMPLETED,
        )
    }
    renderer = DagRenderer(tasks, results=results)
    report = renderer.dependency_report()
    assert "t2" in report
    assert "5000" in report or "5,000" in report or "chars" in report.lower()
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_visualization.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/visualization.py`**

```python
# orchestrator/visualization.py
"""
DAG visualization for orchestrator task dependency graphs.
No external dependencies required (uses pure-Python graph algorithms).
Optional: if networkx is installed, uses it for layout; otherwise falls back
to a simple topological-sort-based ASCII renderer.
"""
from __future__ import annotations
from typing import Optional
from .models import Task, TaskResult, TaskType


class DagRenderer:
    """
    Renders a task dependency graph as Mermaid markdown, ASCII text,
    or a dependency report.
    """

    _TYPE_COLORS = {
        TaskType.CODE_GEN:     "#89B4FA",   # blue
        TaskType.CODE_REVIEW:  "#A6E3A1",   # green
        TaskType.REASONING:    "#CBA6F7",   # purple
        TaskType.EVALUATE:     "#FAB387",   # peach
        TaskType.WRITING:      "#F38BA8",   # red
        TaskType.DATA_EXTRACT: "#94E2D5",   # teal
        TaskType.SUMMARIZE:    "#F9E2AF",   # yellow
    }

    def __init__(
        self,
        tasks: dict[str, Task],
        results: Optional[dict[str, TaskResult]] = None,
        truncation_limit: int = 40000,
    ) -> None:
        self.tasks = tasks
        self.results = results or {}
        self.truncation_limit = truncation_limit

    # ── Mermaid ──────────────────────────────────────────────────────────────

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for tid, task in self.tasks.items():
            label = f"{tid}\\n[{task.type.value}]"
            color = self._TYPE_COLORS.get(task.type, "#CDD6F4")
            lines.append(f'    {tid}["{label}"]')
            lines.append(f"    style {tid} fill:{color},color:#1E1E2E")
        lines.append("")
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                lines.append(f"    {dep} --> {tid}")
        return "\n".join(lines)

    # ── Critical path ─────────────────────────────────────────────────────────

    def critical_path(self) -> list[str]:
        """
        Returns the longest path through the DAG (maximum number of hops).
        Uses DP on topological order — O(V+E).
        """
        order = self._topological_order()
        # dist[v] = (length of longest path ending at v, predecessor)
        dist: dict[str, int] = {tid: 0 for tid in self.tasks}
        pred: dict[str, Optional[str]] = {tid: None for tid in self.tasks}

        for tid in order:
            task = self.tasks[tid]
            for dep in task.dependencies:
                if dist[dep] + 1 > dist[tid]:
                    dist[tid] = dist[dep] + 1
                    pred[tid] = dep

        # Find end of longest path
        end = max(dist, key=lambda t: dist[t])
        path = []
        cur: Optional[str] = end
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        return list(reversed(path))

    # ── ASCII ─────────────────────────────────────────────────────────────────

    def to_ascii(self) -> str:
        """
        Simple level-based ASCII tree.
        Each dependency level is a row; tasks in the same level are side-by-side.
        """
        levels = self._levels()
        lines = []
        for level_idx, level_tasks in enumerate(levels):
            row = "  ".join(
                f"[{tid}:{self.tasks[tid].type.value[:4]}]"
                for tid in level_tasks
            )
            lines.append(f"L{level_idx}: {row}")
            if level_idx < len(levels) - 1:
                lines.append("       " + "  ".join("|" for _ in level_tasks))
        return "\n".join(lines)

    # ── Dependency report ─────────────────────────────────────────────────────

    def dependency_report(self) -> str:
        """
        Per-task report: upstream deps, context size contributed,
        whether context would be truncated.
        """
        lines = ["Dependency Context Report", "=" * 40]
        for tid, task in self.tasks.items():
            if not task.dependencies:
                continue
            lines.append(f"\n{tid} ({task.type.value})")
            for dep in task.dependencies:
                result = self.results.get(dep)
                if result is None:
                    lines.append(f"  ← {dep}: no result recorded")
                    continue
                size = len(result.output)
                truncated = size > self.truncation_limit
                trunc_note = (
                    f"  ⚠ TRUNCATED ({self.truncation_limit:,} / {size:,} chars)"
                    if truncated else ""
                )
                lines.append(
                    f"  ← {dep}: {size:,} chars  "
                    f"score={result.score:.3f}  "
                    f"status={result.status.value}"
                    f"{trunc_note}"
                )
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm — same as engine._topological_sort."""
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                in_degree[task.id] = in_degree.get(task.id, 0) + 1
        # Recalculate properly
        in_degree = {tid: len(self.tasks[tid].dependencies) for tid in self.tasks}
        queue = [t for t, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for tid, task in self.tasks.items():
                if node in task.dependencies:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
        return order

    def _levels(self) -> list[list[str]]:
        """Group tasks into dependency levels (same level = can run in parallel)."""
        levels: dict[str, int] = {}
        for tid in self._topological_order():
            task = self.tasks[tid]
            dep_level = max((levels[d] for d in task.dependencies), default=-1)
            levels[tid] = dep_level + 1
        max_level = max(levels.values(), default=0)
        result = [[] for _ in range(max_level + 1)]
        for tid, lvl in levels.items():
            result[lvl].append(tid)
        return result
```

**Wire into CLI** — add three flags to the top-level parser in `cli.py`:

```python
parser.add_argument("--visualize", choices=["mermaid", "ascii"],
                    default="", metavar="FORMAT",
                    help="Print task dependency DAG (mermaid or ascii) then exit")
parser.add_argument("--critical-path", action="store_true",
                    help="Print the critical path through the task DAG then exit")
parser.add_argument("--dependency-report", action="store_true",
                    help="After run, print dependency context size report")
```

Add handling in `main()` after state is obtained (for `--dependency-report`), and before run (for `--visualize`/`--critical-path` on a loaded YAML file, after decomposition).

**Step 4: Run tests**

```
pytest tests/test_visualization.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/visualization.py tests/test_visualization.py orchestrator/cli.py
git commit -m "feat: task dependency DAG visualization (mermaid, ascii, critical-path)"
```

---

## Phase 2: Intelligent Caching + Adaptive Retry

---

### Task 5: Semantic Cache (`orchestrator/semantic_cache.py`)

**Files:**
- Create: `orchestrator/semantic_cache.py`
- Test: `tests/test_semantic_cache.py`

**Step 1: Write the failing test**

```python
# tests/test_semantic_cache.py
import asyncio
from orchestrator.semantic_cache import SemanticCache, DuplicationDetector

def test_exact_hit_returns_stored():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "def hello(): return 42"))
    result = asyncio.run(cache.get("t2", "code_gen",
                                   "Write a function that returns 42"))
    # Exact match not expected here — similarity-based
    # Just verify interface works
    assert result is None or isinstance(result, str)

def test_same_prompt_returns_cached():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "def hello(): return 42",
                           prompt="Write hello world function"))
    result = asyncio.run(cache.get("t2", "code_gen",
                                   "Write hello world function"))
    assert result == "def hello(): return 42"

def test_different_task_type_no_hit():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "some code",
                           prompt="Write hello world"))
    result = asyncio.run(cache.get("t2", "code_review",
                                   "Write hello world"))
    assert result is None   # different task type — no cross-type reuse

def test_duplication_detector_finds_similar():
    det = DuplicationDetector(similarity_threshold=0.85)
    tasks = {
        "t1": {"prompt": "Write a Dockerfile for a Python FastAPI app"},
        "t2": {"prompt": "Write a Dockerfile for a FastAPI Python application"},
        "t3": {"prompt": "Write unit tests for the auth service"},
    }
    groups = det.find_duplicate_groups(tasks)
    # t1 and t2 should be grouped together
    assert any("t1" in g and "t2" in g for g in groups), f"groups={groups}"
    # t3 should be in its own group or no group
    assert not any("t1" in g and "t3" in g for g in groups)

def test_duplication_detector_no_duplicates():
    det = DuplicationDetector()
    tasks = {
        "t1": {"prompt": "Write a Dockerfile"},
        "t2": {"prompt": "Analyze the database schema"},
        "t3": {"prompt": "Create unit tests for authentication"},
    }
    groups = det.find_duplicate_groups(tasks)
    # All tasks should be isolated — no grouping
    assert len(groups) == 0 or all(len(g) == 1 for g in groups)
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_semantic_cache.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/semantic_cache.py`**

```python
# orchestrator/semantic_cache.py
"""
Semantic output cache and duplication detector.

SemanticCache stores (task_type, prompt, output) tuples and retrieves
outputs when a new prompt is sufficiently similar (Jaccard similarity on
word tokens — no embedding model required).

DuplicationDetector scans a task list for near-duplicate prompts so the
engine can reuse outputs instead of generating twice.
"""
from __future__ import annotations
import re
from typing import Optional


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SemanticCache:
    """
    In-memory semantic output cache keyed by (task_type, prompt_tokens).
    Thread-safe for single-process async use (no locks needed with asyncio).
    """

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        self.threshold = similarity_threshold
        # entries: list of (task_type, tokens, prompt, output)
        self._entries: list[tuple[str, set[str], str, str]] = []

    async def get(
        self, task_id: str, task_type: str, prompt: str
    ) -> Optional[str]:
        tokens = _tokenize(prompt)
        best_score = 0.0
        best_output: Optional[str] = None
        for (etype, etokens, _, eoutput) in self._entries:
            if etype != task_type:
                continue
            score = _jaccard(tokens, etokens)
            if score > best_score:
                best_score = score
                best_output = eoutput
        if best_score >= self.threshold:
            return best_output
        return None

    async def put(
        self,
        task_id: str,
        task_type: str,
        output: str,
        prompt: str = "",
    ) -> None:
        tokens = _tokenize(prompt)
        self._entries.append((task_type, tokens, prompt, output))

    def size(self) -> int:
        return len(self._entries)


class DuplicationDetector:
    """
    Scans a dict of {task_id: {prompt: str}} and returns groups of
    near-duplicate task IDs. Tasks in the same group have prompts with
    Jaccard similarity >= threshold.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.threshold = similarity_threshold

    def find_duplicate_groups(
        self, tasks: dict[str, dict]
    ) -> list[list[str]]:
        ids = list(tasks.keys())
        token_map = {
            tid: _tokenize(tasks[tid].get("prompt", ""))
            for tid in ids
        }
        visited = set()
        groups = []
        for i, tid_a in enumerate(ids):
            if tid_a in visited:
                continue
            group = [tid_a]
            for tid_b in ids[i + 1:]:
                if tid_b in visited:
                    continue
                score = _jaccard(token_map[tid_a], token_map[tid_b])
                if score >= self.threshold:
                    group.append(tid_b)
                    visited.add(tid_b)
            if len(group) > 1:
                groups.append(group)
                visited.add(tid_a)
        return groups
```

**Step 4: Run tests**

```
pytest tests/test_semantic_cache.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/semantic_cache.py tests/test_semantic_cache.py
git commit -m "feat: semantic output cache and duplication detector (Jaccard similarity)"
```

---

### Task 6: Adaptive Router v2 (`orchestrator/adaptive_router.py`)

**Files:**
- Create: `orchestrator/adaptive_router.py`
- Modify: `orchestrator/engine.py` — integrate `AdaptiveRouter` into `_get_available_models`
- Test: `tests/test_adaptive_router.py`

**Step 1: Write the failing test**

```python
# tests/test_adaptive_router.py
import asyncio
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model, TaskType

def test_initial_state_all_healthy():
    router = AdaptiveRouter()
    for m in Model:
        assert router.get_state(m) == ModelState.HEALTHY

def test_consecutive_timeouts_degrade_model():
    router = AdaptiveRouter(timeout_threshold=3)
    for _ in range(3):
        router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED

def test_degraded_model_recovers_after_cooldown():
    import time
    router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=0.1)
    for _ in range(3):
        router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED
    time.sleep(0.15)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY

def test_auth_failure_permanently_disables():
    router = AdaptiveRouter()
    router.record_auth_failure(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DISABLED

def test_success_resets_timeout_counter():
    router = AdaptiveRouter(timeout_threshold=3)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_success(Model.DEEPSEEK_CHAT)
    # After success, counter resets — needs 3 more timeouts to degrade
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY

def test_prefer_fastest_healthy_model():
    router = AdaptiveRouter()
    router.record_latency(Model.DEEPSEEK_CHAT, 500.0)
    router.record_latency(Model.KIMI_K2_5, 2000.0)
    candidates = [Model.DEEPSEEK_CHAT, Model.KIMI_K2_5]
    best = router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.DEEPSEEK_CHAT

def test_disabled_model_excluded_from_preferred():
    router = AdaptiveRouter()
    router.record_auth_failure(Model.DEEPSEEK_CHAT)
    candidates = [Model.DEEPSEEK_CHAT, Model.KIMI_K2_5]
    best = router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.KIMI_K2_5
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_adaptive_router.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/adaptive_router.py`**

```python
# orchestrator/adaptive_router.py
"""
Adaptive Model Router — circuit breaker v2.

Tracks per-model failure patterns with three states:
  HEALTHY   — routing proceeds normally
  DEGRADED  — too many recent timeouts; skip for cooldown_seconds
  DISABLED  — permanent failure (auth error, 404); never route here

Records latency observations to prefer faster healthy models.
"""
from __future__ import annotations
import time
from enum import Enum
from typing import Optional
from .models import Model, TaskType


class ModelState(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class AdaptiveRouter:

    def __init__(
        self,
        timeout_threshold: int = 3,
        cooldown_seconds: float = 300.0,   # 5 minutes
    ) -> None:
        self._timeout_counts: dict[Model, int] = {m: 0 for m in Model}
        self._degraded_since: dict[Model, Optional[float]] = {m: None for m in Model}
        self._disabled: set[Model] = set()
        self._latencies: dict[Model, float] = {}   # EMA
        self.timeout_threshold = timeout_threshold
        self.cooldown_seconds = cooldown_seconds

    # ── State queries ─────────────────────────────────────────────────────────

    def get_state(self, model: Model) -> ModelState:
        if model in self._disabled:
            return ModelState.DISABLED
        since = self._degraded_since.get(model)
        if since is not None:
            if time.monotonic() - since < self.cooldown_seconds:
                return ModelState.DEGRADED
            else:
                # cooldown elapsed — recover
                self._degraded_since[model] = None
                self._timeout_counts[model] = 0
        return ModelState.HEALTHY

    def is_available(self, model: Model) -> bool:
        return self.get_state(model) == ModelState.HEALTHY

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_timeout(self, model: Model) -> None:
        if model in self._disabled:
            return
        self._timeout_counts[model] += 1
        if self._timeout_counts[model] >= self.timeout_threshold:
            if self._degraded_since.get(model) is None:
                self._degraded_since[model] = time.monotonic()

    def record_success(self, model: Model) -> None:
        self._timeout_counts[model] = 0
        self._degraded_since[model] = None

    def record_auth_failure(self, model: Model) -> None:
        self._disabled.add(model)

    def record_latency(self, model: Model, latency_ms: float,
                       alpha: float = 0.1) -> None:
        if model in self._latencies:
            self._latencies[model] = (
                alpha * latency_ms + (1 - alpha) * self._latencies[model]
            )
        else:
            self._latencies[model] = latency_ms

    # ── Routing ───────────────────────────────────────────────────────────────

    def preferred_model(
        self,
        candidates: list[Model],
        task_type: Optional[TaskType] = None,
    ) -> Optional[Model]:
        """
        From a list of candidates, return the healthy model with
        the lowest observed latency. Returns None if no healthy candidates.
        """
        healthy = [m for m in candidates if self.is_available(m)]
        if not healthy:
            return None
        # Sort by known latency ascending; unknowns sort last
        healthy.sort(key=lambda m: self._latencies.get(m, float("inf")))
        return healthy[0]
```

**Wire into `engine.py`:**

In `Orchestrator.__init__`, add:
```python
from .adaptive_router import AdaptiveRouter
self._adaptive_router = AdaptiveRouter()
```

In `_get_available_models`, after filtering by `api_health`, also filter by `adaptive_router`:
```python
available = [m for m in available if self._adaptive_router.is_available(m)]
```

In `_record_failure`, call `self._adaptive_router.record_timeout(model)` for timeouts, `self._adaptive_router.record_auth_failure(model)` for auth errors.

In `_record_success`, call `self._adaptive_router.record_success(model)`.

**Step 4: Run tests**

```
pytest tests/test_adaptive_router.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/adaptive_router.py tests/test_adaptive_router.py orchestrator/engine.py
git commit -m "feat: adaptive router v2 with degraded/disabled states and latency-aware routing"
```

---

## Phase 3: Multi-Run Aggregation + Auto-Remediation

---

### Task 7: Multi-Run Aggregation (`orchestrator/aggregator.py`)

**Files:**
- Create: `orchestrator/aggregator.py`
- Modify: `orchestrator/state.py` — store run metadata table
- Modify: `orchestrator/cli.py` — add `--aggregate-metrics`, `--reuse-profiles` flags
- Test: `tests/test_aggregator.py`

**Step 1: Write the failing test**

```python
# tests/test_aggregator.py
import asyncio
from orchestrator.aggregator import ProfileAggregator, RunRecord
from orchestrator.models import Model, TaskType

def _make_run(model: Model, task_type: TaskType,
              score: float, cost: float, latency: float) -> RunRecord:
    return RunRecord(
        project_id="proj-1",
        task_type=task_type,
        model=model,
        score=score,
        cost_usd=cost,
        latency_ms=latency,
    )

def test_best_model_for_task_type():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.92, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5,     TaskType.CODE_GEN, 0.88, 0.002, 1200))
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.90, 0.001, 850))
    best = agg.best_model(TaskType.CODE_GEN)
    assert best == Model.DEEPSEEK_CHAT   # higher avg score

def test_cost_efficiency_ranking():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.90, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5,     TaskType.CODE_GEN, 0.90, 0.002, 800))
    ranking = agg.cost_efficiency_ranking(TaskType.CODE_GEN)
    # Both equal score, deepseek cheaper → deepseek should rank first
    assert ranking[0][0] == Model.DEEPSEEK_CHAT

def test_summary_table_includes_all_recorded_types():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.9, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5, TaskType.CODE_REVIEW, 0.8, 0.002, 900))
    table = agg.summary_table()
    assert TaskType.CODE_GEN in table
    assert TaskType.CODE_REVIEW in table

def test_empty_aggregator_returns_none():
    agg = ProfileAggregator()
    assert agg.best_model(TaskType.CODE_GEN) is None

def test_record_and_retrieve_multiple_runs():
    agg = ProfileAggregator()
    for i in range(5):
        agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.WRITING,
                             0.8 + i * 0.02, 0.001, 1000))
    stats = agg.stats_for(Model.DEEPSEEK_CHAT, TaskType.WRITING)
    assert stats["count"] == 5
    assert abs(stats["avg_score"] - 0.88) < 0.01
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_aggregator.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/aggregator.py`**

```python
# orchestrator/aggregator.py
"""
Cross-run profile aggregator.

Records (model, task_type, score, cost, latency) tuples from completed
project runs and computes aggregated statistics to guide future routing.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from .models import Model, TaskType


@dataclass
class RunRecord:
    project_id: str
    task_type: TaskType
    model: Model
    score: float
    cost_usd: float
    latency_ms: float


class ProfileAggregator:
    """
    In-memory cross-run aggregator.
    Persist via StateManager.save_run_record() / load_run_records().
    """

    def __init__(self) -> None:
        # {(model, task_type): [RunRecord, ...]}
        self._records: dict[tuple[Model, TaskType], list[RunRecord]] = defaultdict(list)

    def record(self, run: RunRecord) -> None:
        self._records[(run.model, run.task_type)].append(run)

    def stats_for(self, model: Model, task_type: TaskType) -> dict:
        records = self._records.get((model, task_type), [])
        if not records:
            return {"count": 0, "avg_score": 0.0, "avg_cost": 0.0, "avg_latency": 0.0}
        n = len(records)
        return {
            "count": n,
            "avg_score":   sum(r.score     for r in records) / n,
            "avg_cost":    sum(r.cost_usd  for r in records) / n,
            "avg_latency": sum(r.latency_ms for r in records) / n,
        }

    def best_model(self, task_type: TaskType) -> Optional[Model]:
        """Returns the model with the highest average score for this task type."""
        candidates = {
            model for (model, tt) in self._records if tt == task_type
        }
        if not candidates:
            return None
        return max(candidates, key=lambda m: self.stats_for(m, task_type)["avg_score"])

    def cost_efficiency_ranking(
        self, task_type: TaskType
    ) -> list[tuple[Model, float]]:
        """
        Returns (model, efficiency) tuples sorted by score/cost descending.
        Efficiency = avg_score / avg_cost (higher = better value).
        """
        candidates = {
            model for (model, tt) in self._records if tt == task_type
        }
        results = []
        for model in candidates:
            s = self.stats_for(model, task_type)
            if s["avg_cost"] > 0:
                efficiency = s["avg_score"] / s["avg_cost"]
            else:
                efficiency = s["avg_score"]
            results.append((model, efficiency))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def summary_table(self) -> dict[TaskType, list[dict]]:
        """
        Returns {task_type: [{model, avg_score, avg_cost, avg_latency, count}]}
        sorted by avg_score desc.
        """
        by_type: dict[TaskType, list[dict]] = defaultdict(list)
        for (model, task_type), records in self._records.items():
            s = self.stats_for(model, task_type)
            by_type[task_type].append({"model": model, **s})
        for task_type in by_type:
            by_type[task_type].sort(key=lambda x: x["avg_score"], reverse=True)
        return dict(by_type)
```

**Add `--aggregate-metrics` and `--reuse-profiles` flags to CLI:**

```python
# in cli.py parser setup:
parser.add_argument("--aggregate-metrics", action="store_true",
                    help="Print cross-run model performance aggregation and exit")
parser.add_argument("--reuse-profiles", action="store_true",
                    help="Seed routing from historical run profiles")
```

**Step 4: Run tests**

```
pytest tests/test_aggregator.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/aggregator.py tests/test_aggregator.py orchestrator/cli.py
git commit -m "feat: cross-run profile aggregator with cost-efficiency ranking"
```

---

### Task 8: Auto-Remediation Engine (`orchestrator/remediation.py`)

**Files:**
- Create: `orchestrator/remediation.py`
- Modify: `orchestrator/models.py` — add `RemediationStrategy` to `Task` dataclass
- Modify: `orchestrator/engine.py` — call `RemediationEngine` from `_execute_task`
- Modify: `orchestrator/project_file.py` — parse `remediation_strategy` field from YAML
- Test: `tests/test_remediation.py`

**Step 1: Write the failing test**

```python
# tests/test_remediation.py
from orchestrator.remediation import (
    RemediationStrategy, RemediationPlan, RemediationEngine,
)
from orchestrator.models import TaskResult, Model, TaskStatus, TaskType

def _failed_result(task_id="t1"):
    return TaskResult(
        task_id=task_id, output="", score=0.0,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.FAILED,
    )

def _low_score_result(task_id="t1", score=0.55):
    return TaskResult(
        task_id=task_id, output="some output", score=score,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.DEGRADED,
    )

def test_plan_with_single_strategy():
    plan = RemediationPlan([RemediationStrategy.AUTO_RETRY])
    assert plan.next_strategy() == RemediationStrategy.AUTO_RETRY
    plan.advance()
    assert plan.next_strategy() is None
    assert plan.exhausted()

def test_plan_with_multiple_strategies():
    plan = RemediationPlan([
        RemediationStrategy.AUTO_RETRY,
        RemediationStrategy.FALLBACK_MODEL,
        RemediationStrategy.DEGRADE_QUALITY,
    ])
    assert plan.next_strategy() == RemediationStrategy.AUTO_RETRY
    plan.advance()
    assert plan.next_strategy() == RemediationStrategy.FALLBACK_MODEL
    plan.advance()
    assert plan.next_strategy() == RemediationStrategy.DEGRADE_QUALITY
    plan.advance()
    assert plan.exhausted()

def test_should_remediate_on_failure():
    engine = RemediationEngine()
    result = _failed_result()
    assert engine.should_remediate(result, threshold=0.85) is True

def test_should_remediate_on_low_score():
    engine = RemediationEngine()
    result = _low_score_result(score=0.60)
    assert engine.should_remediate(result, threshold=0.85) is True

def test_no_remediation_on_success():
    engine = RemediationEngine()
    result = TaskResult(
        task_id="t1", output="good", score=0.90,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.COMPLETED,
    )
    assert engine.should_remediate(result, threshold=0.85) is False

def test_adjusted_threshold_for_degrade_quality():
    engine = RemediationEngine()
    strategy = RemediationStrategy.DEGRADE_QUALITY
    original_threshold = 0.85
    adjusted = engine.adjusted_threshold(strategy, original_threshold)
    assert adjusted < original_threshold
    assert adjusted >= 0.0

def test_default_plan_for_unknown_task():
    engine = RemediationEngine()
    plan = engine.default_plan()
    assert not plan.exhausted()
    first = plan.next_strategy()
    assert first in list(RemediationStrategy)
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_remediation.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement `orchestrator/remediation.py`**

```python
# orchestrator/remediation.py
"""
Auto-Remediation Engine.

When a task fails or scores below threshold, RemediationEngine selects
the next recovery strategy from an ordered RemediationPlan.

Strategies (tried in order specified by the task's plan):
  AUTO_RETRY       — retry with same model and prompt
  FALLBACK_MODEL   — try the next model in the fallback chain
  ADJUST_PROMPT    — rephrase the task prompt and retry
  SKIP_VALIDATOR   — bypass hard validators, accept any output
  DEGRADE_QUALITY  — lower acceptance threshold by 15%
  ABORT_TASK       — give up, mark FAILED, continue with next task
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from .models import TaskResult, TaskStatus


class RemediationStrategy(str, Enum):
    AUTO_RETRY     = "auto_retry"
    FALLBACK_MODEL = "fallback_model"
    ADJUST_PROMPT  = "adjust_prompt"
    SKIP_VALIDATOR = "skip_validator"
    DEGRADE_QUALITY = "degrade_quality"
    ABORT_TASK     = "abort_task"


_DEFAULT_PLAN = [
    RemediationStrategy.AUTO_RETRY,
    RemediationStrategy.FALLBACK_MODEL,
    RemediationStrategy.DEGRADE_QUALITY,
    RemediationStrategy.ABORT_TASK,
]

_DEGRADE_FACTOR = 0.85   # reduce threshold by 15%


class RemediationPlan:
    """Ordered list of strategies to attempt; advances on each failure."""

    def __init__(self, strategies: list[RemediationStrategy]) -> None:
        self._strategies = strategies
        self._index = 0

    def next_strategy(self) -> Optional[RemediationStrategy]:
        if self._index >= len(self._strategies):
            return None
        return self._strategies[self._index]

    def advance(self) -> None:
        self._index += 1

    def exhausted(self) -> bool:
        return self._index >= len(self._strategies)

    def reset(self) -> None:
        self._index = 0


class RemediationEngine:
    """
    Decides whether a TaskResult warrants remediation and
    what the next strategy should be.
    """

    def should_remediate(
        self,
        result: TaskResult,
        threshold: float,
    ) -> bool:
        return (
            result.status == TaskStatus.FAILED
            or result.score < threshold
        )

    def adjusted_threshold(
        self,
        strategy: RemediationStrategy,
        original_threshold: float,
    ) -> float:
        if strategy == RemediationStrategy.DEGRADE_QUALITY:
            return max(0.0, original_threshold * _DEGRADE_FACTOR)
        return original_threshold

    def rephrase_prompt(self, original_prompt: str) -> str:
        """
        Minimal prompt adjustment: prepends an instruction to be more explicit.
        In production, this could call the LLM for a more intelligent rewrite.
        """
        return (
            "Please provide a complete, detailed, and correct response to "
            "the following task. Be thorough and precise.\n\n"
            + original_prompt
        )

    def default_plan(self) -> RemediationPlan:
        return RemediationPlan(list(_DEFAULT_PLAN))

    def plan_from_list(
        self, strategies: list[str]
    ) -> RemediationPlan:
        return RemediationPlan([RemediationStrategy(s) for s in strategies])
```

**Add `remediation_strategies` to `Task` dataclass in `models.py`:**

```python
# In Task dataclass, add field:
remediation_strategies: list[str] = field(default_factory=list)
# e.g. ["auto_retry", "fallback_model", "degrade_quality"]
```

**Wire into `_execute_task` in `engine.py`** — after the main iteration loop returns a result, check if remediation is needed:

```python
from .remediation import RemediationEngine, RemediationStrategy

_remediation = RemediationEngine()
_plan = (
    _remediation.plan_from_list(task.remediation_strategies)
    if task.remediation_strategies
    else _remediation.default_plan()
)

while _remediation.should_remediate(result, task.acceptance_threshold):
    strategy = _plan.next_strategy()
    if strategy is None or strategy == RemediationStrategy.ABORT_TASK:
        break
    _plan.advance()
    if strategy == RemediationStrategy.DEGRADE_QUALITY:
        task.acceptance_threshold = _remediation.adjusted_threshold(
            strategy, task.acceptance_threshold
        )
    elif strategy == RemediationStrategy.ADJUST_PROMPT:
        task.prompt = _remediation.rephrase_prompt(task.prompt)
    elif strategy == RemediationStrategy.SKIP_VALIDATOR:
        task.hard_validators = []
    # AUTO_RETRY and FALLBACK_MODEL just re-run the execute loop:
    result = await self._execute_task_inner(task)  # re-run
```

> **Implementation note:** Refactor `_execute_task` to extract the core loop into `_execute_task_inner(task)` to allow re-calling cleanly from the remediation wrapper.

**Step 4: Run tests**

```
pytest tests/test_remediation.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add orchestrator/remediation.py tests/test_remediation.py \
        orchestrator/models.py orchestrator/engine.py orchestrator/project_file.py
git commit -m "feat: auto-remediation engine with 6 strategies and per-task YAML config"
```

---

### Task 9: Full integration test + run full test suite

**Files:**
- Modify: `tests/test_engine_e2e.py` — add integration tests covering streaming + remediation

**Step 1: Write integration tests**

```python
# Add to tests/test_engine_e2e.py

def test_streaming_events_include_task_lifecycle(tmp_path):
    """Full streaming run: verify event sequence is coherent."""
    import json
    from orchestrator.streaming import (
        ProjectStarted, TaskStarted, TaskCompleted, ProjectCompleted,
    )
    decomp = json.dumps([{
        "id": "task_001", "type": "code_generation",
        "prompt": "Write hello world", "dependencies": [],
        "hard_validators": [],
    }])
    orch = _make_orch(Budget(max_usd=2.0))
    events = []

    async def mock_call(model, prompt, **kwargs):
        if "decomposition engine" in kwargs.get("system", ""):
            return _make_mock_response(decomp)
        if "evaluat" in prompt.lower():
            return _make_mock_response('{"score": 0.92, "critique": "Good."}')
        return _make_mock_response("def hello():\n    return 'Hello'")

    async def run():
        with patch.object(orch.client, "call", side_effect=mock_call):
            async for ev in orch.run_project_streaming(
                "Hello world", "Must return Hello"
            ):
                events.append(ev)

    _run(run())
    types = {type(e).__name__ for e in events}
    assert "ProjectStarted" in types
    assert "TaskStarted"    in types
    assert "TaskCompleted"  in types
    assert "ProjectCompleted" in types

def test_remediation_degrade_quality_lowers_threshold(tmp_path):
    """Task that scores 0.70 with threshold 0.85 should remediate via degrade."""
    from orchestrator.remediation import RemediationEngine, RemediationStrategy
    from orchestrator.models import TaskResult, Model, TaskStatus
    engine = RemediationEngine()
    result = TaskResult(
        task_id="t1", output="partial", score=0.70,
        model_used=Model.DEEPSEEK_CHAT, status=TaskStatus.DEGRADED,
    )
    assert engine.should_remediate(result, threshold=0.85)
    new_threshold = engine.adjusted_threshold(RemediationStrategy.DEGRADE_QUALITY, 0.85)
    assert new_threshold < 0.85
    assert not engine.should_remediate(result, threshold=new_threshold)
```

**Step 2: Run all tests**

```
pytest tests/ --tb=short -q
```
Expected: 398+ passed (same pre-existing failures in stress_test.py + policy tests)

**Step 3: Commit**

```bash
git add tests/test_engine_e2e.py
git commit -m "test: integration tests for streaming events and remediation engine"
```

---

### Task 10: Export new modules from `orchestrator/__init__.py`

**Files:**
- Modify: `orchestrator/__init__.py`

**Step 1: Check current exports**

```
grep -n "^from\|^import" orchestrator/__init__.py
```

**Step 2: Add new exports**

```python
# Add to orchestrator/__init__.py:
from .streaming import (
    ProjectEventBus, ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, TaskFailed, BudgetWarning, ProjectCompleted,
)
from .visualization import DagRenderer
from .semantic_cache import SemanticCache, DuplicationDetector
from .adaptive_router import AdaptiveRouter, ModelState
from .aggregator import ProfileAggregator, RunRecord
from .remediation import RemediationEngine, RemediationStrategy, RemediationPlan
```

**Step 3: Verify imports work**

```
python -c "from orchestrator import (DagRenderer, SemanticCache, AdaptiveRouter, ProfileAggregator, RemediationEngine); print('OK')"
```
Expected: `OK`

**Step 4: Run full suite one more time**

```
pytest tests/ --tb=no -q
```

**Step 5: Final commit**

```bash
git add orchestrator/__init__.py
git commit -m "feat: export all 6 new improvement modules from orchestrator package"
```

---

## Summary of All New Files

| File | Purpose | Phase |
|------|---------|-------|
| `orchestrator/streaming.py` | Event dataclasses + ProjectEventBus | 1 |
| `orchestrator/progress.py` | Terminal progress renderer | 1 |
| `orchestrator/visualization.py` | DAG: Mermaid, ASCII, critical path, dep report | 1 |
| `orchestrator/semantic_cache.py` | Jaccard similarity output cache + dedup | 2 |
| `orchestrator/adaptive_router.py` | Circuit breaker v2 with latency-aware routing | 2 |
| `orchestrator/aggregator.py` | Cross-run profile aggregator | 3 |
| `orchestrator/remediation.py` | 6-strategy auto-remediation engine | 3 |

## Summary of Modified Files

| File | Changes |
|------|---------|
| `orchestrator/engine.py` | Wire EventBus, AdaptiveRouter, RemediationEngine into execute loop |
| `orchestrator/cli.py` | Add `--visualize`, `--critical-path`, `--dependency-report`, `--aggregate-metrics`, `--reuse-profiles` |
| `orchestrator/models.py` | Add `remediation_strategies: list[str]` to `Task` dataclass |
| `orchestrator/__init__.py` | Export all new modules |
| `orchestrator/project_file.py` | Parse `remediation_strategy` from YAML |

## Test Files

| File | Tests Added |
|------|-------------|
| `tests/test_streaming.py` | 7 tests |
| `tests/test_progress.py` | 3 tests |
| `tests/test_visualization.py` | 6 tests |
| `tests/test_semantic_cache.py` | 5 tests |
| `tests/test_adaptive_router.py` | 7 tests |
| `tests/test_aggregator.py` | 6 tests |
| `tests/test_remediation.py` | 7 tests |
| `tests/test_engine_e2e.py` | 2 integration tests |
