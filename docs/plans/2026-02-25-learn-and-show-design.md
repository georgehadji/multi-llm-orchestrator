# Learn & Show: Persistent Cross-Run Learning + CLI Dashboard

**Date:** 2026-02-25
**Status:** Approved for implementation
**Author:** Georgios-Chrysovalantis Chatzivantsidis

---

## Goal

Build a single cohesive sprint that delivers both **persistent cross-run learning** and
**CLI observability** for the multi-LLM orchestrator.

Every new run starts warm — routing decisions benefit from the full history of every prior
run. Operators see exactly what the system learned, why it's routing the way it is, and
what they should change.

---

## Strategic Context

`ModelProfile` telemetry (quality_score, trust_factor, latency, cost) is currently
**in-memory only** — reset on every run. This means:

- Routing quality is identical on run 1 and run 1000
- No competitive lock-in — switching to another orchestrator costs nothing
- No visibility into which models are improving or degrading

Fixing this creates the **Nash stability moat**: the longer a customer uses this
orchestrator, the better it routes specifically for their workload. A competitor
starting fresh is always cold; this system is always warm.

---

## Approved Design

### Persistence Layer

**New file:** `~/.orchestrator_cache/telemetry.db` (SQLite, append-only)

Two tables:

```sql
CREATE TABLE model_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           TEXT    NOT NULL,
    model                TEXT    NOT NULL,
    task_type            TEXT    NOT NULL,
    quality_score        REAL    NOT NULL,
    trust_factor         REAL    NOT NULL,
    avg_latency_ms       REAL    NOT NULL,
    latency_p95_ms       REAL    NOT NULL,
    success_rate         REAL    NOT NULL,
    avg_cost_usd         REAL    NOT NULL,
    call_count           INTEGER NOT NULL,
    failure_count        INTEGER NOT NULL,
    validator_fail_count INTEGER NOT NULL,
    recorded_at          REAL    NOT NULL
);

CREATE TABLE routing_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT    NOT NULL,
    task_id      TEXT    NOT NULL,
    task_type    TEXT    NOT NULL,
    model_chosen TEXT    NOT NULL,
    reviewer     TEXT,
    score        REAL    NOT NULL,
    cost_usd     REAL    NOT NULL,
    iterations   INTEGER NOT NULL,
    det_passed   INTEGER NOT NULL,
    recorded_at  REAL    NOT NULL
);

CREATE INDEX idx_snapshots_model ON model_snapshots(model, task_type, recorded_at);
CREATE INDEX idx_routing_model   ON routing_events(model_chosen, task_type, recorded_at);
```

All writes are append-only (INSERT only, never UPDATE/DELETE).
Never block the main execution path — writes are `asyncio.create_task` fire-and-forget.

### New Module: `orchestrator/telemetry_store.py`

Owns all reads and writes to `telemetry.db`. The rest of the codebase imports
only this module. No other module touches `telemetry.db` directly.

Key API:

```python
class TelemetryStore:
    async def record_snapshot(project_id, model, task_type, profile: ModelProfile) -> None
    async def record_routing_event(project_id, task_id, result: TaskResult) -> None
    async def load_historical_profile(model, task_type) -> HistoricalProfile | None
    async def model_rankings(days=30) -> list[ModelRanking]
    async def task_type_leaders(days=30) -> dict[TaskType, ModelRanking]
    async def recommendations(days=30) -> list[Recommendation]
```

### Warm-Start Blending

Called in `Orchestrator.__init__()` after profiles are built from defaults:

| Historical calls | Blend ratio | Latency override |
|---|---|---|
| < 10 | Ignore — cold start, use defaults | No |
| 10–49 | 40% historical / 60% default | No |
| ≥ 50 | 100% historical | Yes |

Latency is only overridden at high confidence (≥50 calls) because it is noisier
than quality and matters most for SLA enforcement.

Blend formula (low-confidence example):
```python
profile.quality_score = 0.4 * hist.quality_score + 0.6 * DEFAULT_QUALITY
profile.trust_factor  = 0.4 * hist.trust_factor  + 0.6 * DEFAULT_TRUST
```

### Collection Points

Two insertion points in `engine.py`:

1. **After `run_project()` completes** — snapshot each `ModelProfile` that had ≥1 call
   this run into `model_snapshots`
2. **After each task result** — insert one row into `routing_events` from `TaskResult`

### CLI Dashboard

New command: `orchestrator dashboard`

**Default view** (`orchestrator dashboard`):

```
────────────────────────────────────────────────────────────────────────
 48 runs  │  $312.40 total  │  avg quality 0.871  │  last 30 days
────────────────────────────────────────────────────────────────────────

MODEL RANKINGS  (value = quality / cost_per_quality_point)

  ●  deepseek-chat      2.16  ▲+8%   HOT   312 calls  $0.41/k  q:0.887
  ●  deepseek-reasoner  1.84  ▲+3%   HOT    88 calls  $1.23/k  q:0.903
  ●  claude-sonnet      0.89  ─ 0%   HOT   189 calls  $2.10/k  q:0.871
  ○  gemini-flash       0.72  ▼-11%  WARM   41 calls  $0.18/k  q:0.798
  ○  gpt-4o             0.41  ▼-6%   WARM   64 calls  $3.40/k  q:0.842
  ·  claude-opus        0.31  ─       COLD    8 calls  (insufficient data)
  ·  kimi-k2.5          —             COLD    2 calls  (insufficient data)

  ● HOT ≥50 calls (full historical weight)
  ○ WARM 10–49 calls (40% blend)
  · COLD <10 calls (defaults only)

────────────────────────────────────────────────────────────────────────

TASK-TYPE LEADERS

  code_gen    deepseek-chat  0.91  ●   reasoning   deepseek-r1   0.93  ●
  code_review claude-opus    0.94  ·   summarize   gemini-flash  0.88  ○
  writing     claude-sonnet  0.88  ●   evaluate    deepseek-chat 0.89  ●

────────────────────────────────────────────────────────────────────────

RECOMMENDATIONS  (3)

  [1] Route reasoning tasks to deepseek-reasoner instead of claude-opus
      same quality (0.93 vs 0.91), saves ~$1.97/k tokens  →  +$47/mo est.

  [2] gemini-flash quality declining (-11% over 30d) — consider removing
      from summarize routing or adding a policy latency floor

  [3] claude-opus has only 8 calls — warm-start blending not yet active.
      Run 2 more reasoning projects to reach confidence threshold.

────────────────────────────────────────────────────────────────────────
```

**Task breakdown** (`orchestrator dashboard --by-task`): ranked list per task type

**Cost efficiency** (`orchestrator dashboard --cost`): quality-per-dollar bar chart

**Implementation:** extends `metrics.py` `ConsoleExporter`. Reads from `telemetry.db`
via `telemetry_store.py`. No new dependencies — pure stdlib.

### Value Score Formula

```
value_score = quality_score / (avg_cost_usd + ε)
```

Where ε = 0.0001 to prevent division by zero. Trend computed as percentage
change vs same window 30 days prior.

---

## What This Is NOT

- No web UI (CLI only)
- No automatic policy mutation (recommendations are advisory only)
- No cross-tenant data sharing (per-installation only)
- No ML model training (EMA aggregation only)

---

## Files Touched

| File | Change |
|---|---|
| `orchestrator/telemetry_store.py` | **New** — TelemetryStore class |
| `orchestrator/engine.py` | Add snapshot + event writes after run/task |
| `orchestrator/cli.py` | Add `dashboard` subcommand |
| `orchestrator/metrics.py` | Extend ConsoleExporter for dashboard views |
| `tests/test_telemetry_store.py` | **New** — unit tests for store |
| `tests/test_dashboard.py` | **New** — dashboard rendering tests |

---

## Success Criteria

1. After 10 runs, `ModelProfile.quality_score` for HOT models reflects historical data (not defaults)
2. `orchestrator dashboard` renders within 200ms
3. Recommendations section correctly identifies the cheapest model with ≥ equivalent quality
4. All new code covered by tests; full suite still passes (649+)
5. No blocking I/O on the hot path — telemetry writes are always fire-and-forget
