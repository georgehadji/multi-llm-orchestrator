# T15 — integrations/, vcs/, ide_backend/, dashboard_core/, commands/, cli*, entrypoints/ — Inventory

Seventh of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, continuing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7 across backend files with no individual disposition
recorded in earlier tiers.

## Phase 0 — Scope

82 files by direct `find` count (the plan estimated ~78): `commands/` (24), `integrations/`
(13), `vcs/` (7), `ide_backend/` (16), `dashboard_core/` (6), `kanban/` (3), `hitl/` (3),
`connectors/` (3), `entrypoints/` (3), root `cli*.py` (4). Framed by the plan as
"External-facing adapters and CLI/dashboard surfaces... exercised by users directly" —
calibrated higher-priority than T13/T14's "auxiliary" framing, given defects here are
directly user-visible even when isolated from core orchestration logic.

## Phase 1-3 — Survey, candidates, trigger/innocence

A background agent surveyed all 82 files, giving full depth to three pre-identified handoffs
(T10's `integrations/mcp_server.py`, T13's `commands/nash.py`'s `nash_backup` crash, T13's
`cli.py` docstring) plus a systematic sweep of all 24 `commands/` files and the root
`cli*.py`/`entrypoints/` files, using both a top-level `importlib.import_module` sweep AND a
static AST-based import-resolution sweep (the latter specifically to catch lazy imports
inside function bodies, which a naive import check misses entirely — and which is exactly
where two of this tier's most significant findings were hiding). All findings were
independently re-verified from source and, for every fix implemented, empirically
re-executed (the actual command handler called with constructed args) before being applied.

Given the unusually large number of well-evidenced, high-confidence, directly user-facing
findings this wave produced (7 fixes, versus 2-5 in most prior tiers), the fix count here is
driven by genuine severity distribution rather than a target — this tier's user-facing
framing predicted exactly this kind of result.

## Phase 0 handoffs — resolved

### Handoff 1 — `integrations/mcp_server.py` (T10) — confirmed broken, feature-convergence left as a decision

5 wrong-depth relative imports (root-level modules referenced with one dot instead of two)
make this file fully unimportable. Confirmed it is a dead fork of the live, canonical root
`orchestrator/mcp_server.py` — but a **feature-richer** fork, adding a snapshot-store
integration and two extra MCP tools (`orch_project_status`, `orch_project_snapshots`) the
root copy lacks. Neither file has any caller anywhere in the repo (0% test coverage on both,
confirmed via `coverage_html/status.json`) and no process starts either. **Not fixed**: which
direction to converge (backport the 2 extra tools into root and shim this copy away, or give
this copy its own launcher) is a product decision about which MCP feature set should be
canonical, not a mechanical import-depth fix. `[REQUIRES HUMAN REVIEW]`.

### Handoff 2 — `commands/nash.py`'s `nash backup` crash (T13) — fixed (C6 below)

### Handoff 3 — `cli.py`'s stale docstring (T13) — fixed (C7 below)

## Phase 4 — Fixes (VERIFIED DEFECT)

### C1 — `Orchestrator(..., verbose=...)` crashed two live entry points

**Files:** `orchestrator/entrypoints/chat_cli.py`, `orchestrator/dashboard_core/chat_view.py`.

`Orchestrator.__init__` (`engine.py:420-433`) has never accepted a `verbose` parameter — no
`**kwargs`, no such name anywhere in its 11-parameter signature. Both
`entrypoints/chat_cli.py::_launch_build()` and `dashboard_core/chat_view.py::_run_build()`
called `Orchestrator(budget=..., verbose=True/False)`, crashing with `TypeError:
Orchestrator.__init__() got an unexpected keyword argument 'verbose'` the moment either was
reached. `chat_cli.py`'s call site already carried a `# type: ignore[call-arg]` — evidence
this was previously flagged by mypy and silenced rather than fixed. **Trigger, path 1:**
`orchestrator chat` → complete the interactive spec-gathering conversation without
`--dry-run` → crash at the build handoff. **Trigger, path 2:** the dashboard's live
`/ws/chat` websocket endpoint (registered at `dashboard_core/core.py:337-338`) → same crash.
Empirically confirmed removing the kwarg (verified: `Orchestrator(budget=Budget(max_usd=1.0))`
proceeds past construction to the expected, environment-specific "LLM API key not found"
check) is a complete, correct fix — `verbose` was never read or used anywhere, so nothing is
lost by dropping it.

**Fix:** removed `verbose=...` from both call sites.

### C2 — the `dashboard` console script was broken on every invocation

**Files:** `orchestrator/dashboard.py`, `orchestrator/cli_dashboard.py`.

A three-way signature mismatch: `cli_dashboard.py` (the actual `dashboard` console-script
entry point registered in `pyproject.toml`) called `run_dashboard(host=, port=,
open_browser=)`; `orchestrator/dashboard.py`'s backward-compat shim aliased that name to
`dashboard_core/mission_control.py::create_view()`, which takes **zero** arguments; the
correctly-parameterized implementation (`dashboard_core/core.py::run_dashboard(view="",
host="0.0.0.0", port=8888)`) was never wired to either name, and doesn't accept
`open_browser` either — that parameter has never been implemented anywhere in the real code
(grepped for `open_browser`/`webbrowser` across the whole dashboard stack: zero hits outside
the one broken call site). This `TypeError` is not caught by `cli_dashboard.py`'s own
`except ImportError` — it propagates as a raw crash instead of the friendly
missing-dependency message the file was clearly designed to show. Confirmed this is one of
only 3 console-scripts this project ships (`orchestrator`, `mllm`, `dashboard`).

**Fix:** `dashboard.py`'s shim now imports the real `run_dashboard` from `dashboard_core`
(the package's own `__init__.py` already correctly re-exports it from `.core`) instead of
aliasing the zero-arg `create_view`. `cli_dashboard.py`'s call site now passes only
`host=`/`port=`, matching the real signature; the `--no-browser` flag was removed rather than
kept as a second silently-ignored flag (per-flag it never had any effect, since the command
always crashed before that flag could matter — leaving it registered post-fix would have
created exactly the "flag parsed but does nothing" shape flagged separately in C5). Verified
the fixed call binds cleanly (`inspect.signature(run_dashboard).bind(host=..., port=...)`
raises no `TypeError`).

### C3 — `commands/kanban.py` broke all four kanban subcommands

**File:** `orchestrator/commands/kanban.py`.

Two lazy imports (`from .kanban.board import KanbanBoard`, `from .kanban.dispatcher import
KanbanDispatcher`) used single-dot depth from inside `orchestrator/commands/kanban.py` — a
flat module, not a package — so Python looked for `orchestrator.commands.kanban.board`
(nonexistent) instead of the real `orchestrator.kanban.board` (two dots up). This broke
**all four** kanban subactions (`enqueue`, `list`, `stats`, `start`) unconditionally, with no
try/except catching it. `kanban/board.py`'s own docstring confirms `orchestrator kanban
start` is its only intended entry point. Confirmed via direct call
(`kanban.execute(Namespace(command="list", status=None))`) that this crashed with
`ModuleNotFoundError: No module named 'orchestrator.commands.kanban.board'; 'orchestrator.
commands.kanban' is not a package` pre-fix, and now runs end-to-end (`"No tasks found."`)
post-fix — a real, working 441-line SQLite-backed work queue (atomic claim via
`UPDATE...RETURNING`, per-board worker isolation) was completely unreachable by any user due
to a wrong dot count.

**Fix:** corrected both imports to `..kanban.board`/`..kanban.dispatcher`.

### C4 — `commands/gateway.py` had the identical bug shape

**File:** `orchestrator/commands/gateway.py`.

`from .gateway.run import OrchestratorGateway, GatewayConfig` — same root cause as C3 (a
basename collision between the flat `commands/gateway.py` module and the real
`orchestrator/gateway/` package it needs to reach two dots up, not one). Broke both `gateway
start` and `gateway status`. Confirmed via direct call that `gateway status` now runs
end-to-end (`"Gateway: no platforms connected"`, `"Gateway running: True"`).

**Fix:** corrected to `..gateway.run`.

### C5 — `commands/codebase.py`'s `--budget` flag was silently ignored

**File:** `orchestrator/commands/codebase.py`.

`register()` defines `--budget` (default `10.0`) for the `modify` subcommand, but
`execute()`/`_run_modify()` never read `args.budget` — the budget was hardcoded to
`Budget(max_usd=10.0)` regardless of what the user passed. A user running `orchestrator
modify --budget 1.0` expecting a strict $1 ceiling, or `--budget 50.0` expecting a higher
one, got $10.0 either way. Matches CLAUDE.md's own recorded "Policy system... not fully
integrated" limitation and this hunt's established "flag exists but is silently a no-op"
pattern (T1, T10 precedent).

**Fix:** threaded `budget` through `execute()` → `_run_modify(..., budget=getattr(args,
"budget", 10.0))` → `Budget(max_usd=budget)`.

**What this fix does NOT claim:** `_run_modify()`'s actual call to
`Orchestrator.modify_codebase(...)` targets a method that has never existed anywhere in this
repository (confirmed via `git log -S"modify_codebase"` — the only hit is a design document's
markdown code sample, never implemented). This is a separate, much larger gap — a planned
feature that was never built — and is **not** fixed here; inventing a real implementation
from scratch is out of this hunt's scope (matching the same discipline applied to Handoff 2's
`nash_backup`). The broad `except Exception` around this call also means `modify`'s failure
is printed to stdout with exit code 0 (a script/CI wrapper would see "success") — also not
fixed, since changing exit-code conventions could affect every command's error handling
uniformly and deserves a deliberate, repo-wide decision rather than a single-file patch.
Both residuals documented below.

### C6 — `commands/nash.py`'s `nash backup` crashed instead of failing cleanly (Handoff 2)

**File:** `orchestrator/commands/nash.py`.

`from orchestrator.nash_backup import get_backup_manager` — `orchestrator/nash_backup.py` has
never existed in this repository's git history (re-confirmed this tier: `git log -S` across
the whole repo returns only T13's own commit that first recorded this handoff). No try/except
guarded it, so every invocation of the live, registered `nash backup [--list|--restore|
--value]` subcommand crashed with a raw `ModuleNotFoundError` traceback. Exhaustively searched
`nash/` and `operations/` for anything resembling a backup manager (`BackupManager`,
`BackupManifest`, `estimate_switching_cost`, `total_size_bytes`) — nothing exists anywhere.
Per this hunt's standing discipline against inventing unbuilt features, the fix does **not**
implement a backup manager — it converts the crash into a clear, bounded message, mirroring
the defensive pattern the *dead* sibling `orchestrator/cli_nash.py` (a second, unreachable
Click-based Nash CLI, confirmed via the survey to have zero callers) already uses for the
exact same broken import.

**Fix:** wrapped the import in `try/except ImportError`, printing "Nash backup/restore is not
implemented yet" and returning cleanly instead of crashing.

### C7 — `cli.py`'s docstring named a nonexistent dispatcher module (Handoff 3)

**File:** `orchestrator/cli.py`.

Three locations (module docstring ×2, `main()`'s docstring) said the dispatcher is
`application/cli_dispatch.py` / `application.cli_dispatch.run()`; `orchestrator/application/
cli_dispatch.py` has never existed. The actual, correct, already-working import
(`from .entrypoints.cli_dispatch import run`) was untouched — purely a documentation
accuracy fix, zero behavioral change.

**Fix:** corrected all three references to `entrypoints.cli_dispatch`.

## Phase 4 — Residual, surveyed but not fixed

- **`Orchestrator.modify_codebase()` missing entirely**, and `commands/modify`'s silent
  exit-code-0-on-failure — see C5's "what this fix does NOT claim." `[REQUIRES HUMAN REVIEW]`.
- **`integrations/mcp_server.py`'s feature-convergence direction** — see Handoff 1.
  `[REQUIRES HUMAN REVIEW]`.
- **`integrations/compat.py`** — 4 broken imports (wrong depth, 3 distinct targets), 2
  silently swallowed to `None` via `except ImportError`, 1 unguarded and crashes the module.
  Fully dead (zero callers anywhere) — real-world impact nil today.
- **`integrations/swiftstack_integration.py`** — an unshimmed, byte-identical duplicate of
  the live root file, carrying the same 6 wrong-depth imports the root copy doesn't have.
  Dead on both sides (zero callers of either copy).
- **`git_sync.py` (root) vs `vcs/sync.py`** — a genuine "fix landed on one side, never
  backported" divergence (root correctly distinguishes `files=None` from `files=[]`; `vcs/`'s
  `targets = files or []` collapses both). Unusual direction: `vcs/*` is normally canonical
  here (its two neighbors, `git_hooks.py`/`git_service.py`, are both correctly-labeled shims
  to `vcs.hooks`/`vcs.service`), but `git_sync.py` is not a shim — it's the more-correct
  independent copy, and an existing test (`tests/test_new_modules.py::TestTwoWaySync`)
  already imports from it. Both copies have zero production callers today.
- **Dead "command center" cluster** (`commands/{center,server,integration,registry}.py` +
  root `command_center*.py`, 7 files) — confirmed zero live callers from outside the cluster;
  internally inconsistent shim direction (`commands/integration.py` imports from the *root*
  `command_center_server.py` even though `commands/server.py` independently defines its own
  diverged version). `[UNK]`/low priority — fully dead, but worth resolving whenever someone
  next touches it.
- **Minor/cosmetic, not fixed:** a stale CLAUDE.md example (`--analyze-codebase` doesn't
  exist; the real command is `orchestrator analyze --path`) plus `cli.py`'s own dead
  `--analyze-codebase` Click command referencing two modules that never existed
  (`codebase_understanding.py`, root `improvement_suggester.py`) — this Click `cli()` object
  isn't even the registered entry point, so it's moot in practice; a typo'd deprecation
  warning in `multi_tenant_gateway.py` ("gatewa" missing the "y"); dead `execute_stats()` in
  `commands/cache_stats.py`; 3 `test_*.py`-named files inside `orchestrator/ide_backend/`
  that `pytest tests/` never collects (`testpaths = ["tests"]`); `orchestrator/gateway.py`
  (root) permanently shadowed by the sibling `gateway/` package — already recorded by an
  earlier tier, reconfirmed here as the reason `integrations/gateway.py` has no reachable
  root twin.

## Phase 4 — Cleared (innocent)

All 24 `commands/` files import cleanly at the top level (confirmed via direct
`importlib.import_module`); the `discover_command_modules()` registration mechanism itself is
correct (18 auto-discovered subcommands, zero name collisions) — every defect found this tier
was inside a lazily-imported handler body, invisible to that top-level check alone.
`ide_backend/`'s 16 files fail to import in this sandbox only because the `dashboard` extra
(`fastapi`) was not installed here — a genuine, correctly-declared optional dependency per
`pyproject.toml`, not a code defect. `orchestrator/connectors.py` has the same
file-vs-package shadowing shape as `gateway.py` but is harmless (a correctly-labeled shim
whose effect the live package's own `__init__.py` already achieves independently).

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used. `fix_revisions`: all 7 fixes (C1-C7) correct on first pass,
RED→GREEN verified on the first attempt. No survey severity claim required correction this
tier.
