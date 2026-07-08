---
name: orchestrator-run-and-operate
description: Deep operations manual for RUNNING the Multi-LLM Orchestrator — CLI subcommand anatomy and discovery mechanism, the primary `python -m orchestrator --project ... --criteria ... --budget ...` invocation and its flags, `--resume`, YAML project-spec format (projects/*.yaml), where outputs land and what the delivery pipeline does to them, the two different "dashboard" launch paths and their port drift, the API gateway/server (admin secret, WS token, rate limiting), the MCP server, and the operational hazards that cost real money or real time (eager OPENROUTER_API_KEY validation, HITL fail-closed, dual budgets, cache TTL claims vs enforcement). Load this before running any real project, before adding a new CLI subcommand, before starting the dashboard/gateway/MCP server, or when symptoms match: "command not found after adding a new commands/*.py file", "dashboard says connection refused on 8888 but I ran `dashboard`", "401 from OpenRouter but client constructed fine", "orchestrator --resume does nothing", "where do generated files end up", "is the security scan blocking my build". For flag/env-var *reference* (not operation), use orchestrator-config-and-flags instead.
---

# orchestrator-run-and-operate

Deep manual for **running** the Multi-LLM Orchestrator: CLI anatomy, project specs,
output pipeline, dashboard, gateway/API server, MCP server, and the operational traps
that burn money or wall-clock time. Ground truth verified 2026-07-08 against
`feat/response-healing` branch.

## When NOT to use this skill

- Flag-by-flag / env-var catalog with wired-or-dead status → `orchestrator-config-and-flags`.
- Installing dependencies, extras, pre-commit, CI setup → `orchestrator-build-and-env`.
- Debugging a specific failure once something has already gone wrong → `orchestrator-debugging-playbook`.
- Architecture/why-questions (Mediator, hexagonal layering, DI container) → `orchestrator-architecture-contract`.
- Gates you must not weaken (coverage, contracts, root-file freeze) → `orchestrator-change-control`.
- Incident history / "didn't we already hit this" → `orchestrator-failure-archaeology`.
- Domain theory (routing math, budget model, eval science) → `llm-orchestration-reference`.
- Quick 20-line cheat sheet → `orchestrator-run` (this skill is the expanded manual; it does not contradict that one, it explains the mechanisms behind it).
- Web mission-control quick start → `dashboard-start` (this skill adds the port-drift trap that one doesn't cover).

---

## 1. CLI anatomy

### 1.1 Two layers of commands

The CLI has two coexisting command mechanisms, both wired from `orchestrator/entrypoints/cli_dispatch.py::run()`:

1. **Legacy flat flags** — `--project`, `--criteria`, `--budget`, `--resume`, `--list-projects`,
   `--visualize`, `--file`, `--dry-run`, etc. — parsed directly on the top-level `argparse.ArgumentParser`.
   These are the primary, most-used path (see §2).
2. **Auto-discovered subcommands** — `python -m orchestrator <subcommand> ...` — one module per
   subcommand under `orchestrator/commands/`.

### 1.2 Subcommand discovery mechanism

`orchestrator/commands/__init__.py::discover_command_modules()` walks the package with
`pkgutil.iter_modules` and registers every module that:
- is not a sub-package (`ispkg is False`), **and**
- does not start with `_` (private helpers — e.g. `_path_utils.py` — are skipped), **and**
- is not in the hardcoded exclusion set `{"center", "integration", "server", "registry"}`
  (these four are re-exported for backward compatibility via `from .center import *` etc.
  in the same `__init__.py`, but are not auto-registered as subcommands).

`cli_dispatch.run()` then does, for every discovered module name:
```python
mod = import_module(f"..commands.{cmd_mod}", __package__)
mod.register(subparsers)
```
So **to add a new subcommand**: drop a new `orchestrator/commands/<name>.py` exposing
`register(subparsers)` (adds an `argparse` subparser and sets `.set_defaults(func=execute)`)
and `execute(args)`. No other file needs to change — it is picked up automatically on next run.
If registration raises, `cli_dispatch.py` catches it and only logs a warning (`logger.warning`),
so a broken new command module fails silently rather than crashing the whole CLI — check logs
if a command you just added doesn't show up in `--help`.

As of 2026-07-08 the discovered (auto-registered) command modules are:
`agent, analyze, build, cache_stats, chat, codebase, dashboard, gateway, kanban, meta, nash,
nexus, nexusscope, slash, website`
(alphabetical; `_path_utils.py` and the four excluded names are not in this list).

### 1.3 Every real subcommand (verified against `orchestrator/commands/*.py`)

| Subcommand | Module | Purpose |
|---|---|---|
| `agent "<intent>"` | `agent.py` | Autonomous agent-mode run from a free-text intent |
| `analyze <path>` | `analyze.py` | Analyze an existing codebase |
| `build "<desc>"` | `build.py` | Build an app from a description (thin wrapper) |
| `cache-stats` | `cache_stats.py` | Print DiskCache hit/miss stats |
| `chat` | `chat.py` | Interactive chat REPL |
| `codebase ...` | `codebase.py` | Codebase-related utilities |
| `dashboard` | `dashboard.py` | **Prints a text report** (`render_dashboard`) of cross-run model rankings — NOT the web dashboard. `--days N` (default 30) sets the lookback window. See §4 for the actual web UI. |
| `gateway {start,status}` | `gateway.py` | Multi-platform messaging gateway (`orchestrator.gateway.run.OrchestratorGateway`); `-p/--platforms name:port ...` |
| `kanban enqueue "<desc>"` | `kanban.py` | Kanban task queue |
| `meta ...` | `meta.py` | Meta-learning / rollout controls |
| `nash status` | `nash.py` | Nash-equilibrium routing status |
| `nexus search "<query>"` | `nexus.py` | Web search integration |
| `nexusscope sessions` / `report` | `nexusscope.py` | NexusScope profiling session listing / report export |
| `slash ...` | `slash.py` | Slash-command dispatch (used by chat REPL, not typically invoked directly) |
| `website "<desc>"` | `website.py` | Website generation shortcut |

`center.py`, `integration.py`, `server.py`, `registry.py` are **not** subcommands — they are
library modules re-exported for backward compat (`server.py` hosts `CommandCenterServer`,
a raw `websockets`-based server started programmatically, not via CLI — see §5.3).
`_path_utils.py` is a private helper, never a command.

---

## 2. Primary invocation

```bash
# New project (the flow used ~90% of the time)
python -m orchestrator --project "Build a FastAPI REST API" --criteria "All endpoints tested" --budget 2.0

# Resume a crashed/interrupted run by project_id
python -m orchestrator --resume <project_id>

# List saved projects
python -m orchestrator --list-projects

# Load a full spec from YAML instead of flat flags
python -m orchestrator --file projects/scientific_nbody_symplectic.yaml
```

Equivalent entry points (`pyproject.toml [project.scripts]`): `orchestrator` and `mllm` both
map to `orchestrator.cli:main`. `python -m orchestrator` always works without install; prefer
it during development.

### 2.1 Startup cost (UNVERIFIED exact number — re-measure before quoting precisely)

A cold `python -m orchestrator --help` takes roughly **12-13s warm** (i.e. after Python's own
import caches are warm, not a true cold boot) as of 2026-07-08, dominated by the `openai`
SDK import chain pulled in transitively by `orchestrator/infrastructure/llm_client.py`.
`instructor` (structured-output library) used to add ~30s on top of that on Windows; it is now
**lazy-imported** — `orchestrator/infrastructure/llm_client.py` imports `instructor` inside
functions (`_instructor_mode()`, the two `instructor.from_openai(...)` call sites), not at
module scope, specifically to avoid paying that cost on every CLI invocation. Re-verify with:
```bash
python -c "import time; t=time.time(); import orchestrator.entrypoints.cli_dispatch; print(time.time()-t)"
```

### 2.2 Key flags (legacy flat-flag path)

| Flag | Default | Notes |
|---|---|---|
| `--project` / `-p` | — | required unless `--resume`/`--file`/`--list-projects` |
| `--criteria` / `-c` | — | required alongside `--project` |
| `--budget` / `-b` | `8.0` | USD ceiling for this run's `Budget` (per-run, see §6.1) |
| `--time` | `5400` (s) | wall-clock ceiling |
| `--resume` | `""` | resume by `project_id` |
| `--file` / `-f` | `""` | load a `projects/*.yaml` spec (see §3) instead of flat flags |
| `--output-dir` / `-o` | auto (`_default_output_dir`) | where generated files land |
| `--tdd-first` | off | enable Test-First Generation for code tasks |
| `--visualize {mermaid,ascii}` | — | render the task DAG and exit (no execution) |
| `--critical-path` | off | print critical path (combine with `--visualize` or standalone) |
| `--dry-run` | off | plan only, no execution (`orch.dry_run(...)`) |
| `--new-project` | off | skip resume-detection gate, force a fresh run |
| `--raw-tasks` | off | bypass `AppBuilder`, use the legacy flat task-file path |
| `--quiet` / `-q` | off | suppress progress rendering |
| `--concurrency` | `3` | max simultaneous API calls |
| `--agent-profile` | — | `standard\|max\|creative\|conservative\|research` — preset quality_mode/iteration_cap/temperature bundles |

Full flag catalog with wired/dead status lives in `orchestrator-config-and-flags` — this table
is the operationally load-bearing subset.

### 2.3 What `--project` actually does by default

Unless `--raw-tasks` is passed, `_async_new_project()` routes through **`AppBuilder`**
(`orchestrator/app_builder.py`), not the raw `Orchestrator.run_project_streaming()` loop
directly. Before that, two gates run:
1. **Resume-detection gate** (`_check_resume`, unless `--new-project`): extracts keywords from
   the description, queries `StateManager.find_resumable()` with a **0.2s timeout**
   (`asyncio.wait_for(..., timeout=0.2)` — a slow state DB silently skips resume-detection rather
   than hanging), and on an exact keyword match resumes automatically without prompting; on
   partial matches it prompts interactively (`input()` — breaks in non-interactive/CI contexts
   unless piped or `--new-project` is set).
2. **Enhancement pass** (unless `--no-enhance`): `ProjectEnhancer` may rewrite `description`/`criteria`
   before the build starts — this happens (and may call an LLM) *before* decomposition.

### 2.4 `--resume`

```bash
python -m orchestrator --resume <project_id>
```
Loads project state via `StateManager.load_project(project_id)`; exits with `sys.exit(1)` and
`"Project {id} not found."` if missing. Re-runs `Orchestrator.run_project()` with the *original*
`project_description`/`success_criteria` and the *same* `project_id` — the engine's own
task/result state is what makes this idempotent (already-completed tasks are not re-run;
this depends on `StateManager` persistence in `~/.orchestrator_cache/state.db`, see §3.3).
`--budget`/`--time` on the resume invocation apply to the **new** `Budget` object — they are
not automatically inherited from the original run.

---

## 3. Project specs and outputs

### 3.1 `projects/*.yaml` schema

Loaded via `orchestrator/project_file.py::load_project_file()`, invoked with `--file <path>`.
Verified against `projects/scientific_nbody_symplectic.yaml`:

```yaml
project: |
  <multi-line free-text project description>

criteria: |
  <multi-line free-text acceptance criteria>

budget_usd: 3.5            # → Budget.max_usd
time_seconds: 4800          # → Budget.max_time_seconds
concurrency: 3
project_id: "scientific-nbody-symplectic-v1"   # stable, human-chosen ID (enables --resume)
output_dir: "./outputs/scientific_nbody_symplectic"

quality_targets:             # per-task-type evaluation floor (0.0-1.0)
  code_generation:   0.88
  code_review:       0.85
  complex_reasoning: 0.90
  evaluation:        0.90

policies:
  - name: no_training
    allow_training_on_output: false
```

`projects/` also has `example_simple.yaml` and `example_full.yaml` as minimal/maximal
references. Other real specs in the directory (`backend_rest_api.yaml`,
`frontend_react_dashboard.yaml`, `analysis_*.yaml`, etc.) are good genre-specific examples —
skim one matching your task type before hand-writing a new spec.

### 3.2 Output pipeline (what happens after generation)

For every successful path (`--resume`, `--file`, plain `--project`), the CLI dispatch does,
in order:
1. `write_output_dir(state, output_dir, project_id=...)` — writes generated files to disk.
2. `organize_project_output(path, auto_generate_tests=True, run_tests=True, ...)` —
   `orchestrator/output_organizer.py::OutputOrganizer`, which:
   - **Formats** generated code (`output/formatter.py`: `ruff --fix` then `black` for `.py`;
     `prettier` best-effort for web files) — failures here are logged and appended to
     `report.errors` but **never block delivery**.
   - **Security-scans** the output tree via `orchestrator/safety/generated_output_scanner.py::scan_output_dir`.
     `CRITICAL`/`HIGH` findings are logged (`[SEC] ... blocking`) and appended to
     `report.errors` as `"security: N blocking finding(s) in generated output"` — but this is
     **advisory, not a hard gate**: the scan step is wrapped in `try/except` specifically so it
     "never crash[es] delivery on the scan itself," and the organizer does not delete or refuse
     to write flagged files. Treat a non-empty `report.errors` security entry as a **must-review
     before shipping**, not an automatic abort — verify manually if you see one.
   - Moves `task_*.py/md/json` into `tasks/` (creates `tasks/__init__.py`).
   - Optionally auto-generates and runs tests, iterating up to `--max-fix-iterations` (default 3)
     to reach `--min-pass-rate` (default 0.7).

### 3.3 State DB and cache locations

All cache/state paths flow through `orchestrator/infrastructure/path_provider.py::CachePathProvider`
— the single source of truth (replaces 21+ historical hardcoded paths). Default root:
`Path.home() / ".orchestrator_cache"`, overridable via **`ORCH_CACHE_HOME`** env var (used by
tests to sandbox state). Key files under that root:

| File | Purpose |
|---|---|
| `state.db` | `StateManager` — project/task/result persistence (`orchestrator/infrastructure/state.py`; root `orchestrator/state.py` is a backward-compat shim re-exporting from there) |
| `cache.db` | `DiskCache` (`orchestrator/infrastructure/cache.py`) — prompt-hash → LLM response cache, dedupes identical calls, `$0` cost on hit |
| `cache_l2.db`, `secure_cache.db`, `kanban.db`, `patterns.db`, `telemetry.db`, `trajectories.db`, `skills.db`, `events.db`, `budget.db` | other subsystem stores, same provider |

**Cache TTL caution**: `cache_ttl_hours: int = 48` exists as a *config field* in both
`orchestrator/crosscutting/config.py` and `orchestrator/plugins/cost_optimization.py`
(asserted `> 0` by `tests/unit/test_cost_reduction.py::test_cache_ttl_is_positive`), but the
actual `DiskCache` implementation in `orchestrator/infrastructure/cache.py` **does not read or
enforce this field** — `get()`/`put()` have no expiry check at all; entries live until
`clear()` is called explicitly. Do not assume stale LLM responses self-expire after 48h in
practice — this is a declared-but-unwired setting (cross-reference `orchestrator-config-and-flags`
for the full wired/dead audit). If you need cache correctness after a prompt/model change,
clear `cache.db` manually or call `DiskCache.clear()`.

---

## 4. Dashboard (two different things, one port collision)

There are **two unrelated "dashboard" surfaces** — do not confuse them:

1. **`python -m orchestrator dashboard`** (CLI subcommand, `orchestrator/commands/dashboard.py`) —
   prints a **text report** to stdout via `render_dashboard(TelemetryStore(), days=N)`. No
   server, no port, no browser. `--days N` (default 30).
2. **Web Mission Control dashboard** — an actual HTTP+WebSocket server. Three ways to start it,
   verified to converge on the same `orchestrator/dashboard_core/core.py::run_dashboard` /
   `dashboard_mission_control` machinery:

```bash
# A. Standalone launcher script (defaults to port 8888)
python scripts/utils/start_dashboard.py            # or scripts\batch\start_dashboard.bat on Windows
python scripts/utils/start_dashboard.py --port 9000 --no-browser

# B. Installed console-script entry point (defaults to port 8080 — DIFFERENT DEFAULT, see below)
dashboard --port 8888

# C. Direct module invocation
python -m orchestrator.cli_dashboard --port 8888
```

**Port default drift (verified 2026-07-08, not yet reconciled in code):**
- `scripts/utils/start_dashboard.py` and `orchestrator/dashboard_core/core.py::run_dashboard()`
  both default to **port 8888**.
- The installed `dashboard` script / `orchestrator/cli_dashboard.py::main()` — which is what
  `pyproject.toml`'s `[project.scripts] dashboard = "orchestrator.cli_dashboard:main"` actually
  invokes — defaults `--port` to **8080**, and passes that straight through to
  `orchestrator.dashboard.run_dashboard(host=..., port=args.port, ...)`.

**Practical consequence**: if you run bare `dashboard` (the pip-installed script) expecting
`http://localhost:8888` (as `dashboard-start`/`orchestrator-run` document), you'll get a server
on `:8080` instead, or "connection refused" on 8888. Always pass `--port 8888` explicitly with
the `dashboard` script, or use `scripts/utils/start_dashboard.py` / `python -m orchestrator.cli_dashboard --port 8888` for the documented port. This drift is a candidate cleanup, not yet fixed —
don't silently "fix" it by editing a default without going through `orchestrator-change-control`
(it's a public CLI default; changing it is a behavior change).

`pip install -e ".[dashboard]"` installs `fastapi`, `uvicorn[standard]`, `websockets`, `httpx`.

---

## 5. API gateway / server, and the WebSocket command center

### 5.1 `APIServer` (`orchestrator/api_server.py`)

A REST API (aiohttp-based) started **programmatically**, not via a CLI subcommand — there is no
`if __name__ == "__main__"` in this file. Construct and `await server.start()` from your own
script/service. Key constructor args: `port=8000`, `host="localhost"`, `cors_origins=[]`
(same-origin only unless overridden), `auth_required=True` (logs a warning if disabled — dev
only), `rate_limit=100` requests per `rate_window=60`s (token-bucket), `max_request_size=10MB`.

- **Admin key registration**: `POST /register_key` requires the **`ORCHESTRATOR_ADMIN_SECRET`**
  env var to be set; if unset, the endpoint responds `"Admin key registration disabled
  (ORCHESTRATOR_ADMIN_SECRET not set)"` — registration is fail-closed by default.
- **Auth**: `Authorization: Bearer <api_key>`; keys are SHA-256-hashed before storage/lookup
  (`_verify_api_key`), never compared or stored in plaintext.
- **Rate limiting**: token-bucket middleware (`_rate_limit_middleware`) wraps every route;
  exceeding it returns `429`-style `"Rate limit exceeded"` with a `Retry-After`.

### 5.2 `orchestrator gateway {start,status}` CLI subcommand

Thin wrapper (`orchestrator/commands/gateway.py`) around
`orchestrator.gateway.run.OrchestratorGateway` — a **multi-platform messaging gateway**
(different concern from `APIServer`/HTTP gateway facade mentioned in `CLAUDE.md`'s pattern
table). `-p/--platforms name:port ...` registers platforms; `start` blocks on
`asyncio.sleep(1)` in a loop until `Ctrl+C`.

### 5.3 `CommandCenterServer` (`orchestrator/commands/server.py`)

A raw `websockets`-based real-time alert/metrics server (`websockets.serve`), started
programmatically via `CommandCenterServer().start(host="127.0.0.1", port=8765)` — **not**
auto-registered as a CLI subcommand (it's in the discovery exclusion list, see §1.2). Every
client connection **requires token auth**: the server refuses to start accepting clients
without **`ORCHESTRATOR_WS_TOKEN`** set (`websocket.close(code=4001, reason="Server token not
configured")` if unset), and compares the client's first-message token with
`hmac.compare_digest` (constant-time, timing-attack-resistant). 5-second auth timeout.

### 5.4 MCP server (`orchestrator/mcp_server.py`)

```bash
# stdio (subprocess — e.g. Claude Desktop MCP client config)
python -m orchestrator.mcp_server

# HTTP (shared, long-lived)
python -m orchestrator.mcp_server --http --port 8181
```
Requires the `mcp` SDK (`from mcp.server import Server` — guarded by `HAS_MCP`, degrades if
absent). Exposes tools: `orch_search`, `orch_query`, `orch_get`, `orch_status`, `orch_memory`,
`orch_persona`, `orch_session`. There is a second `orchestrator/integrations/mcp_server.py` —
if both exist, check which one your MCP client config actually points at before assuming
behavior; this skill only verified `orchestrator/mcp_server.py` in depth.

---

## 6. Operational cautions — read before a real (money-spending) run

### 6.1 Real runs spend real money — dual budget

Every run enforces **two independent budgets simultaneously** (see `llm-orchestration-reference`
for the theory): the per-run `Budget` (`--budget`/`--time`, or `budget_usd`/`time_seconds` in a
YAML spec) caps *this* `run_project()` call; `BudgetHierarchy` (if wired by the caller) caps
spend across runs (org/team/job). **Always pass `--budget` explicitly** — the flag default is
`8.0` USD, high enough to be a real bill if you forget it, not a safe accidental default.

### 6.2 HITL is fail-closed by default

Human-in-the-loop decision gates (`orchestrator/hitl/`) fail closed: an unanswered gate blocks
rather than silently proceeding. `ORCH_HITL_AUTOAPPROVE=true` is a legacy escape hatch that
auto-approves every gate — **dev/CI only**, never set it for an unattended run against a real
budget/environment you care about. See `orchestrator-failure-archaeology` (commit `11deb573`)
for why this defaults fail-closed: it used to silently auto-approve.

### 6.3 `OPENROUTER_API_KEY` is validated eagerly at `UnifiedClient` construction (as of 2026-07-07)

`orchestrator/infrastructure/llm_client.py`'s client constructor does:
```python
self._api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
if not self._api_key:
    raise AuthenticationError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
```
This only checks that **some string is present** — a dummy/placeholder key satisfies
construction but will 401 on the first real API call, and the client's retry/fallback logic
will burn real wall-clock time retrying a doomed call before giving up (bit a test fixture on
2026-07-08). Two consequences to plan around:
- **Set a real key before any run that will actually call an LLM**, not just before you expect
  the "interesting" part of the pipeline to start.
- **Architecture-rules generation and similar pre-decomposition steps may already construct a
  client and make real LLM calls at project start, before decomposition even begins** — a bad
  key fails loud-but-late (401s during setup), not at the point you'd naturally think to check
  credentials. If a run fails immediately with retries/401s, check the key **before** assuming
  the pipeline logic is broken.

### 6.4 Resume auto-detection can trigger interactive `input()`

See §2.3 — a partial keyword match on a fresh `--project` invocation (without `--new-project`)
can prompt `input()` for confirmation. This will hang or raise `EOFError` in non-interactive
contexts (CI, piped scripts) with no stdin available; the code catches `EOFError`/`KeyboardInterrupt`
and treats it as "don't resume," so it won't hang forever, but it silently discards your
intended resume — pass `--new-project` explicitly in any non-interactive/scripted invocation to
avoid depending on this fallback behavior.

---

## 7. Cost hygiene

- **Free-tier-first routing is the expectation, not a suggestion.** VFM (Value-for-Money)
  routing prefers `:free`-suffixed OpenRouter model variants where quality allows — locked by
  `tests/unit/test_vfm_routing.py`. If a run is burning non-trivial cost on a task type that
  should route free-tier-first, that's a routing-config bug, not expected behavior — see
  `orchestrator-config-and-flags` for the drift trap (config JSON keys must exactly match
  `Model` enum values or entries silently drop) before assuming the routing logic itself is at
  fault.
- **Cache warm-up matters for iterative work.** Repeated runs against the same prompt/model/params
  hit `DiskCache` ($0 cost) — but only on an **exact** prompt-hash match
  (`prompt_hash(model, prompt, max_tokens, system, temperature)` — see
  `orchestrator/models.py`); changing temperature or trimming whitespace in a prompt is a cache
  miss. Don't expect cache hits across prompt-enhancement passes that reword the description.
- Empty/blank LLM responses are **never cached** (`DiskCache.put()` early-returns on
  falsy/whitespace-only `response`) — a flaky empty reply won't poison future identical calls.

---

## Provenance and maintenance

Re-verify before trusting any of the following if this file is more than a few weeks old
(today: 2026-07-08):

```bash
# Discovered CLI subcommand list
python -c "from orchestrator.commands import discover_command_modules; print(discover_command_modules())"

# Confirm dashboard port drift still exists
grep -n "port" orchestrator/cli_dashboard.py orchestrator/dashboard_core/core.py scripts/utils/start_dashboard.py

# Confirm eager API-key validation still present
grep -n "OpenRouter API key not found" orchestrator/infrastructure/llm_client.py

# Confirm cache_ttl_hours is still declared-but-unwired in DiskCache
grep -n "ttl" orchestrator/infrastructure/cache.py

# Confirm cache path provider root
grep -n "ORCH_CACHE_HOME" orchestrator/infrastructure/path_provider.py

# Re-time cold import (see §2.1)
python -c "import time; t=time.time(); import orchestrator.entrypoints.cli_dispatch; print(time.time()-t)"

# Confirm VFM free-tier lock test still exists
pytest tests/unit/test_vfm_routing.py -q
```
