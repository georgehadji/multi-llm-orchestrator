---
name: orchestrator-config-and-flags
description: Complete catalog of every configuration axis in the Multi-LLM Orchestrator — the five JSON configs (costs/routing/fallbacks/limits/thresholds.json) and their silent-drop drift trap, orchestrator/config.py runtime constants, the ORCH_-prefixed pydantic FeatureFlags, the full environment-variable table (USE_* OpenRouter flags, ORCH_HITL_AUTOAPPROVE, ENABLE_PR_COMMENTS, ENABLE_AUTO_COMMIT, ORCH_CACHE_HOME, ORCHESTRATOR_ADMIN_SECRET, ORCHESTRATOR_WS_TOKEN) with wired-or-dead status, API-key loading (.env override=True, eager OPENROUTER_API_KEY validation), and the dependency-declaration rule. Load this when you see symptoms like "model silently never routed to", "config entry ignored", "fallback never fires", "is USE_PROVIDER_SORTING doing anything", "USE_RESPONSE_HEALING not taking effect", "AuthenticationError: OpenRouter API key not found", "tests fail without API key", "CI collection died on import", "which env var controls X", "where do I add a new model/task type", or before editing anything under orchestrator/config/ or adding an environment flag.
---

# Orchestrator Configuration & Flags Catalog

Every knob in this repo, where it is read, and whether it actually does anything.
All claims verified against the working tree on branch `feat/response-healing`, 2026-07-08.
Repo root: the directory containing `pyproject.toml` and `orchestrator/`.

## 0. Map of all configuration axes

| # | Axis | Location | Mechanism | Owner section |
|---|------|----------|-----------|---------------|
| 1 | Model/routing/cost data | `orchestrator/config/*.json` (5 files) | JSON, loaded lazily | §1 |
| 2 | Runtime constants | `orchestrator/config.py` | Plain Python classes | §2 |
| 3 | Feature flags (ORCH_*) | `orchestrator/crosscutting/config.py` | pydantic-settings, env prefix `ORCH_` | §3 |
| 4 | Ad-hoc env vars | scattered `os.getenv` call sites | env | §4 |
| 5 | API keys + .env | `orchestrator/cli.py` `load_dotenv(override=True)` | env / `.env` file | §5 |
| 6 | Python dependencies | `pyproject.toml` `[project.dependencies]` | pip | §6 |

## 1. The five JSON configs (`orchestrator/config/`)

| File | Schema (top level) | Meaning |
|------|--------------------|---------|
| `costs.json` | `{ "<model-id>": {"input": float, "output": float} }` | USD per **million** tokens (see cost math in `llm_client.py::call`, divides by 1_000_000) |
| `routing.json` | `{ "<task-type>": ["<model-id>", ...] }` | Ordered preference list per task type; index 0 = lead model |
| `fallbacks.json` | `{ "<model-id>": "<model-id>" }` | Single-step fallback chain: on failure of key, try value |
| `limits.json` | `{ "<task-type>": int }` | Max output tokens per task type |
| `thresholds.json` | `{ "<task-type>": float }` | Quality-accept threshold per task type (0.0–1.0) |

`<model-id>` MUST be a `Model` enum **value** from `orchestrator/models.py` (e.g. `"anthropic/claude-sonnet-5"`).
`<task-type>` MUST be a `TaskType` enum value (e.g. `"code_generation"`). Verified 2026-07-08: 130 Model values, 9 TaskType values.

### Two loader paths (both exist — know which one your code uses)

1. **`orchestrator/models.py`** (primary, used by `ROUTING_TABLE` / `COST_TABLE` / `FALLBACK_CHAIN` / `DEFAULT_THRESHOLDS` / `MAX_OUTPUT_TOKENS`):
   - `_load_static_config(filename)` (models.py ~line 754) reads `orchestrator/config/<file>` with a swallow-all `except Exception: return {}`.
   - Tables are **lazy**: populated on first attribute access via module `__getattr__` (models.py ~line 824) — no I/O at import time (Rule #2 of the Four Unbreakable Rules; see `orchestrator-architecture-contract`).
2. **`orchestrator/infrastructure/adapters/config_adapter.py::JsonConfigAdapter`** — implements the `ConfigPort` used by domain services (e.g. `CostService` inside `UnifiedClient`). Same files, separate in-memory cache, logs a warning (rather than silently returning `{}`) on missing file.

### THE DRIFT TRAP (bites repeatedly — read before editing any JSON)

The models.py builders filter with membership guards:

```python
# models.py, _build_cost_table (~line 775)
return {Model(k): v for k, v in data.items() if k in Model._value2member_map_}
```

Any JSON key (or routing list entry, or fallback key/value) that is not **byte-for-byte equal** to an enum value is **SILENTLY DROPPED** — no error, no log line. Symptoms: a model "mysteriously" never routed to; a fallback that never fires; untracked spend because a routed model has no cost entry.

- History: repeated fixes on 2026-06-23; resync in commit `ab17b5f4`. A 2026-07-07 audit additionally reported an orphan `google/gemini-2.5-flash` key present even on master. As of 2026-07-08 that exact key is gone (only `google/gemini-2.5-flash-image` remains, which IS a valid enum value) — **but drift is live right now**, see next block.
- Full incident narratives live in `orchestrator-failure-archaeology`; this skill owns the rule itself.

**Live drift-checker output (run 2026-07-08, exit code 1):**

```
Checked 133 cost keys, 9 routing keys, 51 fallback pairs against 130 Model values and 9 TaskType values.

HARD DRIFT - 5 config entr(ies) silently dropped by models.py:
  costs.json key not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4.1'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-sonnet-4'
  fallbacks.json['qwen/qwen3-coder'] value not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'

Fix: make the JSON key/value EXACTLY equal the Model/TaskType enum .value.

INFO: 1 Model value(s) have no costs.json entry:
  internal/nano-banana-2
```

Command: `python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_config_drift.py` (add `--strict` to also fail on the INFO direction). Run it after **every** edit to a config JSON or to the `Model`/`TaskType` enums.

### The free-tier lead lock (do not reorder routing.json casually)

`tests/unit/test_vfm_routing.py` locks the VFM (value-for-money) routing doctrine for every text task type:

- `test_lead_candidate_is_free_tier` — `ROUTING_TABLE[task][0].value` MUST end with `":free"` (lines 78–83).
- `test_no_silent_drops` — every routing.json entry must survive into the built table.
- `test_every_routed_model_is_priced` — every routed model must have a costs.json entry.
- `test_has_premium_fallback` — chain length ≥ 2 and the LAST model must NOT be `:free` (reliable paid last resort).
- `:free` is an OpenRouter **endpoint variant** — it reaches the API intact and costs $0; free entries in costs.json must be `{"input": 0.00, "output": 0.00}` (also asserted). Theory: see `llm-orchestration-reference`.

If you reorder or extend `routing.json`, run `pytest tests/unit/test_vfm_routing.py -q` before claiming done.

## 2. `orchestrator/config.py` — runtime constants

Plain classes of class attributes ("RuntimeConfig"), zero external deps, importable everywhere. Key values (verified 2026-07-08):

| Class | Notable values |
|-------|----------------|
| `Timeout` | `API_CALL_SHORT=30.0`, `API_CALL_MEDIUM=60.0`, `API_CALL_LONG=120.0`, `API_CALL_EXTREME=300.0`; phases: `GENERATE_STANDARD=180`, `GENERATE_EXTENDED=240`, `CRITIQUE=240`, `EVALUATE=60`, `DECOMPOSE=160`; `CB_RESET_TIMEOUT=60.0` |
| `TokenLimits` | `CODE_STANDARD=4096`, `CODE_LONG=8192`, `STANDARD=2000`, `CONTEXT_TRUNCATION=8192`, `DECOMPOSE_CONTEXT=1200` |
| `BudgetDefaults` | `MAX_USD_DEFAULT=8.0` (CLI default), `MAX_USD_DRY_RUN=1.0`, `MAX_CONCURRENCY=3`, `MAX_PARALLEL_TASKS=3`, `ARA_FRACTION=0.3`, `COMPETITIVE_FRACTION=0.3` |
| `QualityThresholds` | `ACCEPT=0.7`, `RETRY_TRIGGER=0.7`, `SEMANTIC_CACHE=0.85`, `PREFLIGHT_WARN=0.6`, `PREFLIGHT_BLOCK=0.3` |
| `RetryDefaults` | `MAX_ATTEMPTS=3`, `MIN_WAIT=0.1`, `MAX_WAIT=30.0`, `EXPONENTIAL_BASE=2.0`, `CB_FAILURE_THRESHOLD=3` |
| `AnalysisDefaults` | `MAX_CONTEXT_TOKENS=60000`, `MAX_SECTION_TOKENS=4096`, `MAX_CONCURRENCY=2` |
| `LearningDefaults` | `MAX_PATTERNS=200`, `MAX_FAILURES=50`, `COMPRESSION_THRESHOLD=10` |
| `AgentDefaults` | `MAX_ITERATIONS=3`, `MAX_PARALLEL_AGENTS=3`, `MIN_CONFIDENCE=0.5` |
| `OpenRouterOptimizations` | frozen dataclass of the seven `USE_*` env flags (§4.1), all default `False`; module-level singleton `OPENROUTER_OPTS = OpenRouterOptimizations.from_env()` — **evaluated once at import time**, so setting a `USE_*` env var after import has no effect |
| `MemoryConfig` | `memory_dir=~/.orchestrator`, `auto_save_interval=50`, `max_patterns=200`, `cache_ttl_hours=1` |

Beware: these are per-task-type-agnostic defaults; the JSON files in §1 override per task type where both exist (e.g. `limits.json` vs `TokenLimits`).

## 3. `orchestrator/crosscutting/config.py` — pydantic ORCH_* flags

Two `BaseSettings` singletons created at import time: `flags = FeatureFlags()` and `settings = OrchestratorSettings()`. Both use `env_prefix="ORCH_"` and read `.env` directly (`env_file=".env"`), `extra="ignore"`.

- Usage: `from orchestrator.crosscutting.config import flags, settings`.
- `ORCH_CONTEXT_COMPRESSION=true` → `flags.context_compression`, etc.
- **Notable defaults** (verified 2026-07-08): `context_compression=False`, `pattern_injection=False`, `batch_parallelism=False`, `plugin_sandbox=True`, `audit_log=True`; optional-module gates mostly `True` (`a2a_enabled`, `red_team_enabled`, `tdd_enabled`, `cost_optimization_enabled`, ...); `tracing_enabled=False`; all `vs_*` Verbalized Sampling gates `False` (VS pipeline documented as retired — see `orchestrator-failure-archaeology`); `taste_skill_enabled=True`.
- `OrchestratorSettings`: `max_concurrency=3`, `default_budget_usd=10.0`, `cache_ttl_hours=48` (the response DiskCache TTL — the real cost win, see `llm-orchestration-reference`), `semantic_cache_threshold=0.85`, `dashboard_port=8000`, `cache_home=""` (→ `ORCH_CACHE_HOME`).
- **Duplication warning:** `FeatureFlags` also declares `use_json_schema_responses` ... `use_embedding_cache` (lowercase, `ORCH_USE_*` env names) mirroring §4.1's `USE_*` flags. Neither copy is consumed in the LLM call path (verified §4.1). Do not "fix" one copy and assume the other follows.

## 4. Environment-flag table (grep-verified 2026-07-08)

Legend — **Wired**: read AND affects a production code path. **Helper-only**: read, feeds a tested helper that no production code calls yet. **Dead**: read into a value nothing consumes.

### 4.1 `USE_*` OpenRouter flags (all read in `orchestrator/config.py::OpenRouterOptimizations.from_env`, all default `false`)

| Flag | Status (2026-07-08) | Evidence |
|------|---------------------|----------|
| `USE_JSON_SCHEMA_RESPONSES` | **Dead** | Only read site is `from_env()`. `OPENROUTER_OPTS` is imported by `engine.py:54` and `engine_core/engine_deps.py:19` but **never referenced after import** in either file. |
| `USE_MODEL_VARIANTS` | **Dead** | Same. Related helper `_resolve_provider_variant` (`llm_client.py:475`, handles `:nitro`/`:floor`/`:exacto`) exists and is unit-tested (`tests/unit/test_provider_variants.py`) but has no production caller. |
| `USE_NATIVE_FALLBACKS` | **Dead** | Same — no consumer. |
| `USE_PROVIDER_SORTING` | **Dead — confirmed by trace** | Declared in `from_env()`; zero other references in `orchestrator/` outside `config.py` and the unused-import sites above. Matches the long-standing "dead flag" finding; do not document it as a cost lever. |
| `USE_STREAMING` | **Dead** | Same — no consumer. |
| `USE_EMBEDDING_CACHE` | **Dead** | Same — no consumer. (Semantic cache is separately NOT wired into the call path; see `llm-orchestration-reference`.) |
| `USE_RESPONSE_HEALING` | **Helper-only (in flight)** | `_maybe_add_response_healing(request_params, opts)` in `orchestrator/infrastructure/llm_client.py:493` appends `{"id": "response-healing"}` to `extra_body.plugins` for non-streaming requests carrying `response_format`. Unit-tested (`tests/unit/test_response_healing.py`), exported in `__all__`, **but no production call site exists** — `UnifiedClient._dispatch` does not call it. Wiring it end-to-end is the active work of branch `feat/response-healing`. Until then, setting this env var changes nothing at runtime. |

Extra trap: `engine_core/engine_deps.py:19` does `from .config import OPENROUTER_OPTS` — relative to `engine_core`, i.e. `orchestrator.engine_core.config`, which **does not exist** (verified) — so that module's `OPENROUTER_OPTS` is always `None` via the `except ImportError` fallback. Even future consumers in `engine_deps.py` would see `None` until that import is fixed to `..config`.

### 4.2 Named operational flags

| Flag | Default | Read site | Status | Prod vs experimental |
|------|---------|-----------|--------|----------------------|
| `ORCH_HITL_AUTOAPPROVE` | unset (fail-closed) | `orchestrator/hitl/gate.py:67` | **Wired** | Dev/test ONLY. `=true` swaps in `AutoApproveChannel` when a task `requires_approval` and no real `DecisionChannel` is configured. Never set in production — this is the legacy escape hatch created after the silent auto-approval incident (fail-closed doctrine: `llm-orchestration-reference`; incident: `orchestrator-failure-archaeology`). Without it and without a channel, approval requests **raise** (`hitl/channel.py:39`). |
| `ENABLE_PR_COMMENTS` | `true` | `orchestrator/git_service.py:101` AND duplicate `orchestrator/vcs/service.py:101` | **Wired** | Production. Gates PR review comments in git integration. |
| `ENABLE_AUTO_COMMIT` | `false` | `orchestrator/git_service.py:102` AND `orchestrator/vcs/service.py:102` | **Wired** | Production, deliberately off. Leave `false` unless a human explicitly opts in. |
| `ORCH_CACHE_HOME` | unset → `~/.orchestrator_cache` | `orchestrator/infrastructure/path_provider.py:37` (`CachePathProvider`) | **Wired** | Production + tests (tests use it to isolate cache dirs). Overrides the root for state.db, cache.db, telemetry.db, hitl/, checkpoints/, etc. Also surfaced as `settings.cache_home` (§3). Note: `ORCHESTRATOR_CACHE_DIR` is a *different*, mostly-informational var checked only by `operations/diagnostics.py:172` — don't confuse them. |
| `ORCHESTRATOR_ADMIN_SECRET` | unset (feature disabled) | `orchestrator/api_server.py:687` | **Wired, fail-closed** | Production. Required (Bearer header, compared with `hmac.compare_digest`) to register API keys; when unset the endpoint returns 503 "Admin key registration disabled". |
| `ORCHESTRATOR_WS_TOKEN` | unset (fail-closed) | `orchestrator/commands/server.py:190` | **Wired, fail-closed** | Production. WebSocket server closes connections with code 4001 "Server token not configured" if unset; clients authenticate via first message/query param, hmac-compared. |

### 4.3 Other env vars that exist (not exhaustively traced — verify before relying)

`os.getenv`/`os.environ` grep across `orchestrator/` (2026-07-08) also surfaces: `XAI_API_KEY`/`GROK_API_KEY` (wired — §5), `JWT_SECRET`, `JWT_EXPIRY_MINUTES`, `ORCHESTRATOR_HOST`, `PORT`, `HOST`, `ALLOWED_ORIGINS`, `LOG_LEVEL`, `LOG_FORMAT`, `DASHBOARD_URL`, `REQUIRE_HUMAN_APPROVAL`, `ORCH_UNATTENDED_GUARD`, `ORCH_NO_CHECKPOINT_ACK`, `ORCH_WORKTREE_ISOLATION`/`ORCH_WORKTREE_MAX_LIVE`/`ORCH_WORKTREE_GIT_TIMEOUT`, `ORCHESTRATOR_PROFILING`, `ORCHESTRATOR_STRICT_INTEGRATIONS`, `ORCHESTRATOR_INTEGRATIONS_SILENT`, `ORCHESTRATOR_SLACK_WEBHOOK_URL`/`ORCHESTRATOR_SLACK_SIGNING_SECRET`, `GIT_*` / `GITHUB_*` integration vars, `ISSUE_TRACKER_*`, `META_*` (meta-optimization subsystem), `NEXUS_*` / `NEXUSSCOPE_*` (knowledge search), `NETLIFY_TOKEN`, `VERCEL_TOKEN`. Wired-status of these is UNVERIFIED here — trace the specific read site before documenting behavior.

## 5. API keys and `.env` loading

- `orchestrator/cli.py:21` runs `load_dotenv(override=True)` at import. **`override=True` means values in `.env` BEAT already-exported shell environment variables** for any CLI invocation. If a flag "won't turn off", check `.env` first.
- The pydantic settings in §3 *also* read `.env` independently (via `env_file=".env"`), so the file is consulted even on non-CLI entry paths that import `crosscutting.config`.
- There is **no `.env.example` in this repo** (verified `Test-Path .env.example` → False, 2026-07-08) despite CLAUDE.md saying "Copy .env.example to .env" — create `.env` by hand. (Do not add the example file without going through `orchestrator-change-control`.)

### Which keys actually matter in the call path

| Key | Role | Wiring |
|-----|------|--------|
| `OPENROUTER_API_KEY` | **The** key. All providers are reached through OpenRouter. | Validated **eagerly at `UnifiedClient` construction** — `orchestrator/infrastructure/llm_client.py:180-184` raises `AuthenticationError("OpenRouter API key not found...")` if neither the constructor arg nor the env var is set. This eager check landed with the feat/response-healing remediation (commit `ab17b5f4`, 2026-07-07). |
| `XAI_API_KEY` (fallback alias `GROK_API_KEY`) | Optional direct xAI client (`llm_client.py:416`); without it, xAI models fall back to OpenRouter. Also used by `knowledge/xai_search.py`, `rate_limiter.py`. | Wired. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEEY`... | **NOT in the LLM call path.** Appear only in `operations/diagnostics.py` (health checks), `generators/secrets_manager.py`, `slash_integrations.py`, and example files. Don't tell users they're required for runs. | Peripheral. |

### Tests and CI

`tests/conftest.py:17-20`: because of the eager validation above, the suite sets a dummy key for keyless environments:

```python
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
```

A real key in the environment takes precedence (`setdefault`). Tests mock clients and never make real calls. If you write a new test that constructs `UnifiedClient` and it fails with `AuthenticationError` in CI, you bypassed conftest (e.g. a subprocess) — set the dummy key explicitly.

## 6. Dependency-declaration rule (2026-07-07 instructor incident)

Two rules, both born from the same incident on this branch:

1. **Any module-level import in `orchestrator/` MUST be declared in `pyproject.toml` `[project.dependencies]`** (or an extras group whose absence the import site tolerates with try/except). Incident: `instructor` was imported by `infrastructure/llm_client.py` while undeclared → fresh CI env couldn't even **collect** tests (ImportError at collection kills the entire pytest run, not just one test). Fixed: `pyproject.toml:43` now declares `"instructor>=1.0,<2.0"` with a comment naming the importing module. Follow that pattern — comment WHY each dependency exists.
2. **Heavy libraries must be imported lazily** inside the function/method that needs them. `instructor` costs ~30 s to import cold on Windows; `llm_client.py` defers it (`UnifiedClient._instructor_mode`, plus local `import instructor` at client-creation sites, lines 419/443) so `import orchestrator` and CLI startup stay fast for paths that never create a client. Same idiom used for `aiohttp` (`validate_model_available`) and `hmac`/`os` in server handlers.

Also remember: **no new dependencies without explicit approval** (stdlib-first / ponytail doctrine) — that gate lives in `orchestrator-change-control`. Related pin: `grimp>=3.3,<3.4` (3.4+ Rust-panics on Windows with this codebase — pyproject.toml:51).

## 7. How to add a config axis (checklist — route through orchestrator-change-control first)

Adding a **model**:
1. Add the enum member to `Model` in `orchestrator/models.py` FIRST. The enum is the source of truth; JSON follows it, never the reverse.
2. Add `costs.json` entry — key EXACTLY `Model.X.value`. Free variants (`...:free`) must be `{"input": 0.00, "output": 0.00}`.
3. (Optional) add to `routing.json` lists and/or `fallbacks.json` — again exact enum values on BOTH sides of a fallback pair.
4. Run the drift check: `python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_config_drift.py` → must exit 0.
5. Run the locked tests: `pytest tests/unit/test_vfm_routing.py tests/unit/test_cost_reduction.py -q`. If the model leads a text-task route it MUST end `:free`; the last chain entry must NOT.
6. Verify the id exists on the live catalogue: `python scripts/audit_openrouter_models.py` (some ids resolve server-side despite being absent from `/models` — the script documents these "runtime-resolvable ids"; don't delete an id just because the snapshot lacks it).
7. TDD + commit per `orchestrator-change-control`; update docs owned by `orchestrator-docs-and-writing`.

Adding a **task type**: enum member in `TaskType` first, then entries in `routing.json`, `limits.json`, `thresholds.json`; same drift check; extend `_TEXT_TASKS` coverage in `test_vfm_routing.py` if it is a text task.

Adding an **env flag**:
1. Prefer a `FeatureFlags` field (§3) over a raw `os.getenv` — one source of truth, testable, `.env`-aware.
2. Default must be the SAFE value (off for behavior changes, fail-closed for anything security/approval-shaped — see `ORCHESTRATOR_ADMIN_SECRET`/`ORCH_HITL_AUTOAPPROVE` precedents).
3. Wire a consumer in the same PR. A flag with no consumer is a lie in the docs — this repo already has seven of them (§4.1); do not add an eighth.
4. Add a unit test that flips the flag and observes behavior; document the flag here (this SKILL.md) and in whatever doc `orchestrator-docs-and-writing` designates.

## 8. When NOT to use this skill

- **Deciding WHERE code lives / layering rules / DI container** → `orchestrator-architecture-contract`.
- **Whether a change is allowed, gates, TDD/commit process, dependency-approval** → `orchestrator-change-control`.
- **The story of past incidents** (BOM, auto-approve, drift history, dead VS pipeline) → `orchestrator-failure-archaeology`.
- **WHY routing/budget/caching/HITL are designed this way** (theory: `:free` semantics, dual budget, self-consistency vs cache) → `llm-orchestration-reference`.
- **How to prove a change works, markers, coverage, locked/golden tests in general** → `orchestrator-validation-and-qa`.
- **Diagnosing a live failure** → `orchestrator-debugging-playbook` (in flight as of 2026-07-08).
- **Running the orchestrator / dashboard** → skills `orchestrator-run`, `dashboard-start`.

## 9. Provenance and maintenance

All facts verified 2026-07-08 on branch `feat/response-healing` (Windows 11 dev box; CI is ubuntu-latest — path separators in commands below are PowerShell-style; swap `\` for `/` on Linux). Re-verify before trusting anything volatile:

| Claim | Re-verify with |
|-------|----------------|
| JSON↔enum drift state | `python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_config_drift.py` (exit 0 = clean; was exit 1 with 5 orphans on 2026-07-08) |
| Free-tier lead lock + no silent drops | `pytest tests/unit/test_vfm_routing.py -q` |
| Cost-reduction levers locked | `pytest tests/unit/test_cost_reduction.py -q` |
| Model ids live on OpenRouter | `python scripts/audit_openrouter_models.py` |
| USE_* flags still dead / healing helper still uncalled | `rg -n "OPENROUTER_OPTS|_maybe_add_response_healing|_resolve_provider_variant" orchestrator tests` — a new caller in `orchestrator/` (outside `config.py`) means §4.1 is stale |
| Eager key validation + conftest dummy key | `rg -n "OpenRouter API key not found" orchestrator; rg -n "test-key-not-real" tests/conftest.py` |
| `instructor` declared + lazy | `rg -n "instructor" pyproject.toml orchestrator/infrastructure/llm_client.py` |
| Env-var inventory | `rg -n "os\.getenv|os\.environ" orchestrator` |
| `.env.example` still absent | `Test-Path .env.example` |
| config.py constant values | `Read orchestrator/config.py` (single file, ~210 lines) |
| ORCH_* flag defaults | `Read orchestrator/crosscutting/config.py` |
| Response-healing wiring landed? | `pytest tests/unit/test_response_healing.py -q` plus the rg above; once `_maybe_add_response_healing` is called from `UnifiedClient._dispatch` (or equivalent), promote `USE_RESPONSE_HEALING` from "Helper-only" to "Wired" here |
