---
name: orchestrator-debugging-playbook
description: Symptom-to-triage playbook for live failures in the Multi-LLM Orchestrator, ranked by cost — silent failures/auto-approve first, then model-id/config drift, then refactor fallout, then environment/encoding traps. Load this when something is actually broken in front of you and you need the fastest discriminating experiment, not theory. Symptom keywords — "task silently produced nothing", "run hangs", "model never called", "$0 cost for a paid model", "ImportError after refactor", "circular import engine container", "works on Windows fails on Linux CI", "CLI slow at startup", "import instructor takes forever", "integration test times out despite mocked decompose", "evaluation score looks wrong / always the same", "AuthenticationError in tests", "grimp Rust panic", "black --check fails only in CI", "ModuleNotFoundError instructor". For WHY things are designed this way use llm-orchestration-reference; for the full incident chronicle use orchestrator-failure-archaeology; for gates/process use orchestrator-change-control; for config/flag reference use orchestrator-config-and-flags; for layering rules use orchestrator-architecture-contract; for how to prove a fix use orchestrator-validation-and-qa.
---

# Orchestrator Debugging Playbook

Runbooks only. For root-cause narratives and "why does this keep happening" see
`orchestrator-failure-archaeology`. For "why is the system built this way" see
`llm-orchestration-reference`. This skill exists to get you from symptom to fix fast.

All facts verified against branch `feat/response-healing` on 2026-07-08 (Windows 11 dev
box; CI is `ubuntu-latest`). Commands below are PowerShell-style (`\` path separators);
swap `/` on Linux/CI where noted.

## How to use this table

Failure classes are ordered by **cost of not catching them**, per the project's own
ranking: (1) silent failures — you don't even know something is wrong; (2) config
drift — routing silently degrades; (3) refactor fallout — the codebase doesn't run;
(4) environment/encoding traps — works for you, breaks for everyone else. Start at the
top only if you have no better lead; otherwise jump straight to the matching symptom.

| # | Symptom | Jump to |
|---|---------|---------|
| 1 | Task/run produced nothing, or hung | §1 |
| 2 | Model never called / cost is exactly $0 for a paid model | §2 |
| 3 | ImportError / circular import after a refactor | §3 |
| 4 | Works on Windows, fails on Linux CI | §4 |
| 5 | CLI or test suite slow / times out at startup | §5 |
| 6 | Integration test times out despite decompose being mocked | §6 |
| 7 | Evaluation scores look wrong / suspiciously identical | §7 |
| 8 | `AuthenticationError` in tests that shouldn't call any API | §8 |

---

## 1. "Task silently produced nothing" / "run hangs"

**Why this is #1:** an unattended run that neither completes nor errors is the worst
outcome — nobody is paged, nothing shows red. Three unrelated causes produce the same
surface symptom. Do NOT guess; run the discriminating experiment first.

### Symptoms as an engineer sees them
- CLI process still running well past expected completion, no new log lines.
- Process exits 0 but the expected output artifact / PR / file never appeared.
- A `requires_approval=True` step never got a decision and the run is stuck.

### Discriminating experiment
```powershell
# 1. Is it actually hung, or just slow? Check for HITL approval file/log first —
#    this rules in/out cause A in one grep.
rg -n "requires_approval|DecisionChannel|ORCH_HITL_AUTOAPPROVE" orchestrator/hitl/gate.py

# 2. Check what channel got selected for THIS run (fail-closed vs auto-approve vs real):
$env:ORCH_HITL_AUTOAPPROVE   # if unset/empty and no real DecisionChannel wired,
                              # hitl/channel.py:39 RAISES on first approval request —
                              # that's a crash, not a silent hang. If it "hangs" instead
                              # of raising, you are blocked on a real DecisionChannel
                              # (e.g. web UI) waiting for a human — check that UI/queue.

# 3. Rule out the P5 batch-truthiness class (cause B): is a BatchClient in the path?
rg -n "BatchStatus.COMPLETED|request\.result" orchestrator/cost_optimization/batch_client.py
# Fixed in 416b9e18 -- gate is now `if request.status == BatchStatus.COMPLETED`.
# If you're on an OLDER checkout or a fork that reverted this, `if request.result:`
# treats a legitimately falsy result (0, "", {}, []) as "not ready" and spins for the
# full POLL timeout (300s in the old code / max_wait=600s in _poll_batch_results).

# 4. Rule out stale cache serving a phantom "success" for cause C:
rg -n "cache_ttl_hours|DiskCache" orchestrator/crosscutting/config.py orchestrator/infrastructure
python -c "from orchestrator.infrastructure.path_provider import CachePathProvider; print(CachePathProvider().root)"
# Default cache root: ~/.orchestrator_cache  (override via ORCH_CACHE_HOME)
```

### The traps and their stories

| Cause | Trap | Story |
|-------|------|-------|
| A. HITL fail-closed vs auto-approve | Before `11deb573`, an approval gate **silently auto-approved** every request — the run "succeeded" having never asked a human. The fix flips default behavior to **fail-closed**: no channel configured → `FailClosedChannel` raises immediately (`orchestrator/hitl/channel.py:39`) instead of hanging or auto-approving. `ORCH_HITL_AUTOAPPROVE=true` is a **dev/test-only** escape hatch (`orchestrator/hitl/gate.py:67`) — never set it in production; if you see it set and a run auto-completed without a human decision, that's expected-but-dangerous behavior, not a bug. |
| B. P5 falsy-result truthiness | `orchestrator/cost_optimization/batch_client.py` used to gate "is this request done" on `if request.result:` — Python truthiness, not a status check. A batch call that legitimately returned `""`, `0`, `{}`, or `[]` was treated as "still pending" and the caller spun until the poll timeout. Fixed in `416b9e18`: gate is now `if request.status == BatchStatus.COMPLETED`. If you see this pattern reintroduced anywhere else in the codebase (`rg -n "if .*\.result:" orchestrator/`), it is the same bug class — fix by checking an explicit status enum, not a value's truthiness. |
| C. Stale cache | The response DiskCache (48h TTL, `OrchestratorSettings.cache_ttl_hours` in `orchestrator/crosscutting/config.py`) can serve a cached "success" from a prior run that doesn't match current code/config — the run finishes fast with output that looks stale or wrong rather than actually hanging. Distinguish from A/B by timing: A and B block for a long time; C returns suspiciously *fast*. Purge with `Remove-Item -Recurse -Force "$HOME\.orchestrator_cache\cache.db"` (or the `ORCH_CACHE_HOME`-relative path) and rerun. |

**Fix priority:** confirm which cause with the experiment above BEFORE touching code —
A is a config/env issue (do not "fix" fail-closed behavior, it is correct by design;
see `llm-orchestration-reference` for the fail-closed doctrine), B is a code bug pattern
to grep for elsewhere, C is an operational cache-bust, not a bug.

---

## 2. "Model never called" / "$0 cost for a paid model"

**Why this is #2:** routing silently degrading to the wrong (or no) model is invisible
until someone notices a suspiciously cheap invoice or a suspiciously bad output.

### Symptoms as an engineer sees them
- A model you expect to be used never shows up in telemetry/cost logs.
- Total run cost is implausibly low or exactly $0.
- A fallback you configured never fires when the lead model fails.

### Discriminating experiment (exact, copy-pasteable)
```powershell
# One-liner: is the model id you care about actually surviving into the built table?
python -c "from orchestrator.models import Model, ROUTING_TABLE, TaskType; print([m.value for m in ROUTING_TABLE[TaskType.CODE_GEN]])"

# Full drift audit (checks costs.json, routing.json, fallbacks.json against the enum):
python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_config_drift.py
# Exit 0 = clean. Exit 1 = at least one JSON key/value is not byte-for-byte equal to a
# Model or TaskType enum .value and was SILENTLY DROPPED by the loader's membership
# guard (models.py::_build_cost_table etc: `if k in Model._value2member_map_`).

# Confirm the locked VFM routing contract still holds after any routing.json edit:
pytest tests/unit/test_vfm_routing.py -q
```
If `check_config_drift.py` reports the id as dropped, the fix is: make the JSON key (or
fallback value) **exactly** equal `Model.X.value` — never guess, always paste the enum
value. Full mechanism and drift-trap ownership: `orchestrator-config-and-flags` §1
("THE DRIFT TRAP"). Do not re-derive that mechanism here — this skill only tells you how
to detect and confirm the fix.

### The trap and its story
Config JSON keys must be byte-for-byte equal to `Model`/`TaskType` enum `.value`s. The
builders in `orchestrator/models.py` filter with a membership guard and swallow anything
that doesn't match — **no error, no log line**. This has bitten the project repeatedly:
fixed 2026-06-23, resynced in `ab17b5f4`.

**2026-07-07 incident (verify with `git log --oneline -- orchestrator/config/routing.json`):**
`routing.json` on `feat/response-healing` had collapsed several task types' lead model
to `google/gemini-2.5-flash` — an id that does not match any `Model` enum value (the
valid enum member is `google/gemini-2.5-flash-image`, a different model). A second orphan
`google/gemini-2.5-flash` key was found even on `master`. Because of the silent-drop
guard, the entries didn't error — they just vanished from `ROUTING_TABLE`, and requests
fell through toward `gpt-4o-mini`-shaped defaults instead of a free-tier lead. Fix:
re-key the entry to `google/gemini-3.5-flash` (a real, priced enum value) and promote
the actual `:free` variant (`meta-llama/llama-3.3-70b-instruct:free`) to lead position,
per the VFM free-tier-first doctrine. **As of 2026-07-08 this fix is present in the
working tree but UNCOMMITTED** (`git status --short orchestrator/config/routing.json`
shows `MM`) — verify it's still there and gets committed before relying on it:
```powershell
git diff orchestrator/config/routing.json | Select-String "gemini-2.5-flash`",gemini-3.5-flash"
```
Locked regression coverage: `tests/unit/test_vfm_routing.py` — `test_lead_candidate_is_free_tier`
asserts `ROUTING_TABLE[task][0].value` ends with `:free` for every text task; run it
after touching `routing.json` or the `Model`/`TaskType` enums, every time, no exceptions.

---

## 3. "ImportError / circular import after a refactor"

### Symptoms
- `ImportError: cannot import name 'Orchestrator' from partially initialized module 'orchestrator.engine'`
- A test that constructed `Orchestrator` directly (not through the container) started
  failing after touching `engine.py` or `engine_core/container.py`.
- `lint-imports` fails locally or in CI with a contract violation.

### Discriminating experiment
```powershell
# 1. Is this the KNOWN engine<->container cycle, or a NEW violation?
#    engine.py imports container lazily (inside a method), by design:
rg -n "from .engine_core.container import ServiceContainer" orchestrator/engine.py
#    -> orchestrator/engine.py:432, INSIDE a method body, not at module top. If your
#    new code imports container at module top level in engine.py, that's what broke.

# 2. Run the actual gate CI uses (not your own reasoning about layering):
lint-imports
#    5 contracts checked (.importlinter, root_package=orchestrator): domain-purity,
#    application-no-concrete-infra, application-services-no-engine,
#    engine-core-no-loose-infra, root-modules-no-infra.

# 3. Did you add a new root-level orchestrator/*.py module? (Rule #4, CLAUDE.md)
python scripts/check_new_root_files.py --baseline origin/master
```

### The trap and its story
`engine.py` and `orchestrator/engine_core/container.py` have a known circular
dependency: the container builds services that reference the engine, and the engine
needs the container to wire them. The codebase manages this by importing
`ServiceContainer` **lazily inside a method** in `engine.py` (line 432) rather than at
module top level — moving that import to the top of the file reintroduces the cycle.
This is a documented weak point (see `orchestrator-architecture-contract`), not
something to "finally fix" without going through `orchestrator-change-control` — it is
architectural surgery, not a bug fix.

**Known casualty:** `tests/test_phase6_10_comprehensive.py` carries 10 `pytest.mark.skip`
markers (verified via `grep -c "pytest.mark.skip" tests/test_phase6_10_comprehensive.py`
= 10, 2026-07-08) with reasons `"Relies on container.py imports which have circular deps"`
and `"Importing Orchestrator from engine.py has circular imports"`. If you can make one
of these pass by restructuring imports, that's real progress — but if it starts passing
*without* code changes (XPASS), see `orchestrator-validation-and-qa` for what that means
for the xfail/skip ledger discipline before deleting the skip marker.

**Windows-local trap:** `grimp` (the graph library `import-linter` uses) is pinned to
`>=3.3,<3.4` in `pyproject.toml` because **3.4+ Rust-panics on Windows with this
codebase** (comment at `pyproject.toml:50-51`). If `lint-imports` crashes locally with a
Rust panic/traceback instead of a clean pass/fail report, check your installed grimp
version before assuming the contracts themselves are broken — **CI (ubuntu-latest) is
authoritative**, not your local Windows run:
```powershell
pip show grimp   # must report 3.3.x
```

---

## 4. "Works on Windows, fails on Linux CI"

### Symptoms
- `SyntaxError: invalid non-printable character U+FEFF` or similarly cryptic import
  failure, only in CI, never locally.
- `black --check` fails in CI on a file that `black` accepted locally.
- A file you edited on Windows suddenly won't import on Linux.

### Discriminating experiment
```powershell
# BOM scan (cheap, run after any bulk Windows file operation):
python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_bom.py
# Exit 0 = clean, 1 = BOM(s) found (lists offending files), 2 = scan dir missing.

# Black version drift check — compare what CI uses vs what you have locally:
black --version
```
```yaml
# .github/workflows/ci.yml:20-22 (verified 2026-07-08):
# "Pin black so CI and local pre-commit format identically (a floating
#  black version reformats files that pass locally and breaks the gate)."
pip install "black==26.1.0" ruff
```
`requirements-dev.txt:16` pins `black==24.10.0` — **a different version than CI's
26.1.0**. `pyproject.toml`'s `[project.optional-dependencies] dev` group pins a *range*,
`black>=23.7,<25.0`, which is also below CI's pin. This is a live, verified drift as of
2026-07-08 — if `black --check` passes locally and fails in CI (or vice versa), this
version gap is the first thing to check, not your code. Install the CI-pinned version
before trusting a local `black --check`:
```powershell
pip install "black==26.1.0"
black --check orchestrator/ tests/
```

### The traps and their stories
| Cause | Trap | Story |
|-------|------|-------|
| BOM bytes | Commit `d913d136` removed UTF-8 BOM (`EF BB BF`) from 32 `.py` files. BOMs are invisible in most Windows editors/tools but Python on Linux raises a cryptic `SyntaxError` at the very first line. Some Windows tooling (PowerShell `Out-File`, certain editors) re-introduces them silently on save. Run `check_bom.py` after any bulk Windows-side file operation (mass find/replace, PowerShell-generated files), not just before commit. |
| black version drift | CI pins `black==26.1.0` explicitly in the workflow (with a comment explaining why: a floating/mismatched version reformats files differently and breaks the `--check` gate). Local `requirements-dev.txt` (`24.10.0`) and `pyproject.toml`'s dev extra (`<25.0`) are both **behind** that pin as of 2026-07-08. Do not "fix" this by loosening the CI pin — align local tooling to CI's pinned version instead (`orchestrator-change-control` governs gate changes). |
| `ModuleNotFoundError: instructor` (2026-07-07 incident) | WIP code on `feat/response-healing` added a module-level `import instructor` in `orchestrator/infrastructure/llm_client.py` without declaring `instructor` in `pyproject.toml`. Locally the package happened to already be installed (leftover from another env), so nothing failed. On a fresh CI runner, the undeclared import died **at pytest collection time** — which kills the *entire* test run, not just tests touching that module, because conftest's import chain pulls in `llm_client.py` transitively. Fix (now in place, verify): `pyproject.toml:43` declares `"instructor>=1.0,<2.0"` with a comment naming the importing module. **Lesson:** any module-level import in `orchestrator/` must be declared in `[project.dependencies]` (or tolerated via try/except for an optional extra) — verify with `rg -n "instructor" pyproject.toml orchestrator/infrastructure/llm_client.py`. |

---

## 5. "CLI / test suite slow or times out at startup"

### Symptoms
- `python -m orchestrator --help` (or any CLI invocation) takes many seconds before
  doing anything.
- A smoke test that just imports the package and asserts it doesn't crash times out.

### Discriminating experiment
```powershell
python -X importtime -c "import orchestrator" 2>&1 | Sort-Object { [double]($_ -split '\|')[1] } -Descending | Select-Object -First 15
```
Read the output for any single import costing an outsized chunk of total time (hundreds
of ms to multiple seconds). `instructor` was ~30s cold on Windows before the fix below —
that will dominate the whole `-X importtime` trace if it regresses.

### The trap and its story (2026-07-07 incident)
A **top-level** `import instructor` in `orchestrator/infrastructure/llm_client.py` added
~30s to cold Python startup on Windows (the package pulls in a large dependency tree).
This blew past smoke-test timeout budgets even though the test never actually needed
structured-output functionality. Fix: `instructor` is imported **lazily**, only inside
the function/method that actually needs it (`orchestrator/infrastructure/llm_client.py`:
`UnifiedClient._instructor_mode`, plus local `import instructor` at client-creation call
sites — verified at lines 153, 419, 443). Same idiom already used for `aiohttp`
(`validate_model_available`) and other heavy/optional imports.

**Lesson — applies to all future heavy dependencies:** if a library costs more than a
trivial amount to import and is only needed on some code paths, import it **inside the
function that uses it**, not at module top level. Verify a specific import is still lazy:
```powershell
rg -n "^import instructor|^from instructor" orchestrator/infrastructure/llm_client.py
# Should return NOTHING at column 0 (module top level) — only indented `import instructor`
# lines inside function bodies (lines 153, 419, 443 as of 2026-07-08).
```

---

## 6. "Integration test times out despite decompose being mocked"

### Symptoms
- A test that explicitly mocks `orch.decompose(...)` (or equivalent) still takes
  20-40+ seconds and blows a `<15s` fail-fast bound, or times out entirely.
- The mock you added is clearly never hit before the slowdown starts.

### Discriminating experiment
```powershell
# Check what tests/integration/conftest.py actually mocks for its orchestrator fixture:
rg -n "AsyncMock|_generate_architecture_rules|OPENROUTER_API_KEY" tests/integration/conftest.py
```
If the fixture sets a dummy `OPENROUTER_API_KEY` (`tests/integration/conftest.py:125`,
`monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")`) but does **not** also
mock every LLM-calling phase your test's orchestrator instance will actually traverse,
you have found the same bug class as the 2026-07-08 incident below.

### The trap and its story (2026-07-08 incident)
`tests/integration/conftest.py`'s shared `orchestrator_fixture` added a dummy
`OPENROUTER_API_KEY` purely to satisfy `UnifiedClient`'s **eager** key-validation at
construction (see §8) — that unblocked *instantiation*. But a dummy key is not a mock:
it lets `UnifiedClient` **construct successfully and then attempt a real network call**
the moment any phase invokes it. In this incident, `_generate_architecture_rules` ran
*before* decomposition in the pipeline and made a real (retried) network call using the
fake key, burning 30-40s in connection/retry backoff before the test's mocked-decompose
assertion was ever reached — well past a `<15s` fail-fast bound. Fix
(`tests/integration/conftest.py:143-144`):
```python
# the architecture-rules generation phase deliberately can override
# `orch._generate_architecture_rules`.
orch._generate_architecture_rules = AsyncMock(return_value=None)
```

**Lesson — generalizes to any fixture:** a dummy API key satisfies *construction-time*
eager validation. It does **not** satisfy a real network call made later by some other
phase your orchestrator instance will traverse. When writing or extending a shared test
fixture, audit **every** LLM-calling phase the fixture's orchestrator object can reach
during the test — not just the one phase under test — and mock each one explicitly.
`rg -n "async def _generate\|async def _decompose\|async def _critique\|async def _evaluate" orchestrator/engine.py` to enumerate candidate phases before trusting a fixture.

---

## 7. "Evaluation scores look wrong" / "always the same value"

### Symptoms
- Self-consistency evaluation with 3+ runs returns a score that looks like it ignored
  everything past the first run.
- Two evaluation runs of the same input on different days return byte-identical scores
  when you'd expect at least minor variance.

### Discriminating experiment
```powershell
pytest tests/unit/test_bug_scan.py -q -k aggregate
git show e863f0c8 --stat
```

### The traps and their stories
| Cause | Trap | Story |
|-------|------|-------|
| `_aggregate` discarding runs | `EvaluatorService._aggregate` (in `orchestrator/services/evaluator.py`) used to just `return scores[0]` for 3+ self-consistency runs — every run past the first was computed, then thrown away. Fixed in `e863f0c8` (`test(bug-scan): proactive invariant suite + fix _aggregate dropping runs`) with explicit 0/1/2/3+ run handling and a median for 3+. If a score looks like it never reflects more than one sample, check which code path you're actually running against — verify the fix is present: `rg -n "def _aggregate" orchestrator/services/evaluator.py` and read the branch logic. |
| Response cache defeating self-consistency | The response DiskCache (48h TTL) caches by request content. If self-consistency is implemented as "call the same prompt N times", a cache hit returns the **same** cached response for all N calls — you get zero variance, which looks identical to a broken aggregation but is actually a caching interaction. Distinguish by checking cache hit/miss telemetry, or by clearing `~/.orchestrator_cache/cache.db` and rerunning. Full theory (why this happens, why it's an accepted tradeoff) lives in `llm-orchestration-reference`; this entry only tells you how to tell the two causes apart. |

---

## 8. "`AuthenticationError` in tests that shouldn't call any API"

### Symptoms
- A test that never intends to hit a real LLM fails with
  `AuthenticationError: OpenRouter API key not found...`.
- Only happens for a specific test file or when running a test in isolation
  (`pytest tests/some_file.py::test_x`) but not the full suite.

### Discriminating experiment
```powershell
rg -n "OpenRouter API key not found" orchestrator/infrastructure/llm_client.py
rg -n "test-key-not-real" tests/conftest.py tests/integration/conftest.py
```
`orchestrator/infrastructure/llm_client.py:180-184` validates `OPENROUTER_API_KEY`
**eagerly at `UnifiedClient` construction** — not at call time. `tests/conftest.py:20`
sets a dummy key via `os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")`
(setdefault = a real key already present wins). If your failing test:
- runs via a **subprocess** (e.g. `subprocess.run([...])`) — the subprocess gets a fresh
  environment that never went through `conftest.py`'s `setdefault`. Set the env var
  explicitly for the subprocess call.
- imports `orchestrator.infrastructure.llm_client` at **collection time** in a way that
  constructs a client before any fixture runs — check import order.
- is in `tests/integration/` — that directory has its **own** conftest doing the same
  thing via `monkeypatch.setenv` (`tests/integration/conftest.py:125`), scoped to
  whatever fixture requests it; a test that doesn't request the fixture won't get the key.

Full ownership of this pattern (eager validation rationale, wiring): `orchestrator-config-and-flags` §5.

---

## Reproducing a failed task in isolation

1. Find the project/run id from the failure (CLI output, or list recent state):
   ```powershell
   python -c "
   import asyncio
   from orchestrator.infrastructure.state import StateManager
   async def main():
       sm = StateManager()
       for p in await sm.list_projects():
           print(p)
   asyncio.run(main())
   "
   ```
   (`StateManager.list_projects()` verified at `orchestrator/infrastructure/state.py:482`.)
2. Load that project's persisted state directly:
   ```powershell
   python -c "
   import asyncio
   from orchestrator.infrastructure.state import StateManager
   async def main():
       sm = StateManager()
       state = await sm.load_project('<project_id>')
       print(state)
   asyncio.run(main())
   "
   ```
   (`load_project` / `load_latest_checkpoint` verified at `orchestrator/infrastructure/state.py:417,427`.)
3. The state DB lives at `~/.orchestrator_cache/state.db` by default
   (`orchestrator/infrastructure/path_provider.py:13,57` — `CachePathProvider().state_db`),
   overridable via `ORCH_CACHE_HOME`. Copy it aside before poking at it if the run is one
   you cannot afford to lose.
4. For the actual re-run mechanics (`--resume`, project YAML specs, where output lands),
   this skill does not own that — see **`orchestrator-run-and-operate`**.

## docs/debugging/DEBUGGING_GUIDE.md cross-reference

CLAUDE.md's "Key References" section cites `docs/debugging/DEBUGGING_GUIDE.md` as the
debugging guide. **Verified 2026-07-08: this path does not exist in the repo**
(`docs/debugging/` is absent — `find docs -iname "*debug*"` returns nothing). Either the
doc was never created or was removed without updating CLAUDE.md. Do not chase this path;
this SKILL.md is the current debugging reference. If you create that doc, update
CLAUDE.md's reference and consider whether it should defer to this skill instead of
duplicating it (route through `orchestrator-change-control` first).

## When NOT to use this skill

- **The full incident chronicle / "has this exact bug been re-litigated before"** →
  `orchestrator-failure-archaeology`.
- **WHY the system is designed this way** (fail-closed doctrine, VFM routing theory,
  cache-vs-self-consistency tradeoff, dual budgets) → `llm-orchestration-reference`.
- **The complete flag/config catalog and wired-vs-dead status** →
  `orchestrator-config-and-flags`.
- **Whether a fix is allowed, gates, TDD/commit process** → `orchestrator-change-control`.
- **Layering rules, DI container, where new code should live** →
  `orchestrator-architecture-contract`.
- **How to prove a fix works — markers, coverage, locked/golden tests** →
  `orchestrator-validation-and-qa`.
- **Running the orchestrator/dashboard, CLI subcommand anatomy, `--resume` mechanics** →
  `orchestrator-run-and-operate`, `orchestrator-run`, `dashboard-start`.
- **Diagnostic scripts themselves (what `check_bom.py` / `check_config_drift.py` check,
  flag inventory tooling)** → `orchestrator-diagnostics-and-tooling` (this skill only
  tells you *when* to run them).

## Provenance and maintenance

All facts verified 2026-07-08 on branch `feat/response-healing`. Re-verify anything
volatile before trusting it:

| Claim | Re-verify with |
|-------|----------------|
| Commit hashes exist and touch the claimed files | `git show --stat --oneline <hash>` |
| P5 batch_client fix still present | `git show 416b9e18 -- orchestrator/cost_optimization/batch_client.py`; `rg -n "BatchStatus.COMPLETED" orchestrator/cost_optimization/batch_client.py` |
| HITL fail-closed default still in place | `rg -n "ORCH_HITL_AUTOAPPROVE" orchestrator/hitl/gate.py orchestrator/hitl/channel.py` |
| Config drift state | `python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_config_drift.py` |
| routing.json 2026-07-07 fix committed (currently uncommitted as of 2026-07-08) | `git status --short orchestrator/config/routing.json`; `git diff orchestrator/config/routing.json` |
| VFM routing lock | `pytest tests/unit/test_vfm_routing.py -q` |
| engine↔container lazy import still lazy | `rg -n "from .engine_core.container import ServiceContainer" orchestrator/engine.py` (must be inside a method, not at column 0) |
| grimp pin | `rg -n "grimp" pyproject.toml`; `pip show grimp` (must be 3.3.x) |
| skip count in test_phase6_10_comprehensive.py | `grep -c "pytest.mark.skip" tests/test_phase6_10_comprehensive.py` (was 10 on 2026-07-08 — the brief that seeded this skill said 6; trust the live count) |
| BOM scanner clean | `python .claude\skills\orchestrator-diagnostics-and-tooling\scripts\check_bom.py` |
| black version drift (CI vs local) | `rg -n "black==" .github/workflows/ci.yml requirements-dev.txt pyproject.toml` |
| instructor declared + lazy | `rg -n "instructor" pyproject.toml orchestrator/infrastructure/llm_client.py` |
| CLI import time | `python -X importtime -c "import orchestrator"` |
| integration conftest mocks architecture-rules phase | `rg -n "_generate_architecture_rules" tests/integration/conftest.py` |
| `_aggregate` fix present | `rg -n "def _aggregate" orchestrator/services/evaluator.py` |
| eager key validation + conftest dummy key | `rg -n "OpenRouter API key not found" orchestrator/infrastructure/llm_client.py`; `rg -n "test-key-not-real" tests/conftest.py tests/integration/conftest.py` |
| state DB default path | `python -c "from orchestrator.infrastructure.path_provider import CachePathProvider; print(CachePathProvider().state_db)"` |
| docs/debugging/DEBUGGING_GUIDE.md still absent | `Test-Path docs/debugging/DEBUGGING_GUIDE.md` |
