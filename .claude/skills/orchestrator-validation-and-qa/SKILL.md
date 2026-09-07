---
name: orchestrator-validation-and-qa
description: The evidence-and-testing doctrine for the Multi-LLM Orchestrator. Load this BEFORE claiming any change "works", before writing or placing a new test, when deciding which pytest markers/directories to use, when an xfail test unexpectedly turns green (XPASS failure), when tempted to add a file to the pytest --ignore list, when the coverage gate fails (--cov-fail-under), when you need to know what a "contract test" or "golden/locked inventory" test protects, or when asked to judge the quality of generated websites/apps. Symptom keywords - "how do I prove this works", "which marker do I use", "where does this test go", "XPASS(strict)", "test_preexisting_problems", "coverage below fail_under", "tests/contracts failing", "test_vfm_routing failed after config edit", "lint-imports passes locally but what about CI", "grimp panic on Windows", "is the website good enough".
---

# Orchestrator Validation & QA — What Counts as Evidence

Doctrine: **"it looks right" is never evidence.** Every claim of correctness must be
backed by a green gate output you actually ran (or CI ran). This skill defines the
evidence bar, the test taxonomy as actually practiced in this repo, the xfail ledger
discipline, contract tests, the coverage ratchet, and what to do when quality is
subjective (generated websites/apps).

All facts verified against the repo on 2026-07-07 (branch `feat/response-healing`).
Re-verification one-liners are in "Provenance and maintenance" at the bottom.

---

## 1. The Evidence Bar

A change is proven only when ALL of the following are green. Partial evidence is not
evidence.

| # | Gate | Exact command | Blocking? |
|---|------|---------------|-----------|
| 1 | Tests on the CI marker expression | see below | Yes (CI `test` job) |
| 2 | Contract tests | `pytest tests/contracts/ -v --tb=short --no-cov` | Yes (CI `test` job, run with all API-key env vars set to empty) |
| 3 | Import-boundary contracts | `lint-imports` | Yes (CI `architecture` job) |
| 4 | Root-module freeze | `python scripts/check_new_root_files.py --baseline origin/master` | Yes (CI `architecture` job) |
| 5 | mypy on gated layers | `mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py --ignore-missing-imports --no-strict-optional --python-version=3.12` | Yes (CI `typecheck` job; full-codebase mypy is informational only) |
| 6 | bandit HIGH | `bandit -r orchestrator/ --severity-level high --confidence-level medium -f txt` | Yes (CI `security` job) |
| 7 | black + ruff | `black --check orchestrator/ tests/` and `ruff check orchestrator/ tests/` | Yes (CI `lint` job; black pinned to `26.1.0` in CI so a floating local version can disagree — trust CI's pin) |
| 8 | Model-id audit | `python scripts/audit_openrouter_models.py` | Blocking on dead ids; exit code 2 (network unreachable) is tolerated as a warning |

The exact CI pytest command (`.github/workflows/ci.yml`, `test` job — quote, do not
paraphrase):

```bash
pytest -m "not slow and not requires_api and not stress and not e2e" \
  --tb=short -q \
  --ignore=tests/test_service_observability.py \
  --cov=orchestrator \
  --cov-report=xml \
  --cov-report=term-missing \
  --cov-fail-under=6
```

Windows notes (repo dev machine is Windows 11; CI is ubuntu-latest):

- `lint-imports` depends on `grimp`, pinned `>=3.3,<3.4` in `pyproject.toml` because
  grimp 3.4+ has a Rust panic on Windows with this codebase. If `lint-imports`
  panics locally, **CI is authoritative** — do not treat a local grimp crash as a
  pass OR a fail; push and read the CI `architecture` job.
- The pytest command above works unchanged in PowerShell if you drop the `\`
  line-continuations (use one line or PowerShell backticks).

What is NOT evidence: a diff that "reads correctly", a single passing test file run
in isolation, a screenshot, an LLM (including you) saying the code is fine, a local
black/ruff pass with a version different from CI's pin.

Gate ownership: the RULES about gates (never weaken a gate to pass, escalation path
when a gate legitimately blocks you) live in **orchestrator-change-control**. This
skill tells you how to PRODUCE evidence; that skill tells you what you may not do to
the gates.

---

## 2. Test Taxonomy As Actually Practiced

### 2.1 Markers (registered in `pyproject.toml [tool.pytest.ini_options] markers`)

`--strict-markers` is on: an unregistered marker is an error.

| Marker | Meaning | In CI default run? |
|--------|---------|--------------------|
| `unit` | Single function/method, no I/O | Yes |
| `integration` | Multi-module wiring | Yes |
| `contract` | Architecture contract tests | Yes (plus dedicated `tests/contracts/` CI step) |
| `mock` | Uses mocked dependencies | Yes |
| `edge_case` | Boundary tests | Yes |
| `asyncio` | Explicit asyncio marking (mostly redundant, see 2.4) | Yes |
| `slow` | > 1 second | **Excluded** (`-m "not slow ..."`) |
| `requires_api` | Needs real API keys | **Excluded** |
| `e2e` | End-to-end | **Excluded** |
| `stress` | Stress tests | **Excluded** |
| `load`, `benchmark` | Load/benchmark tests | Not in the exclusion expression, but conventionally also marked `slow` |

If your test costs money, needs a network, or takes > 1 s, it MUST carry the
matching exclusion marker — otherwise it runs on every CI push.

### 2.2 Directory layout

| Directory | Contents (verified) |
|-----------|--------------------|
| `tests/unit/` | 66 files — the primary home for new unit tests |
| `tests/contracts/` | Port + architecture contract tests (section 4) |
| `tests/integration/` | Cross-service wiring: golden paths, resume-after-crash, full run, pipeline runner, skill manager (has its own `conftest.py`) |
| `tests/regression/` | Pinned-behavior regressions (currently `test_vs_regression.py`) |
| `tests/smoke/` | Fast sanity: `test_api_contracts.py`, `test_cli.py` |
| `tests/` root | ~50 legacy modules predating the layout (e.g. `test_models.py`, `test_phase6_10_comprehensive.py`). **Do not add new tests here** — new tests go in a subdirectory. |

### 2.3 The pyproject `--ignore` list

`[tool.pytest.ini_options] addopts` carries **24** `--ignore=tests/...` entries
(`test_api.py`, `test_assembler_debug.py`, `test_cli_debug.py`, `verify_frontend.py`,
`verify_syntax.py`, …). CI adds a 25th: `--ignore=tests/test_service_observability.py`.

**What it is:** a quarantine for legacy debug/one-off scripts that are not real test
modules — they `sys.exit(1)` at module level, manipulate `__builtins__`, or have
import/fixture errors. They were written as manual verification scripts, not pytest
suites.

**The rule: NEVER add a file to this list to hide a failing test.** The list is a
museum of pre-existing junk, not an escape hatch. If a real test fails, fix the code
or the test. Adding to the ignore list is a gate-weakening move and falls under
**orchestrator-change-control** (requires explicit approval). Shrinking the list
(rehabilitating or deleting an entry) is always welcome.

### 2.4 `asyncio_mode = "auto"` implications

Set in `pyproject.toml` with `asyncio_default_fixture_loop_scope = "function"`.

- Any `async def test_*` is collected and run automatically — **no
  `@pytest.mark.asyncio` decorator needed** (you'll see the marker in older files;
  it's harmless but redundant).
- Async fixtures also work without decoration.
- **Do NOT override the `event_loop` fixture** — pytest-asyncio >= 0.23 removed
  support for it (there is an explicit warning comment in `tests/conftest.py`).
  Loop scope is per-function via the pyproject setting.

---

## 3. The xfail(strict) Ledger — `tests/unit/test_preexisting_problems.py`

**Location gotcha: the file lives in `tests/unit/`, not the `tests/` root.**

### 3.1 The discipline

When an autonomous bug hunt or audit finds a real bug that is NOT being fixed right
now, it does not go into a TODO comment — it goes into the ledger:

1. Write a test that asserts the **CORRECT** behavior (not the buggy behavior).
2. Mark it `@pytest.mark.xfail(strict=True, reason="P<n>: <one-line bug summary>")`.
3. Add a numbered header comment (P1, P2, …) with: file:line of the bug, the wrong
   behavior, the correct behavior, and the discovery date.

Why `strict=True` matters: while the bug exists the test fails → reported as
`xfail` → suite stays green. The moment anyone fixes the bug (even accidentally),
the test passes → strict xfail reports **XPASS as a FAILURE**. The fixer is forced
to open the ledger, remove the marker, and update the entry. Bugs are therefore both
documented and impossible to silently "fix and forget". This is the repo's answer to
the classic failure mode where a fix lands with no record that the bug ever existed.

### 3.2 How to add an entry

```python
# ──────────────────────────────────────────────────────────────────────────────
# Problem P6: <one-line title>.
#   File: orchestrator/<module>.py:<line>
#   <What the code does wrong, what correct behavior is, blast radius.>
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="P6: <summary> — see header comment")
def test_<correct_behavior_description>():
    ...assert the CORRECT behavior...
```

Run it: it must show `xfail`, not `fail` or `pass`. If it errors at collection
(ImportError etc.) it is not a valid ledger entry — fix the import first.

Also record the bug in **orchestrator-failure-archaeology** if it caused an incident.

### 3.3 How to retire an entry — P4/P5 worked example (commit `416b9e18`)

P4 (HierarchyManager id collision: ids were `f"{type}_{len(self.nodes)}"`, so a
delete let a new node silently overwrite an existing one) and P5 (BatchClient poll
loop tested `if request.result:` truthiness, so a valid-but-falsy result — `""`,
`{}`, `0` — hung 300 s) were fixed in `416b9e18` ("fix(p4,p5): eliminate both
pre-existing catalogued problems"). The retirement pattern from that commit:

1. Fix the bug in the source (P4: monotonic `_next_id()` counter; P5: gate on
   `request.status == BatchStatus.COMPLETED`).
2. **Remove the xfail marker** — the test now runs as a permanent regression guard.
3. Rewrite the header comment: `Problem P4 [FIXED]: ...` and append
   "This is now a regression guard (no longer xfail)." — keep the bug description
   so the history is readable in place.
4. Commit message states the before/after xfail count
   ("603 passed, 0 failed, 0 xfailed (was 601 passed + 2 xfail)").

### 3.4 Current state (verified by running the file, 2026-07-07)

`pytest tests/unit/test_preexisting_problems.py -q --no-cov` → **5 passed, 0 xfailed**.
All five catalogued problems (P1 sync-handler counting, P2 cron weekday convention,
P3 `*/0` step crash, P4, P5) are fixed; every ledger test is now a regression guard.
Note: the module docstring and the P1–P3 header comments still read as if P1–P3 were
open — stale-doc drift; the code fixes are in (e.g. `(t.tm_wday + 1) % 7` in
`orchestrator/operations/automations.py`). Trust the test outcome over the comments.
The DISCIPLINE remains active for the next catalogued bug.

Related invariance suite from the same effort: `tests/unit/test_bug_scan.py`
(commit `e863f0c8`) — cross-cutting invariants over pure logic (parse_score ∈ [0,1],
`_aggregate` never discards runs, VerificationGate/CompletionJudge fail closed,
CronParser never raises). That commit also fixed
`orchestrator/services/evaluator.py::EvaluatorService._aggregate` (3+ consistency
runs now aggregate via median instead of returning `scores[0]`). Careful: there are
TWO evaluators — `orchestrator/services/evaluator.py` (has the median fix) and
`orchestrator/application/evaluator.py` (2-run mean-or-lower only). Verify which one
your call path uses before citing aggregation behavior.

---

## 4. Contract Tests — `tests/contracts/`

Run in CI as a dedicated step with all API-key env vars set to `""` (proves nothing
in the contracts touches a real provider):

```bash
pytest tests/contracts/ -v --tb=short --no-cov
```

| File | What it locks |
|------|---------------|
| `test_architecture_invariants.py` | Executable architecture guards beyond import-linter: embeds the root-kernel `KERNEL_ALLOWLIST` (synced with `scripts/check_new_root_files.py`) and higher-level invariants that module-to-module contracts can't express. If this fails, read **orchestrator-architecture-contract** before touching anything. |
| `test_llm_client_port.py` | The `LLMClient` protocol: `call()` signature (model, prompt, system, max_tokens, temperature, timeout), returns a response, tolerates minimal args. Written as a reusable base class (`LLMClientContract`) — subclass + override the `client` fixture to certify a new adapter. |
| `test_cache_port.py` | The `CachePort` protocol: get-miss returns `None`, put-then-get round-trips. Same base-class pattern (`CachePortContract`). Contains two legitimate non-strict xfails for `NullCache` (which discards `put()` by design) — these are intentional, not ledger entries. |
| `test_state_port.py` | The `StatePort` protocol: `save_project` / `load_project` round-trip preserving `project_id`. Base-class pattern (`StatePortContract`). |

**When you write a new adapter for a port, you MUST subclass the matching contract
base class** and make it pass — that is the definition of "implements the port" here.
The five import-linter contracts (`domain-purity`, `application-no-concrete-infra`,
`application-services-no-engine`, `engine-core-no-loose-infra`,
`root-modules-no-infra` in `.importlinter`) are the sibling mechanism at the import
graph level; **orchestrator-architecture-contract** owns their rationale.

---

## 5. Coverage Ratchet

Two numbers exist, deliberately (verified 2026-07-07):

| Where | Value | Applies to |
|-------|-------|-----------|
| `pyproject.toml [tool.coverage.report] fail_under` | **7** | Local full-suite runs (pytest addopts include `--cov`) |
| CI `--cov-fail-under=6` | **6** | The CI test job |

The pyproject value is the ratchet ("Phase A baseline was 6. Raised to 7 after
Phase B tests… Raise in ~5% steps" — comment in pyproject.toml). CI trails by one
point of slack; the CI file itself says "Never lower this number — only raise it."

Rules:

1. **Only-up.** Neither number may ever decrease. Lowering either to make a red
   build green is gate-weakening → **orchestrator-change-control**.
2. **How to raise safely:** land the tests first, run the full CI marker expression
   locally, note the achieved total percentage, then set `fail_under` comfortably
   BELOW it (2–3 points of headroom) — coverage totals fluctuate slightly with
   collection changes, and a floor set at the exact current value flakes.
3. Raise pyproject first; bump the CI value in the same or a following commit,
   keeping CI ≤ pyproject.
4. Single-file test runs will "fail" coverage locally (one file can't cover 7% of
   the codebase) — use `--no-cov` for targeted runs:
   `pytest tests/unit/test_foo.py -q --no-cov`.

Known doc drift: `.claude/skills/quality-check/SKILL.md` says "fail_under = 12%" —
stale as of 2026-07-07; trust `pyproject.toml`.

---

## 6. How to Add a Test — Checklist

1. **RED first.** CLAUDE.md Rule 3: TDD without exceptions. Write the failing test,
   run it, confirm it fails **with the expected error** (not an ImportError).
2. **GREEN.** Minimal implementation. Re-run the test.
3. **No regressions.** Run the CI marker expression (section 1) before committing.
4. **Commit** with a detailed message; RED and GREEN may be one commit, but the test
   must exist in the same commit as (or before) the implementation.

Placement and marking:

- [ ] Directory: `tests/unit/` for single-module tests; `tests/integration/` for
      cross-service wiring; `tests/contracts/` only for port/architecture contracts;
      `tests/regression/` for pinned bug regressions; never the `tests/` root.
- [ ] Marker: at least one of `unit` / `integration` / `contract`; add
      `slow` / `requires_api` / `e2e` / `stress` if it must not run in default CI.
      `--strict-markers` will reject typos.
- [ ] Async: just write `async def test_...` — no decorator needed (section 2.4).
- [ ] Money/network: anything hitting a real provider MUST be `requires_api`.
- [ ] Targeted run: `pytest tests/unit/test_<name>.py -v --no-cov`.

Fixtures available from `tests/conftest.py` (verified list — use these instead of
rebuilding mocks):

| Fixture | Provides |
|---------|----------|
| `small_budget` / `large_budget` | `Budget(max_usd=5.0, max_time_seconds=300)` / `Budget(max_usd=100.0, max_time_seconds=3600)` |
| `code_task` / `review_task` | A `CODE_GEN` task / a `CODE_REVIEW` task depending on it |
| `sample_tasks` | 3 tasks with a dependency chain (CODE_GEN → CODE_GEN → TEST_GEN) |
| `task_result_ok` | A completed `TaskResult` (score 0.95, `Model.GPT_4O`) |
| `circuit_breaker` | `CircuitBreaker` with low thresholds (failure_threshold=2, reset_timeout=0.05) |
| `temp_dir` / `temp_project_dir` | tmp path / tmp path pre-seeded with `src/main.py`, `tests/`, `README.md` |
| `mock_client` | `MagicMock` LLM client; `call` is an `AsyncMock` returning text `'{"score": 0.85, "issues": []}'`, cost 0.01, 100/50 tokens |
| `mock_telemetry` | `MagicMock` TelemetryCollector |
| `mock_state_manager` | `MagicMock` with async `save_project`/`load_project`/`save_checkpoint`/`close` |
| `default_profiles` | `build_default_profiles()` output |

`tests/integration/` has its own `conftest.py` with additional integration fixtures —
read it before writing an integration test.

---

## 7. Golden / Locked Inventories

These tests pin CONFIGURATION and POLICY, not code logic. They exist because config
drifted silently multiple times (see **orchestrator-failure-archaeology**: model-id /
enum drift class). **If one fails after you edited a config JSON or `models.py`, the
test is almost certainly right and your edit is wrong** — the JSON keys must exactly
equal `Model` enum values; builders do not resolve aliases, mismatches silently drop.

| File (all in `tests/unit/`) | Locks |
|------|-------|
| `test_vfm_routing.py` | VFM (value-for-money) routing doctrine: the named 2026 free/ultra-cheap models are declared in the `Model` enum and priced in `costs.json`, and every text task's cascade in `routing.json` tries a free/cheap capable model FIRST, escalating to premium only on failure. No silent drops, no un-priced spend. |
| `test_cost_reduction.py` | The cost levers are actually USED on the live call path: cache hit → `cost_usd == 0` and no provider dispatch; cache write-through; VerificationGate veto skips LLM spend; CompletionJudge routes to the cheapest non-generator candidate; evaluator gate-veto; always-on cost flags stay default-enabled. |
| `test_openrouter_model_audit.py` | Anti-drift vs the live OpenRouter catalogue: a `KNOWN_DEAD_IDS` frozenset may never resurface as a live reference in the registry/enum/redirect targets (offline, always runs) + a live full audit (skipped without network). Companion script: `scripts/audit_openrouter_models.py` (also a CI job). |
| `test_hitl_gate.py` | The fail-closed HITL (human-in-the-loop) decision gate — **15 test functions** (verified 2026-07-07): no channel → rejected; `ORCH_HITL_AUTOAPPROVE` true/false/absent semantics (absent/false = fail closed); security decisions require explicit opt-in; which channels count as "real". History: silent auto-approval incident, fixed fail-closed in `11deb573` — see **orchestrator-failure-archaeology**. |
| `test_bug_scan.py` | Cross-cutting invariants over core pure logic (section 3.4). |
| `test_preexisting_problems.py` | The bug ledger (section 3). |
| `tests/test_models_no_io_at_import.py` (root) | `models.py` = pure data (Unbreakable Rule 2). Uses SKIP, not xfail, deliberately — xfail bodies still execute and would corrupt `sys.modules` (comment in file explains). |

Rule: when you touch `orchestrator/config/{costs,routing,fallbacks}.json` or
`orchestrator/models.py`, run the drift suite immediately:

```bash
pytest tests/unit/test_vfm_routing.py tests/unit/test_openrouter_model_audit.py -q --no-cov
```

---

## 8. When Quality Is Subjective (Generated Websites / Apps)

Eyeballing generated output is not evidence either. Use the measurable proxies that
already exist, in this order:

1. **Safety scanner as a delivery gate** — `orchestrator/safety/generated_output_scanner.py`,
   wired into `orchestrator/output_organizer.py`. Generated output that trips it
   (hardcoded secrets etc.) does not ship, regardless of how good it looks.
2. **Evaluator scoring** — the generate→critique→revise→evaluate loop scores every
   task against acceptance criteria with 2-pass self-consistency (disagreement →
   conservative lower score). Theory and hazards (score-parsing, cache-defeats-
   self-consistency) live in **llm-orchestration-reference**. An artifact below the
   acceptance threshold iterates; it does not ship on vibes.
3. **Deterministic checks first** — VerificationGate runs deterministic validation
   before any LLM judging; a demonstrably broken artifact never reaches (or bills)
   an eval call (locked by `test_cost_reduction.py`).
4. **Tooling pipeline on the orchestrator's own code** — the `quality-check` skill
   (`.claude/skills/quality-check/SKILL.md`) chains ruff → black → mypy → bandit →
   safety → pre-commit.
5. Generated code is auto-formatted before delivery (ruff/black on Python, prettier
   best-effort on web) via the output formatter wired into the organizer — a
   delivered artifact failing basic lint is a pipeline bug, not a taste issue.

If, after all gates pass, a human still judges the output ugly: that is a prompt/
design-system improvement task (the generated-output-quality track in
**orchestrator-hardest-problems-campaign**), not a reason to bypass gates.

---

## 9. When NOT to Use This Skill

- **Gate is blocking you and you want to change/weaken/exempt it** →
  **orchestrator-change-control** (owns gate rules, change classification,
  escalation path).
- **A test failure smells like a known past bug** (silent approve, BOM, 300s hangs,
  drift…) → **orchestrator-failure-archaeology** (incident chronicle with commits).
- **Deciding WHERE code goes / import-contract failures / engine.py–container
  questions** → **orchestrator-architecture-contract**.
- **WHY routing/eval/budget/caching are designed this way** →
  **llm-orchestration-reference**.
- **Flag and config catalog** (env flags, config file semantics) →
  **orchestrator-config-and-flags**.
- **Environment setup, install, tool versions** → **orchestrator-build-and-env**.
- **Actually running the orchestrator** → **orchestrator-run-and-operate** and the
  `orchestrator-run` skill.

---

## 10. Provenance and Maintenance

All claims verified against the repo on 2026-07-07, branch `feat/response-healing`.
Volatile facts and how to re-check them:

| Fact | Re-verify with |
|------|----------------|
| CI pytest command / marker expression / `--cov-fail-under=6` | `Get-Content .github/workflows/ci.yml` (test job) |
| pyproject `fail_under = 7` | `Select-String -Path pyproject.toml -Pattern "fail_under"` |
| 24 `--ignore` entries in pyproject (+1 CI-only) | `(Select-String -Path pyproject.toml -Pattern '--ignore=').Count` |
| Registered markers / `--strict-markers` / `asyncio_mode = "auto"` | `Select-String -Path pyproject.toml -Pattern "markers|asyncio_mode|strict-markers"` |
| Ledger state (5 passed, 0 xfail) | `pytest tests/unit/test_preexisting_problems.py -q --no-cov` |
| P4/P5 retirement | `git show 416b9e18 --stat` |
| `_aggregate` median fix location (`orchestrator/services/evaluator.py`) | `git show e863f0c8 --stat` |
| 15 HITL gate tests | `Select-String -Path tests/unit/test_hitl_gate.py -Pattern "def test_"` |
| Contract files in `tests/contracts/` | `Get-ChildItem tests/contracts/` |
| 5 import-linter contract names | `Select-String -Path .importlinter -Pattern "^name"` |
| grimp Windows pin | `Select-String -Path pyproject.toml -Pattern "grimp"` |
| conftest fixtures | `Select-String -Path tests/conftest.py -Pattern "^def |@pytest.fixture" -Context 0,1` |
| Golden inventory files exist | `Get-ChildItem tests/unit/test_vfm_routing.py, tests/unit/test_cost_reduction.py, tests/unit/test_openrouter_model_audit.py` |
| Scanner wiring into organizer | `Select-String -Path orchestrator/output_organizer.py -Pattern "generated_output_scanner|scan"` |

Maintenance triggers: update this skill when the coverage floor is raised, when a
new ledger entry is added or retired, when a contract file is added to
`tests/contracts/`, or when the CI marker expression changes. Known stale docs to
fix opportunistically (do not trust them over this skill): the ledger module
docstring/P1–P3 headers (say xfail; actually all fixed), and quality-check's
"fail_under = 12%".
