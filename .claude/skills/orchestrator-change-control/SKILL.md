---
name: orchestrator-change-control
description: Load this skill BEFORE making any change to the Multi-LLM Orchestrator repo — new features, refactors, bug fixes, config edits, dependency additions, or CI/tooling changes. Also load it when a gate blocks you (lint-imports contract failure, "Root-level file violations" from check_new_root_files.py, --cov-fail-under failure, pre-commit hook rejection, bandit HIGH finding, mypy core-layer error) and you are tempted to weaken, exempt, or route around it. Covers change classification (trivial/standard/architectural/security), the Four Unbreakable Rules with their incident histories, the 5 import-linter contracts, the root-file freeze, the coverage ratchet, TDD requirements, HITL/human-review triggers, and the escalation path when a gate legitimately blocks you.
---

# Orchestrator Change Control

How changes are classified, gated, and reviewed in this repository. Every gate here exists because its absence already cost real money, real time, or a silent production failure. The rule of this skill in one sentence: **when a gate blocks you, fix the change — never the gate.**

Jargon, defined once:
- **Gate** — an automated check that must pass before code merges (pre-commit hook, CI job, import contract).
- **Contract** — an import-linter rule in `.importlinter` forbidding certain cross-layer imports.
- **Ratchet** — a numeric floor that may only move in one direction (coverage: up).
- **HITL** — Human-In-The-Loop: the fail-closed approval gate in `orchestrator/hitl/`.
- **Fail-closed** — when no explicit approval channel exists, the system BLOCKS rather than silently approving.

## When NOT to use this skill

| You want... | Use instead |
|---|---|
| Layer map, which module belongs where, port/adapter design | `orchestrator-architecture-contract` |
| The full incident chronicle with root-cause narratives | `orchestrator-failure-archaeology` |
| Flag/env-var/config-file catalog | `orchestrator-config-and-flags` |
| How to run tests, coverage, quality tooling day-to-day | `orchestrator-validation-and-qa` |
| Install/env problems (grimp on Windows, black versions) | `orchestrator-build-and-env` |
| Debugging a runtime failure (not a gate failure) | `orchestrator-debugging-playbook` |

---

## 1. Change classification

Classify BEFORE writing code. When in doubt, classify up, never down.

| Class | Examples | Gates that apply | Human review / HITL |
|---|---|---|---|
| **Trivial** | Typo, comment, docstring, doc-only edit | pre-commit (black/ruff/bandit/lint-imports) + CI | No — but still no direct commit to `master`; work on a branch |
| **Standard** | Bug fix, new function in an existing subpackage, new test | All of the above + **TDD (failing test first)** + full test suite green | Code review recommended; mandatory if it touches evaluator scoring, budget math, or routing |
| **Architectural** | New module, moved module, new layer dependency, anything touching `engine.py`, `models.py`, `.importlinter`, `scripts/check_new_root_files.py`, `engine_core/container.py` | All of the above + read `docs/CODEBASE_MINDMAP.md` first + plan written before code + import contracts must stay KEPT | **Mandatory human review.** Never self-approve edits to the gates themselves |
| **Security** | Auth, secrets handling, subprocess/shell, SQL, SSRF-adjacent HTTP, HITL gate, generated-output scanner (`orchestrator/safety/`) | All of the above + bandit HIGH must be clean (no new `#nosec` without justification) | **Mandatory human review + security-focused review.** HITL gate semantics may never be loosened |

Additional hard triggers for mandatory human review regardless of class:
- Any change to a CI workflow (`.github/workflows/ci.yml`), `.pre-commit-config.yaml`, `pyproject.toml` tool config, `.importlinter`, or `scripts/check_new_root_files.py`.
- Any new third-party dependency (see Unwritten Rule 3, §7).
- Any change that adds an `ignore_imports` entry, an allowlist entry, an `xfail`, or a `--ignore` in test config.
- Any edit under `orchestrator/hitl/` — this gate is fail-closed by design (commit `11deb573`); weakening it recreates the silent-auto-approval incident.

## 2. The Four Unbreakable Rules (CLAUDE.md) — with the incidents behind them

### Rule 1: `engine.py` = Mediator only
New logic goes in a new service module under an existing subpackage; the engine only wires services together.

**Incident.** ARCH-AUDIT-V2 (commit `431fc89c`, 2026-06-24) scored the codebase 5/10 with 2 CRITICAL risks; `engine.py` had grown to ~1,867 lines of accumulated logic. The demolition that followed (`4ac9b4f1`, `b7263052`, `9fb2b826`) removed dead methods and, in the process, uncovered real bugs hiding in the bloat — including a stale `self._container` reference (should have been `self._c`) and a broken planner reference in `set_optimization_backend`. Logic dumped into the mediator doesn't just bloat it; it rots unseen.

*(Earlier snapshot, for context: `docs/ARCHITECTURAL_AUDIT_V5.md` records `engine.py` at
5,036 lines / 104 methods at an even earlier audit than ARCH-AUDIT-V2's ~1,867 — both figures
are true, just at different dates; see `orchestrator-architecture-contract` §1 for the full
size timeline. Current live count as of 2026-07-11: 1,284 lines — `wc -l orchestrator/engine.py`.)*

**What violating it looks like:** your PR adds an `async def _do_something_new` with business logic to `engine.py`. Reviewer must reject; extract to `orchestrator/application/` or `orchestrator/engine_core/`.

### Rule 2: `models.py` = pure data
Only dataclasses and enums. No I/O, no asyncio, no behavior, no imports from outer layers.

**Incident class.** `models.py` defines the `Model` enum whose `.value` strings are the join key for every config JSON (`orchestrator/config/costs.json`, `routing.json`, `fallbacks.json`). Repeated drift incidents (2026-06-23 fix of 9 entries; resync again in `ab17b5f4`, 2026-07-01) happened because config builders do **not** resolve aliases — a JSON key that doesn't exactly equal an enum value **silently drops** the entry: no error, models just vanish from routing/pricing. Purity of `models.py` is what makes it a stable source of truth; the domain-purity import contract (below) enforces the layering half of this rule mechanically.

### Rule 3: TDD without exceptions
First a failing test (RED), then the minimal implementation (GREEN), then commit. No implementation-first "I'll add tests after".

**Incident.** Commit `e863f0c8` (2026-06-25): `EvaluatorService._aggregate` silently discarded self-consistency evaluation runs 3..N, returning `scores[0]` — the orchestrator was paying for extra evaluation LLM calls and throwing the results away. Found only when a proactive invariant test suite was written. A RED test for "3 runs → median of 3" would have caught this the day it was written. Same story for commit `416b9e18`: `BatchClient` used `if request.result:` truthiness, so a *valid falsy* result caused a 300-second hang — the fix gates on `status == COMPLETED`, and the regression test is what keeps it fixed.

### Rule 4: No new root-level `orchestrator/*.py` modules
All new code goes in existing subpackages (`domain/`, `application/`, `engine_core/`, `infrastructure/`, `commands/`, `generators/`, `hitl/`, `safety/`, `quality/`, ...). Enforced mechanically by the root-file freeze (§4).

**Incident.** The "root dump" — **257 modules** at `orchestrator/` depth 1 at the time of the audit (`ARCH-AUDIT-V2.md` line 18/37, not "30+" as an earlier revision of this doc claimed) — was a named CRITICAL finding of ARCH-AUDIT-V2 (`431fc89c`). Live count as of 2026-07-11: **256 files** (`ls orchestrator/*.py | wc -l`) — the root-file freeze has held it roughly flat, not shrunk it meaningfully. Root-level modules import each other freely, which is how the engine↔container circular imports formed — **10 skipped tests** in `tests/test_phase6_10_comprehensive.py` (6 with reason "Importing Orchestrator from engine.py has circular imports" + 4 with reason "Relies on container.py imports", verified 2026-07-11; an earlier revision of this doc undercounted this as "6"). The freeze stops the pile from growing while remediation shrinks it.

**Circular-import caveat:** as of 2026-07-11, direct instantiation
(`OPENROUTER_API_KEY=sk-test-dummy python -c "from orchestrator.engine import Orchestrator; Orchestrator()"`)
succeeds with no `ImportError` — the 10 skips may be misdiagnosing an eager
`AuthenticationError` rather than a live circular import. See
`orchestrator-hardest-problems-campaign` Track A Phase 0.5 for the full reproduction before
treating this as an active blocker.

## 3. The 5 import-linter contracts (`.importlinter`)

`lint-imports` runs in pre-commit AND as a **blocking** CI job ("Architecture Boundaries"). On violation, CI prints the contract name with `BROKEN` and the offending import chain, e.g.:

```
Application layer must not import concrete infrastructure adapters BROKEN

orchestrator.application.task_executor is not allowed to import orchestrator.infrastructure.llm_client:
-   orchestrator.application.task_executor -> orchestrator.infrastructure.llm_client (l.12)

Contracts: 4 kept, 1 broken.
```

Exit code is non-zero; the job fails; merge is blocked. The five contracts, verbatim names from `.importlinter`:

| # | Contract id | Rule (quoted intent) | Typical violation |
|---|---|---|---|
| 1 | `domain-purity` | "Domain layer must not import application or infrastructure" — `orchestrator.domain`, `orchestrator.models`, `orchestrator.exceptions` may not import `orchestrator.infrastructure`, `orchestrator.application`, `orchestrator.engine_core`, or `orchestrator.engine` | Adding `from orchestrator.infrastructure.llm_client import ...` inside a domain module to "just call the API quickly" |
| 2 | `application-no-concrete-infra` | "Application layer must not import concrete infrastructure adapters" — `orchestrator.application` may not import `orchestrator.infrastructure` | A use-case importing a concrete adapter instead of receiving a port via constructor injection |
| 3 | `application-services-no-engine` | "Application services must not import from engine.py directly" — `orchestrator.application` may not import `orchestrator.engine`. Two documented `ignore_imports` exist (`cli_helpers -> meta_integration`, `project_runner -> meta_integration`) for TYPE_CHECKING-only transitive paths | An application service reaching back up to `Orchestrator` — this is how circular imports start |
| 4 | `engine-core-no-loose-infra` | "engine_core pipeline modules must not import infrastructure directly" — `pipeline`, `pipeline_executor`, `pipeline_runner`, `project_planner`, `state_coordinator`, `stages` may not import `orchestrator.infrastructure`. **`engine_core/container.py` is deliberately excluded — it IS the composition root** | Wiring an adapter inside the pipeline instead of in `container.py` |
| 5 | `root-modules-no-infra` | "Root modules must not import infrastructure directly (shims excepted)" — root-level `orchestrator/*.py` may not import `orchestrator.infrastructure`. Its `ignore_imports` list is currently **empty**; keep it that way | A root shim importing an adapter instead of delegating to a subpackage |

Contract 3's comment in `.importlinter` states the design intent explicitly: it uses `source_modules = orchestrator.application` for the whole subpackage "rather than an allow-list that silently exempts new modules added later." That is the philosophy of every gate here.

**If a contract blocks you:** the layering is telling you the code is in the wrong layer. Move the code or introduce a port. Adding an `ignore_imports` entry is an **architectural change requiring human approval** — see §8. All 5 contracts are KEPT as of 2026-07-07; re-verify: `lint-imports` from repo root.

## 4. Root-file freeze guard (`scripts/check_new_root_files.py`)

Mechanism (read the script; it is ~165 lines):
- A `KERNEL_ALLOWLIST` set of 40 filenames is hardcoded in the script — the root files deliberately kept (e.g. `models.py`, `engine.py`, `cli.py`, `config.py`, `validators.py`).
- **CI mode** (`--baseline origin/master`, used by the Architecture Boundaries job after `git fetch --no-tags origin master:refs/remotes/origin/master`): runs `git diff --name-only --diff-filter=A <baseline> -- orchestrator/*.py`, keeps only true root-level paths (`p.count("/") == 1`), and fails (exit 1) if any **newly added** root file is not on the allowlist. Only `--diff-filter=A` (added) is checked — editing existing root files does not trip it, and subpackage files are explicitly allowed.
- **Audit mode** (no `--baseline`): checks ALL current root files against the allowlist.
- `--allowlist` prints the current allowlist and the current root-file count.

Failure output looks like:

```
❌ Root-level file violations (vs. origin/master):
   orchestrator/my_new_module.py

1 violation(s) found.
New root-level files must be added to KERNEL_ALLOWLIST in
scripts/check_new_root_files.py, or placed in a subpackage.
```

**Default response: put the file in a subpackage.** That resolves ~100% of real cases.

**Legitimately adding a root file (rare, requires approval):** if a human maintainer approves a genuine kernel-level module, the change is TWO edits in ONE reviewed commit: (1) add the filename to `KERNEL_ALLOWLIST` in `scripts/check_new_root_files.py` with a trailing comment stating why it must be root-level, (2) add the module itself. It must also satisfy contract 5 (no direct `orchestrator.infrastructure` imports). Expanding this allowlist unilaterally is Unwritten Rule 1 territory (§7) — it is a gate weakening.

Local check before pushing (PowerShell, repo root):

```powershell
python scripts/check_new_root_files.py --baseline origin/master
python scripts/check_new_root_files.py --allowlist   # inspect the list
```

## 5. Coverage ratchet

Two floors exist; know which is which (values verified 2026-07-07):

| Floor | Where | Value | Enforced by |
|---|---|---|---|
| `fail_under = 7` | `pyproject.toml` `[tool.coverage.report]` | 7% | Any local `pytest --cov` run using pyproject's coverage config |
| `--cov-fail-under=6` | `.github/workflows/ci.yml` Test job | 6% | CI (explicit flag overrides pyproject) |

The pyproject comment documents the mechanism: "ratchet floor — Phase A baseline was 6. Raised to 7 after Phase B tests ... Raise in ~5% steps." The CI file says it outright: **"Never lower this number — only raise it."**

Rules:
- **Direction: only up.** Lowering either number to make a failing build pass is a gate weakening (§7) — full stop.
- The pyproject value may run ahead of CI (as now: 7 vs 6). When raising, raise pyproject first, let it soak, then raise the CI flag to match.
- If your change drops coverage below the floor, write tests for what you touched. The floor is intentionally low right now precisely so there is never an excuse to lower it.
- CI test scope: `pytest -m "not slow and not requires_api and not stress and not e2e"` with `--ignore=tests/test_service_observability.py`, plus a separate contract-test step (`pytest tests/contracts/ -v --tb=short --no-cov` with all API keys blanked).

## 6. Pre-commit hooks and CI gates — the full gauntlet

Pre-commit (`.pre-commit-config.yaml`, all `language: system`, all must pass before `git commit` proceeds):

| Hook | Command | Note |
|---|---|---|
| black | `black --check --line-length=100 orchestrator/ tests/` | Check-only; run `black orchestrator/ tests/` to fix. CI pins `black==26.1.0` — match it locally, floating versions caused local-pass/CI-fail drift |
| ruff | `ruff check orchestrator/ tests/` | `ruff check --fix` for auto-fixables |
| bandit | `bandit -r orchestrator/ --severity-level high -x tests,docs` | HIGH severity blocks. 9 real vulns were fixed in `611c1403` (2026-06-23: hmac.compare_digest admin guard, SSRF guards, SQL identifier allowlists, WS bind 127.0.0.1, CSP unsafe-eval removal) — bandit HIGH is now clean and stays clean |
| import-linter | `lint-imports` | The 5 contracts of §3 |

CI (`.github/workflows/ci.yml`), blocking unless noted:

| Job | Blocking? | What it runs |
|---|---|---|
| Lint | Informational in practice (formatting/ruff) | `black --check` (pinned 26.1.0) + `ruff check` |
| Architecture Boundaries | **BLOCKING — "MUST pass — architecture violations block merge"** (quoted from the workflow) | `lint-imports` + `python scripts/check_new_root_files.py --baseline origin/master` |
| Type Check | Blocking for core layers | `mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py`; full-codebase mypy is `continue-on-error: true` (informational) |
| Test | **BLOCKING** | pytest with `--cov-fail-under=6` + contract tests |
| OpenRouter Model Audit | Non-blocking on network failure only (exit 2 tolerated); dead model ids DO fail | `scripts/audit_openrouter_models.py` |
| Security Scan | **BLOCKING** (`continue-on-error` was removed) | `bandit -r orchestrator/ --severity-level high --confidence-level medium` |

Setup: `pre-commit install` after `pip install -e ".[dev]"`. Windows note (dev is Windows 11, CI is ubuntu-latest): `grimp` is pinned `>=3.3,<3.4` in `pyproject.toml` because 3.4+ has a Rust panic on Windows — do not "upgrade" it to fix an unrelated problem. Symptom if your env drifted off the pin: `lint-imports` dies with `pyo3_runtime.PanicException: range end index ... out of range` (observed live on this machine 2026-07-07); fix with `pip show grimp` then `pip install "grimp>=3.3,<3.4" --force-reinstall`, and rely on the ubuntu CI job as the authoritative contract check meanwhile. Also remember commit `d913d136`: BOM characters in 32 `.py` files produced cryptic import errors **on Linux only** — files that work locally on Windows can still break CI; save files as UTF-8 without BOM.

## 7. The three unwritten rules (treat as written now)

### Unwritten Rule 1: Never weaken a gate to pass it
Concretely banned without explicit human approval:
- Lowering `fail_under` or `--cov-fail-under`.
- Adding `ignore_imports` to any `.importlinter` contract, or exempting modules from `source_modules`.
- Expanding `KERNEL_ALLOWLIST` in `scripts/check_new_root_files.py`.
- Converting a failing test to `xfail`/`skip` to go green. (`tests/test_preexisting_problems.py` is an `xfail(strict=True)` *ledger of catalogued, acknowledged bugs* — it is documentation of debt, not a dumping ground for inconvenient failures. Adding to it requires the same review as any catalogued-bug entry.)
- Adding `#nosec`, `# noqa`, `# type: ignore` to silence a finding you haven't understood.
- Adding files to pytest `--ignore` lists.

**Rationale:** every gate in this repo was installed AFTER the failure it prevents had already happened (§2, §3 incidents). Weakening one re-arms the incident. A weakened gate is worse than no gate — it still radiates "this is checked" confidence while checking nothing.

### Unwritten Rule 2: Config JSON keys ≡ Model enum values, and run drift tests after touching either
Keys in `orchestrator/config/costs.json`, `routing.json`, `fallbacks.json` MUST exactly equal `Model.X.value` strings from `orchestrator/models.py`. The config builders do not resolve aliases; a mismatched key **silently drops** the entry — no exception, no log, the model just disappears from pricing/routing/fallbacks.

**Rationale:** this exact silent drop has bitten repeatedly (9 entries fixed 2026-06-23; resync again in `ab17b5f4` 2026-07-01). After touching `models.py` enum values OR any config JSON:

```powershell
pytest tests/ -k "drift or vfm_routing or config" -q --no-cov   # then run the full suite
```

(Test-name selection is a heuristic — confirm the drift/routing suites it collects are green, and see `orchestrator-config-and-flags` for the authoritative config-testing runbook.)

### Unwritten Rule 3: No new dependencies without explicit approval
Stdlib first; existing dependency second; new dependency only with human sign-off (ponytail doctrine — the `ponytail` skill exists in this repo for a reason).

**Rationale:** every dependency is a supply-chain surface (this repo runs bandit + `safety check` for a reason), a Windows/Linux portability risk (see the grimp pin — one dependency version panics on Windows), and a permanent maintenance tax. If you believe a dependency is justified: state what stdlib/existing-dep approach you rejected and why, then ask.

## 8. When a gate blocks you legitimately — escalate, never route around

Sometimes the gate is genuinely wrong for a legitimate change (a true new kernel module; a genuinely needed `ignore_imports`; a coverage floor colliding with a large pure-deletion PR). Procedure:

1. **Stop.** Do not merge a "temporary" weakening. There are no temporary weakenings; there are only weakenings.
2. **Re-check your classification (§1).** A blocked change is usually a mis-layered change. 90%+ of contract failures mean the code belongs in a different layer; 90%+ of root-freeze failures mean the file belongs in a subpackage.
3. **Write the case:** what you're changing, which gate blocks it, why the gate's *intent* (read its comment — every gate in this repo carries one) is not violated by your change, and the minimal gate modification needed.
4. **Escalate to a human maintainer** (repo owner: github.com/georgehadji/multi-llm-orchestrator). For AI agents: surface the case to the user and wait. Do NOT self-approve. The HITL design principle applies to you too: **fail closed** — no explicit approval means no.
5. **If approved:** the gate change ships in its own commit (or the same reviewed commit as its motivating change, per §4), with a comment in the gate file recording why and when, so the next reader inherits the rationale.
6. **Never** use `git commit --no-verify`, `ORCH_HITL_AUTOAPPROVE=true` outside dev/test, or force-push to `master` as an alternative to steps 1–5.

## 9. Pre-merge checklist (all classes)

- [ ] Change classified (§1); if architectural/security, human review arranged
- [ ] Failing test written FIRST and observed RED (Rule 3)
- [ ] No logic added to `engine.py`; no behavior added to `models.py` (Rules 1–2)
- [ ] New files in subpackages, not `orchestrator/` root (Rule 4)
- [ ] `lint-imports` → `Contracts: 5 kept, 0 broken`
- [ ] `python scripts/check_new_root_files.py --baseline origin/master` → ✅
- [ ] `black --check --line-length=100 orchestrator/ tests/` and `ruff check orchestrator/ tests/` clean
- [ ] `bandit -r orchestrator/ --severity-level high -x tests,docs` clean
- [ ] `pytest -m "not slow and not requires_api and not stress and not e2e" --cov=orchestrator -q` passes with coverage ≥ floor
- [ ] If `models.py` or `orchestrator/config/*.json` touched: drift tests run (Unwritten Rule 2)
- [ ] No new dependencies, or approval recorded (Unwritten Rule 3)
- [ ] No gate weakened anywhere in the diff (Unwritten Rule 1) — grep your own diff for `xfail`, `nosec`, `noqa`, `ignore_imports`, `fail_under`, `KERNEL_ALLOWLIST`

## Provenance and maintenance

Facts originally verified 2026-07-07 (branch `feat/response-healing`); the root-file count,
skip count, and engine.py line count were re-verified live on **2026-07-11** and corrected
(root files: 256 not "30+"; skips: 10 not "6"; engine.py: 1,284 lines). Re-verify before
relying on volatile values — numbers in this repo drift fast:

| Fact | Re-verify with |
|---|---|
| 5 import contracts + ignore_imports entries | `Get-Content .importlinter` then `lint-imports` |
| Kernel allowlist contents (40 files as of 2026-07-07) | `python scripts/check_new_root_files.py --allowlist` |
| Coverage floors (pyproject 7 / CI 6) | `Select-String fail_under pyproject.toml` and `Select-String cov-fail-under .github/workflows/ci.yml` |
| Pre-commit hook set | `Get-Content .pre-commit-config.yaml` |
| CI job list / blocking status | `Get-Content .github/workflows/ci.yml` |
| black pin (26.1.0) and grimp pin (>=3.3,<3.4) | `Select-String "black==" .github/workflows/ci.yml; Select-String grimp pyproject.toml` |
| HITL fail-closed + ORCH_HITL_AUTOAPPROVE escape hatch | `Get-Content orchestrator/hitl/gate.py` (module docstring) |
| Incident commits (11deb573, d913d136, 416b9e18, e863f0c8, 611c1403, 431fc89c, 4ac9b4f1, ab17b5f4) | `git log -1 --format='%h %ad %s' --date=short <hash>` |
| Four Unbreakable Rules wording | `Get-Content CLAUDE.md` (repo root) |
| Root-level `orchestrator/*.py` file count (256 as of 2026-07-11) | `ls orchestrator/*.py \| wc -l` (bash) or `(Get-ChildItem orchestrator/*.py).Count` (PowerShell) |
| Circular-import skip count (10 as of 2026-07-11) | `grep -c "pytest.mark.skip" tests/test_phase6_10_comprehensive.py` |
| engine.py line count (1,284 as of 2026-07-11) | `wc -l orchestrator/engine.py` |
