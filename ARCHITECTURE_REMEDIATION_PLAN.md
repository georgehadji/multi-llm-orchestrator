# Architecture Remediation Plan — 5/10 → 9+/10

**Project:** Multi-LLM Orchestrator
**Baseline audit:** 2026-06-23 (ARCH-AUDIT-V2)
**Current score:** 5/10 — Early Production
**Target score:** ≥ 9/10 — Production / Mature
**Owner:** TBD
**Epistemic note:** Every action below references a [VERIFIED] finding from the baseline audit. Score deltas are estimates against the audit rubric, not guarantees.

---

## 0. How the score moves

Rubric anchors:

| Score | Meaning |
|-------|---------|
| 10 | All layers correctly separated, patterns consistent, observable, testable, scalable |
| 8 | Minor drift in 1–2 modules, no critical violations |
| 6 | Moderate drift, 1–2 high-severity violations, scalability concerns |
| **5 (now)** | Between 4 and 6: 1 CRITICAL (root dump) + 2 HIGH (stateful orchestrator, automations bugs) + partial cycle |

To clear 9, **every** of these must hold simultaneously:

1. No CRITICAL violations. (→ Workstream A)
2. No HIGH violations. (→ Workstreams B, D)
3. Layer separation complete + enforced for 100% of modules. (→ Workstreams A, C, F)
4. Patterns consistent (no divergent duplicates, no fix-named modules, retry policy unified). (→ Workstreams A, D)
5. Observable. (→ Workstream E)
6. Testable — coverage materially above the 6% ratchet floor, contract tests green. (→ Workstream E)
7. Scalable — orchestrator stateless, concurrency proven under load. (→ Workstream B)

A defect in any one caps the score at 8. The plan is therefore **gated**: later workstreams do not "buy back" points lost to an unfinished earlier one.

---

## Score-tracking ledger

| Workstream | Removes | Severity cleared | Score ceiling after |
|-----------|---------|------------------|---------------------|
| A — De-dup & root collapse | Root module dump, divergent duplicates | CRITICAL | 6 → 7 |
| B — Stateless orchestrator | Orchestrator bottleneck | HIGH | 7 |
| C — Break engine_core↔application cycle | Partial cycle, infra leak | MEDIUM | 7 → 8 |
| D — Correctness sweep | automations bugs, shim rot, retry drift | HIGH | 8 |
| E — Observability + test ratchet | Untestable/unobservable gaps | — (enables 9) | 8 → 9 |
| F — Contract lockdown | Re-drift risk | — (enables 10) | 9 → 10 |

Workstreams A–D are sequential where they touch the same files; E and F run continuously alongside.

---

## Workstream A — Collapse the root-level module dump

**Target finding:** [VERIFIED — CRITICAL] 298 root-level `orchestrator/*.py` files (34.6% of codebase); 227 stem-duplicates with subpackage equivalents.

**Verified composition of the 227 duplicates:**

| Class | Count | Meaning | Action |
|-------|-------|---------|--------|
| Identical content | 84 | root copy byte-equal to subpackage copy | Delete root copy, redirect imports |
| Root-is-shim (<30 L re-export) | 40 | root already re-exports subpackage (correct direction) | Keep temporarily; remove in Workstream D |
| **Divergent** | **103** | **two real implementations that have drifted** (e.g. `performance.py` differs by 59 lines, `analyzer.py` by 18) | **Reconcile pair-by-pair — this is the hard, risky work** |

> The 103 divergent pairs are drift bombs: a bug fixed in `analysis/performance.py` is NOT fixed in root `performance.py`, and callers split across both. This is the single largest correctness + maintainability hazard in the codebase.

### A1. Freeze (Day 1, blocking)
- Add CI guard: fail the build if a **new** `orchestrator/*.py` (depth-1) file appears that is not on a documented allowlist. Use the existing `.github/workflows/ci.yml` "Check import boundaries" job + a small script comparing `git diff --name-only origin/master` against `orchestrator/[^/]*.py`.
- **Outcome:** the dump stops growing while it is drained. **Risk:** none. **Effort:** S.

### A2. Inventory & canonicalize (Day 1–2)
- Script the classification above into `scripts/audit_root_modules.py` (the audit logic already exists — promote it to a committed tool). Emit `root_module_inventory.json`: `{stem, root_path, sub_path, class, root_importers, sub_importers}`.
- For each stem, pick the **canonical** location: subpackage wins unless the root copy has strictly more importers AND the subpackage copy is the shim. Record decision in the inventory.
- **Outcome:** deterministic migration map. **Risk:** none (read-only). **Effort:** S.

### A3. Eliminate the 84 identical (Week 1)
- For each: delete root copy, replace with no file (update importers to subpackage path) OR a one-line re-export shim if importer count is high (`models` has 17 importers, `log_config` 12 — shim these to avoid a 17-file churn in one PR).
- Run full test suite + import-linter after each batch of ~10.
- **Outcome:** −84 root files, zero behavior change. **Risk:** LOW (content identical). **Effort:** M. **Gate:** `pytest` green, import-linter green.

### A4. Reconcile the 103 divergent (Week 1–3 — the critical path)
Per pair, mechanical procedure:
1. `git diff --no-index root/X.py sub/X.py` → review every hunk.
2. Determine which side is canonical (higher importer count + most-recent git blame on the diverging lines).
3. Port any unique fixes from the loser into the canonical file (this is where latent bugs surface — a fix present in only one copy).
4. Write/extend a unit test that pins the reconciled behavior **before** deleting the loser (TDD: the test must pass against canonical, fail against loser if they truly diverged).
5. Replace loser with re-export shim (or delete + redirect importers).
6. Suite + import-linter green → commit one pair (or a tightly-related cluster) per PR.

- Prioritize by blast radius: reconcile high-importer divergent pairs first (`engine`, `app_detector`, `codebase_profile`).
- **Outcome:** −103 drift bombs; single source of truth per module; latent fix-gaps closed. **Risk:** HIGH per pair (behavior may differ subtly) — mitigated by the pin-test-first step. **Effort:** L (largest single line item).

### A5. Classify & move the remaining ~71 root-only files (Week 3–4)
- Files at root with **no** subpackage twin (298 − 227 = 71, incl. `engine.py`, `hierarchy.py`, `circuit_breaker.py`, `resilience.py`, `automations.py`, `budget.py`, `log_config.py`).
- Assign each to a target layer (table in Workstream F). Move leaf modules (no root-internal dependents) first.
- Leave `engine.py` for last (Workstream B handles it).
- Keep `log_config.py`, `models.py`, `budget.py` as deliberate root-level "kernel" modules IF documented as such in `.importlinter` and CLAUDE.md — a small, named, frozen root kernel is acceptable; an unstructured 298-file dump is not.
- **Outcome:** root reduced to a documented kernel (<15 files). **Risk:** MEDIUM (import churn). **Effort:** L.

**Workstream A exit criteria:**
- Root-level `orchestrator/*.py` count ≤ 15, all on a documented kernel allowlist.
- Zero divergent duplicate pairs (`scripts/audit_root_modules.py` reports 0 in "divergent" class).
- CI freeze guard active.
- Removes the CRITICAL finding → score ceiling rises to 7.

---

## Workstream B — Make the Orchestrator stateless

**Target finding:** [VERIFIED — HIGH] `Orchestrator.__init__` assigns 40 instance attributes incl. per-run mutable state (budget, telemetry, circuit-breaker state). Single coordinator for all task execution → unsafe under concurrency / heavyweight per-request construction. [VERIFIED] Primary 10x-load failure point.

### B1. Separate run-scope state from service-scope wiring (Week 2–3)
- Introduce `RunContext` dataclass: holds everything mutable per `run_project()` call — `Budget`, telemetry accumulators, circuit-breaker snapshot ref, project state handle, output dir.
- `ServiceContainer` keeps only **stateless, shareable** collaborators (clients, selectors, routers, validators, pipeline definition).
- `run_project(project, criteria, budget, ...)` constructs a fresh `RunContext` internally; stages receive `(ctx_pipeline, run_ctx)` rather than reading orchestrator instance state.
- **Outcome:** one `Orchestrator`/`ServiceContainer` safely serves N concurrent runs. **Risk:** HIGH (touches the core loop) — mitigated by integration tests below. **Effort:** L.

### B2. Pool or singleton at the entrypoint (Week 3)
- `api_server.py` / supervisor: build `ServiceContainer` once at startup; reuse across requests; each request gets its own `RunContext`.
- Confirm budget no longer leaks across runs ([HYPOTHESIS in audit] — convert to [VERIFIED] with a test that runs two projects on one container and asserts independent spend).
- **Outcome:** O(1) per-request cost. **Risk:** MEDIUM. **Effort:** M.

### B3. Concurrency proof (Week 3–4)
- Add `tests/load/test_concurrent_runs.py`: launch 10 concurrent `run_project()` against one container with mocked LLM client; assert no cross-run state bleed (budgets, outputs, telemetry counters independent) and no deadlock.
- **Outcome:** scalability claim is [VERIFIED], not [HYPOTHESIS]. **Effort:** M. **Gate:** load test green.

**Workstream B exit criteria:**
- `Orchestrator`/`ServiceContainer` hold zero per-run mutable state (assert via a reflection test: no mutable instance attrs reassigned during `run_project`).
- 10-concurrent-run test green.
- Removes the HIGH "orchestrator bottleneck" finding.

---

## Workstream C — Break the engine_core ↔ application cycle

**Target findings:** [VERIFIED] transitive cycle `engine_core/container → application.* → engine`; [VERIFIED] `engine_core/stages/critique.py:29` imports `application.verbalized_sampling` directly; [VERIFIED] `application/skill_store.py:20` imports `aiosqlite` directly.

### C1. Port-ify VerbalizedSampler (Week 2)
- Define `VSSamplerPort` in `domain/ports.py`. `engine_core/stages/{generate,critique}.py` depend on the port; `ServiceContainer` injects the `application.verbalized_sampling` implementation.
- **Outcome:** `engine_core/stages` no longer imports `application`. **Risk:** LOW. **Effort:** S.

### C2. Port-ify skill_store persistence (Week 2)
- `SkillStorePort` already exists in `domain/ports.py` (audit-confirmed). Move the `aiosqlite` calls behind a concrete adapter under `infrastructure/`; `application/skill_store.py` depends on the port only.
- **Outcome:** `application-no-concrete-infra` contract genuinely covers this path (currently passes only because `aiosqlite` isn't under `orchestrator.infrastructure`). **Risk:** LOW. **Effort:** M.

### C3. Reclassify the thick "application" drivers (Week 3)
- `chat_cli.py` and `cli_dispatch.py` import `engine.Orchestrator` (import-linter exempts them). They are driving adapters, not use-cases. Move both to a new `entrypoints/` package (Workstream F target). Remove the two `.importlinter` `ignore_imports` exemptions once moved.
- **Outcome:** no application module imports `engine`; the exemption list shrinks to empty. **Risk:** MEDIUM (CLI wiring). **Effort:** M.

**Workstream C exit criteria:**
- `engine_core/__init__.py` no longer re-exports from `application` (the backward-compat shim is deleted in Workstream D).
- `.importlinter` `application-services-no-engine` contract has **zero** `ignore_imports` entries.
- Removes the MEDIUM cycle finding → score ceiling rises to 8.

---

## Workstream D — Correctness & consistency sweep

**Target findings:** [VERIFIED] `automations.py` bugs (sync-handler silent fail, cron weekday off-by-one, `*/0` ZeroDivisionError); [VERIFIED] backward-compat shim rot; [VERIFIED] stage retry inconsistency (`generate.py` retries=2, `critique.py` retries=1, no shared constant).

### D1. Fix automations.py (Week 1 — can run parallel to A)
- Sync handler: apply the same `asyncio.iscoroutine(result)` guard already shipped in `events/triggers.py`. Lines 128 & 152.
- Cron weekday: convert Python `tm_wday` (Mon=0) to cron convention (Sun=0) via `(tm_wday + 1) % 7` before comparison.
- `*/0`: guard `if step == 0: return False` before the modulo.
- Promote the three `xfail(strict=True)` tests in `tests/unit/test_preexisting_problems.py` to passing (strict-xfail flips them to failures the moment they're fixed — remove the markers).
- **Outcome:** scheduler reliable; 3 known bugs closed. **Risk:** LOW. **Effort:** S.

### D2. Audit & retire backward-compat shims (Week 4, after A & C)
- Shims: `api_clients.py`, `engine_core/__init__.py`, the 7 sub-1KB `engine_core/*.py` stubs, plus the 40 root-is-shim files from Workstream A.
- For each: grep callsites → migrate importers to the canonical path → delete shim. Where a shim has many importers, migrate importers first, delete last.
- Document a hard deletion date for any shim that must survive the sprint.
- **Outcome:** single import path per symbol; indirection layer gone. **Risk:** MEDIUM (import churn). **Effort:** M.

### D3. Unify retry & resilience policy (Week 4)
- Define `RETRY_POLICY` constants (per task tier) in one place (`resilience.py` or `crosscutting/config.py`). Stages reference the constant, not literals.
- **Outcome:** consistent failure semantics. **Risk:** LOW. **Effort:** S.

### D4. Rename fix-named module (Week 1)
- `infrastructure/state_fix_bug001.py` → `infrastructure/state_migration.py` (or inline). Update importers.
- **Outcome:** no load-bearing "fix" artifact. **Risk:** LOW. **Effort:** S.

**Workstream D exit criteria:**
- `test_preexisting_problems.py` has zero `xfail` markers (all promoted to passing).
- Zero modules named `*_fix_*` / `*_bug*` in production paths.
- Stage retry values sourced from a single constant.
- Removes the HIGH automations finding.

---

## Workstream E — Observability & testability (enables 9)

**Target rubric clauses:** "observable, testable." Current coverage ratchet floor is 6% — far below the "testable" bar for a 9.

### E1. Coverage ratchet (continuous, Weeks 1–6)
- Raise `fail_under` in `pyproject.toml` in steps as modules are migrated and pinned: 6 → 15 → 30 → 50. The existing comment already plans this — execute it.
- Every reconciled divergent pair (A4) and every bug fix (D) ships with tests, so coverage rises as a side effect of A–D.
- **Target:** ≥ 50% line coverage, ≥ 70% on `domain/`, `engine_core/`, `application/`. **Effort:** continuous.

### E2. Contract tests as first-class gate (Week 2)
- The CI already runs import-linter ("Check import boundaries") and contract tests. Expand contract tests to assert the **new** invariants: orchestrator statelessness (B), zero cycle (C), root kernel allowlist (A).
- **Outcome:** architecture invariants are executable, not aspirational. **Effort:** M.

### E3. Observability baseline (Week 5)
- Confirm structured logging + tracing (`tracing.py`, OpenTelemetry deps present) are wired through the pipeline stages and the `RunContext`. Add span coverage for generate/critique/evaluate and budget checkpoints.
- Emit per-run metrics (cost, tokens, model fallbacks, circuit-breaker trips) to the telemetry store.
- **Outcome:** "observable" clause satisfied. **Risk:** LOW. **Effort:** M.

**Workstream E exit criteria:**
- `fail_under` ≥ 50; core layers ≥ 70%.
- Contract tests assert statelessness + zero-cycle + kernel allowlist.
- Every pipeline stage emits a trace span and per-run metrics.

---

## Workstream F — Contract lockdown & target-state layout (enables 10)

**Goal:** make the achieved architecture impossible to silently regress.

### F1. Final layer layout
```
orchestrator/
  <kernel>          # ≤15 documented root files: models, log_config, budget, constants, exceptions-reexport
  domain/           # ports, value objects, enums — stdlib-only (already clean)
  application/      # use-cases, services, schedulers — depends on ports, never engine/infra
  engine_core/      # orchestration pipeline — depends on ports + application via ports only
  infrastructure/   # adapters: LLM clients, persistence, cache, telemetry, tracing
  entrypoints/      # cli, api_server, webhooks, chat_cli, cli_dispatch  (NEW — driving adapters)
```

### F2. Import-linter contracts covering 100% of modules (Week 6)
- Add contracts:
  - `entrypoints` may import any inward layer; nothing imports `entrypoints`.
  - `engine_core` forbidden from importing `application` **except via `domain.ports`** (independence contract).
  - Root kernel: forbidden from importing any subpackage except `domain`.
  - Layered contract (import-linter `layers` type) enforcing `entrypoints > infrastructure > engine_core > application > domain` direction.
- Remove every `ignore_imports` exemption (each represents residual drift).
- **Outcome:** no module escapes boundary enforcement. **Risk:** LOW. **Effort:** M.

### F3. Documentation sync (Week 6)
- Update `docs/CODEBASE_MINDMAP.md` and `CLAUDE.md` "Four Unbreakable Rules" to reflect the realized layout (esp. the new `entrypoints/` layer and the documented kernel).
- **Outcome:** docs match code. **Effort:** S.

**Workstream F exit criteria:**
- import-linter contracts cover all packages, zero `ignore_imports`.
- Layered contract green in CI.
- Docs updated.
- Removes re-drift risk → score ceiling reaches 10.

---

## Sequencing & timeline

```
Week 1: A1 freeze · A2 inventory · A3 identical-dedupe · D1 automations · D4 rename fix-module
Week 2: A4 divergent (high-importer first) · C1 VS port · C2 skill_store port · E2 contract tests
Week 3: A4 continues · A5 root-only moves · B1 RunContext · B2 entrypoint pooling · C3 reclassify drivers
Week 4: A4/A5 finish · B3 load test · D2 shim retirement · D3 retry policy
Week 5: E1 ratchet to 30 · E3 observability
Week 6: E1 ratchet to 50 · F1/F2 layout + contracts · F3 docs
```

Critical path = **A4 (reconcile 103 divergent pairs)**. It is the longest, riskiest line item and gates the CRITICAL-finding removal. Everything else can parallelize around it.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Divergent-pair reconciliation introduces regressions | HIGH | HIGH | Pin-test-first (A4 step 4); one pair per PR; full suite gate |
| RunContext refactor breaks crash-recovery/resume | MEDIUM | HIGH | Integration test for resume before + after; B3 concurrency test |
| Shim removal breaks an un-grepped dynamic import | MEDIUM | MEDIUM | grep for `importlib`, `__import__`, string module names before deleting any shim |
| Coverage ratchet blocks unrelated PRs | MEDIUM | LOW | Raise `fail_under` only after a workstream lands, not preemptively |
| Scope creep into the 71 root-only files | MEDIUM | MEDIUM | A5 strictly leaf-first; `engine.py` deferred to B |

---

## Definition of Done (score ≥ 9 verified)

A re-run of ARCH-AUDIT-V2 must report:
1. **Phase 2:** zero CRITICAL, zero HIGH rows.
2. **Phase 3:** zero circular dependencies; zero layer leaks; `ServiceContainer` no longer flagged as god-wiring with per-run state.
3. **Phase 4:** orchestrator [VERIFIED] stateless; concurrency [VERIFIED] by load test (not [HYPOTHESIS]).
4. **Phase 5:** zero anti-patterns with code evidence (no root dump, no stateful orchestrator, no divergent duplicates, no fix-named modules, no temporal-coupling without guards).
5. **Phase 6:** score ≥ 9, maturity "Production" or "Mature".

Reaching **10** additionally requires Workstream F complete (100% contract coverage, zero exemptions) and Workstream E core-layer coverage ≥ 70%.
