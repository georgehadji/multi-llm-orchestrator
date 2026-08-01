# AI Orchestrator — Autonomous Testing Engine Implementation Plan

**Date:** 2026-07-28
**Status:** Proposed
**Target:** Multi-LLM Orchestrator v6.x → v7.0
**Scope:** Deterministic verification of generated software — consolidation of duplicated test runners, workspace-scoped verification, suite-validity gating, repair-loop hardening, mutation-based quality measurement, green-anchored refinement of code that passes, and a production-readiness gate that decides whether generated applications are fit to deploy.

> **Note on file naming.** The previous contents of `implementation_plan.md` ("Evidence-Driven Self-Improvement Implementation Plan", 2026-07-22) were preserved verbatim as [`implementation_plan_self_improvement.md`](implementation_plan_self_improvement.md) before this document replaced them. The two plans are complementary: that one governs *evaluator and harness co-evolution*; this one supplies the *deterministic oracle* those loops depend on.

---

## 1. Executive Summary

### 1.1 Problem statement

The orchestrator generates code autonomously but cannot presently prove that generated code works. Five verification checks are wired into the delivery path — syntax, build, security, lint, type — and **none of them executes a test**. The one component that does execute tests, `TestFirstGenerator`, is a capable 1,859-line implementation that is (a) configured with `sandbox=None` at its only call site, (b) restricted to single-file artifacts, and (c) blocking the asyncio event loop while it runs. Meanwhile three separate `TestRunner` classes exist with divergent output parsing, and the CLI-facing helper `run_project_tests()` is a stub that prints "no runner is configured" and returns an empty list.

The consequence is a specific, reproducible failure mode: **a task whose generated tests fail is scored 0.8 and ships.** `TaskResult.score` is set to `1.0 if tdd_result.test_result.passed else 0.8` ([`task_executor.py:295`](orchestrator/application/task_executor.py:295)), and 0.8 sits above the typical `acceptance_threshold` of 0.7. The deterministic floor mechanism that exists to prevent exactly this (`VerificationGate.FAIL_SCORE_FLOOR = 0.15`) never sees test results, because the gate's contract is `run(artifact: str)` — a single string — and tests require a materialized multi-file workspace.

### 1.2 Governing principle

> A generated artifact may be scored by an LLM only after a deterministic oracle has been executed against it, and the oracle itself must be proven non-vacuous before its verdict is trusted.

The second clause is the one most autonomous-testing designs omit. The test suite is authored by the same model family that authors the implementation; a green suite proves agreement between two LLM outputs, not conformance to the specification. Suite validity must therefore be established mechanically — RED-gating, assertion floors, and mutation sampling — before a passing result is allowed to raise a score.

### 1.3 Recommended target state

```text
Task ──► generate suite ──► RED-GATE ──► generate impl ──► EXECUTE (sandboxed)
                               │                                 │
                    (vacuous tests discarded)          ┌─────────┴──────────┐
                                                       │                    │
                                                  failures              green
                                                       │                    │
                                             bounded repair loop     MUTATION SAMPLE
                                             (tests immutable,             │
                                              plateau → escalate)     REGRESSION
                                                       │                    │
                                                       └──────► VerificationGate
                                                                (test_execution: REQUIRED)
                                                                       │
                                                             ┌─────────┴──────────┐
                                                        fail │                    │ green + mutation ≥ θ
                                                             ▼                    ▼
                                                    score floor 0.15    GREEN-ANCHORED REFINEMENT
                                                                        measure → propose → apply
                                                                        → re-verify → keep or revert
```

The final stage inverts the usual relationship between tests and code quality. Today the suite is consumed as a pass/fail gate and then discarded; a green, mutation-validated suite is precisely the licence to restructure code safely, and refinement is where that licence is spent.

### 1.4 Scope summary

| Class | Count | Effort |
|---|---|---|
| Defect fixes (F-1 … F-8) | 8 | ~6 days |
| Enhancements (E-1 … E-8) | 8 | ~14 days |
| Refinement stage (E-9 … E-12) | 4 | ~4.5 days |
| Production readiness (P-1 … P-12) | 12 | ~14 days |
| Total | 35 work items across 7 phases | ~39 engineer-days |

Two items are security-classified (F-2 in-process `exec()` of model output; F-3 unisolated execution of model output) and are scheduled in Phase 1 regardless of dependency order.

### 1.5 Non-goals

- No new test *authoring* strategy for the orchestrator's own suite. This plan concerns the orchestrator's ability to test **generated output**; the repo's own 200+ test modules are affected only by additions.
- No replacement of the LLM evaluator. The oracle constrains and floors the evaluator; it does not replace it.
- No Kubernetes/remote execution backend. Sandbox tiering stops at Docker.
- Impact-based test selection (E-7) is explicitly deferred to Phase 5 and may be dropped without affecting correctness — it is a latency optimization.
- Refinement (Phase 6) never runs against an unvalidated oracle. Restructuring code whose suite has not passed the RED-gate and mutation threshold would ship regressions under a green checkmark; the entry condition is a hard gate, not a recommendation.
- Phase 7 does **not** deploy. The orchestrator emits and verifies deployment configuration — Dockerfiles, workflows, health endpoints, migrations — and proves they work in a sandbox. Pushing images, provisioning infrastructure, and running releases stay outside the system boundary.
- Phase 7 does not invent a bespoke quality standard. Its rubric encodes established practice (12-factor, OWASP ASVS L1, SLSA build provenance, container hardening baselines); every requirement cites what it derives from.

---

## 2. Current Architecture Assessment

### 2.1 Layering and enforcement (healthy)

The hexagonal boundary is real and mechanically enforced, which is rare and worth preserving carefully:

- **5 import-linter contracts** in [`.importlinter`](.importlinter): domain purity, application-no-concrete-infra, application-no-engine, engine_core-pipeline-no-infra, root-modules-no-infra. Enforced in CI (`lint-imports`) and pre-commit.
- **Root module freeze** via `scripts/check_root_module_freeze.py` and `scripts/check_new_root_files.py` — no new `orchestrator/*.py` at depth 1.
- **Composition root** at [`engine_core/container.py`](orchestrator/engine_core/container.py) wires infrastructure into application services; `container.py` is intentionally exempt from Contract 4.
- **Coverage ratchet** `fail_under = 7` ([`pyproject.toml:361`](pyproject.toml:361)) — a floor that only moves up.
- **Marker discipline** enforced by `scripts/check_test_markers.py`; 13 registered markers with `--strict-markers`.

**Every design decision below is constrained by these.** In particular: the new port declaration goes in `orchestrator/domain/`, orchestration in `orchestrator/application/`, all subprocess and Docker code in `orchestrator/infrastructure/`, and no file is added at `orchestrator/*.py`.

### 2.2 Verification path (structurally sound, functionally incomplete)

`VerificationGate` ([`application/verification_gate.py:105`](orchestrator/application/verification_gate.py:105)) is well-built: chain-of-responsibility, runs all checks regardless of early failure, emits structured `ExecutionReceipt`s with artifact hash and duration, supports a `VerificationPolicy` with `REQUIRED`/`RECOMMENDED`/`NOT_RUN` outcomes, and caps failing artifacts at `FAIL_SCORE_FLOOR = 0.15`. It is wired at [`container.py:647`](orchestrator/engine_core/container.py:647) into `EvaluatorService`.

The limitation is its contract: `async def run(self, artifact: str, policy=...)`. A string is sufficient for syntax/lint/type/security. It is structurally insufficient for test execution, which needs a directory containing implementation modules, test modules, fixtures, and a dependency manifest. **This is the central architectural gap, not a missing check.**

### 2.3 Test execution capability inventory

| Component | Location | State |
|---|---|---|
| `TestFirstGenerator` | [`testing/first_generator.py`](orchestrator/testing/first_generator.py) | Functional. 5 frameworks (pytest, jest/vitest/mocha, go, cargo), framework auto-detection, cost tracking per phase, bounded repair. **Tests are immutable during repair** — `_repair_to_pass_tests` returns implementation only ([:1500](orchestrator/testing/first_generator.py:1500)). Preserve this property. |
| `TestRunner` (A) | [`runtime/sandbox.py:80`](orchestrator/runtime/sandbox.py:80) | Unwired. Regexes `(\d+) passed` from `-q` output. Returns `success` from exit code but `total=0` yields no error. |
| `TestRunner` (B) | [`quality/quality_control.py:370`](orchestrator/quality/quality_control.py:370) | Uses `--json-report`; parses coverage from stderr. Duplicated verbatim at [`quality_control.py:365`](orchestrator/quality_control.py:365). |
| `run_project_tests()` | [`quality/run_tests.py`](orchestrator/quality/run_tests.py) | Stub. Prints, returns `[]`. Silent false-negative for any caller. |
| `PreSubmissionTester` | [`quality/pre_submission_testing.py:42`](orchestrator/quality/pre_submission_testing.py:42) | Named as a test gate; runs zero tests. Checks output non-empty, score ≥ threshold, absence of the literal string "TODO". |
| `SandboxExecutor` | [`runtime/sandbox.py:34`](orchestrator/runtime/sandbox.py:34) | Unwired. Subprocess + tempdir + timeout. No rlimits, no network isolation, no cwd control. |
| `SandboxExecutor` (different) | [`safety/sandbox_executor.py:31`](orchestrator/safety/sandbox_executor.py:31) | Name collision, different responsibility (diff/review/merge). |
| `AutonomousDebugger` | [`autonomous_debugger.py:115`](orchestrator/autonomous_debugger.py:115) | Wired into the output organizer's test-fix loop. Duplicated root/package. |

Four implementations of "run tests", none authoritative, one silently returning success-shaped emptiness.

### 2.4 Data flow, as executed today

```text
task_executor._try_tdd_generation()          [enable_tdd_first flag]
  └─► TestFirstGenerator(sandbox=None, max_test_iterations=3)   ← host execution
        ├─ Phase 1  generate test suite       (LLM)
        ├─ Phase 2  generate implementation   (LLM)   ← no RED gate between 1 and 2
        ├─ Phase 3  write main.py + test_main.py to tempdir, rewrite imports to `main`
        │           subprocess.run(pytest)  ← blocking call inside async def
        └─ Phase 4  repair loop (tests immutable, collection errors correctly skipped)
  └─► TaskResult(score = 1.0 if passed else 0.8)     ← failing suite still ships
        └─► EvaluatorService ─► VerificationGate.run(artifact_str)
                                 syntax | build | security | lint | type_check
                                 (no test signal reaches the gate)
```

### 2.5 Technical debt register (testing subsystem)

| ID | Debt | Evidence | Severity |
|---|---|---|---|
| D-1 | Blocking I/O in async path — `subprocess.run(timeout=120)` inside `async def` stalls the event loop, serializing all concurrent tasks | [`first_generator.py:951`](orchestrator/testing/first_generator.py:951) | High |
| D-2 | `exec(compile(artifact))` of model output in the orchestrator process | [`verification_checks.py:242`](orchestrator/infrastructure/verification_checks.py:242) | Critical (security) |
| D-3 | Model-generated code executed on host without isolation | [`task_executor.py:271`](orchestrator/application/task_executor.py:271) (`sandbox=None`) | Critical (security) |
| D-4 | Failing suite scores 0.8, above acceptance threshold | [`task_executor.py:295`](orchestrator/application/task_executor.py:295) | High (correctness) |
| D-5 | Four divergent test-execution implementations | §2.3 | High (maintainability) |
| D-6 | Silent stub returning `[]` | [`quality/run_tests.py`](orchestrator/quality/run_tests.py) | High (silent failure) |
| D-7 | Regex parsing of human-readable pytest output | [`runtime/sandbox.py:110`](orchestrator/runtime/sandbox.py:110) | Medium |
| D-8 | Single-file constraint — `main.py` + `test_main.py`, imports rewritten | [`first_generator.py:910`](orchestrator/testing/first_generator.py:910) | Medium (capability ceiling) |
| D-9 | No suite-validity check — vacuous suites pass | absent | High (correctness) |
| D-10 | Root/package duplication (`quality_control.py`, `output_organizer.py`, `pre_submission_testing.py`, `autonomous_debugger.py`) | §2.3 | Medium |
| D-11 | `max_test_iterations` default 5, call site passes 3; no plateau detection | [`:315`](orchestrator/testing/first_generator.py:315) vs [`:273`](orchestrator/application/task_executor.py:273) | Low (cost) |

### 2.6 Scalability, security, observability posture

- **Scalability.** D-1 is the binding constraint: any concurrency in the task pipeline is nullified while a 120-second test run blocks the loop. Fixing it converts test execution into genuinely parallel work bounded by a semaphore.
- **Security.** D-2 and D-3 together mean model-authored code executes with full orchestrator privileges — filesystem, network, environment (including provider API keys in `os.environ`). Bandit's HIGH-severity CI gate does not catch D-2 because the call site carries a `# nosec B102` suppression justified by "isolated namespace", which prevents name leakage but not side effects.
- **Observability.** `ExecutionReceipt` already carries `check_name`, `outcome`, `reason`, `duration_ms`, `artifact_hash`. Test execution has no equivalent record — no per-test outcomes, no isolation level, no flake history. Telemetry exists (`telemetry` passed to services) and is the natural sink.

### 2.7 Post-test capability gap — nothing improves code that passes

The delivery pipeline ([`output_organizer.py:145`](orchestrator/output_organizer.py:145) `organize_project`) runs:

```text
organize files → generate missing tests → FORMAT (black / ruff --fix / prettier)
              → security scan → run tests → fix FAILING tests → move tests → deliver
```

Two observations follow from the ordering. First, the only mechanical improvement pass (`_format_code`) runs **before** tests by design — the inline comment at [`:182`](orchestrator/output_organizer.py:182) states it explicitly, "so the suite validates the formatted code" — which is correct for formatting but means the pipeline's one code-modifying step has no verification behind it at the moment it runs. Second, after the suite goes green there is no stage at all. Repair stops at green; improvement never starts.

Components that resemble a refinement stage but are not one:

| Component | Actual behavior | Why it does not close the gap |
|---|---|---|
| `TestFixer` / [`AutonomousDebugger`](orchestrator/autonomous_debugger.py:115) | iterates until the suite passes | terminates at green — repair, not improvement |
| critique → revise cycle | per-task generate/critique/revise | executes **before** any test runs; LLM opinion with no oracle behind it |
| [`MAPElitesPipeline`](orchestrator/engine_core/stages/map_elites.py:50) | quality-diversity search over 9 variants | scored by [`_heuristic_score`](orchestrator/engine_core/stages/map_elites.py:170) — string heuristics, not execution; also pre-test |
| [`QualityController.run_quality_gate`](orchestrator/quality/quality_control.py:587) | complexity, duplication, doc/type coverage, issue list | **measures, never acts.** Emits `QualityReport`; no consumer applies it. Wired only into [`project_analyzer.py:629`](orchestrator/project_analyzer.py:629) (the `--analyze-codebase` path), not into generation delivery |
| [`format_output_dir`](orchestrator/output/formatter.py:222) | ruff --fix + black + prettier | cosmetic, pre-test, and exception-swallowed by design ("never block delivery on formatting") |

The orchestrator therefore pays the full cost of generating and executing a test suite and collects none of the structural payoff. Metrics are computed and discarded; the safety net that would make restructuring verifiable exists and goes unused. Phase 6 closes this, and only after Phase 3 has made the oracle trustworthy enough to lean on.

### 2.8 Production posture of generated applications

Three delivery paths exist, with materially different output guarantees:

| Path | Entry point | Emits |
|---|---|---|
| **Default** | `python -m orchestrator --project …` | `output_writer` → [`ProjectAssembler`](orchestrator/project_mgmt/assembler.py:136) for Python, [`WebProjectAssembler`](orchestrator/web_assembler.py) for web |
| **App build** | `orchestrator build` ([`commands/build.py`](orchestrator/commands/build.py:11)) | advisor → scaffold → generate → assemble → resolve deps → install + test + startup + optional Docker verify |
| **Website** | `--website` | `WebsiteGenerator` + `website_validator` + formatter + secret scanner |

Only the middle path verifies the application runs. The default path — the documented one — never starts what it built.

#### 2.8.1 Existing strengths

`ProjectAssembler` is a capable scaffold and should be extended, not replaced. It already emits a multi-stage Dockerfile with a non-root user, `HEALTHCHECK`, and `.dockerignore`; a layered `src/domain|application|infrastructure` tree; `pyproject.toml` carrying `--cov-fail-under=80` ([:991](orchestrator/project_mgmt/assembler.py:991)); a Makefile; a GitHub Actions workflow with lint, type, security, and a 3-version test matrix; `.env.example`; a pre-commit config; an exception hierarchy; a logging config; and a Pydantic Settings config layer. Delivery is additionally gated by a blocking secret scan ([`generated_output_scanner.py:270`](orchestrator/safety/generated_output_scanner.py:270)).

That covers roughly the first two thirds of production readiness. The remaining third is where deployments actually fail.

#### 2.8.2 Gap register

| ID | Gap | Evidence | Severity |
|---|---|---|---|
| G-1 | No lock file — image builds from unpinned `pyproject` ranges, so builds are not reproducible | [`assembler.py:1700`](orchestrator/project_mgmt/assembler.py:1700) | Critical |
| G-2 | Base image on a mutable tag, not a digest | [`:1680`](orchestrator/project_mgmt/assembler.py:1680) | Critical |
| G-3 | Security scan cannot fail the build — `bandit … \|\| true` | [`:1908`](orchestrator/project_mgmt/assembler.py:1908) | High |
| G-4 | No `permissions:` block ⇒ default write-scoped `GITHUB_TOKEN`. The correct emitter exists but is unwired — [`cicd_generator.py:151`](orchestrator/generators/cicd_generator.py:151) already emits `permissions: contents: read` | [`:1848`](orchestrator/project_mgmt/assembler.py:1848) | High |
| G-5 | Dependency CVE scan absent from CI — `safety` appears only as a Makefile note, "run manually" | [`:1642`](orchestrator/project_mgmt/assembler.py:1642) | High |
| G-6 | Health check is vacuous — `python -c "import {pkg}"` passes while the server is dead, so no orchestrator will ever restart a hung container | [`:1737`](orchestrator/project_mgmt/assembler.py:1737) | High |
| G-7 | No `/health` + `/ready` endpoints; no liveness/readiness distinction; no dependency probes | absent | High |
| G-8 | No graceful shutdown — no SIGTERM handling, no connection draining ⇒ every deploy drops in-flight requests | absent | High |
| G-9 | No metrics, no tracing, no correlation IDs. Logging is configured but carries no request context | absent | High |
| G-10 | No migration tooling, seed data, or backup procedure. [`database_generator.py`](orchestrator/generators/database_generator.py) is capable but reachable only through the orphan [`swiftstack_integration.py`](orchestrator/integrations/swiftstack_integration.py) | absent from wired path | High |
| G-11 | Model-authored dependency manifests are installed **unsandboxed on the host** — arbitrary package install is arbitrary code execution | [`verifier.py:111`](orchestrator/appbuilder/verifier.py:111) | Critical (security) |
| G-12 | Default path never boots the app; no smoke test, no API contract test, no load or a11y budget | §2.8 table | High |
| G-13 | Web output is unbuilt — raw `index.html`/`style.css`/`script.js`, no bundling/hashing/minification, no `robots.txt`/`sitemap.xml`/404, CSP permits `'unsafe-inline'` and cdnjs, no SRI | [`web_assembler.py:10`](orchestrator/web_assembler.py:10) | Medium-High |
| G-14 | `pyproject` declares MIT but no `LICENSE` file is written; no `SECURITY.md`, `CODEOWNERS`, NOTICE, or dependency-license check | [`:857`](orchestrator/project_mgmt/assembler.py:857) | Medium |
| G-15 | No runbook, ADRs, OpenAPI reference, deployment/rollback guide, or SLO definitions | absent | Medium |
| G-16 | Assembly failure degrades silently — `except Exception → "Continuing with task files only"` — the product drops from "production project" to "a folder of files" and still reports success | [`output_writer.py:325`](orchestrator/output_writer.py:325) | Medium (honesty) |
| G-17 | No SBOM, image scanning, signing, or build provenance | absent | Medium |

#### 2.8.3 Root causes

1. **The best implementation is rarely the wired one.** `cicd_generator`, `docker_generator`, `database_generator`, `logging_generator`, `fullstack_generator`, and `testing_templates` are reachable only through `swiftstack_integration.py`, which nothing imports. This is the same disease as the four `TestRunner` classes in §2.3 — production readiness is substantially a *wiring* problem, not a writing problem.
2. **Verification asks "does it run?", never "is it operable?"** Install + tests + startup is a build check. Production readiness is about what happens at 03:00 — health, restart, rollback, observability, schema change.
3. **No encoded definition of done.** Nothing in the system expresses what "production-grade" means, so nothing can gate on it, and each generator improvises its own partial answer.

---

## 3. Detailed Implementation Plan

### 3.1 Phase overview

| Phase | Theme | Items | Duration | Exit gate |
|---|---|---|---|---|
| **0** | Baseline & characterization | B-1 … B-3 | 1.5 d | Current behavior locked by tests; every defect has a failing/xfail test |
| **1** | Security + consolidation | F-1, F-2, F-3, F-5, F-6, F-7 | 4 d | One runner. No host execution of model output. No silent stubs. |
| **2** | Workspace & gate integration | E-1, E-6, F-4, F-8 | 5 d | Test execution reaches `VerificationGate`; failing suite floors the score |
| **3** | Oracle validity & repair hardening | E-2, E-3, E-5 | 5 d | Vacuous suites rejected; repair loop bounded and escalating; flakes quarantined |
| **4** | Quality measurement & observability | E-4, E-8 | 3 d | Mutation score recorded and gateable; receipts carry isolation + mutation data |
| **5** | Optimization (optional) | E-7 | 2 d | Impact-selected regression runs with full-suite fallback |
| **6** | Green-anchored refinement | E-9 … E-12 | 4.5 d | Passing code is measurably improved without behavior change; every candidate is revertible |
| **7** | Production readiness | P-1 … P-12 | 14 d | Generated apps are reproducible, observable, operable, and legally shippable — proven by probe, not by template presence |

**Dependency graph:**

```text
B-1,B-2,B-3
     │
     ├─► F-2 ─┐                          (security, no dependencies)
     ├─► F-3 ─┤
     ├─► F-1 ─┤
     ├─► F-5 ─┼─► F-6 ─► F-7 ─► E-1 ─► E-6 ─► F-4
     │        │                    │      │
     │        │                    │      └─► E-8
     │        │                    ├─► E-2 ─► E-3
     │        │                    ├─► E-5
     │        │                    └─► E-4 (needs E-1 + E-2)
     └────────┘                           ├─► E-7 (needs E-1 + E-6)
                                          │
                                          └─► E-9 ─► E-10 ─► E-11 ─► E-12
                                              (Phase 6 — hard-gated on E-2 + E-4:
                                               refinement requires a validated oracle)

Phase 7 (needs E-1 Workspace + F-3 sandbox tiers + E-6 gate):

  P-1 ─► P-2 ─┬─► P-4 ─► P-5                       (supply chain, CI)
              ├─► P-3 ─┬─► P-6                     (live probes → operability)
              │        ├─► P-7                     (data layer)
              │        └─► P-8                     (verification depth; P-8a blocks on F-3)
              ├─► P-9                              (web pipeline)
              ├─► P-10, P-11                       (compliance, handover — no deps)
              └─────────────────────► P-12         (gate integration + honesty fix — last)
```

### 3.2 Target module layout

All additions respect Contracts 1–5 and the root-module freeze.

```text
orchestrator/
  domain/
    ports.py                       # + TestExecutorPort, SandboxPort, MetricCollectorPort,
                                   #   BenchmarkPort (Protocol, no I/O)
    testing_models.py              # + SuiteReport, TestOutcome, Workspace, TestSelection
                                   #   frozen dataclasses; stdlib only
    refinement.py                  # + MetricSnapshot, RefinementCandidate, AcceptanceVerdict,
                                   #   RefinementOutcome + pure decision functions (Phase 6)
    readiness.py                   # + AppArchetype, ReadinessLevel, Requirement, Evidence,
                                   #   RequirementOutcome, ReadinessReport (Phase 7)
  application/
    testing/
      __init__.py
      service.py                   # TestingService — orchestration, port-injected
      suite_validator.py           # RED-gate + assertion floor (AST, no I/O)
      repair_policy.py             # plateau detection, escalation, immutability lock
    refinement/                    # Phase 6
      __init__.py
      service.py                   # RefinementService — measure → propose → apply → verify
      operators/
        base.py                    # RefinementOperator Protocol (Strategy)
        dead_code.py               # mechanical tier
        deduplicate.py             # structural tier
        extract_function.py        # structural tier
        flatten_nesting.py         # structural tier
      acceptance.py                # Chain of Responsibility: green → ratchet → no-new-findings
      ledger.py                    # Command history + revert (Memento over Workspace)
    readiness/                     # Phase 7
      __init__.py
      service.py                   # ReadinessService — assess → remediate → re-assess
      rubric.py                    # Composite requirement tree, archetype-scoped
      requirements/                # Specification objects — one module per concern
        supply_chain.py            # lock file, digest pin, SBOM, action pinning
        ci_enforcement.py          # blocking scans, least-privilege token, dep audit
        operability.py             # health/ready, shutdown, metrics, tracing, config
        data_layer.py              # migrations, seed, backup procedure
        web_delivery.py            # build pipeline, robots/sitemap/404, CSP, SRI
        compliance.py              # LICENSE, SECURITY.md, CODEOWNERS, dep licences
        handover.py                # runbook, ADRs, API reference, SLOs
      remediations.py              # deterministic auto-fixes (Command, ledger-backed)
    verification_gate.py           # + CheckScope, workspace-aware dispatch (additive)
  infrastructure/
    test_runners/
      __init__.py                  # framework registry (factory)
      base.py                      # shared machine-readable-output parsing
      pytest_runner.py             # --json-report
      jest_runner.py               # --json
      go_runner.py                 # go test -json
      cargo_runner.py              # --format json
    sandboxes/
      __init__.py                  # tier selection policy
      subprocess_sandbox.py        # rlimits, timeout, scrubbed env, temp HOME
      docker_sandbox.py            # --network=none --read-only --cap-drop=ALL
    metrics/                       # Phase 6 — MetricCollectorPort adapters
      __init__.py                  # collector registry
      ast_metrics.py               # ast.NodeVisitor: complexity, nesting, size, duplication
      dead_code.py                 # vulture adapter
      bench_runner.py              # BenchmarkPort adapter (pytest-benchmark / hyperfine)
    readiness_probes/              # Phase 7 — ProbePort adapters
      __init__.py                  # probe registry
      static_probes.py             # file presence + real parsing (TOML/YAML/Dockerfile)
      supply_chain_probes.py       # lock resolution, digest pins, SBOM, action SHAs
      live_probes.py               # boot in sandbox, HTTP /health + /ready, SIGTERM drain
      license_probes.py            # dependency licence extraction + compatibility
    artifact_emitters/             # Phase 7 — ArtifactEmitterPort adapters
      __init__.py                  # emitter registry (archetype → emitter set)
      cicd_emitter.py              # Adapter over generators/cicd_generator (unwires the orphan)
      docker_emitter.py            # Adapter over generators/docker_generator
      database_emitter.py          # Adapter over generators/database_generator (Alembic)
      observability_emitter.py     # Adapter over generators/logging_generator + OTel
      compliance_emitter.py        # LICENSE / SECURITY.md / CODEOWNERS / NOTICE
      web_build_emitter.py         # bundling, hashing, robots/sitemap/404, SRI
    verification_checks.py         # + _make_test_execution_check (WORKSPACE scope)
  engine_core/
    container.py                   # composition root: instantiate + inject
```

**Deletions and shims (Phase 1):**

| Path | Action |
|---|---|
| `orchestrator/runtime/sandbox.py` | `TestRunner` deleted; `SandboxExecutor` superseded by `infrastructure/sandboxes/subprocess_sandbox.py`; module becomes a deprecation shim |
| `orchestrator/quality/quality_control.py::TestRunner` | Delegates to `TestExecutorPort` |
| `orchestrator/quality_control.py` | Confirmed re-export shim only |
| `orchestrator/quality/run_tests.py` | Stub replaced with real delegation; raises `TestRunnerUnavailableError` when no runner resolves — never returns `[]` |
| `orchestrator/output_organizer.py`, `orchestrator/pre_submission_testing.py`, `orchestrator/autonomous_debugger.py` | Verified as shims re-exporting the package copies (pattern already used by `test_fixer.py`/`test_validator.py`) |

### 3.3 Core interface design

```python
# orchestrator/domain/testing_models.py — pure data, stdlib only (Contract 1)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"


class IsolationLevel(str, Enum):
    """Recorded on every report so downstream gates know what protection applied."""
    NONE = "none"              # never selectable — reserved for legacy receipts
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


@dataclass(frozen=True)
class TestOutcome:
    node_id: str
    status: TestStatus
    duration_ms: float
    message: str = ""


@dataclass(frozen=True)
class SuiteReport:
    """Structured result of one suite execution. Never derived from regex."""
    passed: bool
    exit_code: int
    outcomes: tuple[TestOutcome, ...] = ()
    collection_errors: tuple[str, ...] = ()   # distinct from failures — not LLM-repairable
    line_coverage: float | None = None
    mutation_score: float | None = None
    flaky_node_ids: tuple[str, ...] = ()
    isolation: IsolationLevel = IsolationLevel.SUBPROCESS
    duration_ms: float = 0.0
    truncated_output: str = ""

    @property
    def executed(self) -> int:
        return len(self.outcomes)

    @property
    def is_vacuous_result(self) -> bool:
        """Zero executed tests is never success — closes the D-7 false-positive."""
        return self.executed == 0


@dataclass(frozen=True)
class Workspace:
    """A materialized, multi-file project tree ready for execution."""
    root: Path
    framework: str
    source_files: tuple[Path, ...] = ()
    test_files: tuple[Path, ...] = ()
    manifest: Path | None = None          # pyproject.toml / package.json / go.mod
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TestSelection:
    node_ids: tuple[str, ...] = ()        # empty ⇒ full suite
    reason: str = "full"
```

```python
# orchestrator/domain/ports.py  (additions)
from typing import Protocol

class TestExecutorPort(Protocol):
    """Executes a suite in a workspace. Implementations live in infrastructure."""
    async def run(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        timeout_s: float = 120.0,
    ) -> SuiteReport: ...

    def supports(self, framework: str) -> bool: ...


class SandboxPort(Protocol):
    """Provides the isolation boundary a runner executes inside."""
    level: IsolationLevel
    async def exec(
        self, argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float
    ) -> tuple[int, str, str]: ...
```

**Gate extension — backward compatible by construction.** Rather than widening `VerificationGate.run()`'s signature (which would break the 8 existing test modules that construct gates directly), a scope discriminator is added to `VerificationCheck`:

```python
# orchestrator/application/verification_gate.py  (additive)
class CheckScope(str, Enum):
    ARTIFACT = "artifact"     # existing checks — unchanged, default
    WORKSPACE = "workspace"   # needs a materialized tree

@dataclass
class VerificationCheck:
    name: str
    run: CheckFn
    command: str | None = None
    scope: CheckScope = CheckScope.ARTIFACT      # default preserves all call sites

# gate.run(artifact, policy=None, workspace=None)
#   workspace is None  → WORKSPACE-scoped checks emit CheckOutcome.NOT_RUN receipts
#   workspace present  → they execute
```

`NOT_RUN` is already a modeled outcome with receipt semantics, so no new state is introduced. Every existing test continues to pass unmodified — an explicit acceptance criterion for E-6.

### 3.4 Refinement design (Phase 6)

#### 3.4.1 Paradigm — functional core, imperative shell

Refinement is a decision problem wrapped in an I/O problem, and the two must not mix. Contract 1 (domain purity) already forces this shape, and the design leans into it rather than working around it:

- **Functional core** (`domain/refinement.py`, `application/refinement/operators/*`, `acceptance.py`): frozen dataclasses and pure functions. Given a `MetricSnapshot` before and after, decide accept or reject. No filesystem, no subprocess, no clock. Every acceptance rule is a total function that is trivially testable and trivially auditable — which matters because these rules are what stand between "improved" and "silently broken".
- **Imperative shell** (`application/refinement/service.py`, `infrastructure/metrics/*`): materializes workspaces, shells out to collectors, calls the LLM, writes files, reverts.

The practical consequence: the entire accept/reject logic is unit-testable with no subprocess and no network, and every refinement decision can be replayed from recorded snapshots.

#### 3.4.2 Pattern selection and rationale

| Concern | Pattern | Why this one, here |
|---|---|---|
| One transform per improvement kind (dead code, dedupe, extract, flatten) | **Strategy** | Open/closed: new operators register without touching the service. Mirrors `image_optimizer.py`'s `ImageOptimizerStrategy`, already the repo's idiom for pluggable transforms |
| Apply / revert a single candidate atomically | **Command** | Each candidate is an object with `apply()` and `revert()`. Enables an ordered ledger, partial rollback, and replay — impossible with in-place mutation |
| Snapshot the workspace before a candidate | **Memento** | Already the repo's persistence idiom (`state.py`, `checkpoints.py`). Revert restores a snapshot rather than attempting inverse transforms, which are unreliable for LLM-authored edits |
| Accept/reject sequence (green → ratchet → no-new-findings → API-surface unchanged) | **Chain of Responsibility** | Identical to `VerificationGate`'s existing composition; every rule runs and reports, so a rejected candidate yields a full diagnosis rather than a single first failure |
| Metric extraction over the AST | **Visitor** | `ast.NodeVisitor` is the stdlib realization; complexity, nesting depth, function length, and duplication are four visitors over one parse, not four parses |
| Tool adapters (vulture, radon, benchmark runners) | **Ports & Adapters** | `MetricCollectorPort` in `domain/`, adapters in `infrastructure/`. Contract 2 forbids the application layer from importing them directly |
| Emitting refinement events | **Observer / EventBus** | Reuses the existing bus; refinement telemetry lands in the same sink as test telemetry (E-8) |
| Staged tier execution (mechanical → structural → performance) | **Pipeline (`BasePipeline`)** | Consistent with `engine_core/stages/`; each tier is skippable and independently flagged |

Deliberately **not** used: no Template Method inheritance hierarchy for operators (composition over inheritance — operators are Protocol implementations, not subclasses); no Observer-driven mutation (events are reports, never triggers); no global registry singleton (the registry is constructed in `container.py` like every other collaborator).

#### 3.4.3 Invariants

1. **Entry gate.** Refinement runs only when the suite is green **and** `mutation_score ≥ ORCH_REFINE_MIN_MUTATION` (default 0.6). A weak oracle cannot license restructuring.
2. **Test immutability.** The E-3 hash lock is *reused*, not reimplemented. Tests are read-only for the entire refinement pass; a candidate that touches a test file is rejected before execution.
3. **Behavior preservation is verified, not asserted.** Every candidate re-runs the same suite. Green is necessary but not sufficient — see (4).
4. **Ratchet.** A candidate is accepted only if the targeted metric strictly improves, no other tracked metric regresses, mutation score does not drop, and no new lint/type/security finding appears. This is the same monotone-floor discipline as the repo's coverage ratchet.
5. **Public API surface is frozen.** Candidates that add, remove, or re-sign a public symbol are rejected. Refinement restructures internals; it does not redesign interfaces.
6. **Zero-finding ⇒ zero cost.** Measurement gates entry to every LLM-backed tier. A clean project completes Phase 6 with no model calls at all.
7. **Every candidate is independently revertible.** Failure of candidate *N* never invalidates accepted candidates 1…*N*−1.

#### 3.4.4 Core types

```python
# orchestrator/domain/refinement.py — frozen, stdlib only (Contract 1)
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RefinementTier(str, Enum):
    MECHANICAL = "mechanical"     # deterministic, no LLM
    STRUCTURAL = "structural"     # LLM-proposed, suite-verified
    PERFORMANCE = "performance"   # benchmark-gated


@dataclass(frozen=True)
class MetricSnapshot:
    """One measurement of a workspace. Comparable, not mutable."""
    cyclomatic_mean: float
    cyclomatic_max: int
    max_nesting_depth: int
    longest_function_lines: int
    duplicated_blocks: int
    dead_symbols: int
    total_lines: int
    bundle_bytes: int | None = None
    benchmark_ns: dict[str, float] | None = None


@dataclass(frozen=True)
class RefinementCandidate:
    """A single proposed change. Immutable description; the Command applies it."""
    operator: str                 # "extract_function", "dead_code", …
    tier: RefinementTier
    target_file: str
    rationale: str
    diff: str
    predicted_metric: str         # which MetricSnapshot field this must improve


@dataclass(frozen=True)
class AcceptanceVerdict:
    accepted: bool
    rule_results: tuple[tuple[str, bool, str], ...]   # (rule, passed, reason)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(reason for _, ok, reason in self.rule_results if not ok)


# ── pure decision core — no I/O, total functions ──────────────────────────────

def metric_improved(before: MetricSnapshot, after: MetricSnapshot, field: str) -> bool:
    """Targeted metric strictly improved (lower is better for every tracked field)."""
    ...

def no_metric_regressed(
    before: MetricSnapshot, after: MetricSnapshot, tolerance: float = 0.0
) -> bool:
    """No non-targeted metric got worse beyond tolerance."""
    ...
```

```python
# orchestrator/domain/ports.py  (Phase 6 additions)
class MetricCollectorPort(Protocol):
    name: str
    async def collect(self, workspace: Workspace) -> MetricSnapshot: ...


class BenchmarkPort(Protocol):
    async def measure(self, workspace: Workspace, *, runs: int = 5) -> dict[str, float]: ...
```

```python
# orchestrator/application/refinement/operators/base.py — Strategy + Command
class RefinementOperator(Protocol):
    """Proposes candidates. Pure: reads a snapshot and source text, returns proposals."""
    name: str
    tier: RefinementTier

    def applicable(self, snapshot: MetricSnapshot) -> bool: ...
    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]: ...
```

#### 3.4.5 Control flow

```text
entry gate (green ∧ mutation ≥ θ)  ── fail ─► exit, receipt records why
        │
   MEASURE  (MetricCollectorPort fan-out — deterministic, no LLM)
        │
   no findings ─► exit (zero model calls — the common case)
        │
   for each operator where applicable(snapshot):          [Strategy]
        propose candidates (ranked by predicted gain / cost)
        for each candidate, up to ORCH_REFINE_MAX_CANDIDATES:
            snapshot workspace                             [Memento]
            apply                                          [Command]
            re-run the SAME suite (tests hash-locked)
            re-measure
            acceptance chain:                              [Chain of Responsibility]
              suite green? → targeted metric improved? → no metric regressed?
              → mutation score held? → no new lint/type/security finding?
              → public API surface unchanged?
            accept  → commit to ledger, snapshot becomes the new baseline
            reject  → restore memento, record the rejection reason, continue
        │
   emit refinement receipt + events                        [Observer]
```

### 3.5 Production-readiness design (Phase 7)

#### 3.5.1 The organizing idea — one requirement, two faces

§2.8.3 identifies the structural cause of the gap: generators improvise partial answers to "is this production-grade?" and nothing checks the result. Emitting a Dockerfile and checking a Dockerfile are currently unrelated pieces of code in unrelated modules, which is exactly how [`cicd_generator.py:151`](orchestrator/generators/cicd_generator.py:151) can emit the correct `permissions:` block while the wired assembler emits none.

The design principle that removes this class of drift: **every requirement is declared once and owns both its probe and its remediation.**

```python
Requirement(
    id="ci.least_privilege_token",
    derives_from="GitHub hardening guidance / least privilege",
    archetypes={PYTHON_SERVICE, FULLSTACK, WEB_STATIC},
    level=ReadinessLevel.DEPLOYABLE,
    probe=probe_workflow_declares_read_only_token,     # verifies
    remediation=add_permissions_block,                 # fixes, deterministically
)
```

A requirement without a probe cannot be claimed. A requirement with a deterministic remediation is never merely reported — it is fixed and then re-probed. The rubric is the single source of truth; emitters become adapters that satisfy it rather than independent opinions about what a project needs.

#### 3.5.2 Paradigm — declarative rubric over imperative checklist

The rubric is **data plus pure predicates**, not a script. Requirements are frozen dataclasses in a Composite tree; probes are the only components permitted I/O, and they are ports. Consequences that matter:

- The full requirement set is enumerable and diff-able — "what does the orchestrator consider production-grade?" is answered by reading one tree, not by tracing seven generators.
- Scoring, archetype filtering, and level roll-up are pure functions, unit-testable with fabricated evidence and zero filesystem.
- Adding a requirement is a data change plus two functions, not a new branch inside an assembler.

This is the same functional-core / imperative-shell split adopted in §3.4.1, and it is what Contract 1 (domain purity) already forces.

#### 3.5.3 Pattern selection and rationale

| Concern | Pattern | Why this one, here |
|---|---|---|
| Requirement tree with roll-up scoring (category → requirement → probe) | **Composite** | A category's outcome is derived from its children by the same interface as a leaf. Already the repo's idiom for hierarchical policy — `BudgetHierarchy` in `cost.py` (Org → Team → Job) |
| "Is this requirement satisfied?" | **Specification** | Each requirement is a self-contained, composable predicate over `Evidence`. Combinable with and/or for conditional rules ("needs migrations **if** a datastore is declared") without nested conditionals in a service |
| Which requirements apply to which output | **Strategy**, keyed by `AppArchetype` | A CLI needs no `/health`; a static site needs no migrations. Archetype-scoped rule sets prevent the false-requirement noise that makes checklists get ignored |
| Deterministic auto-fixes | **Command**, over the Phase 6 ledger | Remediation is apply-and-verify-or-revert, identical mechanics to refinement candidates. Reuses `ledger.py` — no second rollback mechanism |
| Wrapping the orphan generators without rewriting them | **Adapter** (`ArtifactEmitterPort`) | `cicd_generator`, `docker_generator`, `database_generator`, `logging_generator` are competent and unwired. Wrapping is cheaper and lower-risk than rewriting, and it deletes the duplicate inline templates in `assembler.py` |
| Probe implementations (filesystem, parse, HTTP, container) | **Ports & Adapters** | `ProbePort` in `domain/`, adapters in `infrastructure/`. Contract 2 forbids the application layer from importing them |
| Gate participation | **Chain of Responsibility** | Readiness becomes one more `WORKSPACE`-scoped `VerificationCheck` (§3.3), not a parallel gating system |
| Report emission | **Observer / EventBus** | Same sink as testing and refinement telemetry (E-8) |

Deliberately **not** used: no rules engine or DSL (YAGNI — Python predicates are already declarative enough and type-checked); no plugin auto-discovery by filesystem scan (registration happens in `container.py`, explicit and greppable); no inheritance hierarchy of requirement subclasses (composition of predicate functions).

#### 3.5.4 Readiness levels

Levels are cumulative, and the gate threshold is set per archetype rather than globally — a demo static page and a payment service should not face the same bar.

| Level | Means | Representative requirements |
|---|---|---|
| **L0 — Prototype** | Runs on the author's machine | Imports resolve, tests pass, no hardcoded secrets |
| **L1 — Deployable** | Someone else can build and run it identically | Lock file, digest-pinned base image, `.dockerignore`, LICENSE, CI that can fail, least-privilege token |
| **L2 — Operable** | Survives contact with an orchestrator and an on-call engineer | Real `/health` + `/ready`, graceful shutdown, structured logs with correlation IDs, metrics, config fail-fast, migrations, runbook |
| **L3 — Hardened** | Withstands audit | SBOM, image scan clean, signed provenance, dependency licence compatibility, ASVS L1 controls, SLOs defined, load-tested |

Default gate thresholds: `PYTHON_SERVICE`/`FULLSTACK` → **L2**; `WEB_STATIC` → **L1**; `PYTHON_CLI`/`LIBRARY` → **L1**; L3 is opt-in per project. Thresholds live in `config/limits.json` and are covered by `scripts/check_config_drift.py`.

#### 3.5.5 Core types

```python
# orchestrator/domain/readiness.py — frozen, stdlib only (Contract 1)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, FrozenSet


class AppArchetype(str, Enum):
    PYTHON_SERVICE = "python_service"
    PYTHON_CLI = "python_cli"
    LIBRARY = "library"
    WEB_STATIC = "web_static"
    FULLSTACK = "fullstack"


class ReadinessLevel(IntEnum):
    PROTOTYPE = 0
    DEPLOYABLE = 1
    OPERABLE = 2
    HARDENED = 3


class ProbeResult(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"   # probe could not run — never counts as satisfied


@dataclass(frozen=True)
class Evidence:
    """What a probe observed. Reports cite evidence; they never assert bare verdicts."""
    probe: str
    detail: str
    location: str | None = None       # file:line, URL, or container id
    raw: str = ""                     # truncated observation


@dataclass(frozen=True)
class RequirementOutcome:
    requirement_id: str
    result: ProbeResult
    evidence: tuple[Evidence, ...] = ()
    remediable: bool = False


@dataclass(frozen=True)
class Requirement:
    """Specification object: declares its own probe and optional deterministic fix."""
    id: str
    title: str
    derives_from: str                                  # standard or practice cited
    level: ReadinessLevel
    archetypes: FrozenSet[AppArchetype]
    blocking: bool = True                              # False ⇒ advisory at this level


@dataclass(frozen=True)
class ReadinessReport:
    archetype: AppArchetype
    achieved_level: ReadinessLevel
    required_level: ReadinessLevel
    outcomes: tuple[RequirementOutcome, ...] = ()
    remediations_applied: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.achieved_level >= self.required_level

    @property
    def blocking_violations(self) -> tuple[RequirementOutcome, ...]:
        return tuple(o for o in self.outcomes if o.result is ProbeResult.VIOLATED)


# ── pure roll-up: a level is achieved only if every blocking requirement
#    at that level and all levels below it is satisfied ────────────────────────
def achieved_level(
    outcomes: tuple[RequirementOutcome, ...], rubric: "Rubric"
) -> ReadinessLevel: ...
```

```python
# orchestrator/domain/ports.py  (Phase 7 additions)
class ProbePort(Protocol):
    """Observes a workspace (or a booted instance of it) and returns evidence."""
    requirement_id: str
    async def probe(self, workspace: Workspace, ctx: "ProbeContext") -> RequirementOutcome: ...


class ArtifactEmitterPort(Protocol):
    """Writes production artifacts into a workspace. Idempotent by contract."""
    name: str
    def applies_to(self, archetype: AppArchetype) -> bool: ...
    async def emit(self, workspace: Workspace, profile: "AppProfile") -> list[str]: ...
```

#### 3.5.6 Control flow

```text
DETECT ARCHETYPE  (reuses appbuilder/detector.py — no second detector)
        │
EMIT     archetype → emitter set                        [Strategy + Adapter]
        │           (idempotent: re-emission of an existing correct artifact is a no-op)
        ▼
ASSESS   rubric.applicable(archetype) → probes fan out  [Composite + Specification]
        │           static probes → parse, never glob-and-hope
        │           live probes   → boot in the F-3 sandbox, HTTP-probe, SIGTERM, observe drain
        ▼
REMEDIATE  for each violated requirement with a deterministic fix:
        │      snapshot → apply Command → re-probe → keep or revert   [Command + Memento]
        │      (no LLM: these are template and config edits)
        ▼
RE-ASSESS  achieved_level recomputed from fresh outcomes
        │
        ├─ achieved ≥ required → PASS   → receipts + events           [Observer]
        └─ below required      → report blocking violations with cited evidence;
                                 gate fails in `enforce`, records only in `shadow`
```

Two rules keep the gate honest. `INDETERMINATE` never counts as satisfied — a probe that could not run is a failure to verify, not a pass. And remediation always re-probes: an applied fix is not a satisfied requirement until observed as one.

---

## 4. Task Breakdown Structure (WBS)

Each item follows the mandated template. Effort in engineer-days. All work is TDD (Rule 3): failing test first, minimal implementation, full-suite regression, commit.

---

### Phase 0 — Baseline & Characterization (1.5 d)

#### B-1 — Characterization tests for current test-execution behavior
**Objective.** Lock present behavior before refactoring, so consolidation regressions are detectable.
**Affected components.** `tests/unit/test_testing_baseline.py` (new).
**Design changes.** None — pure test addition.
**Tasks.** Characterize `TestFirstGenerator._parse_pytest_output` across 6 real pytest output shapes (all pass, mixed, collection error, no tests collected, error in fixture, timeout kill). Characterize `quality_control.TestRunner._run_pytest` JSON and fallback paths.
**Testing strategy.** `@pytest.mark.unit`; fixtures are captured stdout strings, no subprocess.
**Acceptance criteria.** ≥ 12 characterization assertions green; module registered in marker check.
**Rollback.** Delete file — no production code touched.

#### B-2 — Failing tests for every registered defect
**Objective.** Every F-item has a red test before its fix (Rule 3).
**Affected components.** `tests/unit/test_testing_defects.py` (new).
**Tasks.** `xfail(strict=True)` tests for D-1 (event loop not blocked), D-4 (failing suite scores below 0.7), D-6 (`run_project_tests` does not silently return `[]`), D-7 (`total=0` is not success). Strict xfail means these turn into CI failures the moment they're fixed without being retired — the promotion signal.
**Acceptance criteria.** 4+ strict-xfail tests; `pytest -m unit` green.
**Rollback.** Delete file.

#### B-3 — Duplication inventory and shim plan
**Objective.** Enumerate every duplicated testing module with authoritative-copy decisions recorded.
**Affected components.** `docs/testing_consolidation.md` (new).
**Tasks.** Run `scripts/find_duplicates.py`; diff root vs package copies of `quality_control.py`, `output_organizer.py`, `pre_submission_testing.py`, `autonomous_debugger.py`; record which copy is canonical and whether the copies have diverged.
**Acceptance criteria.** Table of 4+ pairs with canonical designation and diff summary; any divergence explicitly resolved before Phase 1 begins.
**Rollback.** Documentation only.

---

### Phase 1 — Security & Consolidation (4 d)

#### F-1 — Unblock the async test path
**Objective.** Test execution must not stall the event loop.
**Affected components.** [`testing/first_generator.py`](orchestrator/testing/first_generator.py) (`_run_pytest_locally`, `_run_npm_tests_locally`, `_run_go_tests_locally`, `_run_cargo_tests_locally`).
**Design changes.** Replace `subprocess.run(...)` with `asyncio.create_subprocess_exec` + `asyncio.wait_for`. On timeout: `process.kill()`, drain pipes, return a `SuiteReport` with `collection_errors=("timeout",)` — never leave a zombie (repo has `scripts/check_subprocess_cleanup.py` for exactly this class of bug; run it as part of acceptance).
**Implementation tasks.** (1) Extract a single `_exec_async(argv, cwd, timeout)` helper. (2) Convert 4 call sites. (3) Add explicit process-group kill on Windows (`CREATE_NEW_PROCESS_GROUP`) and POSIX (`start_new_session=True`) so child pytest workers die with the parent.
**Refactoring requirements.** The 4 runner methods collapse into one parameterized path — prerequisite for F-6.
**Testing strategy.** Unit test asserting a concurrent `asyncio.sleep(0.05)` task completes while a 2-second fake test command runs (proves non-blocking); timeout test asserting the child process is dead after kill.
**Acceptance criteria.** B-2's D-1 xfail flips to pass; `scripts/check_subprocess_cleanup.py` clean; no `subprocess.run` remains under `orchestrator/testing/`.
**Rollback.** Single-commit revert; behavior returns to blocking-but-working.

#### F-2 — Remove in-process `exec()` of model output *(security)*
**Objective.** Eliminate arbitrary code execution inside the orchestrator process.
**Affected components.** [`infrastructure/verification_checks.py:232`](orchestrator/infrastructure/verification_checks.py:232) `_make_build_check`.
**Design changes.** Replace `exec(compile(artifact), {})` with an out-of-process import check: write the artifact to a tempfile, run it under the subprocess sandbox with a 15-second timeout, a scrubbed environment (no `*_API_KEY`), and a temporary `HOME`/`CWD`. Failure classification (SyntaxError / ImportError / other) is derived from the child's exit code and captured traceback, preserving the current reason strings.
**Implementation tasks.** (1) New `_make_build_check` using `SandboxPort`. (2) Delete the `# nosec B102` suppression. (3) Scrub env allowlist: `PATH`, `SYSTEMROOT`, `TMPDIR`/`TEMP`, `LANG` only.
**Refactoring requirements.** Introduces the first `SandboxPort` consumer — implement `subprocess_sandbox.py` here.
**Testing strategy.** Unit: artifact containing `os.environ["OPENAI_API_KEY"]` at import time must not be able to read a value; artifact writing to `Path.home()` must not touch the real home; timing test for the 15 s bound. Security: bandit HIGH clean without suppression.
**Acceptance criteria.** No `exec(` on model-derived strings anywhere under `orchestrator/`; existing `test_verification_checks.py::_make_build_check` cases pass unmodified.
**Rollback.** Env flag `ORCH_BUILD_CHECK_MODE=inprocess|subprocess` defaulting to `subprocess`; revert = flip flag. Flag is removed at the end of Phase 2 (no permanent dual path — YAGNI).

#### F-3 — Never execute model output unisolated *(security)*
**Objective.** Close the `sandbox=None` host-execution path.
**Affected components.** [`task_executor.py:271`](orchestrator/application/task_executor.py:271), new `infrastructure/sandboxes/`, [`container.py`](orchestrator/engine_core/container.py).
**Design changes.** Sandbox tier ladder resolved at composition time, not at the call site:

| Tier | Selected when | Protections |
|---|---|---|
| `DOCKER` | Docker daemon reachable and `ORCH_SANDBOX_TIER != subprocess` | `--network=none --read-only --cap-drop=ALL --pids-limit=256 --memory=512m`, tmpfs workdir, non-root user |
| `SUBPROCESS` | Docker unavailable | `RLIMIT_AS`, `RLIMIT_NPROC`, `RLIMIT_FSIZE` (POSIX), Job Object memory cap (Windows), scrubbed env, temp `HOME`, no inherited fds, timeout + process-group kill |

Selection is logged once at startup and stamped on every `SuiteReport.isolation`. There is no `NONE` tier: if neither backend initializes, `TestingService` raises rather than executing on the host.
**Implementation tasks.** (1) `subprocess_sandbox.py` with per-platform limits. (2) `docker_sandbox.py` with image pinned by digest, workspace bind-mounted read-only plus a writable tmpfs overlay. (3) Tier resolver in `sandboxes/__init__.py`. (4) `container.py` constructs the sandbox and injects it; `task_executor` stops passing `sandbox=None`.
**Refactoring requirements.** `TestFirstGenerator.__init__(sandbox=...)` becomes required (keyword-only, no default) — a deliberate breaking change inside the package, guarded by the fact it has one call site.
**Testing strategy.** Unit with a fake sandbox asserting the runner never shells out directly. Integration (`@pytest.mark.integration`, skipped without Docker) asserting a test that attempts `socket.create_connection(("1.1.1.1", 80))` fails under `DOCKER` and that a fork bomb is contained under `SUBPROCESS`. Security review sign-off required (repo rule: security-sensitive code → `security-reviewer`).
**Acceptance criteria.** No code path constructs `TestFirstGenerator` without a sandbox; `IsolationLevel.NONE` unreachable; integration tests green where Docker is present, cleanly skipped where absent.
**Rollback.** `ORCH_SANDBOX_TIER=subprocess` forces the lower tier; full revert restores `sandbox=None` (documented as security-regressing and requiring explicit approval).

#### F-5 — Replace the silent `run_project_tests` stub
**Objective.** No caller may receive success-shaped emptiness.
**Affected components.** [`quality/run_tests.py`](orchestrator/quality/run_tests.py) and its callers.
**Design changes.** Delegate to `TestingService`. When no runner supports the detected framework, raise `TestRunnerUnavailableError` (new, `orchestrator/exceptions.py`) instead of returning `[]`. `print()` replaced with `logger`.
**Implementation tasks.** (1) Add exception. (2) Rewrite delegation. (3) Audit and update all callers to handle the exception explicitly.
**Testing strategy.** Unit asserting the raise; caller tests asserting the error surfaces rather than being swallowed.
**Acceptance criteria.** B-2's D-6 xfail flips; no `print(` in `orchestrator/quality/`.
**Rollback.** Single-commit revert.

#### F-6 — Consolidate four runners into one port + adapters
**Objective.** One authoritative execution path.
**Affected components.** `runtime/sandbox.py`, `quality/quality_control.py`, `quality_control.py`, `testing/first_generator.py`, new `infrastructure/test_runners/`.
**Design changes.** Abstract-factory registry keyed by framework, returning `TestExecutorPort` implementations. `TestFirstGenerator` stops owning execution and consumes the port. Deleted `TestRunner` classes become shims that emit `DeprecationWarning` and delegate (the pattern already established by [`test_fixer.py`](orchestrator/test_fixer.py)).
**Implementation tasks.** (1) `base.py` with shared timeout/kill/decode logic. (2) Four adapters. (3) Registry with `supports()` dispatch. (4) Convert `TestFirstGenerator` to injection. (5) Shim the old classes. (6) Update `container.py` wiring.
**Refactoring requirements.** ~400 lines net deleted. Watch Contract 2: `application/testing/service.py` may import the *port*, never `infrastructure.test_runners`.
**Testing strategy.** B-1 characterization tests must pass against the new adapters unchanged — the consolidation's proof of behavioral equivalence. Contract test asserting exactly one non-shim `class TestRunner` remains in the tree.
**Acceptance criteria.** `lint-imports` green; characterization suite green; duplicate-class contract test green; coverage on `infrastructure/test_runners/` ≥ 80%.
**Rollback.** Shims mean old import paths keep working; revert of the adapter package restores previous behavior in one commit.

#### F-7 — Machine-readable result parsing only
**Objective.** Eliminate regex parsing of human output and the `total=0 ⇒ success` false positive.
**Affected components.** `infrastructure/test_runners/*`.
**Design changes.** `pytest --json-report --json-report-file=<tmp>` (file, not stdout — stdout mixes with captured test output), `jest --json --outputFile`, `go test -json`, `cargo test --format json` with a documented fallback. `SuiteReport.passed` requires `exit_code == 0 and not is_vacuous_result and not collection_errors`. Regex parsing survives only as an explicitly-labeled degraded path when the JSON plugin is absent, and that path sets `passed=False` if it cannot determine counts.
**Implementation tasks.** (1) JSON parsing per adapter. (2) `pytest-json-report` added to the `dev` extra and to generated-project scaffolds. (3) Degraded-path labeling on the report.
**Testing strategy.** Fixture-driven: real captured JSON reports for pass/fail/collection-error/empty cases per framework.
**Acceptance criteria.** B-2's D-7 xfail flips; no `re.search` against test output in non-degraded paths.
**Rollback.** Degraded path already exists as fallback; revert is low-risk.

---

### Phase 2 — Workspace & Gate Integration (5 d)

#### E-1 — `Workspace` abstraction and multi-file materialization
**Objective.** Lift the single-file ceiling (D-8) and give tests a real project tree.
**Affected components.** `domain/testing_models.py`, `application/testing/service.py`, `infrastructure/test_runners/*`.
**Design changes.** `WorkspaceBuilder` materializes: source files at their declared `target_path` (preserving package structure), test files under `tests/`, a minimal manifest (`pyproject.toml` / `package.json` / `go.mod` / `Cargo.toml`), and `conftest.py` inserting the workspace root on `sys.path` — removing the need to rewrite imports to `main`. Dependency resolution reuses the existing `dependency_resolver` already held by `task_executor`.
**Implementation tasks.** (1) `Workspace` + builder. (2) Import-rewriting logic deleted (`_fix_test_imports`), replaced by real package layout. (3) Prior-task artifacts from `results` seeded into the workspace so cross-file tasks can be tested together. (4) Workspace lifecycle: created under the OS temp root, removed on completion, retained on failure when `ORCH_KEEP_FAILED_WORKSPACES=1` (debuggability).
**Refactoring requirements.** `TestFirstGenerator` Phases 3–4 operate on a `Workspace`, not on two strings.
**Testing strategy.** Unit: two-module task where `b.py` imports `a.py` and tests exercise both — impossible today. Unit: workspace cleanup on both success and exception paths.
**Acceptance criteria.** Multi-file test passes; `_fix_test_imports` deleted; no regression in single-file characterization tests.
**Rollback.** `ORCH_WORKSPACE_MODE=single|multi`, default `multi` after a one-release soak.

#### E-6 — Wire test execution into `VerificationGate`
**Objective.** Make test results reach the deterministic score floor.
**Affected components.** `application/verification_gate.py`, `infrastructure/verification_checks.py`, `container.py`.
**Design changes.** `CheckScope` discriminator (§3.3); `_make_test_execution_check()` with `scope=WORKSPACE`; policy entry added at [`container.py:634`](orchestrator/engine_core/container.py:634):

```python
checks={
    "syntax": CheckOutcome.REQUIRED,
    "security": CheckOutcome.REQUIRED,
    "test_execution": CheckOutcome.REQUIRED,   # new
    "build": CheckOutcome.RECOMMENDED,
    "lint": CheckOutcome.RECOMMENDED,
    "type_check": CheckOutcome.RECOMMENDED,
}
```

Rollout is tri-state via `ORCH_TEST_GATE=off|shadow|enforce`. **`shadow` is the default for one release**: the check runs, receipts are recorded and logged, but the outcome does not affect the score. This mirrors the repo's existing HARD/SOFT/MONITOR policy vocabulary and produces the data needed to justify `enforce`.
**Implementation tasks.** (1) `CheckScope` + gate dispatch + `NOT_RUN` receipts when no workspace. (2) Check factory. (3) Policy + flag in container. (4) `EvaluatorService` passes the workspace through when the task carries one.
**Refactoring requirements.** None to existing checks — `scope` defaults to `ARTIFACT`.
**Testing strategy.** All 8 existing gate/verification test modules must pass **unmodified** (hard acceptance criterion). New: workspace-absent ⇒ `NOT_RUN`; workspace-present-failing ⇒ score ≤ 0.15 in `enforce`, unchanged score in `shadow`.
**Acceptance criteria.** Above, plus telemetry shows `test_execution` receipts on ≥ 95% of code-generation tasks in shadow mode.
**Rollback.** `ORCH_TEST_GATE=off` — one env var, no redeploy.

#### F-4 — Honest scoring for failing suites
**Objective.** A failing suite must not produce a shippable score.
**Affected components.** [`task_executor.py:295`](orchestrator/application/task_executor.py:295), `PreSubmissionTester`.
**Design changes.** Remove the hardcoded 0.8. Score derives from the gate: failing tests ⇒ `FAIL_SCORE_FLOOR` (0.15). `tests_run == 0` ⇒ score is not raised by TDD at all; the task falls through to standard generation + LLM evaluation with a receipt recording why. `PreSubmissionTester` gains a real `tests_passed` check sourced from the `SuiteReport` rather than inferred.
**Implementation tasks.** (1) Delete the literal. (2) Thread `SuiteReport` into `TaskResult` (new optional field on the result payload — `models.py` stays pure data, so this is a dataclass field, not behavior). (3) `PreSubmissionTester` check #5.
**Testing strategy.** Unit: failing suite ⇒ score < any default `acceptance_threshold`. Unit: zero-tests-run ⇒ no artificial score inflation. Regression: existing `task_executor` tests updated with justification in the commit message.
**Acceptance criteria.** B-2's D-4 xfail flips; no numeric score literals remain in `_try_tdd_generation`.
**Rollback.** Coupled to `ORCH_TEST_GATE`; in `shadow` the old scoring is retained, making this reversible without a code change.

#### F-8 — Configuration coherence
**Objective.** Remove the 3-vs-5 iteration mismatch and centralize testing knobs.
**Affected components.** `task_executor.py`, `first_generator.py`, `config/limits.json`.
**Design changes.** Single source of truth in `config/limits.json` (`testing.max_repair_iterations`, `testing.suite_timeout_s`, `testing.mutation_sample_size`), read through the existing config loader. Defaults are documented, not duplicated.
**Testing strategy.** Contract test asserting no testing limit is hardcoded at more than one site.
**Acceptance criteria.** `scripts/check_config_drift.py` extended to cover the new keys and green.
**Rollback.** Trivial revert.

---

### Phase 3 — Oracle Validity & Repair Hardening (5 d)

#### E-2 — RED-gate and assertion floor *(highest correctness value)*
**Objective.** Prove the suite is non-vacuous before trusting it.
**Affected components.** `application/testing/suite_validator.py` (new), `first_generator.py` Phase 1→2 boundary.
**Design changes.** Two mechanical gates between suite generation and implementation generation:

1. **RED-gate.** Materialize the workspace with a *stub* implementation (module exists; every public symbol raises `NotImplementedError`, generated by AST from the test file's referenced symbols). Run the suite. Any test that **passes** against the stub is vacuous — it asserts nothing about the implementation. Discard it. If more than 50% of tests are vacuous, regenerate the suite once with an explicit corrective prompt; if still vacuous, fail the task rather than proceeding with a decorative oracle.
2. **Assertion floor.** Pure-AST, zero-cost: every `test_*` function must contain ≥ 1 `assert` (or framework equivalent), and at least one asserted expression must reference a symbol imported from the implementation module. Rejects `assert True`, bare-call tests, and `assert x is not None`-only suites.

**Implementation tasks.** (1) Stub synthesizer (AST). (2) RED execution + per-test vacuity classification. (3) AST assertion analyzer (framework-parameterized: `assert` / `expect(` / `t.Error` / `assert_eq!`). (4) Corrective regeneration path, capped at one retry. (5) Vacuity/assertion statistics on `SuiteReport`.
**Refactoring requirements.** Inserts a phase into `generate_with_tests`; the phase numbering in logs and docstrings updates accordingly.
**Testing strategy.** Unit: each degenerate suite form is rejected (`assert True`, no-assertion call, `is not None`-only, empty body). Unit: a genuine suite survives with zero tests discarded. Unit: stub synthesis for classes, functions, async functions, and constants. Cost test: RED-gate adds one test execution and zero LLM calls in the happy path.
**Acceptance criteria.** All four degenerate forms rejected; no false-positive discards on a 20-test golden suite; measured token cost delta ≈ 0 for valid suites.
**Rollback.** `ORCH_RED_GATE=off|warn|enforce`, default `warn` for one release (logs vacuity rate without discarding).

#### E-3 — Repair-loop hardening
**Objective.** Bounded, escalating, non-cheating repair.
**Affected components.** `application/testing/repair_policy.py` (new), `first_generator._repair_to_pass_tests`.
**Design changes.**
- **Immutability lock (formalize existing behavior).** Test file content is hashed after the RED-gate; the hash is asserted before and after every repair iteration. Any mismatch raises — the agent cannot weaken its own oracle. Test changes require a separate arbiter call that must cite a specification clause; arbiter-approved changes are logged as first-class events.
- **Plateau detection.** Normalize each failure signature (`exception type` + `test node_id` + first traceback frame). Two consecutive identical signatures ⇒ stop repairing at this model tier.
- **Escalation.** On plateau, escalate one tier via the existing `FALLBACK_CHAIN` / model-cascade rather than retrying the same model. One escalation maximum, then fail with a full diagnostic artifact.
- **Collection errors stay excluded** from repair (existing behavior at [`:497`](orchestrator/testing/first_generator.py:497) — preserved and given a test).

**Implementation tasks.** (1) Signature normalizer. (2) Hash lock. (3) Escalation hook into the cascade. (4) Structured failure artifact (`SuiteReport` + last diff + signature history).
**Testing strategy.** Unit: repeated identical failure stops at iteration 2, not 5. Unit: a repair response that modifies the test file is rejected. Unit: escalation invoked exactly once. Cost test: worst-case LLM calls per task bounded and asserted.
**Acceptance criteria.** Worst-case repair cost per task provably ≤ (2 × tier₁ + 2 × tier₂) calls; hash-lock violation covered by test.
**Rollback.** `ORCH_REPAIR_POLICY=legacy|hardened`.

#### E-5 — Determinism and flake quarantine
**Objective.** Ensure the oracle is reproducible, and prevent the agent from "fixing" nondeterminism by weakening assertions.
**Affected components.** `infrastructure/sandboxes/*`, `application/testing/service.py`.
**Design changes.** Pinned execution environment: `PYTHONHASHSEED=0`, `TZ=UTC`, `LC_ALL=C.UTF-8`, `SOURCE_DATE_EPOCH`, `-p no:randomly`, network denied (Docker tier). Failures are re-run **once**; pass-on-rerun ⇒ `flaky_node_ids`, excluded from the repair loop, surfaced in the report and telemetry. Three flake observations for the same node id across runs ⇒ quarantine recorded in the project state.
**Implementation tasks.** (1) Env pinning in both sandbox tiers. (2) Re-run policy in the service. (3) Flake accounting persisted through `StateManager`.
**Testing strategy.** Unit: a deliberately nondeterministic test (`random.random() > 0.5`) is classified flaky, not failing, and is not sent to repair. Unit: env pins present in the child process.
**Acceptance criteria.** Flaky tests never reach the repair loop; re-run adds at most one execution per failing suite.
**Rollback.** `ORCH_FLAKE_RERUN=0`.

---

### Phase 4 — Quality Measurement & Observability (3 d)

#### E-4 — Mutation-score gate (sampled, budget-capped)
**Objective.** Measure whether the suite can actually detect defects — the metric line coverage cannot provide.
**Affected components.** `infrastructure/test_runners/mutation.py` (new), `application/testing/service.py`.
**Design changes.** Sampled mutation testing rather than exhaustive: generate *N* mutants (default 12, from `config/limits.json`) via AST operators — comparison flip, boundary shift, arithmetic swap, constant replacement, `return None` injection — restricted to lines the suite actually covers. Score = killed / total. Below threshold (default 0.6) the suite is judged weak: **the implementation is frozen** and one test-strengthening call is made citing the surviving mutants. Wall-clock is bounded by running mutants concurrently under the existing semaphore and capping total mutation time at 60 s.
**Implementation tasks.** (1) AST mutation operators. (2) Coverage-restricted mutant selection (reuses E-7's coverage data when present, falls back to all lines). (3) Concurrent execution + scoring. (4) Test-strengthening prompt path. (5) `SuiteReport.mutation_score` populated.
**Testing strategy.** Unit: a strong suite scores ≥ 0.8 on a known implementation; a vacuous suite scores ≈ 0.0. Unit: mutation respects the time cap. Cost test: zero extra LLM calls when the threshold is met.
**Acceptance criteria.** Mutation score recorded on every code-generation `SuiteReport`; gate configurable and defaulting to **report-only** until a full release of observed data justifies enforcement.
**Rollback.** `ORCH_MUTATION=off|report|enforce`, default `report`.

#### E-8 — Observability for the testing subsystem
**Objective.** Make oracle behavior measurable rather than anecdotal.
**Affected components.** `events/`, telemetry sinks, `verification_gate` receipts.
**Design changes.** Emit structured events: `test.suite.generated` (count, vacuity rate), `test.suite.executed` (passed/failed/skipped, duration, isolation level), `test.repair.iteration` (signature, tier, cost), `test.mutation.scored`. `ExecutionReceipt` for `test_execution` carries `isolation`, `executed`, `mutation_score`, `flaky_count`. Metrics: suite pass rate, mean repair iterations, escalation rate, flake rate, mutation-score distribution, cost per verified task.
**Implementation tasks.** (1) Event definitions. (2) Emission points. (3) Receipt field extension. (4) Dashboard panel additions.
**Testing strategy.** Unit: each event emitted exactly once per lifecycle stage with a fake bus. No assertions on sink internals.
**Acceptance criteria.** A single task run produces a complete, ordered event trace; no PII or artifact content in event payloads (only hashes and counts).
**Rollback.** Events are additive; disable at the sink.

---

### Phase 5 — Optimization, deferred (2 d)

#### E-7 — Impact-based test selection
**Objective.** Reduce regression latency on large generated projects.
**Affected components.** `application/testing/selection.py` (new).
**Design changes.** `pytest --cov-context=test` builds a persisted line→test map in the workspace. On modification, select tests touching changed lines. **Confidence rules:** map missing, stale, or covering < 80% of changed lines ⇒ full suite. Full suite always runs before final delivery, unconditionally. Selection is a latency optimization and never a correctness claim.
**Implementation tasks.** (1) Coverage-context capture. (2) Changed-line extraction from the diff. (3) Selection with fallback. (4) Mandatory full-suite delivery gate.
**Testing strategy.** Unit: stale map ⇒ full suite. Unit: delivery always runs the full suite regardless of selection. Benchmark: selection reduces mid-loop latency on a 200-test workspace without changing outcomes.
**Acceptance criteria.** Zero cases where selection changes a pass/fail outcome versus the full suite across a 50-run comparison.
**Rollback.** `ORCH_TEST_SELECTION=off` (the default until benchmarked).

---

### Phase 6 — Green-Anchored Refinement (4.5 d)

*Hard prerequisite: E-2 (RED-gate), E-3 (repair hardening), E-4 (mutation scoring), E-6 (gate wiring). Refinement without a validated oracle is a regression generator.*

#### E-9 — Measurement harness and refinement domain model (1.0 d)
**Objective.** Make code quality a measured, comparable quantity so that "improved" is a fact rather than a claim.
**Affected components.** `domain/refinement.py` (new), `domain/ports.py`, `infrastructure/metrics/` (new), `container.py`.
**Design changes.** `MetricSnapshot` as a frozen, comparable value object (§3.4.4). `MetricCollectorPort` with three adapters: `ast_metrics.py` (four `ast.NodeVisitor` passes over one parse — cyclomatic complexity, max nesting, longest function, duplicated-block detection via normalized subtree hashing), `dead_code.py` (vulture adapter, graceful absence handling), and later `bench_runner.py` (E-12). Collectors fan out concurrently and merge into one snapshot. The existing [`StaticAnalyzer`](orchestrator/quality/quality_control.py:370) metrics are *reused*, not re-derived — its `QualityMetrics` output maps into `MetricSnapshot` fields, finally giving that component a consumer.
**Implementation tasks.** (1) Domain types + pure comparison functions. (2) Port declaration. (3) AST collector with subtree-hash duplication detection. (4) Vulture adapter with availability probe. (5) Collector registry and concurrent fan-out. (6) Container wiring.
**Refactoring requirements.** None destructive. `StaticAnalyzer` gains an adapter, keeps its interface.
**Testing strategy.** `unit`: each visitor against hand-built fixtures with known complexity/nesting/length; duplication detection on near-miss pairs (same structure, different identifiers ⇒ detected; different structure ⇒ not). `unit`: comparison functions are total — every field pair, including `None` benchmark fields. `unit`: missing vulture degrades to `dead_symbols=None`, never crashes.
**Acceptance criteria.** Snapshot computed for a 20-file workspace in < 2 s; all comparison functions pure (no imports beyond stdlib in `domain/refinement.py`, asserted by a contract test); `lint-imports` green.
**Rollback.** Additive package — deleting it affects nothing else.

#### E-10 — Mechanical refinement tier, post-test (0.5 d)
**Objective.** Apply deterministic, zero-LLM improvements *after* the suite passes, with re-verification — closing the §2.7 ordering gap.
**Affected components.** `application/refinement/service.py`, `operators/dead_code.py`, `output_organizer._format_code`.
**Design changes.** Mechanical tier: unused-import removal, dead-symbol removal, and a second `ruff --fix`/`black` pass, each executed as a `Command` with a workspace `Memento` and followed by a full suite re-run. Crucially this does **not** replace the existing pre-test format pass — pre-test formatting stays (tests must validate formatted code); post-test mechanical refinement is additive and verified. Failure reverts and is reported rather than swallowed, unlike the current `except Exception: logger.warning` at [`output_organizer.py:231`](orchestrator/output_organizer.py:231).
**Implementation tasks.** (1) `Command` + `Memento` primitives in `ledger.py`. (2) Dead-code operator. (3) Formatter re-invocation as an operator. (4) Service loop for the mechanical tier only. (5) Receipt emission.
**Refactoring requirements.** `_format_code`'s blanket exception swallow narrowed: formatting failure still does not block delivery, but is recorded as a failed refinement candidate rather than a warning line.
**Testing strategy.** `unit`: dead-symbol removal that breaks a test is reverted and the workspace hash matches the pre-candidate snapshot exactly. `unit`: accepted candidate leaves the suite green and `dead_symbols` strictly lower. `integration`: full mechanical pass over a generated project.
**Acceptance criteria.** Zero LLM calls in this tier (asserted by a fake client that raises on call); byte-exact revert verified by hash; no regression in `output_organizer` integration tests.
**Rollback.** `ORCH_REFINE=off`.

#### E-11 — Structural refinement tier (2.0 d)
**Objective.** Reduce complexity, duplication, and nesting in code that already passes — the payoff the test suite was paid for.
**Affected components.** `application/refinement/operators/{deduplicate,extract_function,flatten_nesting}.py`, `acceptance.py`, `ledger.py`, `service.py`.
**Design changes.** Three `Strategy` operators, each gated by `applicable(snapshot)` so no model is called when the corresponding metric is already healthy (function ≤ 50 lines, nesting ≤ 4, no duplicated blocks — the thresholds the repo's own review standard uses). Each operator proposes candidates ranked by predicted gain; the service applies them one at a time through the acceptance chain of §3.4.5. Candidate prompts carry the *specific* measured finding ("function `handle()` at `svc.py:88` has cyclomatic complexity 19 and 5 nesting levels"), never a generic "improve this code" instruction — the difference between a targeted transform and a rewrite.
Budget: refinement draws from a dedicated sub-budget derived from `BudgetHierarchy`; exhaustion ends the tier cleanly with accepted candidates retained.
**Implementation tasks.** (1) Three operators. (2) Acceptance chain with six rules (§3.4.3 invariant 4 + API-surface freeze). (3) Public-API extractor (AST: module-level and class-level public symbols with signatures) for the freeze rule. (4) Candidate ranking. (5) Sub-budget enforcement. (6) Ledger persistence so accepted candidates survive a crash.
**Refactoring requirements.** Reuses the E-3 test hash lock verbatim — a candidate touching a test file is rejected pre-execution, not detected after.
**Testing strategy.** `unit`: each acceptance rule in isolation, including the case where the suite stays green but a non-targeted metric regresses ⇒ reject. `unit`: candidate that renames a public symbol ⇒ rejected by the API freeze. `unit`: candidate that edits a test file ⇒ rejected before execution. `unit`: budget exhaustion mid-tier retains prior acceptances. `integration`: a deliberately convoluted 120-line function is reduced and the suite stays green. `benchmark`: token cost per accepted candidate recorded.
**Acceptance criteria.** No accepted candidate ever changes suite outcome or public API; every rejection carries a reason string; a clean workspace produces zero model calls; ledger replay reconstructs the final state from snapshots.
**Rollback.** `ORCH_REFINE=mechanical` drops back to E-10 behavior; `off` disables entirely.

#### E-12 — Benchmark-gated performance tier (1.0 d)
**Objective.** Allow performance work only where improvement can be measured, and forbid it everywhere else.
**Affected components.** `infrastructure/metrics/bench_runner.py`, `operators/` (perf operator), `domain/ports.py` (`BenchmarkPort`).
**Design changes.** The tier is **inert unless the workspace ships a benchmark** (`tests/benchmarks/`, `pytest-benchmark`, or a declared `bench` script). No benchmark ⇒ tier skipped with a receipt stating so; guessing at optimizations without measurement is how correctness bugs enter under the banner of performance. When present: baseline measured over *n* runs (default 5), median compared, and a candidate is accepted only if the median improves beyond noise (default 5%) **and** every other invariant of §3.4.3 holds. Web outputs additionally track `bundle_bytes`.
**Implementation tasks.** (1) `BenchmarkPort` + adapter with warm-up runs and median (not mean — outlier resistance, matching the repo's existing median-aggregation decision in the evaluator). (2) Noise-floor estimation from baseline variance. (3) Perf operator with measured-hot-path context in the prompt. (4) Bundle-size collector for web projects.
**Testing strategy.** `unit`: absent benchmark ⇒ tier skipped, zero cost, receipt emitted. `unit`: improvement inside the noise floor ⇒ rejected. `unit`: median resists a single outlier run. `benchmark`: end-to-end on a workspace with a known-slow implementation.
**Acceptance criteria.** No performance candidate is ever accepted without a measurement; noise floor derived from observed variance, not hardcoded; tier adds zero cost to projects without benchmarks.
**Rollback.** `ORCH_REFINE=structural` excludes this tier; it is not part of the `full` default until benchmarked in production.

---

### Phase 7 — Production Readiness (14 d)

*Hard prerequisites: E-1 (Workspace — probes need a materialized tree), F-3 (sandbox tiers — live probes boot model-authored code), E-6 (gate wiring — readiness participates as a `WORKSPACE`-scoped check).*

Every item closes a gap from the §2.8.2 register. Item titles cite the gap IDs they retire.

#### P-1 — Readiness domain model and rubric engine (1.5 d) — *foundation*
**Objective.** Encode "production-grade" once, as an enumerable, archetype-scoped, evidence-backed rubric.
**Affected components.** `domain/readiness.py` (new), `domain/ports.py`, `application/readiness/rubric.py` (new), `config/limits.json`.
**Design changes.** Types of §3.5.5. `Rubric` as a Composite tree of categories and `Requirement` leaves; `applicable(archetype)` filters; `achieved_level()` rolls up with the rule that a level is achieved only when every blocking requirement at that level *and all lower levels* is `SATISFIED`. `INDETERMINATE` is never satisfying. Archetype detection **reuses** [`appbuilder/detector.py`](orchestrator/appbuilder/detector.py) — no second detector, no second source of truth about what kind of app this is.
**Implementation tasks.** (1) Domain types. (2) `Rubric` Composite with registration API. (3) Pure roll-up + scoring. (4) `ProbePort` / `ArtifactEmitterPort` declarations. (5) Level thresholds per archetype in `config/limits.json`. (6) Archetype adapter over the existing detector.
**Refactoring requirements.** None destructive; the detector gains a mapping function, keeps its interface.
**Testing strategy.** `unit`: roll-up correctness across the full matrix — a satisfied L2 requirement with a violated L1 requirement yields L0, not L2 (the classic error). `unit`: `INDETERMINATE` blocks level achievement. `unit`: archetype filtering excludes `/health` for `PYTHON_CLI`. `contract`: `domain/readiness.py` imports stdlib only; every registered requirement has a non-empty `derives_from`.
**Acceptance criteria.** Full rubric printable as a tree (`orchestrator readiness --explain`); scoring functions pure; `lint-imports` green.
**Rollback.** Additive package; nothing consumes it until P-12.

#### P-2 — Static probe adapters (1.0 d) — *retires G-1, G-2, G-4, G-14 detection*
**Objective.** Verify artifacts by parsing them, never by asserting a filename exists.
**Affected components.** `infrastructure/readiness_probes/static_probes.py`, `supply_chain_probes.py`, `license_probes.py`.
**Design changes.** Every static probe parses: `pyproject.toml` via `tomllib`, workflows via `yaml.safe_load`, Dockerfile via a small line-directive parser, `package.json` via `json`. A probe reports `INDETERMINATE` when a file is unparseable — distinct from `VIOLATED` — so a malformed workflow is never mistaken for a missing feature. Every outcome carries `Evidence` with `file:line`.
Supply-chain probes: lock file present *and* resolving (`pip install --dry-run --require-hashes` or `uv lock --check`); base image referenced by digest; GitHub Actions pinned to 40-hex SHAs; SBOM present and parseable.
**Implementation tasks.** (1) Parser helpers per format. (2) ~18 static probes across supply chain, CI, and compliance. (3) Evidence construction with locations. (4) Probe registry.
**Testing strategy.** `unit`: fixture workspaces — one satisfying, one violating, one malformed per probe; the malformed case must yield `INDETERMINATE`. `unit`: a probe never raises; failures degrade to `INDETERMINATE` with the exception text as evidence.
**Acceptance criteria.** Zero probes implemented by substring search on file content where a parser exists; all three fixture cases covered per probe.
**Rollback.** Additive.

#### P-3 — Live probe adapters (1.5 d) — *retires G-6, G-7, G-8 detection*
**Objective.** Prove the application actually serves, actually reports health honestly, and actually shuts down cleanly.
**Affected components.** `infrastructure/readiness_probes/live_probes.py`.
**Design changes.** Boot the workspace inside the F-3 sandbox (Docker tier preferred; subprocess tier permitted with the isolation level stamped on the evidence). Then:
- **Health honesty probe** — the defect at [`assembler.py:1737`](orchestrator/project_mgmt/assembler.py:1737) is that a health check can pass while the service is dead. The probe therefore asserts both directions: `/health` returns 200 while serving, **and** returns non-200 (or the container reports unhealthy) after the application's main listener is stopped. A check that cannot fail is not a check.
- **Readiness probe** — `/ready` must return non-200 while a declared dependency is unavailable, distinguishing it from liveness.
- **Graceful shutdown probe** — issue SIGTERM during an in-flight request; assert the request completes and the process exits within the grace period rather than being killed.
All live probes are time-boxed, and a boot failure is `VIOLATED` with the captured startup log as evidence — not `INDETERMINATE`, because failure to boot is a genuine verdict.
**Implementation tasks.** (1) Boot-and-wait helper with readiness polling and a hard timeout. (2) Three probes above. (3) Port allocation that cannot collide under concurrency. (4) Guaranteed teardown on every exit path.
**Testing strategy.** `integration` (Docker-skipped): fixture app with an honest health check passes; fixture app with an `import`-only health check **fails** the honesty probe. `integration`: app without SIGTERM handling fails the shutdown probe. `unit`: teardown runs on timeout, on boot failure, and on probe exception.
**Acceptance criteria.** No leaked containers or ports after a full run (asserted); honesty probe demonstrably rejects the orchestrator's own current template.
**Rollback.** `ORCH_READINESS_LIVE=0` reduces Phase 7 to static probes only.

#### P-4 — Supply-chain hardening emitters (1.0 d) — *retires G-1, G-2, G-17*
**Objective.** Make generated builds reproducible and attestable.
**Affected components.** `infrastructure/artifact_emitters/docker_emitter.py`, `project_mgmt/assembler.py` (template removal).
**Design changes.** Lock file generation (`uv lock`, falling back to `pip-compile --generate-hashes`) as part of dependency resolution; base images pinned by digest with the resolved tag recorded in a comment; SBOM emission (CycloneDX) as a build artifact; a container-scan step (`trivy`) wired into the generated CI as a **failing** job at HIGH severity. Wraps [`generators/docker_generator.py`](orchestrator/generators/docker_generator.py) through `ArtifactEmitterPort` rather than reimplementing it, and **deletes** the inline Dockerfile f-string at [`assembler.py:1674`](orchestrator/project_mgmt/assembler.py:1674) so one emitter owns the artifact.
**Implementation tasks.** (1) Lock-file resolution with tool detection. (2) Digest resolution at emit time with an offline fallback that marks the requirement `INDETERMINATE` rather than emitting a mutable tag silently. (3) SBOM emitter. (4) Trivy CI job. (5) Adapter + inline-template deletion.
**Refactoring requirements.** Dockerfile generation exists in three places today (assembler inline, `docker_generator`, [`verifier.py:263`](orchestrator/appbuilder/verifier.py:263) minimal fallback). Consolidate to one emitter with the other two as thin delegations — the same consolidation discipline as F-6.
**Testing strategy.** `unit`: emitted Dockerfile has a digest-pinned base; lock file present and hash-complete. `integration`: two builds of the same workspace produce identical image digests (reproducibility, the actual claim). `unit`: offline digest resolution degrades to `INDETERMINATE`, never to a mutable tag.
**Acceptance criteria.** Exactly one Dockerfile emitter in the tree; reproducibility test green; P-2 supply-chain probes pass on emitted output.
**Rollback.** Emitter registry entry removed; inline template restorable from one commit.

#### P-5 — CI enforcement via the wired emitter (0.5 d) — *retires G-3, G-4, G-5*
**Objective.** Make the generated pipeline capable of failing, and give it least privilege.
**Affected components.** `infrastructure/artifact_emitters/cicd_emitter.py`, `project_mgmt/assembler.py:1848` (template removal).
**Design changes.** Adopt [`generators/cicd_generator.py`](orchestrator/generators/cicd_generator.py) as the single CI emitter — it already emits `permissions: contents: read` ([:151](orchestrator/generators/cicd_generator.py:151)); it was simply never wired. Extend it with: `bandit` **without** `|| true`, a `pip-audit` job that fails on HIGH, `concurrency:` grouping, SHA-pinned actions, and `--cov-fail-under` inherited from the project's own `pyproject`. Delete the inline workflow template from `assembler.py`.
**Implementation tasks.** (1) Adapter. (2) Four workflow additions. (3) Inline-template deletion. (4) Update `_generate_makefile`'s "run safety manually" note ([:1642](orchestrator/project_mgmt/assembler.py:1642)) now that CI enforces it.
**Testing strategy.** `unit`: emitted workflow parses as YAML, contains a `permissions:` block whose scopes are read-only by default, and contains no `|| true` on any security step. `unit`: every `uses:` is SHA-pinned.
**Acceptance criteria.** P-2 CI probes pass on emitted output; the string `|| true` appears nowhere in an emitted workflow.
**Rollback.** Single commit; the old template is one revert away.

#### P-6 — Runtime operability emitters (2.0 d) — *retires G-6, G-7, G-8, G-9*
**Objective.** Emit applications that an orchestrator can run and an engineer can operate.
**Affected components.** `infrastructure/artifact_emitters/observability_emitter.py`, archetype-scoped service templates.
**Design changes.** For `PYTHON_SERVICE` and `FULLSTACK`:
- **Health endpoints.** `/health` (process liveness — no dependency calls, must be cheap) and `/ready` (dependency reachability, migration state). Docker `HEALTHCHECK` switched from `python -c "import pkg"` to an actual HTTP probe of `/health`.
- **Graceful shutdown.** SIGTERM handler, listener drain with a bounded grace period, in-flight completion, framework-appropriate lifespan hooks.
- **Observability.** Structured JSON logging (wraps [`generators/logging_generator.py`](orchestrator/generators/logging_generator.py), currently orphan), correlation-ID middleware propagating a request id into every log record, OpenTelemetry tracing with OTLP export configured by env, and a Prometheus `/metrics` endpoint with RED metrics.
- **Config fail-fast.** The existing Pydantic Settings layer gains startup validation that raises on missing required secrets rather than failing on first use.
- **Resilience defaults.** Request size caps, upstream timeouts, and rate limiting — ported in spirit from the orchestrator's own `resilience.py` and `rate_limiter.py`, which already solve these problems for this codebase.
Emitters are archetype-scoped: a `PYTHON_CLI` receives structured logging and config validation and nothing else.
**Implementation tasks.** (1) Health/ready templates per framework (FastAPI, Flask, Django). (2) Shutdown hooks. (3) Logging adapter + correlation middleware. (4) OTel + metrics wiring. (5) Config validation. (6) Resilience defaults. (7) Archetype scoping.
**Testing strategy.** `integration`: emitted service passes all three P-3 live probes. `integration`: a request in flight at SIGTERM completes. `unit`: correlation id present on every log line emitted during a request. `unit`: missing required env var raises at import/boot, not at first request. `unit`: CLI archetype receives no HTTP artifacts.
**Acceptance criteria.** Emitted services reach L2 on the operability category without manual edits; the emitted health check fails when the listener is stopped (the honesty test of P-3).
**Rollback.** Per-emitter registry entries; each is independently removable.

#### P-7 — Data layer emitters (1.5 d) — *retires G-10*
**Objective.** No generated application ships with an unmanaged schema.
**Affected components.** `infrastructure/artifact_emitters/database_emitter.py`, `requirements/data_layer.py`.
**Design changes.** Wire [`generators/database_generator.py`](orchestrator/generators/database_generator.py) — currently reachable only through the orphan `swiftstack_integration.py` — behind `ArtifactEmitterPort`. Emit Alembic scaffolding with an initial revision generated from the models, a `make migrate` target, seed-data support, connection-pool defaults, and a backup/restore section in the runbook. The data-layer requirements are **conditional Specifications**: they apply only when the workspace declares a datastore, so a stateless service is not penalized.
**Implementation tasks.** (1) Adapter. (2) Alembic scaffold + autogenerated initial revision. (3) Datastore detection (dependency + config inspection) feeding the conditional specification. (4) Pool defaults. (5) Seed hooks. (6) Runbook section.
**Testing strategy.** `integration`: emitted app with a model runs `alembic upgrade head` from empty to current inside the sandbox. `unit`: stateless app marks migration requirements `NOT_APPLICABLE`, not `VIOLATED`. `integration`: `/ready` returns non-200 when migrations are pending — ties P-6 and P-7 together.
**Acceptance criteria.** Any emitted app with a datastore has a runnable migration path proven in the sandbox; `swiftstack_integration.py` either becomes a thin delegation or is deleted.
**Rollback.** Registry entry removal.

#### P-8 — Verification depth (1.5 d) — *retires G-11, G-12*
**Objective.** Verify the product, not just the build — and stop installing model-authored manifests on the host.
**Affected components.** [`appbuilder/verifier.py`](orchestrator/appbuilder/verifier.py), `application/readiness/service.py`.
**Design changes.**
- **P-8a (security, ships first).** `pip install -r requirements.txt` and `npm install` at [`verifier.py:111`](orchestrator/appbuilder/verifier.py:111) execute arbitrary code from model-authored manifests on the host. Route both through the F-3 sandbox with no-network-except-registry, a scrubbed environment, and disk/time caps. This is the same defect class as F-3 at a second entry point and is not staged behind a flag.
- **P-8b.** Smoke test against a booted instance: exercise declared primary routes or the CLI's `--help` and one real invocation, asserting non-error responses.
- **P-8c.** Contract tests: emit an OpenAPI spec for service archetypes and assert the running app conforms (schemathesis or equivalent).
- **P-8d.** Web archetypes gain Lighthouse performance/a11y budgets as `HARDENED`-level requirements.
**Implementation tasks.** (1) Sandbox routing for both installers. (2) Smoke-probe with route discovery. (3) OpenAPI emission + conformance check. (4) Lighthouse budget runner.
**Testing strategy.** `integration`: a `requirements.txt` containing a package with a malicious `setup.py` cannot read `OPENAI_API_KEY` or write outside the sandbox. `integration`: smoke probe catches an app that boots but 500s on every route. `unit`: absent Lighthouse degrades to `INDETERMINATE`.
**Acceptance criteria.** No dependency installation path executes on the host; smoke failures block at `enforce`; `security-reviewer` sign-off on P-8a.
**Rollback.** P-8a is not rollback-eligible (security). P-8b–d gate behind `ORCH_READINESS_LIVE=0`.

#### P-9 — Web production pipeline (1.5 d) — *retires G-13*
**Objective.** Turn web output from a prototype into a deployable site.
**Affected components.** `infrastructure/artifact_emitters/web_build_emitter.py`, [`web_assembler.py`](orchestrator/web_assembler.py), `requirements/web_delivery.py`.
**Design changes.** Add a real build step: bundling and minification with content-hashed filenames, plus `robots.txt`, `sitemap.xml`, and a 404 page. Tighten the injected CSP — remove `'unsafe-inline'` for styles by extracting inline blocks (the assembler already extracts them for `style.css`, so the hard part is done), drop the cdnjs allowance in favor of vendored assets, and add SRI hashes for anything remaining remote. Emit a deploy config for the detected target (Nginx, Netlify, or Vercel) with cache headers matched to the hashed-asset strategy.
**Implementation tasks.** (1) Build emitter with a no-toolchain fallback that still hashes and minifies deterministically. (2) robots/sitemap/404. (3) CSP tightening + inline extraction. (4) SRI computation. (5) Deploy config + cache headers.
**Testing strategy.** `unit`: emitted CSP contains no `'unsafe-inline'` for `style-src`; every remote `<script>` carries `integrity`. `integration`: built site serves under a static server with no console errors and no CSP violations. `unit`: hashed filenames change when content changes and only then.
**Acceptance criteria.** Web archetype reaches L1 without manual edits; CSP violations measured at zero on the emitted site.
**Rollback.** `ORCH_WEB_BUILD=off` restores raw emission.

#### P-10 — Compliance artifacts (0.5 d) — *retires G-14*
**Objective.** Make the licence claim true and the repository legally coherent.
**Affected components.** `infrastructure/artifact_emitters/compliance_emitter.py`, `requirements/compliance.py`.
**Design changes.** Emit the `LICENSE` text matching the `pyproject` declaration ([`:857`](orchestrator/project_mgmt/assembler.py:857) currently declares MIT with no file), plus `SECURITY.md` with a disclosure contact, `CODEOWNERS`, and a `NOTICE` listing third-party attributions. Add a dependency-licence compatibility probe that flags copyleft dependencies in a permissively-licensed project.
**Implementation tasks.** (1) Licence text catalogue (MIT, Apache-2.0, BSD-3, AGPL) with year/author substitution. (2) `SECURITY.md`, `CODEOWNERS`, `NOTICE`. (3) Licence extraction from installed distributions. (4) Compatibility matrix + probe.
**Testing strategy.** `unit`: declared licence and emitted `LICENSE` always agree — mismatch is `VIOLATED`. `unit`: an AGPL dependency in an MIT project is flagged. `unit`: unknown licence ⇒ `INDETERMINATE`, never silently compatible.
**Acceptance criteria.** No emitted project declares a licence it does not ship; compatibility probe covers the top 20 dependency licences.
**Rollback.** Registry entry removal.

#### P-11 — Operational handover documentation (1.0 d) — *retires G-15*
**Objective.** Emit what an on-call engineer needs at 03:00, not only what a reader needs on day one.
**Affected components.** `infrastructure/artifact_emitters/` (docs), `assembler._generate_docs_structure`, `requirements/handover.py`.
**Design changes.** Extend the existing `docs/` scaffold with a runbook (start/stop, health interpretation, common failures with symptoms, rollback procedure, escalation), ADRs capturing the architecture decisions the orchestrator itself made during generation (it uniquely knows them — the advisor's `AppProfile` and the decomposer's rationale are already in state), a generated API reference from the P-8c OpenAPI spec, a deployment guide keyed to the emitted deploy config, and SLO definitions derived from the P-6 metrics. Probes verify **structure and non-emptiness with resolvable cross-references**, not prose quality — an unresolvable link in a runbook is a `VIOLATED` outcome; a boring sentence is not.
**Implementation tasks.** (1) Runbook template populated from the profile and emitted config. (2) ADR emission from generation state. (3) API reference from OpenAPI. (4) Deploy guide. (5) SLO stub from metric names. (6) Structural probes including link resolution.
**Testing strategy.** `unit`: runbook rollback section references the actual emitted deploy mechanism, not a generic placeholder. `unit`: every internal doc link resolves. `unit`: ADRs reference real decisions from state, not invented ones.
**Acceptance criteria.** Docs probes verify structure and links only; every emitted runbook command is one that exists in the emitted Makefile or deploy config.
**Rollback.** Registry entry removal.

#### P-12 — Gate integration and delivery honesty (0.5 d) — *retires G-16*
**Objective.** Make readiness decide delivery, and stop the pipeline from silently downgrading its own output.
**Affected components.** `infrastructure/verification_checks.py`, `container.py`, [`output_writer.py:325`](orchestrator/output_writer.py:325), `application/readiness/service.py`.
**Design changes.** `_make_readiness_check()` with `scope=WORKSPACE`, registered in the policy at `RECOMMENDED` initially and `REQUIRED` after the S7 soak — mechanically identical to E-6, no parallel gating path. Separately, the assembler's `except Exception → "Continuing with task files only"` is replaced with an explicit degraded-delivery outcome: the run still completes, but the report states that production assembly failed, the readiness level is recorded as `PROTOTYPE`, and the gate reflects it. A silent downgrade from "production project" to "a folder of files" is the same silent-failure family as F-5.
**Implementation tasks.** (1) Check factory. (2) Policy + flag wiring. (3) Degraded-delivery outcome type and reporting. (4) `ReadinessReport` attached to `TaskResult` and the run summary. (5) Events for E-8's sink.
**Testing strategy.** `unit`: assembly failure yields a reported degraded delivery, never a silent success. `unit`: below-threshold readiness fails the gate at `enforce` and records only at `shadow`. `unit`: all existing gate tests still pass unmodified.
**Acceptance criteria.** No swallowed assembly exception remains; every delivered project carries a `ReadinessReport` with an achieved level and cited evidence.
**Rollback.** `ORCH_READINESS=off`.

---

## 5. Risk & Mitigation Matrix

| ID | Risk | Likelihood | Impact | Mitigation | Owner phase |
|---|---|---|---|---|---|
| R-1 | Enforcing the test gate collapses task success rates (many generated suites are weak today) | High | High | Ship `ORCH_TEST_GATE=shadow` first; collect a full release of receipts; enforce only when shadow data shows the pass rate is acceptable | 2 |
| R-2 | Consolidating four runners breaks an undiscovered caller | Medium | High | B-1 characterization tests first; deprecation shims keep every old import path working; `lint-imports` + full suite per commit | 1 |
| R-3 | Docker unavailable on developer machines and Windows CI ⇒ integration tests unrunnable | High | Medium | Tier ladder with automatic `SUBPROCESS` fallback; Docker tests marked `integration` and skipped cleanly; isolation level stamped on every report so results are never silently conflated | 1 |
| R-4 | RED-gate discards legitimate tests (false-positive vacuity) | Medium | High | Default `warn` mode logging vacuity rate without discarding; 20-test golden suite as a regression fixture; one corrective regeneration before failing | 3 |
| R-5 | Mutation testing blows the wall-clock or token budget | Medium | Medium | Sampled (N=12) not exhaustive; coverage-restricted; 60 s hard cap; concurrent; `report` mode by default; zero LLM calls on pass | 4 |
| R-6 | Widening the gate signature breaks 8 existing test modules | Medium | Medium | `CheckScope` discriminator with `ARTIFACT` default — signature is additive; "existing tests pass unmodified" is an explicit E-6 acceptance criterion | 2 |
| R-7 | Sandbox escape or resource exhaustion from hostile generated code | Low | Critical | Two-tier isolation, no-network default, read-only mounts, rlimits/Job Objects, pids/memory caps, mandatory `security-reviewer` pass on F-2 and F-3 | 1 |
| R-8 | Coverage ratchet (`fail_under = 7`) fails as large new modules land | Medium | Low | New packages carry ≥ 80% coverage as an acceptance criterion; the ratchet moves up per phase, never down | all |
| R-9 | Import-linter contract violation from the new package layout | Medium | Medium | Port in `domain/`, service in `application/`, adapters in `infrastructure/`, wiring only in `container.py`; `lint-imports` runs pre-commit and in CI | 2 |
| R-10 | Workspace materialization leaks temp directories on crash | Medium | Low | Context-manager lifecycle with `finally` cleanup; retention only under an explicit env flag; cleanup asserted in tests | 2 |
| R-11 | Async conversion introduces zombie processes on Windows | Medium | Medium | Process-group creation + kill on both platforms; `scripts/check_subprocess_cleanup.py` in acceptance for F-1 | 1 |
| R-12 | Plan scope creep into the evaluator/self-improvement plan | Medium | Medium | Explicit non-goals (§1.5); the two plans interface only through `SuiteReport` and gate receipts | all |
| R-13 | Refinement changes behavior the suite does not cover — green checkmark over a real regression | Medium | **Critical** | Hard entry gate on mutation score (not merely green); acceptance requires mutation score to hold, not just tests to pass; public-API freeze; per-candidate revert; refinement is scheduled after Phase 3, never before | 6 |
| R-14 | Refinement burns tokens producing no measurable improvement | Medium | Medium | Measurement gates entry to every LLM tier — a clean workspace costs zero model calls; per-operator `applicable()` predicate; dedicated sub-budget from `BudgetHierarchy`; candidates ranked by predicted gain and capped by `ORCH_REFINE_MAX_CANDIDATES` | 6 |
| R-15 | Metric gaming — the model restructures to satisfy the measured field while making the code worse overall | Medium | High | Ratchet requires the targeted metric to improve **and** no other tracked metric to regress **and** mutation score to hold **and** no new lint/type/security finding; targeted prompts cite a specific measured defect rather than asking for general improvement | 6 |
| R-16 | Revert path leaves a partially-modified workspace | Low | High | Memento snapshot per candidate rather than inverse transforms; post-revert workspace hash asserted byte-equal in tests; ledger replay reconstructs state after a crash; accepted candidates are never invalidated by a later rejection | 6 |
| R-17 | Readiness rubric becomes checkbox theatre — requirements satisfied by file presence while the underlying property is absent (exactly today's `HEALTHCHECK` defect) | **High** | High | Every probe parses or executes; no probe is satisfied by a filename. The health probe asserts both directions (passes when serving, fails when stopped). `INDETERMINATE` never counts as satisfied. Contract test forbids substring-only probes where a parser exists | 7 |
| R-18 | Requirements that do not apply generate noise, and the report gets ignored | High | Medium | Archetype-scoped Strategy sets plus conditional Specifications (migrations only when a datastore is declared); `NOT_APPLICABLE` is a first-class outcome distinct from `VIOLATED`; per-archetype thresholds so a static page is not judged as a payment service | 7 |
| R-19 | Live probes leak containers, ports, or disk across a long run | Medium | Medium | Guaranteed teardown on every exit path including timeout and probe exception; collision-free port allocation; leak assertion in integration tests; `ORCH_READINESS_LIVE=0` disables the whole class | 7 |
| R-20 | Wiring the orphan generators regresses output that currently works | Medium | High | Adapters wrap rather than rewrite; the inline template is deleted only once its emitter passes the same probes; golden-output tests compare emitted artifacts before and after adoption; one generator adopted per commit | 7 |
| R-21 | Enforcing L2 makes most generated projects fail delivery | **High** | High | Same shadow-first discipline as E-6 — `ORCH_READINESS=shadow` for a full release, thresholds set from the observed level distribution rather than aspiration; auto-remediation applied before judgement, so the gate measures what remains after the system has fixed what it can | 7 |
| R-22 | Auto-remediation corrupts a working project while "fixing" it | Low | High | Remediations are deterministic template/config edits only — never LLM-authored; each is a Command over the Phase 6 ledger with snapshot-and-revert; every remediation is re-probed, and a remediation that does not flip its own probe is reverted | 7 |

**Architectural constraints carried throughout:** engine.py wires only (Rule 1); `models.py` stays pure data (Rule 2); TDD without exception (Rule 3); no new root-level modules (Rule 4); all 5 import contracts stay KEPT.

**Backward compatibility commitments:** every deleted class keeps a deprecation shim for one minor release; `VerificationGate.run(artifact)` keeps working with a single positional argument; every behavioral change ships behind an env flag defaulting to current behavior until its soak period ends.

---

## 6. Testing & Quality Assurance Strategy

### 6.1 Test taxonomy for this initiative

| Level | Marker | Scope | Runs |
|---|---|---|---|
| Unit | `unit` | Parsers, AST analyzers, policies, selection logic, refinement acceptance rules. No subprocess, no network. | Every commit |
| Contract | `contract` | Architecture invariants: one `TestRunner`, no `subprocess.run` under `orchestrator/testing/`, no `exec()` on model output, no hardcoded testing limits, `domain/refinement.py` imports stdlib only | Every commit (CI job exists) |
| Equivalence | `unit` | Refinement candidates: post-revert workspace byte-equal to snapshot; accepted candidate preserves public API surface | Every commit (Phase 6) |
| Probe fixtures | `unit` | Three fixture workspaces per readiness probe — satisfying, violating, malformed — with the malformed case asserting `INDETERMINATE` | Every commit (Phase 7) |
| Live readiness | `integration` | Boot, health honesty (both directions), readiness vs liveness, SIGTERM drain, smoke routes, migration path | Every PR, Docker-skipped where absent (Phase 7) |
| Golden output | `unit` | Emitted artifacts (Dockerfile, workflow, docs) compared against approved goldens when an emitter is adopted or changed | Every commit (Phase 7) |
| Integration | `integration` | Real pytest/jest execution in a real workspace; sandbox tiers | Every PR |
| Security | `integration` + review | Network denial, env scrubbing, filesystem containment, resource caps | Phase 1 and Phase 4 gates |
| Regression | `unit` | B-1 characterization suite — behavioral equivalence across consolidation | Every commit |
| Benchmark | `benchmark` | Wall-clock and token cost per verified task | Phase exit |

### 6.2 TDD protocol (mandatory)

Per Rule 3 and the repo's development workflow: write the failing test, verify it fails with the *expected* error (not an import error), implement minimally, run `pytest -m unit` then `-m integration`, commit with a detailed message. Defect fixes additionally require the corresponding B-2 strict-xfail to flip — that flip is the acceptance evidence, not a claim.

### 6.3 Coverage and quality gates

- New packages (`application/testing/`, `application/refinement/`, `application/readiness/`, `infrastructure/test_runners/`, `infrastructure/sandboxes/`, `infrastructure/metrics/`, `infrastructure/readiness_probes/`, `infrastructure/artifact_emitters/`) ≥ 80% line coverage — enforced per-PR by review, not by the global ratchet.
- Global `fail_under` raised in ~5% steps at each phase exit; never lowered.
- `black --check`, `ruff check`, `mypy` (core layers), `bandit -r` HIGH, `lint-imports`, `check_root_module_freeze.py`, `check_test_markers.py`, `check_config_drift.py`, `check_subprocess_cleanup.py` — all green before merge.
- No new entries in the mypy legacy-override list and no new ruff baseline ignores. New code is clean from the start.

### 6.4 Review standards

- Every PR: `code-reviewer` pass. CRITICAL/HIGH findings block.
- F-2, F-3, E-5, P-8a and any sandbox change: mandatory `security-reviewer` pass. This is non-negotiable per the repo's security-review triggers (external process execution, file system operations, untrusted input). P-8a additionally covers dependency resolution, which executes third-party install hooks.
- Architecture-affecting PRs (E-1, E-6): `architect` review against `docs/CODEBASE_MINDMAP.md`, plus a mindmap update in the same PR.

### 6.5 Definition of Done for the initiative

1. A generated artifact whose tests fail cannot receive a passing score.
2. A vacuous test suite cannot produce a passing verdict.
3. Model-generated code never executes on the host without an isolation boundary.
4. Exactly one authoritative test-execution path exists.
5. Test outcomes are parsed from machine-readable formats.
6. Repair cost per task is provably bounded and escalates rather than looping.
7. Flaky tests are quarantined, never repaired.
8. Every suite execution produces a structured, queryable receipt with its isolation level.
9. Code that passes is measurably improved, and every improvement is verified against the unchanged suite or reverted.
10. No refinement runs against an oracle that has not cleared the mutation threshold.
11. Every delivered application carries a readiness level backed by cited evidence, and no requirement is satisfiable by file presence alone.
12. Generated builds are reproducible, generated services are operable, and generated repositories are legally coherent.
13. The pipeline never silently downgrades its own output.
14. All architecture, type, lint, contract, and security gates pass.

---

## 7. Deployment & Rollback Plan

### 7.1 Rollout sequence

| Stage | Gate | Duration | Advance criterion |
|---|---|---|---|
| S1 — Internal | All flags at legacy defaults; new code present but inert | 1 release | Full suite green; no latency regression |
| S2 — Shadow | `ORCH_TEST_GATE=shadow`, `ORCH_RED_GATE=warn`, `ORCH_MUTATION=report` | 1 release | ≥ 95% of code-gen tasks produce `test_execution` receipts; vacuity and mutation distributions recorded |
| S3 — Enforce (tests) | `ORCH_TEST_GATE=enforce`, F-4 scoring live | 1 release | Task success rate within an agreed band of the shadow baseline |
| S4 — Enforce (validity) | `ORCH_RED_GATE=enforce` | 1 release | False-positive vacuity rate ≈ 0 on the golden suite |
| S5 — Enforce (mutation) | `ORCH_MUTATION=enforce` at a data-derived threshold | — | Threshold justified by the S2–S4 distribution, not guessed |
| S6 — Refinement | `ORCH_REFINE=mechanical` → `structural` → `full` | 1 release per step | Each step advances only when the prior step shows zero suite-outcome changes and a positive median metric delta across ≥ 50 tasks |
| S7 — Readiness (report) | `ORCH_READINESS=shadow`, emitters live, auto-remediation on | 1 release | Level distribution recorded across ≥ 50 generated projects; per-archetype thresholds chosen from that distribution, not from aspiration |
| S8 — Readiness (enforce) | `ORCH_READINESS=enforce` at the S7-derived thresholds | — | Post-remediation pass rate acceptable at the chosen level; blocking violations dominated by genuine defects rather than probe noise |

P-8a (sandboxed dependency installation) is a security fix and ships enforced with Phase 7's first commit, outside the S7/S8 staging.

Security fixes (F-2, F-3) are **not** staged — they ship enforced in Phase 1, with `ORCH_BUILD_CHECK_MODE` and `ORCH_SANDBOX_TIER` as escape hatches for environment problems only, not as opt-outs.

### 7.2 Flag inventory

| Flag | Values | Default at ship | Removal |
|---|---|---|---|
| `ORCH_TEST_GATE` | off / shadow / enforce | shadow | Retained (operational control) |
| `ORCH_RED_GATE` | off / warn / enforce | warn | Retained |
| `ORCH_MUTATION` | off / report / enforce | report | Retained |
| `ORCH_SANDBOX_TIER` | docker / subprocess | auto-detect | Retained |
| `ORCH_BUILD_CHECK_MODE` | inprocess / subprocess | subprocess | **Removed end of Phase 2** |
| `ORCH_WORKSPACE_MODE` | single / multi | multi | **Removed end of Phase 3** |
| `ORCH_REPAIR_POLICY` | legacy / hardened | hardened | **Removed end of Phase 4** |
| `ORCH_TEST_SELECTION` | off / on | off | Retained |
| `ORCH_FLAKE_RERUN` | 0 / 1 | 1 | Retained |
| `ORCH_KEEP_FAILED_WORKSPACES` | 0 / 1 | 0 | Retained (debug) |
| `ORCH_REFINE` | off / mechanical / structural / full | mechanical | Retained (operational control) |
| `ORCH_REFINE_MIN_MUTATION` | float 0.0–1.0 | 0.6 | Retained (safety threshold) |
| `ORCH_REFINE_MAX_CANDIDATES` | int | 8 | Retained (cost control) |
| `ORCH_READINESS` | off / shadow / enforce | shadow | Retained (operational control) |
| `ORCH_READINESS_LIVE` | 0 / 1 | 1 | Retained (disables boot/HTTP/SIGTERM probes) |
| `ORCH_READINESS_REMEDIATE` | 0 / 1 | 1 | Retained (deterministic auto-fixes) |
| `ORCH_WEB_BUILD` | off / on | on | **Removed end of Phase 7** (raw web emission is not a supported mode) |

Temporary flags have explicit removal phases — the repo already carries one confirmed dead flag (`use_provider_sorting`), and permanent dual code paths are how that happens. Each removal is a tracked task in its phase, and `scripts/flag_inventory.py` is the verification tool.

### 7.3 Rollback tiers

1. **Configuration rollback (seconds).** Flip the relevant env flag. Covers every behavioral change in Phases 2–5. No redeploy.
2. **Commit revert (minutes).** Each work item is one squashed commit with its tests. Deprecation shims mean reverting an adapter package does not break importers.
3. **Phase revert (hours).** Each phase is a merge commit on `master`; `git revert -m 1` restores the prior phase wholesale. Phase boundaries are chosen so that each is independently revertible — notably, E-6 (gate wiring) is separable from E-1 (workspace), so the gate can be withdrawn while keeping multi-file support.
4. **Data rollback.** None required — no schema migrations. `SuiteReport` and refinement-ledger fields on `TaskResult` are optional and additive; older state files load unchanged.
5. **Candidate rollback (Phase 6, sub-second).** Independent of the four tiers above: any single refinement candidate restores its workspace memento without affecting accepted candidates or requiring a redeploy. This is the innermost rollback loop and runs dozens of times per task by design.

### 7.4 Deployment checks per phase

Pre-merge: full CI matrix (Ubuntu/Windows × Python 3.11/3.12), `lint-imports`, root-module freeze, marker check, config-drift check, bandit HIGH, mypy core.
Post-merge: smoke run of `python -m orchestrator --project "..." --budget 0.5` end-to-end, with the resulting receipt trace inspected manually for the first run of each phase.

---

## 8. Post-Implementation Validation Checklist

### 8.1 Functional

- [ ] A task whose generated tests fail produces a score ≤ 0.15 and is not delivered.
- [ ] A task whose suite is vacuous (all tests pass against a `NotImplementedError` stub) is rejected or regenerated, never accepted.
- [ ] A multi-file task (module B importing module A) generates, executes, and passes tests across both files.
- [ ] Zero executed tests is reported as failure, never as success.
- [ ] Collection errors are reported as infrastructure problems and never enter the repair loop.
- [ ] A deliberately flaky test is quarantined and excluded from repair.
- [ ] A repair loop facing an identical failure signature twice escalates model tier exactly once, then fails with a diagnostic artifact.
- [ ] A repair response attempting to modify the test file is rejected by the hash lock.
- [ ] Refinement does not start when the mutation score is below threshold, and the receipt states why.
- [ ] A refinement candidate that breaks any test is reverted, and the workspace is byte-identical to its pre-candidate snapshot.
- [ ] A refinement candidate that improves its target metric while regressing another is rejected.
- [ ] A refinement candidate that renames, adds, or removes a public symbol is rejected by the API-surface freeze.
- [ ] A workspace with no measured findings completes Phase 6 with zero LLM calls.
- [ ] A workspace with no benchmark skips the performance tier entirely and records that fact.
- [ ] Accepted candidates survive the rejection of a later candidate.

### 8.2 Security

- [ ] No `exec()` or `eval()` is applied to model-derived strings anywhere under `orchestrator/`.
- [ ] Generated code executing `os.environ.get("OPENAI_API_KEY")` at import or test time receives no value.
- [ ] Generated code attempting an outbound connection fails under the Docker tier.
- [ ] Generated code attempting to write outside the workspace fails under both tiers.
- [ ] A fork bomb or memory bomb in generated code is contained and the run terminates within the timeout.
- [ ] Every `SuiteReport` carries a non-`NONE` isolation level.
- [ ] `bandit -r orchestrator/` reports zero HIGH findings with no new suppressions.

### 8.3 Architectural

- [ ] `lint-imports` — all 5 contracts KEPT.
- [ ] `scripts/check_root_module_freeze.py` — no new root modules.
- [ ] `application/testing/` and `application/refinement/` import no `orchestrator.infrastructure` symbol.
- [ ] `domain/testing_models.py` and `domain/refinement.py` import only the standard library.
- [ ] Refinement acceptance rules are pure functions — unit-testable with no subprocess, no filesystem, no clock.
- [ ] Refinement operators are Protocol implementations registered in `container.py`, not subclasses and not a module-level singleton registry.
- [ ] `engine.py` gained no logic — wiring only.
- [ ] Exactly one non-shim test-execution implementation exists in the tree.
- [ ] `docs/CODEBASE_MINDMAP.md` updated with the testing subsystem.

### 8.4 Quality and performance

- [ ] `pytest -m unit` and `-m integration` green on Ubuntu and Windows, Python 3.11 and 3.12.
- [ ] New packages ≥ 80% line coverage; global `fail_under` raised and holding.
- [ ] Test execution no longer blocks the event loop — a concurrency benchmark shows parallel task throughput scaling with the semaphore.
- [ ] No zombie processes after timeout kills (`scripts/check_subprocess_cleanup.py` clean).
- [ ] Worst-case LLM calls per verified task bounded and asserted by test.
- [ ] Mutation scoring completes within the 60-second cap on the benchmark workspace.

### 8.5 Observability and documentation

- [ ] A single task run emits the complete ordered event trace (`generated → executed → [repair…] → scored`).
- [ ] Dashboard shows suite pass rate, repair iterations, escalation rate, flake rate, mutation distribution, cost per verified task.
- [ ] Dashboard shows refinement acceptance rate, median metric delta per accepted candidate, rejection reasons by rule, and token cost per accepted candidate.
- [ ] No artifact content or secrets in event payloads — hashes and counts only.
- [ ] `CLAUDE.md`, `docs/CODEBASE_MINDMAP.md`, and `.env.example` updated with the new flags.
- [ ] Temporary flags (`ORCH_BUILD_CHECK_MODE`, `ORCH_WORKSPACE_MODE`, `ORCH_REPAIR_POLICY`, `ORCH_WEB_BUILD`) removed on schedule and verified absent by `scripts/flag_inventory.py`.

### 8.6 Production readiness (Phase 7)

**Reproducibility and supply chain**

- [ ] Two builds of the same generated workspace produce identical image digests.
- [ ] Every emitted Dockerfile pins its base image by digest; no mutable tag is ever emitted silently (offline resolution yields `INDETERMINATE`).
- [ ] Every emitted project ships a lock file whose resolution is verified, not assumed.
- [ ] Every emitted workflow pins actions by SHA; SBOM emitted and parseable; image scan runs as a failing job.

**CI that can fail**

- [ ] No emitted workflow contains `|| true` on a security step.
- [ ] Every emitted workflow declares a read-only default `permissions:` block.
- [ ] Dependency CVE audit runs in CI, not as a Makefile suggestion.

**Operability**

- [ ] The emitted health check **fails** when the application's listener is stopped — verified in both directions.
- [ ] `/ready` returns non-200 while a declared dependency is unavailable or migrations are pending.
- [ ] A request in flight when SIGTERM arrives completes, and the process exits within the grace period.
- [ ] Every log line emitted during a request carries a correlation id; traces and RED metrics are exported.
- [ ] A missing required secret fails at boot, not at first use.
- [ ] Any app with a datastore has a migration path proven runnable in the sandbox.

**Verification depth**

- [ ] No dependency installation executes on the host — `pip`/`npm` install runs sandboxed with a scrubbed environment.
- [ ] A malicious `setup.py` in a generated manifest cannot read provider API keys or write outside the sandbox.
- [ ] Smoke probes catch an application that boots but errors on every route.
- [ ] Service archetypes emit an OpenAPI spec the running app conforms to.

**Web delivery**

- [ ] Emitted CSP contains no `'unsafe-inline'` for `style-src`; remote scripts carry `integrity`.
- [ ] Built site serves with zero console errors and zero CSP violations.
- [ ] `robots.txt`, `sitemap.xml`, and a 404 page are present; asset filenames are content-hashed.

**Compliance and handover**

- [ ] Declared licence and emitted `LICENSE` always agree; copyleft dependencies in permissive projects are flagged.
- [ ] `SECURITY.md`, `CODEOWNERS`, and `NOTICE` present.
- [ ] Every runbook command exists in the emitted Makefile or deploy config; every internal doc link resolves.

**Gate integrity**

- [ ] `NOT_APPLICABLE` and `INDETERMINATE` are never counted as satisfied in level roll-up.
- [ ] No readiness requirement is satisfiable by file presence alone where a parser or probe is possible.
- [ ] Assembly failure produces a reported degraded delivery with readiness `PROTOTYPE` — never a silent success.
- [ ] Every delivered project carries a `ReadinessReport` with achieved level, required level, and cited evidence.
- [ ] No containers, ports, or temp directories leak after a full readiness run.

---

## Appendix A — Evidence Index

Every claim in §2 traces to a specific location:

| Claim | Location |
|---|---|
| Gate runs 5 checks, none executes tests | [`verification_checks.py:305`](orchestrator/infrastructure/verification_checks.py:305) `default_checks()` |
| Gate policy lists no test check | [`container.py:634`](orchestrator/engine_core/container.py:634) |
| Gate contract is string-scoped | [`verification_gate.py:128`](orchestrator/application/verification_gate.py:128) |
| Score floor exists and is 0.15 | [`verification_gate.py:116`](orchestrator/application/verification_gate.py:116) |
| Failing suite scores 0.8 | [`task_executor.py:295`](orchestrator/application/task_executor.py:295) |
| Sandbox is `None` at the only call site | [`task_executor.py:271`](orchestrator/application/task_executor.py:271) |
| `exec()` of model output in-process | [`verification_checks.py:242`](orchestrator/infrastructure/verification_checks.py:242) |
| Blocking `subprocess.run` in async | [`first_generator.py:951`](orchestrator/testing/first_generator.py:951) |
| Single-file constraint | [`first_generator.py:910`](orchestrator/testing/first_generator.py:910) |
| Tests immutable during repair (preserve) | [`first_generator.py:1500`](orchestrator/testing/first_generator.py:1500) |
| Collection errors excluded from repair (preserve) | [`first_generator.py:497`](orchestrator/testing/first_generator.py:497) |
| Regex output parsing; `total=0` not an error | [`runtime/sandbox.py:110`](orchestrator/runtime/sandbox.py:110) |
| Silent stub returning `[]` | [`quality/run_tests.py`](orchestrator/quality/run_tests.py) |
| `PreSubmissionTester` runs no tests | [`quality/pre_submission_testing.py:76`](orchestrator/quality/pre_submission_testing.py:76) |
| Iteration limit mismatch 5 vs 3 | [`first_generator.py:315`](orchestrator/testing/first_generator.py:315), [`task_executor.py:273`](orchestrator/application/task_executor.py:273) |
| Delivery pipeline order — format before tests, nothing after green | [`output_organizer.py:145`](orchestrator/output_organizer.py:145), [`:182`](orchestrator/output_organizer.py:182) |
| Formatting failure swallowed | [`output_organizer.py:231`](orchestrator/output_organizer.py:231) |
| Quality metrics computed but never acted on | [`quality/quality_control.py:587`](orchestrator/quality/quality_control.py:587), consumer at [`project_analyzer.py:629`](orchestrator/project_analyzer.py:629) |
| MAP-Elites scores variants heuristically, pre-test | [`map_elites.py:170`](orchestrator/engine_core/stages/map_elites.py:170) |
| Existing Strategy idiom for pluggable transforms | [`image_optimizer.py:133`](orchestrator/generators/image_optimizer.py:133) |
| Production scaffold generator (wired) | [`project_mgmt/assembler.py:136`](orchestrator/project_mgmt/assembler.py:136) |
| Unpinned dependency install in emitted Dockerfile | [`assembler.py:1700`](orchestrator/project_mgmt/assembler.py:1700) |
| Mutable base image tag | [`assembler.py:1680`](orchestrator/project_mgmt/assembler.py:1680) |
| Security scan cannot fail the build | [`assembler.py:1908`](orchestrator/project_mgmt/assembler.py:1908) |
| Correct `permissions:` block exists in the unwired generator | [`cicd_generator.py:151`](orchestrator/generators/cicd_generator.py:151) |
| Dependency audit deferred to manual invocation | [`assembler.py:1642`](orchestrator/project_mgmt/assembler.py:1642) |
| Vacuous health check | [`assembler.py:1737`](orchestrator/project_mgmt/assembler.py:1737) |
| Licence declared without a LICENSE file | [`assembler.py:857`](orchestrator/project_mgmt/assembler.py:857) |
| Host-side install of model-authored manifests | [`verifier.py:111`](orchestrator/appbuilder/verifier.py:111) |
| Third Dockerfile generator (fallback) | [`verifier.py:263`](orchestrator/appbuilder/verifier.py:263) |
| Silent degradation to "task files only" | [`output_writer.py:325`](orchestrator/output_writer.py:325) |
| Unbuilt web output | [`web_assembler.py:10`](orchestrator/web_assembler.py:10) |
| Orphan generator hub | [`swiftstack_integration.py`](orchestrator/integrations/swiftstack_integration.py) |
| Import contracts | [`.importlinter`](.importlinter) |
| Coverage ratchet | [`pyproject.toml:361`](pyproject.toml:361) |

## Appendix B — Effort Summary

| Phase | Items | Days | Cumulative |
|---|---|---|---|
| 0 — Baseline | B-1, B-2, B-3 | 1.5 | 1.5 |
| 1 — Security & consolidation | F-1, F-2, F-3, F-5, F-6, F-7 | 4.0 | 5.5 |
| 2 — Workspace & gate | E-1, E-6, F-4, F-8 | 5.0 | 10.5 |
| 3 — Validity & repair | E-2, E-3, E-5 | 5.0 | 15.5 |
| 4 — Measurement & observability | E-4, E-8 | 3.0 | 18.5 |
| 5 — Optimization (optional) | E-7 | 2.0 | 20.5 |
| 6 — Green-anchored refinement | E-9, E-10, E-11, E-12 | 4.5 | 25.0 |
| 7 — Production readiness | P-1 … P-12 | 14.0 | 39.0 |

**Recommended minimum viable slice:** Phases 0–2 (10.5 days). That alone closes both security defects, eliminates the silent-failure paths, unifies the four runners, and makes failing tests block delivery — the highest-value outcomes in the plan. Phase 3 is where the oracle becomes genuinely trustworthy, and should follow without a long gap.

**On sequencing Phase 6.** Phase 5 is optional and may be skipped; Phase 6 may not be *reordered*. Refinement is the only phase that modifies passing code, and its entire safety argument rests on the mutation-validated oracle built in Phases 3–4. Running it earlier would produce exactly the failure it is designed to prevent: confident, green, silently regressed output.

**On sequencing Phase 7.** Phase 7 is the largest phase and the one with the most externally visible payoff — it is what turns "the orchestrator generated an app" into "the orchestrator generated something deployable". It nonetheless depends on Phase 1 (sandbox tiers — live probes boot model-authored code) and Phase 2 (`Workspace` — probes need a real tree, and the gate needs `CheckScope`). Two items may be pulled forward if the schedule demands it, because both are pure defect fixes with no dependency on the rubric:

- **P-8a** (sandboxed dependency installation) — a security defect of the same class as F-3, at a second entry point. Ship it with Phase 1.
- **P-5** (CI enforcement via the already-written `cicd_generator`) — half a day, retires three gap items, and is mostly deletion of a duplicate template.

The rest of Phase 7 should run as a block, because the rubric (P-1) is what keeps the emitters honest and the emitters are what make the rubric passable. Splitting them yields either checks nothing can satisfy or artifacts nothing verifies — the two failure modes §2.8.3 already documents.
