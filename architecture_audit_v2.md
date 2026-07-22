# Architecture Audit — ARCH-AUDIT-V2

**Target:** WBS-1 Verification Subsystem  
**Scope:** `orchestrator/domain/verification.py`, `orchestrator/application/verification_gate.py`, `orchestrator/infrastructure/verification_checks.py`, `orchestrator/application/evaluator.py` (gate integration), `orchestrator/engine_core/container.py` (wiring), `orchestrator/operations/feedback.py` (deterministic field)  
**Epistemic Protocol:** EGFV — every claim tagged [VERIFIED] / [HYPOTHESIS] / [UNKNOWN] / [FALSE]  
**Date:** 2026-07-22  
**Commit:** `2d5af7e0`

---

## Phase 1: Architectural Fingerprinting

### DETECTED ARCHITECTURE: Hexagonal (Ports & Adapters) for Verification Wedge

Supporting evidence:

1. **Domain purity** — `orchestrator/domain/verification.py` imports only stdlib (`collections.abc.Mapping`, `dataclasses`, `enum`, `typing`). Zero imports from application, infrastructure, or engine layers. [VERIFIED]

2. **Application layer depends on domain protocols, not concrete infrastructure** — `orchestrator/application/verification_gate.py` imports only from `..domain.verification` (line 24). The `CheckFn` protocol (`Callable[[str], Awaitable[tuple[bool, str]]]`, line 31) is the abstraction boundary. [VERIFIED]

3. **Infrastructure adapters live in `orchestrator/infrastructure/` with explicit statement they must not be imported by application** — The module docstring at `verification_checks.py:8-9` states: "These live in infrastructure because they execute shell commands — application code should never import from this module directly." [VERIFIED]

4. **Composition root (`container.py`) wires infrastructure into application** — `ServiceContainer.build()` at container.py:619-672 imports infrastructure adapters and wraps them into application-layer `VerificationCheck` objects [VERIFIED]

5. **No cyclic dependencies exist between layers** — Domain → (nothing), Application → Domain, Infrastructure → (nothing, standalone), Container → Application + Domain + Infrastructure. Data flows one direction. [VERIFIED]

### Dependency Graph (actual, from imports):

```
Domain/verification.py → stdlib only
Application/verification_gate.py → Domain/verification.py
Application/evaluator.py → Application/verification_gate.py → Domain
Infrastructure/verification_checks.py → stdlib + subprocess only
EngineCore/container.py → Application + Domain + Infrastructure (composition root, exempt)
Operations/feedback.py → stdlib only (no verification module deps)
```

### Domain Boundary Candidates:
- **Verification domain** — `CheckOutcome`, `ExecutionReceipt`, `VerificationPolicy` form a cohesive cluster. All three model verification-specific concepts with no overlap outside `orchestrator.domain`.
- **Gate orchestration** — `VerificationGate`, `VerificationCheck`, `GateResult` form the application orchestrator layer.
- **Check implementation** — 5 check adapters are interchangeable and isolated; each is a self-contained concern.

### Entry Points:
- `VerificationGate.run(artifact)` — verification_gate.py:128
- `EvaluatorService.evaluate(task, output)` — evaluator.py:81
- `ServiceContainer.build(budget, ...)` — container.py:455

### Data Flow Topology:
- **Synchronous async** — all I/O is asyncio-based. None blocking.
- **Push topology** — gate pushes artifact to each check sequentially, collects results, returns aggregate. No queues, no backpressure.

### Configuration & Secrets:
- **None in this subsystem.** Check adapter commands are hardcoded. All configuration is injected via constructor parameters (verification_gate.py:120-126). No secrets handled.

---

## Phase 2: Compliance Matrix

| Module | Detected Pattern | Intended Pattern | Drift | Violations | Severity | Evidence |
|--------|-----------------|-----------------|-------|------------|----------|----------|
| Domain/verification.py | Pure domain types | Pure domain types | NONE | 0 | — | Zero infra imports. All frozen dataclasses. Stdlib only. |
| Application/verification_gate.py | Application service with protocol-based DI | Application service with protocol-based DI | NONE | 0 | — | Imports only `..domain.verification`. Purely orchestration. |
| Application/evaluator.py | Integration seam | Integration seam | MINOR: `deterministic` field is an untyped dict (feedback.py:81) | 1 | LOW | Schema-unvalidated dict passed between layers. See `feedback.py:81`. |
| Infrastructure/verification_checks.py | Concrete adapters behind protocol | Concrete adapters behind protocol | NONE | 0 | — | Wraps 5 check concerns. All behind `CheckFn` protocol. |
| Operations/feedback.py | Data model | Data model | NONE | 0 | — | Pure data transit. |
| EngineCore/container.py | Composition root | Composition root | NONE | 0 | — | Exempt from layer rules. |

**Drift details:**

- **D1** — `CritiqueReport.deterministic` is typed `Optional[dict]` (feedback.py:81) with no schema contract. The gate populates it with 6 keys (`passed`, `checks`, `reasons`, `artifact_hash`, `failure_summary`, `status_summary`) but this is a convention, not an enforced type. A future change that adds/removes keys silently breaks consumers. **Severity: LOW**

---

## Phase 3: Dependency & Coupling Analysis

### Circular Dependencies
- **None detected.** The dependency graph is strictly acyclic: Domain → Application → Infrastructure (via container). [VERIFIED]

### Layer Leaks
- **None detected.** Infrastructure details never appear in domain or application modules. `verification_gate.py` references only domain types. [VERIFIED]

### Shared Mutable State Risks
- `VerificationGate._checks` (verification_gate.py:125) is a `list[VerificationCheck]` set at `__init__`. `_checks` is read-only during `run()` but is not protected against external mutation by callers holding a reference to the gate. [HYPOTHESIS — no evidence of mutation in practice, but no defense against it]
- `VerificationPolicy.checks` (domain/verification.py:89) is typed `Mapping` but is a `dict` at runtime. `@dataclass(frozen=True)` prevents attribute reassignment but not dict content mutation. [VERIFIED]

### Tight Coupling Hotspots
- **None within the verification subsystem.** The `CheckFn` protocol is minimal (1 callable signature). Each adapter is independent.
- **Boundary service:** `EvaluatorService` (evaluator.py:60-77) accepts `VerificationGate | None` directly via constructor injection, creating a tight-but-explicit coupling. The gate is optional (None = disabled). This is intentional per the design.

### Boundary Violations
- **None detected.** The `container.py` composition root imports from infrastructure (verification_checks.py) which is allowed and intended. No boundary violation because `container.py` is the dedicated wiring point. [VERIFIED]

---

## Phase 4: AI Orchestrator Review

This review applies to the Verification Gate's role within the larger AI Orchestrator.

### Orchestration Model
- **Centralized** — `VerificationGate` is called once per evaluation inside `EvaluatorService._evaluate_inner()` (evaluator.py:106). The gate is not distributed.
- **Routing separated from business logic** — The gate handles deterministic verification only; LLM routing is handled by `ModelSelector` and `TieredModelRouter` outside this scope. [VERIFIED]
- **Provider details isolated** — Not applicable. This subsystem does not interact with LLM providers directly.

### Async and Concurrency
- **Consistent async** — All check adapters are async. `_run_command` uses `asyncio.create_subprocess_exec` (verification_checks.py:42). No sync-over-async blocking. [VERIFIED]
- **No backpressure** — Checks run sequentially. There is no mechanism to refuse new artifacts if the gate is busy. The single `run()` call is synchronous-async from the caller's perspective. [VERIFIED — not a problem for single-task evaluation]
- **Concurrent calls bounded** — `VerificationGate` has no internal concurrency guard. Two concurrent `run()` calls on the same gate iterating `self._checks` are safe under the GIL (reads only), but unsynchronized mutation of `self._policy` between calls is not prevented. [HYPOTHESIS — GIL protects reads, but `_policy` mutation by external code races]

### State and Context
- **Stateless gate** — `VerificationGate` holds only configuration (`_checks`, `_policy`). No session state. Each `run()` call produces a fresh `GateResult`. [VERIFIED]
- **Context is explicit** — `run(artifact, policy=None)` — artifact and optional policy override are the only inputs. [VERIFIED]

### Failure Semantics
- **Retry policies** — Not implemented in the gate. The gate runs once per `evaluate()` call. If a check fails, the gate returns the capped score. The caller (pipeline) decides whether to retry. [VERIFIED]
- **Fallback routing** — Not implemented. The gate is a binary pass/fail floor. No fallback check provider is invoked if one check fails. [VERIFIED]
- **Partial failure states** — Handled: a check that raises is recorded as `BLOCKED` (verification_gate.py:183-201), all other checks still run, the aggregate reflects partial results. [VERIFIED]

### Tool Execution
- **Isolated** — Check adapters are standalone callable functions. The gate does not manage tool lifecycle. [VERIFIED]
- **Tool output validated** — Each adapter validates its own output (e.g., `_make_syntax_check` catches `SyntaxError`, `_make_lint_check` interprets return codes). [VERIFIED]

### Scalability Bottlenecks
- **Single point under 10x load:** The 30-second `_make_type_check` timeout (verification_checks.py:120). If all evaluations trigger a mypy check, and mypy is slow, the gate can bottleneck a 10-task pipeline to 300 seconds of waiting. [VERIFIED]
- **Stateless orchestrator** — `VerificationGate` is stateless across calls. Scales horizontally if the pipeline dispatches tasks across processes. [VERIFIED]

### Stack-Specific Checks
- **FastAPI:** Not applicable — the verification subsystem is CLI/pipeline-driven.
- **Redis:** Not applicable — no caching in this subsystem.
- **Docker:** Not applicable — no container boundaries in this scope.

---

## Phase 5: Anti-Pattern Detection

| Anti-Pattern | Detected? | Evidence | Severity |
|---|---|---|---|
| God module | **No.** None of the 6 modules exceeds 238 lines. Responsibility is clearly separated across layers. | Max module: verification_gate.py (238 lines). | — |
| Hidden monolith | **No.** The verification subsystem is a self-contained wedge within a larger orchestrator. | No internal boundaries cross-cut. | — |
| Shared database coupling | **No.** No database in this subsystem. | — | — |
| Temporal coupling | **No.** Checks are independent — order does not affect correctness. | verification_gate.py:159 — sequential loop but any order produces same aggregate. | — |
| Anemic domain model | **Partially.** `CheckOutcome`, `ExecutionReceipt`, `VerificationPolicy` are data-only with simple property helpers. `ExecutionReceipt` has 2 properties (`is_blocked`, `is_failure`) but no behavioral methods. This is acceptable for DTOs. | domain/verification.py:47-73 — frozen dataclass with no methods beyond `is_blocked`/`is_failure`. | LOW — DTOs are intentionally anemic. |
| Orchestrator bottleneck | **No.** `VerificationGate` runs ~5 checks sequentially, each <1s (except mypy at 30s). Not a bottleneck for typical 2-5 task pipelines. Under 10x load the mypy check becomes a bottleneck, but this is a performance issue, not an architectural anti-pattern. | verification_checks.py:120 — 30s mypy timeout. | LOW (performance, not structure) |
| Infrastructure leakage | **No.** Infrastructure adapters are in `infrastructure/` and are only imported by `container.py` (the composition root). Application code never imports them. | Verified in import graph. | — |
| Premature abstraction | **Yes.** `VerificationCheckAdapter` (verification_checks.py:189-203) mirrors `VerificationCheck` (verification_gate.py:35-42) nearly 1:1. Both carry `name`, `run`/`_run`, and `command`. Two classes for the same concept separated only by import-boundary policy. | Compare verification_gate.py:35-42 vs verification_checks.py:189-203. | LOW — justified by hexagonal boundary policy. Not premature in context. |
| Overengineering | **No.** The design is minimal: 3 domain types, 1 gate, 1 adapter wrapper, 5 factories. Each component has a single clear purpose. | Total: ~580 lines of source across 6 files. | — |
| Underengineering | **No.** All key concerns are addressed: check isolation, error handling, structured results, policy support, audit hashes. | See full module inventory in Phase 1. | — |

---

## Phase 6: Executive Summary

**ARCHITECTURE SCORE: 8 / 10**

Justification: The verification wedge is well-structured with clean hexagonal boundaries, no layer leaks, no circular dependencies, and solid separation of concerns. It loses 2 points for:
1. The untyped `CritiqueReport.deterministic` dict (feedback.py:81) which creates a silent schema-contract gap between producer (gate) and consumer (persistence).
2. The premature abstraction duplication between `VerificationCheck` and `VerificationCheckAdapter` which, while architecturally justified, adds cognitive overhead with zero behavioral difference.

**MATURITY LEVEL: Early Production**

The code is fully tested (91 tests), has CI integration (black, ruff, lint-imports), includes security annotations (`# nosec`), and has a CI guard for subprocess cleanup (scripts/check_subprocess_cleanup.py). However, the `ORCH_VERIFY_ACTS` feature flag referenced in docstrings is not implemented, and there is no production telemetry for gate-disabled-by-import-failure scenarios.

**PRIMARY RISKS:**

1. **MEDIUM** — `proc.wait()` after timeout has no timeout itself (verification_checks.py:62). A stuck kill+wait sequence would block the entire evaluation pipeline.
2. **MEDIUM** — Gate disabled silently on import failure (container.py:658-659). Only a debug log line alerts operators.
3. **MEDIUM** — `CritiqueReport.deterministic` dict is schema-unversioned (feedback.py:81). Future field changes could corrupt persisted state.
4. **LOW** — Empty gate (`checks=[]`) passes everything (verification_gate.py:68). Misconfigured gate cannot be detected from the score.
5. **LOW** — 30s mypy check blocks sequential pipeline (verification_checks.py:120). Under 10x load this becomes a bottleneck.

**CRITICAL VIOLATIONS: None.**

**REFACTOR URGENCY: Backlog**

The architecture is sound. All identified issues are LOW-MEDIUM severity with straightforward fixes (add timeout to kill+wait, emit metric on gate initialization, version the deterministic dict schema). No immediate refactoring is required.

---

## Phase 7: Refactoring Roadmap

### IMMEDIATE (fix before next feature)

- **[PR-6] Add timeout to kill+wait cleanup** — `verification_checks.py:58-62`: Wrap `proc.kill()` and `proc.wait()` in a second `asyncio.wait_for` with a 5-second timeout. If the process ignores kill, the timeout prevents indefinite blocking. [VERIFIED — existing risk documented in V7 hunt]

### HIGH-IMPACT (next sprint)

- **[PR-1] Version the `deterministic` dict schema** — `feedback.py:81`: Add a `"version": 1` key to the deterministic dict, and validate on `from_dict()` that the version matches the expected schema. This prevents silent data corruption from future field changes. [HYPOTHESIS — no corruption has occurred, but the risk exists]

- **[PR-2] Emit metric on gate initialization** — `container.py:658-659`: When the gate is successfully wired, emit a counter metric (`verification.gate.active = 1`). When it fails and becomes `None`, emit `verification.gate.failed = 1`. Wire this into the existing `TelemetryCollector` already available in the container. [VERIFIED — `container.telemetry` is `TelemetryCollector` at container.py:553]

### LONG-TERM (architectural evolution)

- **[PR-3] Implement `ORCH_VERIFY_ACTS` feature flag** — `verification_gate.py:233`: Implement the feature flag referenced in the docstring. The flag should control whether the gate is wired at all (container level) or whether individual checks are enabled. [SPECULATIVE — the flag is mentioned but not implemented; actual requirements are unknown]

- **[PR-4] Add caching by artifact hash** — `verification_gate.py:97-99`: The `compute_artifact_hash` function already computes SHA-256. A cached check bypass (hash → previous result) would eliminate redundant checks when the same artifact is evaluated multiple times. This is relevant for the critique-revise cycle where the same artifact may be re-evaluated. [HYPOTHESIS — caching is not implemented and may interact badly with non-deterministic checks]

### SWITCHING TRIGGERS

- The architecture should switch from sequential to parallel check execution when `verification_checks.py` grows beyond 8 check adapters (current: 5), or when the total check execution time exceeds 50% of the target LLM evaluation latency.
- The architecture should add check result caching when the same artifact is evaluated more than 3 times (the critique-revise cycle default).

---

## Uncertainty Log

| Question | Location | Possible Interpretations | Impact if Wrong |
|----------|----------|--------------------------|-----------------|
| Is `ORCH_VERIFY_ACTS` implemented elsewhere? | `verification_gate.py:233` docstring | A) Planned but not coded. B) Implemented in `orchestrator/crosscutting/config.py` not in scope. | If B, this audit misses a feature flag path that could skip verification. |
| Are mypy/ruff guaranteed in PATH? | `verification_checks.py:92, 118` | A) Optional, graceful degradation. B) Assumed present in production. | If B is wrong, production silently runs only 3 of 5 checks. |
| Is `proc` unbound in some failure path? | `verification_checks.py:56-62` | A) Always bound (line 42 runs first). B) Some paths fail before assignment. | If B, `proc.kill()` raises `NameError` → caught as generic `Exception` → timeout signal lost. |
| Full test suite (non-verification modules) results unknown | Not in scope | Unrelated tests may be failing. | This audit may declare the subsystem clean while the broader codebase has test failures. |
