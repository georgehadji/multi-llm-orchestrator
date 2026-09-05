# T5 — Resilience State Machines — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Budget: 10 candidates. Spent: 1.

## Phase 0 delta

- `orchestrator/resilience.py` (root, 31 lines) is a correct shim of
  `orchestrator/operations/resilience.py` (462 lines, canonical) — no
  divergence in what it re-exports.
- `orchestrator/integration_circuit_breaker.py` (root, 7 lines) is a
  correct shim of `orchestrator/integrations/integration_circuit_breaker.py`
  (406 lines, canonical) — no divergence. (A different circuit-breaker
  concept — CI/CD integration health, not LLM-provider fail-fast — despite
  the similar name.)
- `orchestrator/circuit_breaker.py` (root, 329 lines) vs `orchestrator/
  operations/circuit_breaker.py` (326 lines) — NOT a shim relationship,
  two independent implementations, diverged. See C1.
- `orchestrator/nexus_search/optimization/circuit_breaker.py` exists but
  is a different subsystem (search-result caching), not touched.

## Candidates

### C1 — VERIFIED DEFECT — `operations/circuit_breaker.py` diverged from the live circuit breaker, breaking HALF_OPEN single-probe enforcement AND cross-module exception identity
- **Property violated:** class 3 (concurrency) + threat 7's exact
  namesake — a circuit breaker's HALF_OPEN state must admit exactly one
  probe call, not every concurrent caller.
- **Location:** `orchestrator/operations/circuit_breaker.py` (pre-fix)
  vs `orchestrator/circuit_breaker.py`.
- **Finding, part A (state-machine divergence):** `orchestrator/
  circuit_breaker.py`'s own code carries two explicit, dated fix comments
  — `BUG-001` (in `record_failure()`: clear `probe_in_flight` on a failed
  probe so a future probe isn't permanently blocked) and `BUG-002` (in
  `record_success()`: do NOT clear `probe_in_flight` before the success
  threshold is met, "ensures exactly one probe is active per reset
  window"). `operations/circuit_breaker.py` never received these fixes'
  companion line: `check()`'s `if state == HALF_OPEN:` branch is missing
  `self._state.probe_in_flight = True` after admitting a probe. Also
  missing: `record_failure()`'s already-OPEN no-op guard (lower-severity —
  traced through and found to only cause harmless counter noise, since
  `_open()`/probe-admission logic is gated on state, not on the counters
  this omission mutates).
- **Finding, part B (exception-identity break, found by tracing who
  actually uses this divergent copy):** `orchestrator/operations/
  resilience.py:31` does `from .circuit_breaker import CircuitBreakerOpen,
  CircuitBreakerRegistry` — a same-package import that resolves to the
  divergent `operations/circuit_breaker.py`, not the root module. Every
  live circuit-breaker construction site
  (`engine_core/container.py:39,923`, `infrastructure/llm_client.py:212`,
  `engine_slimming.py:28`) imports from the ROOT module instead. Verified
  directly: `orchestrator.circuit_breaker.CircuitBreakerOpen is
  orchestrator.operations.circuit_breaker.CircuitBreakerOpen` was `False`
  (two independently-defined classes, not a subclass relationship), and
  `except <operations-local CircuitBreakerOpen>` does **not** catch a real
  `orchestrator.circuit_breaker.CircuitBreakerOpen` — confirmed by
  actually raising one and observing it propagate uncaught. `operations/
  resilience.py::run_with_resilience()`'s own `except CircuitBreakerOpen
  as exc:` (line 314, meant to "skip immediately, no retries" per its
  docstring) would silently fail to catch a `CircuitBreakerOpen` raised by
  any breaker constructed the way every live call site actually
  constructs one.
- **Reachability:** part A's HALF_OPEN bug is live the moment
  `operations.circuit_breaker.CircuitBreaker` is used directly (it is the
  class every import of `operations.circuit_breaker`/`operations.
  resilience` resolves to). Part B's exception-mismatch is currently
  dormant in the sense that `run_with_resilience()` itself has zero live
  callers passing a `registry=` argument (grepped: `run_with_resilience(`
  only appears in its own docstring example, without `registry=`, and in
  `application/fallback_handler.py`'s deprecation notice pointing
  developers toward it) — but `fallback_handler.py` explicitly steers new
  code at exactly this landmine, and the underlying class divergence
  (part A) is independently live regardless of `run_with_resilience`.
- **Innocence attempt:** none — this is the exact "two independently
  fixed-then-diverged implementations" pattern already found repeatedly
  in T1-T4 (e.g. T1's C1), here with a security-relevant twist (silent
  loss of the thundering-herd protection this whole class exists to
  provide) rather than a data-loss twist.
- **Fix:** `operations/circuit_breaker.py` rewritten as a re-export shim
  of the canonical `orchestrator.circuit_breaker` (the version every live
  construction site already uses and the one carrying the BUG-001/BUG-002
  fixes) — resolving both the state-machine divergence and the
  exception-identity mismatch with one change, since they share the same
  root cause.
- **Tests:** `test_c1_operations_circuit_breaker_is_canonical`,
  `test_c1_resilience_module_catches_the_real_circuit_breaker_open`,
  `test_c1_half_open_allows_only_one_concurrent_probe` (real trigger: two
  concurrent `check()` calls via `asyncio.gather` against a breaker
  arranged into HALF_OPEN with no probe in flight — pre-fix, both calls
  were admitted; post-fix, exactly one is).

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
