# T5 — Resilience State Machines — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`orchestrator/resilience.py` + `orchestrator/operations/resilience.py`
(full read), `orchestrator/circuit_breaker.py` +
`orchestrator/operations/circuit_breaker.py` (full read, byte-diffed),
`orchestrator/integration_circuit_breaker.py` +
`orchestrator/integrations/integration_circuit_breaker.py` (byte-diffed),
`orchestrator/rate_limiter.py::GrokRateLimiter.acquire()` (read in full —
its check-then-act RPM/TPM logic is entirely inside one `async with
self._lock:` block, confirmed atomic; the file's own `BUG-NEW-001`/
`BUG-NEW-002` comments show it already went through a prior
concurrency-safety hardening pass: never sleep while holding the lock,
refresh limits every loop iteration). `orchestrator/adaptive_router.py`
was not read this tier — see residual note below.

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>          PASS
ruff check <changed files>                                        PASS
lint-imports                                                       PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_test_markers.py                               PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS (58 files, 0 issues)
bandit -lll -r orchestrator/operations/circuit_breaker.py           PASS (0 issues)
python -m pytest tests/unit/test_hunt_t5_resilience.py -m unit    PASS (3/3)
python -m pytest tests/ -k "resilience or circuit_breaker"        PASS (61/61 existing + new,
                                                                     zero regressions)
```

## RED→GREEN verification
Verified via `git stash push --keep-index` on the one fixed source file,
full T5 suite re-run, fix restored via `git stash pop`. All 3 tests failed
against the pre-fix tree for the exact predicted reason: wrong class
identity, a real `CircuitBreakerOpen` not recognized by `isinstance()`
against the operations-local class, and — the most concrete real trigger —
two concurrent `check()` calls against a HALF_OPEN breaker both being
admitted (`2 == 1` failure) instead of exactly one. All 3 pass on the
fixed tree.

## Verdict
- **VERIFIED DEFECT fixed:** 1 — `operations/circuit_breaker.py` had
  silently diverged from the canonical, live `circuit_breaker.py`,
  dropping the `probe_in_flight` enforcement that limits HALF_OPEN to one
  concurrent probe, and — because `operations/resilience.py` sources its
  `CircuitBreakerOpen` from this same divergent copy — breaking exception
  identity against every breaker actually constructed by the live
  pipeline.
- **CLEARED (innocent):** 2 duplicate pairs checked and found to be
  correct, non-diverged shims (`resilience.py`, `integration_circuit_
  breaker.py`).
- **Residual, not independently investigated this tier:**
  `orchestrator/adaptive_router.py` (192 lines) was enumerated in Phase 0
  but not read or candidate-generated against, given the depth already
  spent on the circuit-breaker finding and the explicit instruction to
  keep the tier sequence moving. `[UNK]` whether it has its own state-
  machine defects.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — the resilience/circuit-breaker duplicate
pairs and `rate_limiter.py`'s core `acquire()` loop — no VERIFIED defect
remains unfixed. This does not claim `adaptive_router.py` was audited at
all, nor that every method on `GrokRateLimiter`/`RateLimiter` beyond
`acquire()` was individually reviewed.

## What this tier does NOT claim
- It does not claim `orchestrator/adaptive_router.py` is defect-free —
  it was not examined this tier.
- It does not claim `rate_limiter.py`'s tier-upgrade/spend-tracking logic
  (`_update_tier_from_spend()`, `fetch_current_spend()`) was independently
  re-derived for correctness — only that the specific check-then-act
  pattern in `acquire()` was confirmed atomic.
- It does not claim the now-shimmed `operations/circuit_breaker.py`'s
  removal of its own independent code is risk-free for any caller that
  might have relied on the *specific* (buggy) behavior — none were found,
  but a caller relying on the bug as a feature is not something a grep
  can rule out with certainty.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used — the fix was
correct on first pass, confirmed via the retroactive RED→GREEN stash test
above.
