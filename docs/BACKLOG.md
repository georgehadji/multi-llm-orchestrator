# Backlog

Items deferred from the V7 refactoring scope guard. Tackle these after Phases 0–4 are stable.

---

## P4-3 Orphaned Package Findings

### `orchestrator/graphify-out/` — Delete or relocate (done 2026-09-08)
- **Finding:** Not a Python package (no `__init__.py`). Contains only JSON cache files.
- **Imports:** None — nothing in `orchestrator/` imports it.
- **Action taken:** Moved to `.graphify-out/` at project root, added to `.gitignore`
  there, and dropped the now-unneeded `bandit -x orchestrator/graphify-out` from CI
  (bandit only scans `orchestrator/`). Its 48MB of hash-named cache files had made
  every repo-wide grep/bandit run crawl it (test_suite_remediation_plan.md R9).

### `orchestrator/runtime/` — Test-only; evaluate for removal
- **Finding:** Contains `sandbox.py`. Only imported from `tests/test_capabilities_5_10.py`.
- **Imports:** No imports from main `orchestrator/` code.
- **Action:** Move `sandbox.py` to `tests/fixtures/` and update `test_capabilities_5_10.py`, or delete both if the capability tests are superseded.

---

## Phase 5 — Horizontal Scaling Preparation

Prerequisite: Phases 0–4 complete and stable for ≥ 2 weeks.

- **P5-1:** Add `TaskQueuePort` abstraction with `InProcessTaskQueue` and `RedisTaskQueue` adapters.
- **P5-2:** Serialize SQLite writes via `asyncio.Semaphore(1)` in `StateManager` to prevent concurrent-write corruption.
# Updated on Sat, May 30, 2026  4:22:27 PM
