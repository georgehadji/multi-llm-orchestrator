# Backlog

Items deferred from the V7 refactoring scope guard. Tackle these after Phases 0–4 are stable.

---

## P4-3 Orphaned Package Findings

### `orchestrator/graphify-out/` — Delete or relocate
- **Finding:** Not a Python package (no `__init__.py`). Contains only JSON cache files.
- **Imports:** None — nothing in `orchestrator/` imports it.
- **Action:** Move to project root or add to `.gitignore`. Safe to delete the cache files; the graphify skill will regenerate them.

### `orchestrator/runtime/` — Test-only; evaluate for removal
- **Finding:** Contains `sandbox.py`. Only imported from `tests/test_capabilities_5_10.py`.
- **Imports:** No imports from main `orchestrator/` code.
- **Action:** Move `sandbox.py` to `tests/fixtures/` and update `test_capabilities_5_10.py`, or delete both if the capability tests are superseded.

---

## Phase 5 — Horizontal Scaling Preparation

Prerequisite: Phases 0–4 complete and stable for ≥ 2 weeks.

- **P5-1:** Add `TaskQueuePort` abstraction with `InProcessTaskQueue` and `RedisTaskQueue` adapters.
- **P5-2:** Serialize SQLite writes via `asyncio.Semaphore(1)` in `StateManager` to prevent concurrent-write corruption.
