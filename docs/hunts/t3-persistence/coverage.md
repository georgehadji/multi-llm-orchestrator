# T3 — Persistence & Resume — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`orchestrator/state.py`, `orchestrator/state_mgmt/` (all 9 modules, read or
grepped), `orchestrator/checkpoints.py` + `orchestrator/state_mgmt/
checkpoints.py`, `orchestrator/async_event_store.py` + `orchestrator/events/
async_event_store.py`, `orchestrator/infrastructure/state.py` (the real
`StateManager` — `save_project`/`save_checkpoint`/`load_project`/
`load_latest_checkpoint`/`list_projects`/`find_resumable` all read),
`orchestrator/application/resumption_service.py` (read, not independently
audited beyond confirming its existence and role), `entrypoints/
cli_dispatch.py`'s resume-gate logic (`_check_resume`).

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>              PASS
ruff check <changed files>                                            PASS
lint-imports                                                           PASS (5/5 KEPT, 824 files/1389 deps)
python scripts/check_root_module_freeze.py                            PASS (256/256)
python scripts/check_test_markers.py                                  PASS
mypy orchestrator/domain/ .../application/ .../container.py           PASS (58 files, 0 issues)
bandit -lll -r <changed files>                                         PASS (0 issues)
python -m pytest tests/unit/test_hunt_t3_persistence.py -m unit       PASS (6/6)
python -m pytest tests/ -q -m "unit or integration"                   see docs/hunts/INVENTORY.md
```

## RED→GREEN verification
Both fixes retroactively verified via `git stash push --keep-index` on the
two fixed source files, full T3 suite re-run, fixes restored via `git stash
pop`. All 6 tests failed against the pre-fix tree for the exact predicted
reason (`ImportError`/`AssertionError` for C1's missing class; `TypeError:
object NoneType can't be used in 'await' expression` for all 4 of C2's tests,
since the pre-fix method was sync and always returned `None`). All 6 pass on
the fixed tree.

## Verdict
- **VERIFIED DEFECTs fixed:** 2 — C1 (checkpoint-rollback divergence
  between root and subpackage, real content-restoring rollback only on
  one side), C2 (`ResumeDetector.find_resumable_project()` was an
  acknowledged, never-finished stub that always returned `None` despite
  its own scoring engine being fully implemented and correctly used
  elsewhere).
- **CLEARED (innocent):** 2 — `orchestrator.state`/`orchestrator.
  state_mgmt.state` (both correct, redundant shims to `infrastructure/
  state.py`, no divergence) and `orchestrator.async_event_store` (correct
  shim to `events/async_event_store.py`). Recorded so a later tier does
  not re-raise them.
- **Residual, not independently investigated further this tier:**
  - `infrastructure/state.py`'s actual write path (`save_project`,
    `save_checkpoint`) does use `BEGIN`/`commit`/`rollback` with WAL
    checkpointing after critical writes (`_checkpoint_wal()`, called
    after both `save_project` and `save_checkpoint`) — read in full and
    found no obvious atomicity gap, but this was a read-through, not an
    adversarial crash-injection test (e.g. killing the process mid-
    `execute()` to confirm SQLite's own durability guarantees hold under
    this exact usage pattern). `[UNK]`, not claimed either way beyond
    "the code looks correct on inspection."
  - `application/resumption_service.py` was read for its role
    (constructing `ProjectRunner`'s resume path) but not independently
    audited candidate-by-candidate the way `resume_detector.py` was — a
    later pass could extend this tier's coverage here specifically.
  - CLAUDE.md's own "Known Limitations: Resume detection uses a file
    mod-time heuristic; could be more robust" — this tier found the
    *keyword/Jaccard* resume path (`cli_dispatch.py::_check_resume`,
    `find_resumable`) to be real and working (C2's fix now also
    completes the class-based path), which is a different, more
    sophisticated mechanism than a bare mtime heuristic. Whether a
    separate, simpler mtime-based path also exists and is what CLAUDE.md
    refers to was not traced down this tier — `[UNK]`.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — the checkpoint/rollback duplicate pair and
the keyword-based resume-detection path (both the live inline version and
the now-fixed class version) — no VERIFIED defect remains unfixed. This
says nothing about `infrastructure/state.py`'s durability under real crash
conditions (inspected, not adversarially tested) or the mtime-based resume
heuristic CLAUDE.md separately documents.

## What this tier does NOT claim
- It does not claim SQLite/aiosqlite's WAL-mode durability guarantees were
  independently verified under a real crash — only that the calling code's
  transaction structure (`BEGIN`/try/commit/except/rollback) looks correct
  on inspection.
- It does not claim `ContentCheckpointManager`'s snapshot-restore logic is
  bug-free — only that it now exists on both import paths where it
  previously existed on one. Its own internal correctness (does
  `SnapshotStore.restore()` actually round-trip file contents faithfully)
  was not independently tested this tier.
- It does not claim every one of T3's originally-listed files (`state_mgmt/
  capability_logger.py`, `session_lifecycle.py`, `session_watcher.py`,
  `telemetry_store.py`, `workspace.py`, `restore_points.py`) was
  candidate-generated against — they were enumerated in Phase 0 but not
  individually mined for defects, given this tier's two strong,
  well-evidenced candidates and the explicit instruction to keep the tier
  sequence moving.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used — both fixes correct
on first pass, confirmed via the retroactive RED→GREEN stash test above.
