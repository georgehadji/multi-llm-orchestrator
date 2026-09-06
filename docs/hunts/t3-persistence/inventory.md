# T3 — Persistence & Resume — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Budget: 14 candidates (raised from 12 per
§3.1 — least-covered tier, owns crash recovery). Spent: 2.

## Phase 0 delta — census correction

- `orchestrator/state.py` (7 lines) and `orchestrator/state_mgmt/state.py`
  (8 lines) are BOTH correct, redundant shims pointing to the real canonical
  `orchestrator/infrastructure/state.py::StateManager` (664 lines). No
  divergence — checked because the pattern recurred so often in T1/T2.
- `orchestrator/async_event_store.py` (root) is a correct shim of
  `orchestrator/events/async_event_store.py` (canonical). No divergence.
- `orchestrator/checkpoints.py` (root) vs `orchestrator/state_mgmt/
  checkpoints.py` (subpackage) diverged — see C1.
- Neither `orchestrator.checkpoints` nor `orchestrator.state_mgmt.
  checkpoints` had a live importer found anywhere in `orchestrator/`
  (grepped every plausible import path). Fixed anyway per the same logic
  as T1's C1/C2: a landmine that fires the moment either path is wired in
  is worth closing while it's still cheap (a shim, not a data-loss risk),
  and it directly serves this tier's "partial write, bad resume" mandate.

## Candidates

### C1 — VERIFIED DEFECT — `state_mgmt/checkpoints.py` silently lacked real content-restoring rollback
- **Property violated:** threat 4 (persistence — "partial write, bad
  resume, lost paid work").
- **Location:** `orchestrator/state_mgmt/checkpoints.py` (pre-fix, 446
  lines) vs `orchestrator/checkpoints.py` (564 lines).
- **Finding:** identical through `NamedCheckpointManager` (confirmed:
  same class start lines, 32/75/290/325 in both files). Root additionally
  has `ContentCheckpointManager`, which — per its own docstring —
  "Unlike `NamedCheckpointManager.rollback()` which returns metadata and
  says 'File restoration is caller's responsibility,' this method
  actually restores file contents from the snapshot store." The
  subpackage copy never received this class.
- **Reachability:** no live importer found for either path (see Phase 0
  delta) — a landmine, not an active bug, at time of fix.
- **Innocence attempt:** none — no comment marks either copy deprecated.
- **Fix:** `state_mgmt/checkpoints.py` rewritten as a re-export shim of
  `orchestrator.checkpoints` (the superset), preserving every
  pre-existing class plus gaining `ContentCheckpointManager`.
- **Tests:** `test_c1_state_mgmt_checkpoints_content_manager_is_canonical`,
  `test_c1_state_mgmt_checkpoints_exposes_all_manager_classes`.

### C2 — VERIFIED DEFECT — `ResumeDetector.find_resumable_project()` always returned `None`
- **Property violated:** threat 4 + the class's own documented contract.
- **Location:** `orchestrator/state_mgmt/resume_detector.py::
  ResumeDetector.find_resumable_project()` (pre-fix).
- **Finding:** the method's own trailing comment read: "This would be
  called in an async context via async wrapper. For now, return None
  (would be implemented with async/await)." It extracted keywords,
  checked preconditions, then unconditionally returned `None` — never
  calling `_score_candidates()` (defined in the same file, fully
  implemented, Jaccard-similarity + 30-day recency decay) or querying
  `self.state_manager` for any actual candidates.
- **Reachability:** `ResumeDetector`/`find_resumable_project` had zero
  callers anywhere (including tests) — but the identical underlying
  capability is live and correct elsewhere: `entrypoints/cli_dispatch.
  py::_check_resume()` (the actual CLI resume gate) calls `await
  state_mgr.find_resumable(keywords)` (a real, already-existing async
  method on `StateManager`, `infrastructure/state.py:500`) then
  `_score_candidates()` directly, bypassing this class entirely. So the
  live resume path works; this specific class's public method never did.
- **Innocence attempt:** considered leaving this as `[REQUIRES HUMAN
  REVIEW]` (matching T1's C3 disposition for genuinely-dead subsystems)
  since implementing it requires an async signature change — but unlike
  C3, the exact missing piece (`StateManager.find_resumable()`) already
  exists, fully async, purpose-built for this ("Only returns
  PARTIAL_SUCCESS or IN_PROGRESS projects"), and the correct calling
  pattern is already proven live in `cli_dispatch.py`. Completing an
  acknowledged stub by wiring two already-correct, already-tested pieces
  together is a contained fix, not a design decision.
- **Fix:** `find_resumable_project()` made `async def` (safe — zero
  existing callers to break); now calls `self.state_manager.
  find_resumable(list(combined_keywords))`, builds `ResumeCandidate`
  objects, scores via `_score_candidates()`, and returns the top match's
  dict if `overall_score >= self.match_threshold`, mirroring
  `cli_dispatch.py::_check_resume()`'s already-live pattern.
- **Tests:** `test_c2_find_resumable_project_no_longer_always_none`,
  `test_c2_find_resumable_project_returns_none_below_threshold`,
  `test_c2_find_resumable_project_returns_none_without_state_manager`,
  `test_c2_find_resumable_project_handles_no_candidates`.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
