# T15 — integrations/CLI/dashboard — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

82 files (`commands/` 24, `integrations/` 13, `vcs/` 7, `ide_backend/` 16, `dashboard_core/`
6, `kanban/` 3, `hitl/` 3, `connectors/` 3, `entrypoints/` 3, root `cli*.py` 4). A background
agent surveyed all 82, giving full depth to the three pre-identified handoffs and to every
`commands/`/root-`cli*`/`entrypoints/` file, using both a top-level import sweep and a static
AST-based import-resolution sweep that also catches lazy, function-body imports — the
technique that surfaced this tier's two most significant findings (`commands/kanban.py`,
`commands/gateway.py`), neither of which a plain top-level import check could have found.
Every fix was independently re-verified from source and empirically re-executed (the actual
command handler invoked with constructed arguments, not just import-checked) before being
applied.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <9 changed files>                 PASS (1 file needed
                                                                             `black` applied —
                                                                             commands/nash.py;
                                                                             re-verified tests
                                                                             still pass after)
ruff check <9 changed files>                                              PASS
lint-imports                                                              PASS (5/5 KEPT,
                                                                             824 files)
python scripts/check_root_module_freeze.py                               PASS (256/256)
python scripts/check_test_markers.py                                     PASS
mypy orchestrator/domain/ .../application/ .../container.py              PASS — isolated
                                                                             diff empty
bandit -lll -r <9 changed Python source files>                            PASS (0 issues at
                                                                             any severity)
python -m pytest tests/unit/test_hunt_t15_integrations_cli_dashboard.py  PASS (7/7)
python -m pytest tests/ -q -m "unit or integration"                      2565 passed (+7 over
                                                                             T14's 2558), 2
                                                                             pre-registered
                                                                             environmental
                                                                             failures
                                                                             unchanged, 20
                                                                             skipped, 157
                                                                             deselected — zero
                                                                             regressions
```

The full run completed (148s) confirming zero regressions — the 7 new tests accounted for the
entire pass-count increase, with no change to the skip/deselect counts. Every fixed file was
also individually verified end-to-end (not just import-checked) before this run:
`commands/kanban.py`'s `execute()` was called directly and produced real output ("No tasks
found."), `commands/gateway.py`'s likewise ("Gateway running: True"), `dashboard.py`'s
`run_dashboard` was confirmed to bind the exact call shape `cli_dashboard.py` uses,
`Orchestrator(budget=...)` was confirmed to construct past the point the bug crashed at, and
`commands/nash.py`'s `backup()`/`commands/codebase.py`'s `_run_modify()` were both called
directly with representative arguments. No existing test in the repository touches any of
these 9 files (a plain grep found none), so no targeted regression subset applies here beyond
the gate suite itself and this full run.

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 9 fixed source files (new test file stays
present, staged), full T15 test file re-run against the pre-fix tree, fix restored via `git
stash pop`. 6 of 7 tests failed against the pre-fix tree for the exact predicted reason:
- C1: the source of both call sites still contained `verbose=` (assertion failure showing the
  literal pre-fix source).
- C2: `orchestrator.dashboard.run_dashboard` was `create_view`, not the canonical
  `dashboard_core.core.run_dashboard` — identity assertion failed.
- C3, C4: `ModuleNotFoundError: No module named 'orchestrator.commands.kanban.board'` /
  `'...commands.gateway.run'` — the exact original errors.
- C5: `TypeError: _run_modify() got an unexpected keyword argument 'budget'` — confirming the
  parameter didn't exist to receive the value pre-fix.
- C6: `ModuleNotFoundError: No module named 'orchestrator.nash_backup'` — the exact original
  error.

The 7th test (`test_c1_orchestrator_construction_succeeds_without_verbose_kwarg`) passed on
both trees, as expected — it exercises `Orchestrator`'s own constructor directly, which never
had the bug itself (the bug was only in the two callers passing an argument the constructor
never accepted); it serves as a no-regression check confirming the fix didn't touch
`Orchestrator.__init__` itself.

## Verdict

- **VERIFIED DEFECT fixed:** 7 — C1 (`Orchestrator(verbose=...)` crashing two live entry
  points: the `orchestrator chat` CLI flow and the dashboard's `/ws/chat` websocket feature),
  C2 (the `dashboard` console script — one of only 3 this project ships — broken on every
  invocation via a three-way signature mismatch), C3 (`commands/kanban.py`'s wrong-dot
  imports stranding a fully-built 441-line work-queue subsystem behind all four kanban
  subcommands), C4 (`commands/gateway.py`, identical bug shape, both gateway subcommands),
  C5 (`commands/codebase.py`'s `--budget` flag silently ignored, always charging $10
  regardless of user input), C6 (`commands/nash.py`'s `nash backup` crashing with a raw
  traceback instead of a clean message — the confirmed non-existence of any backup-manager
  implementation anywhere in the repo means the underlying feature stays unbuilt, per this
  hunt's standing discipline against inventing features), C7 (`cli.py`'s docstring naming a
  dispatcher module that was renamed years ago, cosmetic).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** `integrations/
  mcp_server.py`'s 5 broken imports (confirmed fixable, but it's a feature-richer dead fork
  of the live root MCP server — which side should be canonical is a product decision);
  `Orchestrator.modify_codebase()` — referenced by `commands/codebase.py` but never
  implemented anywhere in this repository's history (only a design document ever described
  it) — the same "planned but never built" shape as `nash_backup`, correctly left unbuilt;
  the same command's silent exit-code-0-on-failure (changing exit-code conventions could
  affect every command uniformly, deserves a repo-wide decision, not a single-file patch).
- **Cleared (innocent):** all 24 `commands/` files confirmed to import cleanly at the top
  level, with the important caveat that this alone would have missed C3 and C4 (both bugs
  live inside lazily-imported function bodies) — the static AST-based sweep is what actually
  caught them; the `discover_command_modules()` registration mechanism itself (18
  auto-discovered subcommands, zero name collisions); `ide_backend/`'s 16 files failing to
  import in this sandbox specifically because the `dashboard` extra (`fastapi`) isn't
  installed here — a correctly-declared optional dependency, not a code defect;
  `orchestrator/connectors.py`'s file-vs-package shadowing (harmless, unlike `gateway.py`'s
  already-recorded instance of the same shape).
- **Discovered but explicitly out of this tier's fix scope, documented as residual:**
  `integrations/compat.py` (4 broken imports, dead), `integrations/
  swiftstack_integration.py` (unshimmed dead duplicate of the live root file), a `git_sync.py`
  (root) vs `vcs/sync.py` divergence (root has a fix `vcs/`'s copy lacks — an unusual
  direction, since `vcs/` is normally canonical for this codebase's other git-adjacent
  shims), a dead, internally-inconsistent 7-file "command center" cluster, and several
  cosmetic/minor findings (a stale CLAUDE.md example, a typo'd deprecation warning, dead code
  behind a live function, 3 misplaced `test_*.py` files that `pytest tests/` never collects).

## Clean claim this tier is permitted to make, and no more

Within the 82 files in this wave's declared scope: every file received at least a
purpose+reachability pass plus two independent import-resolution checks (direct execution and
static AST analysis); the three handoffs, all 24 `commands/` files, and the fixed files
received a correctness-level read with empirical re-execution. This does **not** claim
uniform deep-audit coverage of all 82 files — most of `ide_backend/`'s internals (blocked
from even importing in this sandbox), `dashboard_core/`'s non-chat views, and several
`integrations/` files (`tenancy.py`, `openrouter_ab_testing.py`, `openrouter_sync.py`,
`slack_integration.py`) were read at header/reachability depth only.

## What this tier does NOT claim

- It does not claim `integrations/mcp_server.py` is fixed — only that its defect is confirmed
  and its correct resolution requires a product decision this tier did not make.
- It does not claim `commands/codebase.py`'s `modify` subcommand works end-to-end — only that
  its budget flag is now honored; the underlying `modify_codebase()` method it calls still
  does not exist.
- It does not claim `ide_backend/`'s 16 files are defect-free — only that they could not be
  exercised in this sandbox's environment (missing optional dependency), not that they were
  found correct.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used. `fix_revisions`: all 7 fixes correct on first pass, RED→GREEN
verified on the first attempt. No survey severity claim required correction this tier — the
survey's own three-tier severity ranking (HIGH/MODERATE/LOW) was adopted as-is after
independent verification confirmed each finding's characterization.
