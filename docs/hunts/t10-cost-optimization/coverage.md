# T10 — Cost-Optimization Remainder — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

19 files (`cost_optimization/`'s 13 files, `costing/analytics.py`,
`rate_limiter.py`, `token_budget.py`, `token_optimizer.py`,
`provisioned_throughput.py`), all in `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
T10 wave definition. A background agent performed the full read +
reachability trace + innocence attempt on all 19; every fix below was
independently re-verified from source (byte-for-byte `diff` against
canonical modules, direct execution to reproduce the `ModuleNotFoundError`,
direct grep for live callers/construction sites) before being applied.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <5 changed/new files>       PASS (after 1 reformat —
                                                                        cosmetic only)
ruff check <5 changed/new files>                                    PASS
lint-imports                                                         PASS (5/5 KEPT, 824 files)
python scripts/check_root_module_freeze.py                          PASS (256/256)
python scripts/check_test_markers.py                                 PASS
mypy orchestrator/domain/ .../application/ .../container.py         PASS — isolated diff against
                                                                        the pre-fix tree is empty
                                                                        (zero output difference);
                                                                        none of this tier's 4
                                                                        changed files are reached
                                                                        by this core-path invocation
bandit -lll -r <4 changed source files>                              PASS (0 issues at HIGH
                                                                        severity threshold; 2
                                                                        pre-existing Low findings,
                                                                        unrelated to this tier)
python -m pytest tests/unit/test_hunt_t10_cost_optimization.py      PASS (5/5)
python -m pytest tests/ -q -m "unit or integration"                  2527 passed (+5 over T9's
                                                                        2522), 2 pre-registered
                                                                        environmental failures
                                                                        unchanged, 21 skipped —
                                                                        zero regressions
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 4 fixed source files (new
test file stays present, staged), full T10 test file re-run against the
pre-fix tree, fix restored via `git stash pop`. 4 of 5 tests failed against
the pre-fix tree for the exact predicted reason:
- C1/C2: `assert via_root is canonical` failed — two distinct class objects.
- C3: `ModuleNotFoundError: No module named
  'orchestrator.cost_optimization.log_config'` at import time.
- C4 (traversal test): the fake docker client's `.containers` attribute
  error fired instead of the containment error — i.e. pre-fix, the
  unsanitized filename sailed straight through to the (fake)
  `client.containers.run()` call with no rejection at all.

The 5th test (`test_c4_docker_sandbox_accepts_normal_filename`, the
no-regression check for a legitimate filename) correctly passed in *both*
states — it asserts the absence of a containment error, which is true
whether or not the containment check exists at all, by design: it exists to
catch a regression in the fix, not to prove the defect.

All 5 pass on the fixed tree.

## Verdict

- **VERIFIED DEFECT fixed:** 4 — C1 and C2 (two more unshimmed root-vs-
  subpackage duplicate pairs, caught before divergence rather than after —
  `token_budget.py` and `provisioned_throughput.py`), C3 (a broken relative
  import blocking `Tier1OptimizationMixin` from ever being importable), C4
  (a path-traversal / arbitrary-host-file-write vector in a Docker sandbox
  execution primitive).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:**
  `docker_sandbox.py`'s missing container-hardening parameters (`read_only`,
  `cap_drop`, `security_opt`, `pids_limit`) if ever wired live;
  `pricing_cache.py`'s architectural fragmentation (a fourth/fifth
  independent pricing mechanism); the `--tdd-first` CLI flag's silent no-op
  (a concrete symptom of T1's already-recorded C3, not a new bug).
- **Cleared (innocent):** `token_optimizer.py` (correct, live shim);
  `rate_limiter.py` (whole module confirmed dead, extending T5/T8's
  per-method clearance); `costing/analytics.py` and the other 8 dead
  `cost_optimization/` files (individually verified, not just at category
  level, extending T1's C6).
- **Handed off, not this tier's territory:** a live, real bug in
  `application/cache_warmup.py` (wave T12) and a broken import in
  `integrations/mcp_server.py` (wave T15) — both discovered incidentally
  while tracing this tier's own scope, recorded in inventory.md for
  whichever future wave covers them.

## Clean claim this tier is permitted to make, and no more

Within the 19 files in this wave's declared scope: every file has a
recorded, independently-verified disposition (4 fixed across 4 files, the
rest cleared or flagged). This does **not** claim the two handoff findings
are fixed — they are explicitly out of this tier's scope and belong to
later waves. It does **not** claim `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
remaining waves (T11–T16, ~750 files) are covered.

## What this tier does NOT claim

- It does not claim `DockerSandbox` is safe to wire into a live execution
  path — only that the specific path-traversal vector found is closed. The
  missing container-hardening parameters remain a real gap if this is ever
  activated.
- It does not claim the `--tdd-first` CLI flag works — it remains a silent
  no-op, now merely re-confirmed rather than newly broken.
- It does not claim `application/cache_warmup.py`'s broken import (found
  while tracing this tier) is fixed — it is a live bug, handed off to
  wave T12, not addressed here.
- It does not claim the cost-optimization pricing fragmentation (up to 5
  independent, uncoordinated pricing mechanisms found across this hunt:
  `models.py::COST_TABLE`, `costing/core.py`, `costing/analytics.py`,
  `cost_optimization/pricing_cache.py`, `cost_optimization/token_budget.py`'s
  own `COST_PER_1K`) is resolved — only that no individual file among
  these was found to contain an internal arithmetic bug.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1/3 survey delegated to one background
agent run). `fix_revisions`: 1/1 per fixed candidate — all four fixes were
correct on first pass, confirmed via the retroactive RED→GREEN stash test
above.
