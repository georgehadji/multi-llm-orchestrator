# T18 — Silent-failure sweep at scale — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

**All 916** broad exception handlers in `orchestrator/`, enumerated by AST and
classified by what each does with the failure. Of the 222 silent ones, tier 1
(money, 7) was audited exhaustively by reading every site; tier 2 (validation
gate, 46) was audited by two mechanical fail-open sweeps across the whole
population plus reading of the highest-risk sites; tiers 3–5 (190) received the
mechanical sweeps but not individual reading.

The mechanical sweeps are exhaustive over the *severe* variant, which is what
makes this wave's central claim strong: **every** handler in the codebase was
tested for the fail-open shape, not a sample.

## Gates

```
black --line-length=100 --check --fast <3 changed files>          PASS (1 file
                                                                      reformatted,
                                                                      re-verified)
ruff check <3 changed files>                                      PASS
lint-imports                                                      PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_duplicate_pairs.py            (T17's gate)    PASS (59 == baseline)
python scripts/check_silent_failure.py             (new this wave) PASS (916 examined,
                                                                      0 fail-open)
python scripts/check_test_markers.py                              PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS — isolated
                                                                      diff empty
bandit -lll -r <changed files>                                    PASS (no issues
                                                                      identified)
python -m pytest tests/unit/test_hunt_t18_silent_failure.py       PASS (4/4)
python -m pytest tests/ -q -m "unit or integration"               2584 passed (+4
                                                                      over T17's 2580),
                                                                      2 pre-registered
                                                                      environmental
                                                                      failures
                                                                      unchanged, 20
                                                                      skipped, 157
                                                                      deselected —
                                                                      zero regressions
                                                                      (257s)
```

**Full-suite note.** The run was still executing when this wave was committed,
so the count above was deliberately left unasserted in that commit and filled in
here once the run landed: 2584 passed, the 4 new tests accounting for the entire
increase, with the skip and deselect counts unchanged.

## RED→GREEN verification

`git stash push` on `website_validator.py`, T18 test file re-run against the
pre-fix tree, `git stash pop`. **Both defect tests failed pre-fix for the exact
predicted reason**, quoting the misleading text verbatim:

- C1: `details` was `'No rate limiting found. Contact forms and registration
  endpoints must include IP-based rate limiting.'` — no indication that files
  were skipped.
- C2: `details` was `'Auth pages found but no email verification flow
  detected.'` — likewise.

The two **gate** tests pass on both trees, correctly: they exercise
`scripts/check_silent_failure.py`, which is a new tool rather than a fix to
existing behaviour, and the fail-open count was already zero before this wave.
They are tool tests, not defect proofs, and are labelled as such.

## Verdict

- **VERIFIED DEFECT fixed:** 2 — C1 and C2, the two remaining silent
  file-skips in `website_validator.py`, now logged and reported as incomplete
  scans. Severity **LOW**: both fail *closed*, so unlike T8's C5 in the same
  file they were never a security hole; the harm is a misreport that sends a
  developer to add protection that may already exist, and silently reduced
  coverage of checks gating `--min-quality` / `--require-all-checks`.
- **FALSE (innocent), recorded:** 3 — `cost.py`'s `_static_estimate` (the 0.0
  sentinel is documented and `cheapest_model` filters zero-cost candidates out
  rather than preferring them), `safety/code_executor.py`'s
  `_is_sandbox_available` (caller fails closed by default and logs loudly on the
  explicit opt-out), `infrastructure/state.py`'s `save_checkpoint` (the silent
  handler guards a rollback inside an outer handler that re-raises).
- **Recorded, not elevated:** 3 — `control_plane.py`'s documented-deliberate
  audit swallow, `streaming_resilient.py`'s fabricated 50% memory reading (same
  shape as T6's fabricated-score item, treated consistently), and
  `batch_client.py`'s bounded polling swallow.
- **Negative result, exhaustive:** **0** fail-open handlers in the entire
  codebase, across both detection shapes.

## Clean claim this wave is permitted to make, and no more

**Within the fail-open silent-failure pattern:** every broad exception handler
in `orchestrator/` at this commit was tested, none reports success on failure,
and a gate now prevents reintroduction. That claim is total over this pattern
because the sweep was mechanical and complete — not sampled.

It says **nothing** about whether the 222 silent handlers are individually
correct. 190 of them were never read; they were only tested for the fail-open
shape. A handler that silently returns `None` where the caller expects data, or
silently skips work whose absence nobody notices, would pass this wave's sweeps
untouched.

## What this wave does NOT claim

- It does not claim the codebase has no silent failures — only that none of them
  takes the form of an unlogged handler returning an affirmative result.
- It does not claim tiers 3–5 (190 silent handlers) are correct; they were not
  individually read.
- It does not claim the two fixed scans are now complete — only that when they
  skip a file they say so. An unreadable file still reduces real coverage.
- It does not claim `streaming_resilient.py`'s fabricated memory reading is
  acceptable; it is recorded with the same disposition T6 gave the analogous
  fabricated-score case.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3. `fix_revisions`: 0. Three candidates were falsified in
Phase 3 before any fix was written; the innocence attempt, not the trigger test,
did most of this wave's work — which is the expected shape for a wave run over a
pattern that earlier tiers have already largely eliminated.

## Next wave

**T19 — subprocess/exec argument-construction sweep.** 70 files touching
`subprocess`/`os.system`/`eval`/`exec`; T7 audited 2 and T9 covered the
`safety/` subset.
