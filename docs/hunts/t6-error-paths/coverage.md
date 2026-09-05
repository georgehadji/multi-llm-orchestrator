# T6 — Error-Path Sweep — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

A background Explore agent surveyed all `except Exception`/`except
BaseException`/bare `except:` sites across `orchestrator/` (excluding
`tests/`) — roughly 977 raw sites per the initial grep, narrowed to ~115
individually read after filtering pre-cleared/genuinely-isolating shapes,
producing 18 ranked candidates. Of those, this tier fully investigated and
fixed 3: `application/verbalized_sampling.py`, `costing/tracker.py`,
`cost.py` (the two explicitly prioritized money modules plus the
highest-confidence example of the exact "silent, zero-logging, real-money"
shape). The other 15 were read by the survey agent but not independently
re-verified or fixed by me this tier — see inventory.md's residual section
for the full list and per-item disposition.

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>          PASS (after 1 reformat —
                                                                     see RED→GREEN note)
ruff check <changed files>                                        PASS
lint-imports                                                       PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_test_markers.py                               PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS (58 files, 0 issues)
bandit -lll -r <3 changed source files>                            PASS (0 issues at HIGH
                                                                     severity threshold; 3
                                                                     pre-existing Low findings
                                                                     unrelated to this tier's
                                                                     edits — 2 pre-existing
                                                                     `assert self._db_path`
                                                                     in cost.py, 1 pre-existing,
                                                                     untouched `except: pass`
                                                                     elsewhere in
                                                                     verbalized_sampling.py)
python -m pytest tests/unit/test_hunt_t6_error_paths.py           PASS (3/3)
python -m pytest tests/ -k "cost or verbalized or budget..."      PASS (155/155 existing +
                                                                     new, 2 pre-existing
                                                                     skips, zero regressions)
```

## RED→GREEN verification
Verified via `git stash push --keep-index` on the three fixed source files
(test file stays present), full T6 test file re-run, fix restored via
`git stash pop`. All 3 tests failed against the pre-fix tree for the exact
predicted reason: zero log records captured by `caplog` in every case
(`got: []`). All 3 pass on the fixed tree.

One correction made before finalizing: the first `black` run reformatted
`verbalized_sampling.py`'s new log line onto a single line — re-checked
line length (exactly 100 chars, the project's configured limit) and
re-ran `black --check --fast`, which then passed cleanly; this was a
formatting-only correction, not a logic change, and does not affect the
RED→GREEN result above (re-run after the reformat, still 3/3 GREEN).

## Verdict
- **VERIFIED DEFECT fixed:** 3 — all the same shape (a broad `except
  Exception` around a money-relevant operation swallowing the failure
  with zero logging), in `verbalized_sampling.py` (a real, already-
  incurred LLM cost silently never charged to budget on a charge-call
  failure), `costing/tracker.py` (corrupt persisted cost history silently
  treated as "no history"), and `cost.py` (a corrupt per-team/per-job
  spend record silently dropped, potentially letting a near-limit
  team/job appear to have full budget again).
- **Residual, triaged but not independently fixed:** 15 candidates from
  the same survey, ranked by the agent from moderate-to-high severity
  down to low-confidence/low-stakes — see inventory.md for the full list
  and per-item reasoning. The single highest-severity residual item is
  `application/evaluator.py`'s fabricated-neutral-score-on-failure
  pattern, deliberately not folded into this tier's diff because its
  correct fix is a scoring-semantics design decision, not a log-line
  addition.

## Clean claim this tier is permitted to make, and no more
Within the 3 specific sites fixed — the money-charging/persistence swallows
in `verbalized_sampling.py`, `costing/tracker.py`, and `cost.py` — no
VERIFIED defect remains unfixed. This does **not** claim the other 15
surveyed candidates are non-issues, nor that the ~862 other broad-except
sites in the codebase not touched by this survey (this agent covered
`orchestrator/`, filtered to money/security/state-adjacent and silent-pass
shapes) are free of the same pattern.

## What this tier does NOT claim
- It does not claim `application/evaluator.py`'s self-consistency scoring
  is fixed — it is explicitly flagged `[REQUIRES HUMAN REVIEW]`, not
  touched.
- It does not claim the website-output secret-scanner false-clean-report
  pattern (`website_validator.py`, `quality_control.py` x2) is fixed —
  same `[REQUIRES HUMAN REVIEW]` status, deliberately deferred to keep
  this tier's diff scoped to the money-charging shape.
- It does not claim every broad-except site in the codebase was surveyed
  — the agent explicitly prioritized files T0-T5 already touched, plus a
  general severity/stakes ranking; sites outside that prioritization may
  not have been read at all.
- It does not claim the fix (adding a log line) makes the underlying
  operation succeed or retried — only that the failure is now visible in
  logs where it was previously undetectable after the fact. Retry/backoff
  semantics for a failed budget charge are out of scope.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used (the Phase 1/2 survey was delegated to one
background agent run, not iterated). `fix_revisions`: 1/1 per fixed site —
each of the 3 fixes was correct on first pass, confirmed via the
retroactive RED→GREEN stash test above; the one correction made
(black reformatting) was cosmetic, not a logic revision.
