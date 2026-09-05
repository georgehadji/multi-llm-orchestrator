# T8 — Remainder, Coverage-Ordered — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

8 candidates carried over from T2/T5/T6/T7's own residual notes in
`docs/hunts/INVENTORY.md` (not a fresh survey — see inventory.md's Phase 0).
A background agent ran Phase 1 (reachability) + Phase 3 (trigger +
innocence) on all 8; every reachability/innocence claim load-bearing for a
fix was independently re-verified from source before fixing (in particular
C6's env-var claim, which contradicts `CLAUDE.md`'s own documented required
keys, was re-derived directly from `infrastructure/llm_client.py` rather
than trusted from the survey alone).

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <6 changed/new files>      PASS (after 1 reformat —
                                                                       cosmetic only, re-verified)
ruff check <6 changed/new files>                                   PASS
lint-imports                                                        PASS (5/5 KEPT, 824 files)
python scripts/check_root_module_freeze.py                         PASS (256/256)
python scripts/check_test_markers.py                                PASS
mypy orchestrator/domain/ .../application/ .../container.py        PASS relative to baseline —
  + adaptive_router.py + log_config.py + diagnostics.py               0 new errors introduced by
                                                                       this tier's 3 files (isolated
                                                                       diff confirmed identical
                                                                       error content pre/post,
                                                                       only line numbers shifted
                                                                       by the added comments);
                                                                       1607 pre-existing errors in
                                                                       container.py/meta_integration.py/
                                                                       meta/ package are unrelated
                                                                       to this diff and were not
                                                                       introduced or touched by it
bandit -lll -r <5 changed source files>                             PASS (0 issues at HIGH
                                                                       severity threshold; 4
                                                                       pre-existing Low findings,
                                                                       unrelated to this tier's edits)
python -m pytest tests/unit/test_hunt_t8_remainder.py               PASS (5/5)
python -m pytest <8 targeted regression files, 158 tests>           PASS (158/158, zero regressions)
python -m pytest tests/ -q -m "unit or integration"                 2517 passed, 2 failed (both
                                                                       pre-registered environmental,
                                                                       §7.3, unchanged), 21 skipped
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 5 fixed source files (new
test file stays present, staged), full T8 test file re-run against the
pre-fix tree, fix restored via `git stash pop`. All 5 tests failed against
the pre-fix tree for the exact predicted reason:
- C1: raw secret string present verbatim in captured log output.
- C4: `ImportError: cannot import name 'Plugin'` (the incidental blocking
  bug found while testing this candidate — confirms the fix was necessary
  just to reach the test, not just to make an assertion pass).
- C5: `check.passed is True` (an unreadable file rolled into a clean scan).
- C6: a real `OPENROUTER_API_KEY`-only setup raised the `ENV001` CRITICAL
  issue.
- C7: `is_available()` returned `True` for a permanently-disabled model
  while a concurrent writer held the lock.

All 5 pass on the fixed tree (re-confirmed after the black reformat, still
5/5 GREEN).

## Verdict

- **VERIFIED DEFECT fixed:** 5 — C1 (SecretsFilter never installed on any
  real logger), C4 (silent seccomp network-rule-install failures, plus an
  incidental blocking import bug in the same file/package), C5 (a live
  secret scanner reporting false-clean on unreadable files, gating real
  `orchestrator website` CLI flags), C6 (a health check requiring API keys
  the live client never reads, contradicting `infrastructure/llm_client.py`'s
  actual behavior), C7 (a lock-ordering bug making a dormant router's
  availability check ignore committed DISABLED/DEGRADED state).
- **Cleared (innocent), no fix needed:** C2 (wrong threat model — hash-based
  lookups, not vulnerable `==` comparisons; also dead code), half of C3
  (`secure_execution.py`, read in full, no defect), C8 (both confirmed
  fully dead by exhaustive grep, fail-safe-low design if ever wired in).
- **Residual, explicitly not fixed — `[REQUIRES HUMAN REVIEW]`:** the other
  half of C3 (`sandbox.py`'s bypassable denylist and unenforced resource
  limits — dead code, building real enforcement into it is a feature
  request, not a bug fix); `configure_logging()` still has zero live callers
  (C1's filter fix only matters once something calls it — wiring it into a
  real entry point is a separate architecture decision); `CLAUDE.md`'s and
  `.env.example`'s own required-API-key documentation still says
  OPENAI/GOOGLE/ANTHROPIC (same stale claim as C6, but in user-facing docs,
  not code — deliberately not rewritten without knowing whether the
  OpenRouter migration was intentional); whether `AdaptiveRouter` should be
  wired into `container.py` at all; `router_integration.py`'s unrelated
  broken `get_adaptive_router` import (dead, zero importers).

## Clean claim this tier is permitted to make, and no more

Within the 8 candidates handed off from T0–T7's residual backlog: all 8 now
have a re-verified, independently-checked disposition (5 fixed, 3 cleared/
correctly deferred). This does **not** claim T8 surveyed any new part of the
codebase — see `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` for the
data-backed accounting of what remains genuinely unexamined (841 of 892
backend files have no individual disposition on record anywhere in this
hunt) and the prioritized wave plan (T9+) for closing that gap.

## What this tier does NOT claim

- It does not claim `configure_logging()` is ever actually called in
  production — only that if/when it is, `SecretsFilter` is now correctly
  attached.
- It does not claim `CLAUDE.md`/`.env.example`'s required-API-key
  documentation is correct — it is very likely still wrong, in the same way
  C6's pre-fix code was, but rewriting user-facing setup docs based on this
  tier's own inference about intent (OpenRouter migration vs. a real gap)
  was judged out of scope without a human decision.
- It does not claim `orchestrator/plugin/`'s isolation/sandboxing subsystem
  is safe to wire in — `_apply_seccomp()`'s network loop now logs its
  failures, but the subsystem's dead-code status and `_requires_isolation()`'s
  self-reported-trust bypass (found by the independently-run T9 survey, not
  this tier) mean it should not be treated as production-ready isolation.
- It does not claim the money-path/handler-wiring findings independently
  surfaced by a concurrent, separate code-review pass (not part of this
  hunt's declared T8 scope) are addressed — those are tracked separately.

## `hunt_iterations` / `fix_revisions`

See inventory.md.
