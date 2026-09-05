# T0 — Census Repair — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`CLAUDE.md` (Testing Strategy, Known Limitations), `.github/workflows/ci.yml`,
`.github/workflows/config-drift-gate.yml`, `pyproject.toml` (markers + dependencies),
`scripts/check_root_module_freeze.py`, `scripts/check_test_markers.py`,
`orchestrator/quality/toml_validator.py` (pulled into scope by C2, which is a direct,
concrete instance of the §7.1 census question this tier exists to resolve).

## Gates (this tier's fixed tree)
```
black --line-length=100 --check orchestrator/quality/toml_validator.py tests/unit/test_hunt_t0_census.py   PASS
ruff check orchestrator/quality/toml_validator.py tests/unit/test_hunt_t0_census.py                         PASS
lint-imports                                                                                                 PASS (5/5 contracts KEPT)
python scripts/check_root_module_freeze.py                                                                  PASS (256/256, no additions)
python scripts/check_test_markers.py                                                                        PASS
mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional --python-version=3.12 --follow-imports=silent            PASS (58 files, 0 issues)
bandit -lll -r orchestrator/quality/toml_validator.py                                                        PASS (0 issues)
python -m pytest tests/unit/test_hunt_t0_census.py -m unit                                                   PASS (5/5)
python -m pytest tests/ -q -m "unit or integration"                                                          see below
```

## Verdict
- **VERIFIED DEFECTs fixed:** 2 (C1 — false documented invariant in CLAUDE.md; C2 —
  Python-version-floor-breaking unconditional stdlib import).
- **CLEARED (innocent):** 1 (C3 — reserved-not-dead pytest markers).
- **Residual, not fixed, explicitly flagged:** 2 (C4 — CI never tests the declared 3.10/3.11
  floor, `[REQUIRES HUMAN REVIEW]` because it's a pipeline change; C5 — four unresolvable
  commit citations in a skill file's incident table, `[UNK]`, out of this tier's declared
  scope).
- §7.1 (runtime version discrepancy) is **resolved**: CI runs Python 3.12 exclusively,
  on every job, and never exercises 3.10 or 3.11. This is now stated as fact, not `[UNK]`,
  and C4 records the resulting gap.
- §7.2 (the false `tests/stress_test.py` invariant) is **resolved**: verified never
  committed, corrected in `CLAUDE.md`, with a regression test.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — `CLAUDE.md`'s testing/limitations claims, the CI Python
version census, the two freeze/marker scripts, and `toml_validator.py`'s import contract —
no VERIFIED defect remains unfixed. This says nothing about the other ~99 T1–T8
candidates or the ~85% of the backend this program will never reach (§8 of the plan).

## What this tier does NOT claim
- It does not claim CI *should not* also gain a 3.10/3.11 job — C4 is a recommendation,
  not an executed fix, because changing the CI pipeline needs explicit sign-off.
- It does not claim the skill-file incident table (C5) is wrong in substance, only that its
  citations do not currently resolve in this repository's git history — `[UNK]`, not
  `[VERIFIED]` in either direction.
- It does not claim any other documented invariant in this repo is true. T0's scope was
  the specific census items in §7.1/§7.2 plus what surfaced while resolving them, not an
  exhaustive audit of every claim in every doc.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used (both fixes correct on first pass;
no rework triggered `[REQUIRES HUMAN REVIEW]`).
