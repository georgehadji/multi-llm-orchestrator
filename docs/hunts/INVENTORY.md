# Defect-Hunt Programme — Running Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §5. Appended once per tier on close. Cleared candidates are
recorded, not deleted, so a later tier does not re-raise a settled false alarm.

## T0 — Census repair (closed)

Full detail: `docs/hunts/t0-census-repair/inventory.md`, `docs/hunts/t0-census-repair/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `CLAUDE.md` cited `tests/stress_test.py` (S2/S6/S7) as an existing file with documented failures; never committed. Corrected in both locations. |
| C2 | **VERIFIED DEFECT — FIXED** | `orchestrator/quality/toml_validator.py` unconditionally imported stdlib `tomllib` (3.11+), breaking on the declared `requires-python = ">=3.10"` floor. Version-gated with a `tomli` fallback (`python_version < '3.11'` marker in `pyproject.toml`). |
| C3 | **CLEARED (innocent)** | `stress`/`load` pytest markers registered, zero usage — deliberately reserved per `projects/stress_test/README.md`, not dead config. Do not re-raise. |
| C4 | **RESIDUAL — `[REQUIRES HUMAN REVIEW]`** | CI runs Python 3.12 exclusively on every job; never tests the declared 3.10/3.11 floor. Resolves §7.1. Not fixed here — a CI/CD pipeline change needs explicit sign-off. |
| C5 | **RESIDUAL — `[UNK]`** | `.claude/skills/orchestrator-failure-archaeology/SKILL.md`'s incident table cites 4 commit hashes that do not resolve in current git history. Out of T0's declared scope; not fixed. |

**Cumulative residual-UNKNOWN set added by T0:** C4, C5 (above).

**Pre-registered known-innocents reconfirmed still failing, still environmental (§7.3):**
`tests/unit/test_openrouter_model_audit.py::test_audit_against_live_catalogue_is_clean`,
`::test_runtime_only_ids_resolve_via_endpoints` — sandbox blocks `openrouter.ai`.

**Gate status at T0 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/
bandit all PASS. `pytest tests/ -m "unit or integration"`: 2460 passed, 2 failed (both
pre-registered environmental), 21 skipped.

**Next tier:** T1 (money) — `budget.py`, `cost.py`, `cost_tracker.py`, `cost_analytics.py`,
`cost_optimization/`. Its Phase 0 delta must re-verify the shared census against the tree
this tier leaves behind, per §4 Step 1.
