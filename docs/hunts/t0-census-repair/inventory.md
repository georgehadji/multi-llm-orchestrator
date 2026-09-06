# T0 — Census Repair — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Scope: `CLAUDE.md`, `.github/workflows/*.yml`,
`pyproject.toml` (markers/deps), `scripts/check_*.py`. Budget: 6 candidates. Spent: 5.
Tree at start of tier: `9566e4c` (branch `claude/llm-orchestrator-website-factory-luk61w`).

## Phase 0 delta

- Taxonomy prune: census integrity only — not one of V7's 8 standard classes. A T0
  candidate is a documented invariant that is false, unverifiable, or misleading to a
  later tier's Phase 3b innocence attempt.
- `hunt_iterations` used: 1. `fix_revisions` used: 1 (both fixes landed first pass).

## Candidates

### C1 — VERIFIED DEFECT — CLAUDE.md cites a nonexistent test file, twice
- **Property violated:** documented invariant must be true (V7 Phase 0 census integrity).
- **Location:** `CLAUDE.md` (Testing Strategy section, Known Limitations section).
- **Trigger:** `find . -iname '*stress*'` under `tests/` returns nothing; `git log --all
  --follow -- tests/stress_test.py` returns no history — the file was never committed.
- **Innocence attempt:** none available — the claim names a specific file, a specific
  suite (S2/S6/S7), and a specific status ("documented, not blocking"). No renamed or
  relocated equivalent exists.
- **Corroboration:** `projects/stress_test/README.md:64` states directly: "`tests/
  stress_test.py` was documented but never committed — these are the replacement."
- **Why this matters beyond wording:** a later tier's Phase 3b innocence defense may cite
  CLAUDE.md's "Known Limitations" to wave off a finding. This entry would have let it wave
  off a finding in a file that does not exist, which proves nothing about any real file.
- **Fix:** corrected both citations to state the verified fact (never committed; the real
  manual-stress surface is `projects/stress_test/*.yaml`). No behavior change — docs only.
- **Tests:** `test_c1_claude_md_does_not_assert_stress_test_py_has_known_failures`,
  `test_c1_stress_test_py_still_does_not_exist`.

### C2 — VERIFIED DEFECT — unconditional `import tomllib` breaks the declared Python floor
- **Property violated:** class 8 (contract/dependency) — a module's imports must be
  satisfiable on every Python version the package claims to support.
- **Location:** `orchestrator/quality/toml_validator.py:13` (pre-fix).
- **Reachability:** `UNKNOWN` via static import graph (no other module imports this one),
  but directly reachable as its own documented entry point: `python -m
  orchestrator.quality.toml_validator fix <file>` (module docstring, line 6).
- **Trigger (real, not simulated):**
  ```
  $ /usr/bin/python3.10 -c "import tomllib"
  ModuleNotFoundError: No module named 'tomllib'
  $ /usr/bin/python3.11 -c "import tomllib"
  <no error>
  ```
  `tomllib` entered the stdlib in Python 3.11 (PEP 680). `pyproject.toml` declares
  `requires-python = ">=3.10"` and lists a `Programming Language :: Python :: 3.10`
  classifier; no `tomli` backport was declared. Any invocation of this module on Python
  3.10 fails at the `import` line, before any of its logic runs.
- **Why CI never caught it:** every job in `.github/workflows/ci.yml` and
  `config-drift-gate.yml` pins `python-version: "3.12"` — this resolves §7.1's open
  question. CI does not exercise 3.10 or 3.11 at all, so a stdlib addition landing on 3.11
  can silently violate a 3.10 promise indefinitely.
- **Innocence attempt:** checked whether `tomli` was already an installed/declared
  dependency pulled in transitively (it was not — `pip show tomli` reports not found) and
  whether any 3.10-excluding guard existed elsewhere (none). No innocence available.
- **Fix:** version-gated import (`sys.version_info >= (3, 11)` → stdlib `tomllib`, else
  `tomli` as a new `python_version < '3.11'`-marked dependency in `pyproject.toml`).
  Constraint check: fix lands in the owning module, not `engine.py`/`models.py` — no
  collision with §6.
- **Tests:** `test_c2_toml_validator_has_version_gated_tomllib_import`,
  `test_c2_pyproject_declares_tomli_for_pre_311`,
  `test_c2_module_still_imports_and_validates_on_current_interpreter`.

### C3 — CLEARED (innocent) — `stress`/`load` pytest markers registered, zero usage
- **Property checked:** class 8 — a registered marker with `--strict-markers` on should be
  either used or removed; dead configuration is a common source of false signal.
- **Location:** `pyproject.toml` markers list (`load`, `stress`); zero
  `@pytest.mark.stress` / `@pytest.mark.load` decorators anywhere in `tests/`.
- **Innocence attempt succeeded:** `projects/stress_test/README.md:63` documents this as
  deliberate — the markers were reserved for a pytest-based stress suite that was replaced
  by the YAML-driven runner (`projects/stress_test/*.yaml`) instead. Not dead
  configuration; a reserved name with a written reason.
- **Disposition:** CLEARED. Recorded here so a later tier does not re-raise it.

### C4 — VERIFIED GAP, NOT FIXED (needs human sign-off) — CI never tests the declared floor
- **Finding:** every CI job runs Python 3.12 exclusively. The package declares
  `requires-python = ">=3.10"` and ships 3.10/3.11/3.12/3.13 classifiers, none of which CI
  verifies except 3.12. C2 is a direct, concrete consequence of this gap.
- **Why not fixed here:** adding a Python version matrix to `.github/workflows/ci.yml` is
  a CI/CD pipeline change (cost, runtime, and policy implications) outside what a
  documentation-and-one-module tier should do unilaterally. Flagging for explicit approval
  rather than pushing an unrequested pipeline change.
- **Disposition:** residual, `[REQUIRES HUMAN REVIEW]`. Recommendation: add at least a
  3.10 job to the Type Check or Test matrix, or narrow `requires-python` to `>=3.11` if 3.10
  support is not actually intended.

### C5 — VERIFIED DISCREPANCY, OUT OF SCOPE, NOT FIXED — stale commit citations in a skill file
- **Finding:** `.claude/skills/orchestrator-failure-archaeology/SKILL.md`'s "Pre-2026-06
  background incidents" table cites four commit hashes (`3143b49b`, `7edcd61f`,
  `8296af32`, `e07467cd`, plus a branch tip `321547ac`) as evidence. **None resolve** in
  this repository's current git history (`git cat-file -t <sha>` fails for all four).
- **Why not fixed here:** this is a full-table discrepancy, not a single typo — the most
  likely explanation is a history rewrite (squash/rebase) between when the skill file was
  authored and now, but that is `[UNK]` without more information, and T0's declared scope
  is `CLAUDE.md`/CI config/marker inventory, not `.claude/skills/`. Guessing at replacement
  hashes would risk citing evidence that is itself wrong.
- **Disposition:** residual `[UNK]`, recorded, not fixed. Flagged for the skill's
  maintainer to regenerate provenance rather than guessed at here.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
