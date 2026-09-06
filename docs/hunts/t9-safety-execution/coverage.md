# T9 — Safety/Execution/Plugin Surface — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

48 files (46 from the wave's declared file list in
`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, plus 2 T7 named but never
opened) — `safety/`, `security/`, `plugin/`+`plugins/`, the `gateway/`
subpackage, and root-level security-adjacent modules. A background agent
performed the full read + reachability trace + innocence attempt on all 48;
every fix below was independently re-verified from source (diffs against
canonical modules, direct grep for live callers) before being applied, not
taken on the survey's word alone.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <8 changed/new files>       PASS
ruff check <8 changed/new files>                                    PASS
lint-imports                                                         PASS (5/5 KEPT, 824 files)
python scripts/check_root_module_freeze.py                          PASS (256/256)
python scripts/check_test_markers.py                                 PASS
mypy orchestrator/domain/ .../application/ .../container.py         PASS relative to baseline —
                                                                        the only content-level
                                                                        change is 8 pre-existing
                                                                        type errors in the now-
                                                                        reachable canonical
                                                                        orchestrator/reference_monitor.py
                                                                        (+3 in specs.py), surfaced
                                                                        because the C1 shim fix
                                                                        makes mypy follow that
                                                                        import for the first time
                                                                        in this invocation — these
                                                                        are real, pre-existing bugs
                                                                        in a live file, not
                                                                        introduced by this tier,
                                                                        and out of scope for a
                                                                        duplicate-shim fix (see
                                                                        inventory.md's C1 note)
bandit -lll -r <7 changed source files>                              PASS (0 issues at HIGH
                                                                        severity threshold; 2
                                                                        pre-existing Low findings,
                                                                        unrelated to this tier)
python -m pytest tests/unit/test_hunt_t9_safety_execution.py        PASS (5/5)
python -m pytest tests/ -q -m "unit or integration"                  2522 passed (+5 over T8's
                                                                        2517), 2 pre-registered
                                                                        environmental failures
                                                                        unchanged, 21 skipped —
                                                                        zero regressions
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 7 fixed source files (new
test file stays present, staged), full T9 test file re-run against the
pre-fix tree, fix restored via `git stash pop`. All 5 tests failed against
the pre-fix tree for the exact predicted reason:
- C1 (×3): `assert via_safety is canonical` failed — two distinct class
  objects, not the same one.
- C2: captured module name was the singular, unimportable
  `orchestrator.plugin.plugins.memory.demo` instead of the plural path.
- C3: `AttributeError: 'ScanReport' object has no attribute 'files_skipped'`
  — the field didn't exist yet.

All 5 pass on the fixed tree.

One correction made before finalizing C3's test: the first version used a
directory named `bad.py` to simulate an unreadable file (the pattern that
worked for T8's C5), but `_iter_scannable_files()` filters to
`path.is_file()` *before* the read attempt, so a directory never reaches
the code path being tested at all — confirmed by watching the naive
version fail with `files_skipped == 0`, not the expected assertion.
Rewritten to monkeypatch `Path.read_text` to raise for one specific real
file, which correctly exercises the `except OSError` branch. This was a
test-design correction, not a logic change to the fix itself.

## Verdict

- **VERIFIED DEFECT fixed:** 4 — C1 (three dead safety/ duplicates with
  broken imports, converted to shims of their live canonicals — the same
  root-vs-subpackage divergence shape T1/T2/T3/T5/T7 already found
  repeatedly), C2 (a wrong module path silently breaking every future
  bundled plugin), C3 (a live secret/insecure-pattern scanner silently
  skipping unreadable files with no signal — the fourth known instance of
  this exact shape across the hunt), C4 (a security-control docstring
  falsely claiming live wiring that doesn't exist).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** the
  actual wiring of `tool_guardrails.py`'s `ToolCallGuardrailController`,
  `command_guard.py`'s `classify_command()`, the `orchestrator/plugin/`
  isolation subsystem (plus its self-reported-trust bypass if ever wired),
  `safety/guardrails.py`'s `ProductionGuardrails`/`KillSwitch`, and
  `gateway/run.py`'s unauthenticated `handle_message()` (currently
  unreachable — no real network listener exists yet) — all genuine,
  independently-verified gaps, all architecture/product decisions about
  whether and how to activate dormant machinery, consistently deferred the
  same way this hunt has deferred every other "wire this dead subsystem
  in" question (T1 C3/C6, T5 adaptive_router, T8 C1).
- **Cleared (innocent):** `safety/sandbox.py`/`safety/secure_execution.py`
  (T7's originally-flagged pair) — confirmed to be plain re-export shims
  with zero divergence risk, resolving T7's open question entirely; several
  dead-but-non-diverged duplicate pairs and standalone dead modules with no
  security-relevant defect found.

## Clean claim this tier is permitted to make, and no more

Within the 48 files in this wave's declared scope: every file has a
recorded, independently-verified disposition (4 fixed across 6 files, the
rest cleared or flagged). This does **not** claim the fixes make any
currently-dormant safety control effective — `tool_guardrails.py`'s
docstring now correctly says it isn't wired in, but it still isn't wired
in. It does **not** claim `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
remaining waves (T10–T16, ~790 files) are covered — see that document for
the accounting.

## What this tier does NOT claim

- It does not claim any of the "fully built but never wired" security
  controls surveyed here (`ToolCallGuardrailController`, `command_guard`,
  the `plugin/` isolation subsystem, `ProductionGuardrails`) provide any
  actual runtime protection — they remain exactly as dormant as they were
  found, only more honestly documented in one case (C4).
- It does not claim the 8 mypy errors surfaced in `orchestrator/
  reference_monitor.py`/`specs.py` are fixed — they are real, pre-existing,
  and now visible to this particular mypy invocation for the first time as
  a side effect of C1's shim fix, not introduced by it.
- It does not claim `gateway/run.py`'s auth gap is safe to leave
  indefinitely — only that it is not reachable today because no real
  network listener exists; the moment one is added, the missing
  auth/rate-limiting becomes live.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1/3 survey delegated to one background
agent run). `fix_revisions`: 1/1 per fixed candidate except C3's test,
which needed one design correction (directory-trick → monkeypatch) before
the assertion itself was ever run against fixed code — recorded above, not
a revision of the fix's logic.
