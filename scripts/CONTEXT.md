# scripts/ — gates, audits, and ops helpers

**Job:** measure the repo instead of eyeballing it. Run before asserting a claim about the code.

## Gate and drift checks (read-only)
| Script | Checks |
|---|---|
| `check_new_root_files.py`, `check_root_module_freeze.py` | No new root-level modules |
| `check_test_placement.py`, `check_test_markers.py` | Test location and markers |
| `check_config_wiring.py` | Config entries are wired |
| `check_silent_failure.py`, `check_shell_injection.py`, `check_subprocess_cleanup.py` | Defect-class scans |
| `check_duplicate_pairs.py`, `find_duplicates.py`, `reachability_census.py` | Duplicates and dead code |
| `audit_openrouter_models.py`, `fetch_openrouter_models.py`, `validate_models.py` | Model id liveness |

## Ops helpers
`git/` (commit and push helpers), `setup/` (scaffold generators), `utils/`, `batch/`, and many `*.bat` launchers (dashboard, server).

Scripts that write or push (`git/*`, `setup/create_*`, `convert_to_shim.py`) change state. Read them before running.

## Reference
Skill `orchestrator-diagnostics-and-tooling` explains each tool and what its output means.

## Human check
Never run a state-changing script without reading it first.
