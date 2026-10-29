---
type: object
cluster: work
universe: ghost
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# JobSpec (models.py version)

An App Builder job spec: description, criteria, app type, docker flag, output dir. Nothing imports it.

## Why this shape

The docstring says it is "separate from the policy-oriented JobSpec in `policy.py`" (`models.py:1227-1229`). `__post_init__` rejects empty or oversized strings (`models.py:1263-1282`).

## Shape

- `description`, `success_criteria`, `app_type=""`, `docker=False`, `output_dir=""` (`models.py:1253-1261`).

Citations: `orchestrator/models.py:1219-1282`

## Why ghost

A grep for `models import ... JobSpec` and `models.JobSpec` across `.py` and `.md` files (tests included, `Search/` and `openevolve-main/` excluded) found no importer. The live `JobSpec` is `policy.JobSpec` (`policy.py:227`), used by `engine.py:88` and `engine.py:1080`. `specs.py` defines `JobSpecV2` (`specs.py:58`). Do not build on the `models.py` one.

## Connected to

- **owns / owned-by / joins:** none found.
- **looks-like-but-is-not:** `policy.JobSpec` (live), `JobSpecV2`.

## If you change this

- **Hits:** nothing found. Check again with a repo-wide grep before deleting; dynamic imports would not show up.
- **Does not hit:** `Orchestrator.run_job`, which takes `policy.JobSpec`.

## Surfaces

| Surface | Role |
|---|---|
| none | none |

## See

- Source: `orchestrator/models.py:1219`; live version `orchestrator/policy.py:227`
