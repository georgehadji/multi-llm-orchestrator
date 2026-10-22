---
type: object
cluster: work
universe: leftover
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# ProjectSpec (models.py version)

Input for App Store asset generation: a name, a description and criteria. It is not the main project input.

## Why this shape

Three plain strings, documented as "Project specification for App Store asset generation" (`models.py:1377-1398`).

## Shape

- `name`, `description=""`, `criteria=""` (`models.py:1394-1398`).

Citations: `orchestrator/models.py:1376-1398`

## Who uses it

`app_store_assets.py:32,114` and `appbuilder/store_assets.py:32,114` only. Narrow use, so `leftover`, not `live` for the main path.

## Connected to

- **owns / owned-by:** none.
- **looks-like-but-is-not:** `ProjectSpec` in `application/conversation_agent.py:40`. That one holds `project_description`, `success_criteria`, `target_users`, `core_features`, `tech_stack`, `platform` and feeds the orchestrator from chat. `entrypoints/chat_cli.py:149` mentions "ProjectSpec" in a docstring; which class it means was not checked.

## If you change this

- **Hits:** the two store-asset generators.
- **Does not hit:** `ProjectState`, `run_project`, or the chat `ProjectSpec`.

## Surfaces

| Surface | Role |
|---|---|
| `app_store_assets.py`, `appbuilder/store_assets.py` | reads |

## See

- Source: `orchestrator/models.py:1376`; chat version `orchestrator/application/conversation_agent.py:40`
