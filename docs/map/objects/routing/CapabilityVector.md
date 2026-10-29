---
type: object
cluster: routing
universe: ghost
status: verified
verified: 2026-10-22 at 26fbe258 (source unchanged since d74829e6)
entity: orchestrator/domain/capability_vector.py
---

# CapabilityVector

A frozen record of a model's capability dimensions: context size, vision, tool use, JSON mode, price, plus four derived scores.

## Why this shape

Phase 0 of the ACR work: the type exists so `ModelProfile` can carry an optional field. The module says hydration and scoring are out of scope (`capability_vector.py:4-7`). It uses stdlib only, to satisfy the domain-purity contract (`:9-10`).

## Shape

- Static: `max_context`, `vision`, `tool_use`, `json_mode`, `price` (`capability_vector.py:26-30`).
- Derived, default `None`: `reasoning`, `legal`, `coding`, `citation` (`:33-36`).

Citations: `orchestrator/domain/capability_vector.py:18-36`

## Why ghost

The only field that uses it is `ModelProfile.capability: CapabilityVector | None = None`, commented "Phase 0: inert, unused by any backend" (`policy.py:131-132`). A grep across `orchestrator/` and `tests/` found no code that sets or reads it besides its own unit test (`tests/unit/test_capability_vector.py`). Do not build routing logic on it.

## Connected to

- **owned-by:** `ModelProfile` in `policy.py:77`, as an optional field. The planner reads profiles: see [ConstraintPlanner](ConstraintPlanner.md).
- **looks-like-but-is-not:** `capability` strings in `capability_logger.py` and `state_mgmt/capability_logger.py` (log events, a different thing).

## If you change this

- **Hits:** `policy.py:28,132` and `tests/unit/test_capability_vector.py`.
- **Does not hit:** model selection. No backend reads it.

## Surfaces

| Surface | Role |
|---|---|
| tests | reads |

## See

- Source: `orchestrator/domain/capability_vector.py:18`
