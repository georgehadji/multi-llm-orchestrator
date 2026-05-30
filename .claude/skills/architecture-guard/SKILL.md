# architecture-guard

Architecture rules and placement guide for the AI Orchestrator. Use before editing `engine.py`, `models.py`, or any core module, and when deciding where new business logic belongs.

## Three Unbreakable Rules

### Rule 1 — `engine.py` = Mediator ONLY
- `engine.py` wires services together. It delegates; it never implements.
- New algorithms, new business logic → **new service module**, not engine.py.
- If you're writing a method longer than ~20 lines inside engine.py → STOP and create a service.

### Rule 2 — `models.py` = Pure data ONLY
- Only `@dataclass` and `Enum` definitions.
- No I/O, no `asyncio`, no behavior, no side effects.
- If you're adding a method that calls anything external → STOP and put it elsewhere.

### Rule 3 — TDD without exceptions
1. Write a failing test first (RED) — confirm it fails with the expected error.
2. Write the minimum code to pass (GREEN).
3. Commit only after GREEN.

## Dependency direction (Hexagonal Architecture)

```
Interfaces → Infrastructure → Application core → Domain models
(cli.py)    (api_clients)     (engine.py)        (models.py)
```

**Inner layers must NOT import from outer layers.**

## Where does new code belong?

| Type of logic | Correct location |
|--------------|-----------------|
| LLM routing / model selection | `model_routing.py` or `planner.py` |
| New pipeline stage | `engine_core/stages/<stage_name>.py` |
| Validation rule | `validators.py` or `preflight.py` |
| Persistence / state | `state.py` or new repository module |
| Event handling | `events.py` or `hooks.py` |
| LLM provider adapter | `api_clients.py` (follow UnifiedClient pattern) |
| HTTP API gateway | `gateway.py` |
| Budget tracking | `cost.py` |
| Resilience / retry logic | `resilience.py` or `rate_limiter.py` |
| Statistical profiling | `infrastructure/nexusscope/` |
| Web search | `infrastructure/nexus_search/` |
| Caching | `infrastructure/cache.py` |
| New domain model | `models.py` (dataclass or enum only) |
| New application service | New file in `orchestrator/` root |

## Pattern reference

| Pattern | Key files |
|---------|-----------|
| Mediator | `engine.py` |
| Strategy | `model_routing.py`, `planner.py` |
| Decorator | `verification.py`, `prompt_enhancer.py` |
| Chain of Responsibility | `validators.py`, `preflight.py` |
| Repository + Memento | `state.py`, `checkpoints.py` |
| Observer / EventBus | `events.py`, `hooks.py` |
| Adapter (LLM) | `api_clients.py` (UnifiedClient) |
| Facade (HTTP) | `gateway.py` |
| Composite (budget) | `cost.py` |
| State Machine | `resilience.py`, `rate_limiter.py` |

## Pre-change checklist

- [ ] Is there a failing test that proves the current behavior is wrong?
- [ ] Does the new code belong in engine.py, or in a dedicated service?
- [ ] Does models.py still only contain dataclasses and enums after my change?
- [ ] Does the import direction follow the hexagonal rule (no inner → outer)?
- [ ] Have I read the relevant section of `docs/CODEBASE_MINDMAP.md`?
- [ ] Is the new module added to `pyproject.toml` mypy overrides list? (Only if legacy; new code must pass strict.)

## Key files to read before large changes

- `docs/CODEBASE_MINDMAP.md` — full architectural reference
- `orchestrator/engine_core/pipeline.py` — TaskPipeline and PipelineContext
- `orchestrator/domain/ports.py` — Protocol interfaces (CachePort, StatePort, LLMClient, EventPort)
- `orchestrator/engine_core/container.py` — ServiceContainer wiring
