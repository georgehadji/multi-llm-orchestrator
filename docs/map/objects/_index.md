# Noun index

One line per noun. Paths are relative to `orchestrator/`. All entries are `stub`: named from an inventory, not yet verified against source. Universe is `live` unless stated.

## work
All verified 2026-10-22 at 80161d2c.
- [ProjectState](work/ProjectState.md) — verified — `models.py:1192`
- [Task](work/Task.md) — verified — `models.py:1077`
- [TaskResult](work/TaskResult.md) — verified — `models.py:1125`
- [AttemptRecord](work/AttemptRecord.md) — verified — `models.py:1177`
- [Verdict](work/Verdict.md) — verified — `models.py:1062`
- [WorkEnums](work/WorkEnums.md) (`TaskType`, `TaskStatus`, `ProjectStatus`) — verified — `models.py:65,838,823`
- [ProjectSpec](work/ProjectSpec.md) — verified, leftover (App Store assets only) — `models.py:1377`
- [JobSpec](work/JobSpec.md) — verified, ghost (no importer; live one is `policy.py:227`) — `models.py:1220`

## routing
- `Model` with `ROUTING_TABLE`, `FALLBACK_CHAIN`, `COST_TABLE` — stub — `models.py`
- `ModelRegistry`, `ModelConfig` — stub — `domain/model_registry.py`
- `ConstraintPlanner` — stub — `planner.py`
- `CapabilityVector` — stub — `domain/capability_vector.py`
- `PhasePolicy`, `Phase` — stub — `domain/phase_policy.py`

## money
All verified 2026-10-22 at 3b5f746d.
- [Budget](money/Budget.md) — verified — `budget.py:36`
- [BudgetHierarchy](money/BudgetHierarchy.md) — verified — `cost.py:136`
- [CostPredictor](money/CostPredictor.md) — verified — `cost.py:499`
- [CostForecaster](money/CostForecaster.md) (with `ForecastReport`, `RiskLevel`) — verified, ghost (no call site found) — `cost.py:580`
- [TokenBudgetManager](money/TokenBudgetManager.md) — verified, ghost (no live caller) — `infrastructure/token_budget.py:136`

## ports
- Port protocols (Cache, State, Event, Config, LLMClient, Planner, Telemetry, Tracing, PolicyEngine, HookRegistry, Validator, TaskExecutor, TaskQueue, SkillStore) — stub — `domain/ports.py`
- `Null*` adapters — stub — `domain/ports.py`
- `StateManager` — stub — `infrastructure/state.py`
- `UnifiedClient`, `APIResponse` — stub — `infrastructure/llm_client.py`

## pipeline
- `TaskPipeline`, `PipelineContext`, `PipelineStage` — stub — `engine_core/pipeline.py`
- Stages (generate, critique, evaluate, validate, preflight, constitution gate) — stub — `engine_core/stages/`
- `ProjectConstitution` — stub — `domain/constitution.py`

## leftover (not nouns, listed so nobody builds on them)
- `state.py`, `api_clients.py` — shims over `infrastructure/`
