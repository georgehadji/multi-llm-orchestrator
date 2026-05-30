# Architectural Refactoring Plan: Hexagonal & Event-Driven Migration

## Objective
Progressively refactor the AI Orchestrator codebase to fully realize its intended Hexagonal (Ports & Adapters) and Event-Driven Agent architecture, using the "Strangler Fig" pattern to minimize disruption.

## Background & Motivation
The architectural audit revealed a system trapped between legacy procedural patterns and a modern, event-driven agent model. Key issues include:
- A massive 3,500+ line God Object (`engine.py`) orchestrating all tasks.
- Three competing event systems (`events/`, `unified_events/`, `nash/events.py`).
- Domain contamination in `models.py` (which houses global routing/cost state).
- Application layers directly calling infrastructure, bypassing domain ports.

## Scope & Impact
- **Impacted Areas:** `orchestrator/engine.py`, `orchestrator/models.py`, `orchestrator/budget.py`, and the event subsystems.
- **Goal:** Strict decoupling, unified messaging, and enhanced testability. This will unblock further agent autonomy features by providing a stable execution foundation.

## Proposed Solution
We will adopt an incremental "Strangler Fig" approach to gradually replace the monolithic components with well-defined Application Services and a unified Event Bus.

## Phased Implementation Plan

### Phase 1: Event System Consolidation [COMPLETED]
*Goal: Establish a single source of truth for all inter-component messaging.*
- **1.1:** Audit all active listeners and publishers in `events/` and `nash/events.py`. **[DONE]**
- **1.2:** Migrate all domain events to `unified_events/core.py`. **[DONE]**
- **1.3:** Reroute application publishing to the `UnifiedEventBus` via the `ServiceContainer`. **[DONE]**
- **1.4:** Deprecate and remove the legacy event systems (replaced with backward-compat shims). **[DONE]**

### Phase 2: Decoupling Domain State [IN PROGRESS]
*Goal: Purge infrastructure configuration and global state from `models.py`.*
- **2.1:** Extract infrastructure-heavy constants from `models.py`:
    - `COST_TABLE` (Provider-specific pricing)
    - `ROUTING_TABLE` (Task-to-model mapping)
    - `FALLBACK_CHAIN` (Cross-model recovery paths)
    - `DEFAULT_THRESHOLDS` (Quality gate settings)
    - `MAX_OUTPUT_TOKENS` (Task-specific limits)
- **2.2:** Update `ServiceContainer` to provide high-level Domain Services:
    - `RoutingService` (abstracts model selection and fallbacks)
    - `CostService` (abstracts pricing and budget calculation)
    - `ConfigurationService` (abstracts task-specific thresholds)
- **2.3:** Implement Infrastructure Adapters (e.g., `JsonConfigurationAdapter`) to load these values from external JSON/YAML files, removing hardcoded state from the Python domain model.
- **2.4:** Resolve the high fan-in circular dependency risks by ensuring `models.py` only contains pure data structures (DTOs).

### Phase 3: Dismantling the God Object
*Goal: Dissolve `engine.py` into cohesive Application layer services.*
- **3.1:** Extract pipeline execution stages from `engine.py` into `orchestrator/application/pipeline_runner.py`.
- **3.2:** Move planning and routing logic fully into `planner.py` and `policy_engine.py`.
- **3.3:** Audit `engine.py` for direct infrastructure imports (e.g., SQLite, `aiosqlite`, LLM client implementations) and replace them with calls to defined domain Ports (e.g., `StatePort`, `LLMClient`).
- **3.4:** Reduce `engine.py` to a thin facade/API entry point that solely relies on the `ServiceContainer` for dependency resolution.

## Verification
- Run existing test suites after each incremental phase.
- Ensure 100% event delivery compliance during Phase 1 using temporary event-shadowing tests.
- Validate that the dependency graph shows zero imports of infrastructure modules from `engine.py` post-Phase 3.

## Migration & Rollback
- Each phase is merged in small, reversible pull requests.
- Shims and adapters are used to maintain compatibility with unrefactored components.
- Side-by-side execution can be utilized for critical systems (like the event bus) to allow quick fallback.
