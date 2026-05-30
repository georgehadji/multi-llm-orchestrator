# Multi-LLM Orchestrator — Development Roadmap

**Date:** 2026-05-28
**Version:** v6.1.0
**Status:** God File Refactoring Phases 1-7 COMPLETE

---

## Milestones

### M1: Structural Foundation ✓ (2026-05-28)
- engine.py reduced 43% (5,313→3,036 lines)
- ServiceContainer wires all 60+ collaborators via dependency injection
- Application layer decoupled into specialized services (Executor, Evaluator, StateCoordinator, ContextService)
- Topological sort and multi-task coordination delegated to ProjectPlanner/PipelineRunner
- Domain protocols enforced for Interface Segregation
- Externalized configuration moved to `orchestrator/config/*.json`
- EventBus unified (HookRegistry + AgentMessageBus)
- Pipeline delegation: _execute_task ~1,200→30 lines

### M2: Quality & Observability (2026-06)
- Contract tests for all 12+ Protocols in domain/ports.py and engine_core/protocols.py
- structlog adoption across hot path modules
- OpenTelemetry tracing wired through pipeline
- Coverage baseline raised to 40%
- CI gating: pytest + ruff + mypy on every PR

### M3: Performance & Scalability (2026-07)
- Async SQLite pool tuned (target: max_parallel_tasks=10)
- Hot-path profiling: identify and fix top 3 bottlenecks
- Lazy numpy import (saves 0.13s on startup)
- Config consolidation: single import target for all config needs

### M4: Documentation & Governance (2026-07)
- ROADMAP.md (this file)
- Architecture Decision Records (ADRs) for key decisions
- Updated CONTRIBUTING.md with coding standards
- CI pipeline with coverage gates

### M5: Config & Knowledge Housekeeping (2026-08)
- Single config entry point enforced
- Knowledge base deduplication complete
- Environment variable reference documented

### M6: Distributed Execution (2026-Q3)
- Multi-process agent coordination via NATS/Redis
- State DB shared across instances
- Auto-scaling model router for GPU-backed endpoints

---

## Current Short-Term Tasks (2-3 weeks)

| Priority | Task | Effort | Owner |
|----------|------|--------|-------|
| P0 | CI gating (.github/workflows/gate.yml) | S | — |
| P0 | Contract tests for remaining Protocols | M | — |
| P1 | Remove last `self.client.call()` in engine.py | S | — |
| P2 | Hot-path profiling with pyinstrument | S | — |
| P2 | Lazy numpy import in telemetry.py | S | — |

---

## Architecture Decision Records

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Hexagonal Architecture with Protocol-based Ports | Draft |
| 002 | ServiceContainer as DI Wiring Layer | Approved |
| 003 | Pipeline Runner as Execution Abstraction | Approved |
| 004 | max_parallel_tasks = 3 (SQLite WAL) | Approved |
| 005 | EventBus Unification | Approved |
| 006 | Budget moved from models.py | Approved |
| 007 | Role Protocols for Interface Segregation | Approved |

---

## Contributing Guidelines (Summary)

### Code Style
- Python 3.10+ with strict type hints
- Line length: 100 (enforced by black + ruff)
- Imports: stdlib → third-party → orchestrator (absolute preferred)
- Testing: pytest, unit tests for all new code

### Architecture Rules
1. `engine.py` = thin mediator — no business logic
2. `models.py` = pure data — no I/O, no asyncio
3. Application layer depends on Protocols only (no concrete infra imports)
4. All new features need a Protocol interface + contract test

---

*Last updated: 2026-05-28*
