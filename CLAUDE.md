# Multi-LLM Orchestrator — CLAUDE.md

**Project**: Multi-LLM Orchestrator with cross-provider routing, budget control, and resilience
**Status**: Active development (v5.1 complete → v6.0 target)
**Repository**: https://github.com/georgehadji/multi-llm-orchestrator

---

## Architecture & Development Philosophy

> **Master reference:** [`docs/ARCHITECTURE_ROADMAP.md`](docs/ARCHITECTURE_ROADMAP.md)
> Διάβασε αυτό το αρχείο **πριν από οποιαδήποτε αρχιτεκτονική απόφαση ή implementation**.
> Περιέχει: hexagonal architecture overview, paradigm ανά component, design patterns, full roadmap με TDD specs για κάθε feature, και architecture rules.

### 3 Κανόνες που δεν παραβιάζονται ποτέ

1. **`engine.py` = Mediator** — νέα business logic πηγαίνει σε νέο service module, όχι στο engine. Το engine μόνο wire-άρει services.
2. **`models.py` = Pure data** — κανένα I/O, κανένο asyncio, κανένο behavior. Μόνο dataclasses και enums.
3. **TDD χωρίς εξαιρέσεις** — πρώτα failing test (RED), μετά implementation (GREEN), μετά commit.

### Implementation Status

**v6.0 — All Phases Complete ✅**

| Phase | Status | Count | Features |
|-------|--------|-------|----------|
| P0 | ✅ Complete | 3/3 | autonomy, model_routing, verification |
| P1 | ✅ Complete | 12/12 | brain, evaluation, escalation, checkpoints, modes, prompt_enhancer, cost_analytics, competitive, tracing, plan-then-build, memory_bank, context_condensing |
| P2 | ✅ Complete | 10/10 | hierarchy, triggers, workspace, gateway, connectors, sandbox, context_sources, api_server, skills, drift |
| P3 | ✅ Complete | 1/1 | browser_testing |

**Overall: 26 Complete ✅ | 0 Partial 🟡 | 0 Missing ❌** (up from 3/9/17 in v5.1)

### Αρχιτεκτονικό Pattern Summary

| Scope | Pattern | Αρχείο |
|-------|---------|--------|
| Overall | Hexagonal Architecture | — |
| Orchestration | Mediator | `engine.py` |
| LLM Routing | Strategy | `model_routing.py`, `planner.py` |
| Optional Features | Decorator | `verification.py`, `prompt_enhancer.py` |
| Validation | Chain of Responsibility | `validators.py`, `preflight.py` |
| Persistence | Repository + Memento | `state.py`, `checkpoints.py` |
| Cross-cutting | Observer / EventBus | `events.py`, `hooks.py` |
| LLM Providers | Adapter + Protocol | `gateway.py` |
| Budget Hierarchy | Composite | `cost.py` |
| Resilience | State Machine | `resilience.py`, `rate_limiter.py` |

---

## Project Overview

A production-grade orchestrator for coordinating multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek) with intelligent routing, budget hierarchy enforcement, and circuit breaker resilience.

**Core Capabilities:**
- Multi-provider routing with quality-aware model selection
- Cross-run budget hierarchy with pre-flight checks
- Resume capability with auto-detection (prevents infinite loops)
- Policy-driven enforcement (compliance, latency, cost constraints)
- Deterministic validation + LLM-based evaluation scoring
- Telemetry collection and circuit breaker health tracking

---

## Recent Fixes (v5.1 SRE Bug-Finding Pass)

### BUG-001: Budget Reservation Leaked on run_project() Failure
**Problem**: `can_afford_job()` reserves org and team budget. If `run_project()` raised, `release_reservation()` was never called, permanently locking budget. Also `release_reservation()` only cleared org-level, not team-level.
**Solution**: `cost.py` — fixed `release_reservation()` to clear `_team_reserved`. `engine.py` — wrapped `run_project()` in try/except to call `release_reservation()` on any error.
**Tests**: `tests/test_bug_fixes_v5_1.py` (5 tests)

### BUG-002: asyncio.gather Without return_exceptions Left Orphan Tasks
**Problem**: `asyncio.gather()` in level execution without `return_exceptions=True` propagated the first exception immediately; sibling coroutines ran as orphaned background tasks and wrote to `self.results` after the checkpoint snapshot was taken.
**Solution**: `engine.py` — added `return_exceptions=True` to the level-execution gather; exceptions logged explicitly after gather completes.
**Tests**: `tests/test_bug_fixes_v5_1.py` (2 tests)

### BUG-003: SearchResult Mutated In-Place During Reranking
**Problem**: Reranking code mutated fused `SearchResult` objects in-place. If reranking raised mid-loop, `fused[:top_k]` in the except branch returned a corrupted mix of reranker and RRF scores.
**Solution**: `hybrid_search_pipeline.py` — creates new `SearchResult` instances instead of mutating in-place.
**Tests**: `tests/test_bug_fixes_v5_1.py` (3 tests)

### BUG-004: BudgetHierarchy.remaining(level="team") Ignored Pending Reservations
**Problem**: `remaining(level="team")` only subtracted actual spend, not pending reservations, returning inflated available budget.
**Solution**: `cost.py` — deducts `_team_reserved.get(key, 0.0)` from team-level remaining, matching org-level logic.
**Tests**: `tests/test_bug_fixes_v5_1_round2.py` (4 tests)

### BUG-005: RateLimiter TOCTOU Between check() and record()
**Problem**: Two concurrent async callers could both pass `check()` before either called `record()`, since the window state isn't updated until `record()`. Both would then proceed, collectively exceeding the limit.
**Solution**: `rate_limiter.py` — `check()` now atomically reserves in-flight slots (`_in_flight_tokens`, `_in_flight_reqs`). `record()` settles the reservation. New `release()` method for error paths. `engine.py` — try/except wraps API calls to call `release()` on failure.
**Tests**: `tests/test_bug_fixes_v5_1_round2.py` (5 tests)

### OPENAI-T: temperature Parameter Forwarded to OpenAI Models
**Problem**: `_call_openai()` passed `temperature=temperature` to the OpenAI API. OpenAI's newer models (o1/o3/o4-series) fix temperature at 1 and reject the parameter with an error.
**Solution**: `api_clients.py` — removed `temperature` from `_call_openai()` completions call.
**Tests**: `tests/test_bug_fixes_v5_1_round2.py` (1 test)

---

## Previous Fixes (v1.0 Resilience Hardening)

### Fix #1: Terminal COMPLETED_DEGRADED Status (95% confidence)
**Problem**: Infinite resume loops when runs complete with failed validation
- PARTIAL_SUCCESS was conflating two scenarios: interrupted mid-run (resumable) vs completed-degraded (not resumable)
- Every load would re-enter completed-degraded runs, causing unbounded re-execution and output destruction
- Forced sequential execution on resumption (3-5x slower than parallel)

**Solution**: Added COMPLETED_DEGRADED terminal status
- Modified: `orchestrator/models.py` (ProjectStatus enum)
- Modified: `orchestrator/engine.py` (_determine_final_status logic)
- PARTIAL_SUCCESS now only returned for genuinely incomplete runs (missing results)
- COMPLETED_DEGRADED returned when all tasks executed but some failed validation
- Tests: `tests/test_terminal_status_fix.py` (2 tests)

**Impact**: Prevents infinite resume loops while preserving legitimate resume capability

---

### Fix #2: Critique Resilience with Graduated Circuit Breaker (92% confidence)
**Problem**: Single transient critique error immediately disabled reviewer
- Any critique failure (429 rate limit, timeout) immediately set `api_health[reviewer] = False`
- Remaining iterations used only self-evaluation with inflated scores (silent quality collapse)
- No recovery path for transient errors
- No audit trail despite logs showing failures

**Solution**: Route critique exceptions through circuit breaker
- Modified: `orchestrator/engine.py` (line 861, critique exception handler)
- Changed from: `self.api_health[reviewer] = False` (1-strike kill)
- Changed to: `self._record_failure(reviewer, error=e)` (3-strike circuit breaker)
- Leverages existing circuit breaker logic:
  - Transient errors (429, timeout) → increment consecutive_failures counter
  - 3 consecutive failures → mark model unhealthy (circuit breaker trips)
  - Success → reset counter (transient error recovery)
  - Permanent errors (401, 404) → immediate mark unhealthy (no counter)
- Tests: `tests/test_critique_resilience_fix.py` (5 tests)

**Impact**: Transient failures become recoverable; persistent issues still blocked

---

### Fix #3: BudgetHierarchy Charging Integration (97% confidence)
**Problem**: `charge_job()` not called from `run_job()`, breaking cross-run budget enforcement
- `can_afford_job()` pre-flight check never saw accumulated spend
- Cross-run budgets completely non-functional
- Org budget tracking stayed at zero forever

**Solution**: Verified `charge_job()` integration
- Implementation: `orchestrator/engine.py` lines 300-305 in `run_job()`
- Calls `self._budget_hierarchy.charge_job(job_id, team, actual_spend)` after each project
- Enables pre-flight check to enforce multi-run budgets correctly
- Tests: `tests/test_budget_hierarchy_integration.py` (2 tests)

**Impact**: Cross-run budget enforcement now functional

---

## Architecture Notes

### ProjectStatus Enum Lifecycle
```
PARTIAL_SUCCESS → genuinely incomplete (missing results, resumable)
COMPLETED_DEGRADED → completed with failed validation (terminal, not resumable)
SUCCESS → all tasks passed validation
BUDGET_EXHAUSTED → halted due to budget
TIMEOUT → halted due to time limit
SYSTEM_FAILURE → unexpected error (no results)
```

### Circuit Breaker (3-Strike Threshold)
- Located in `_record_failure()` and `_record_success()`
- Per-model consecutive failure tracking
- Permanent errors (401, 404) bypass threshold, immediate disable
- Success resets counter, enabling recovery

### Budget Hierarchy Integration
- Org-level budget cap via `BudgetHierarchy(org_max_usd=X)`
- Team-level budgets via `team_budgets={team: amount}`
- Per-job spending tracked and accumulated
- Pre-flight `can_afford_job()` blocks jobs exceeding remaining budget

---

## Codebase Enhancer (Phase 1-2 POC)

**Status**: Complete (Static analysis + Semantic understanding)

A feature enabling the orchestrator to analyze existing codebases, understand their functionality, and generate prioritized improvement suggestions. This is a proof-of-concept implementation of Phases 1-2, with Phase 3 (auto-implementation) deferred to future work.

### Architecture

**Three-Stage Pipeline**:

1. **Static Analysis** (`CodebaseAnalyzer`)
   - Recursive directory scanning with smart filtering (skip .git, node_modules, .venv, etc.)
   - File counting by language (Python, JavaScript, TypeScript, Go, Rust, Java, C++, Ruby, PHP, Swift)
   - Project type detection (FastAPI, Django, Flask, Next.js, React, Vue, Go, Rust)
   - Line-of-code counting with 500-line limits per file to save tokens

2. **Semantic Understanding** (`CodebaseUnderstanding`)
   - Integrates static analysis with DeepSeek Reasoner via orchestrator.run_task()
   - Reads key files (README.md, main.py, app.py, index.js, package.json)
   - Constructs rich prompts combining file counts + key file contents
   - Parses JSON responses to extract: purpose, primary_patterns, anti_patterns, test_coverage, documentation
   - Graceful fallback to heuristic-based analysis if LLM unavailable

3. **Improvement Generation** (`ImprovementSuggester`)
   - Analyzes CodebaseProfile for gaps: test coverage, documentation, anti-patterns, missing infrastructure
   - Generates suggestions with: title, description, impact statement, effort_hours estimate, priority (HIGH/MEDIUM/LOW), category
   - Prioritized output sorted by impact and effort (HIGH priority + low effort first)

### Usage

**CLI Command**:
```bash
python -m orchestrator --analyze-codebase /path/to/project
```

**Programmatic**:
```python
from orchestrator.codebase_understanding import CodebaseUnderstanding
from orchestrator.improvement_suggester import ImprovementSuggester

understanding = CodebaseUnderstanding()
profile = await understanding.analyze(codebase_path="/path/to/project")

suggester = ImprovementSuggester()
improvements = suggester.suggest(profile)

for imp in improvements:
    print(f"{imp.priority}: {imp.title} ({imp.effort_hours}h)")
```

### Components

**Key Classes**:
- `CodebaseAnalyzer`: Static file scanning, language/project detection
- `CodebaseMap`: Dataclass with total_files, total_lines_of_code, files_by_language, project_type, has_tests, has_docs
- `CodebaseProfile`: Semantic profile from LLM analysis
- `CodebaseUnderstanding`: Orchestrates static analysis + LLM understanding
- `Improvement`: Suggestion dataclass (title, description, impact, effort_hours, priority, category)
- `ImprovementSuggester`: Generates prioritized suggestions from profile

**Files**:
- Created: `orchestrator/codebase_analyzer.py` (216 lines)
- Created: `orchestrator/codebase_profile.py` (45 lines)
- Created: `orchestrator/codebase_understanding.py` (180 lines)
- Created: `orchestrator/improvement_suggester.py` (119 lines)
- Modified: `orchestrator/cli.py` (added --analyze-codebase command)
- Modified: `orchestrator/__init__.py` (exported new classes)

### Test Coverage

**16 tests across 5 test files** (all passing):
- `tests/test_codebase_analyzer.py`: 9 tests (file scanning, language detection, project type detection)
- `tests/test_codebase_profile.py`: 2 tests (profile creation, string representation)
- `tests/test_codebase_understanding.py`: 3 tests (async analysis, key file reading, LLM integration)
- `tests/test_cli_analyze.py`: 1 test (CLI command acceptance)
- `tests/test_improvement_suggester.py`: 2 tests (suggestion generation, effort estimates)
- `tests/test_codebase_enhancer_e2e.py`: 4 E2E tests (static analysis workflow, semantic analysis, improvement suggestions, full pipeline)

**Run tests**:
```bash
pytest tests/test_codebase_*.py -v
pytest tests/test_improvement_suggester.py -v
pytest tests/test_cli_analyze.py -v
```

### Design Decisions

**DeepSeek Reasoner for Analysis**: Chose reasoning model (TaskType.REASONING) over cheaper models to understand complex architectural patterns and anti-patterns with high accuracy.

**Graceful LLM Fallback**: Static-analysis-only mode available if DeepSeek unavailable, preventing feature breakage on transient errors.

**Configurable Suggestion Rules**: ImprovementSuggester maps CodebaseProfile fields to actionable suggestions, making it easy to extend with domain-specific rules.

**Effort Estimates as Heuristics**: Effort hours (1-6 range) are teaching tools, not precise forecasts. Users adapt based on team context.

### Future Work (Phase 3+)

- Auto-implementation of suggestions (generate code, run tests, commit)
- Persistent learning across codebases (remember which suggestions succeeded)
- Integration with GitHub workflows (auto-PR suggestions)
- Custom suggestion rules per organization (domain-specific improvement patterns)

---

## v5.1 Intelligence & Multi-tenant Features

Four features implemented via TDD + subagent-driven development, derived from analysis of production-grade OSS projects (LiteLLM, QMD, Mnemo Cortex).

### Feature #1: Content Preflight Gate

**Status**: Complete — `orchestrator/preflight.py` extended

Content-level validation gate that inspects every LLM response before delivery. Actions:
- **PASS**: Response is clean, deliver as-is
- **WARN**: Flag inconsistency or missing context, log but deliver
- **BLOCK**: Factual error or policy violation — reject and request retry

Distinct from the budget preflight (`can_afford_job()`): this is a *quality* gate, not a *cost* gate.

**Files**: `orchestrator/preflight.py`, `tests/test_preflight_integration.py`

---

### Feature #2: Hybrid Search Pipeline (RRF + Query Expansion)

**Status**: Complete

Unified search pipeline combining BM25 keyword search + vector similarity via Reciprocal Rank Fusion (RRF), with optional LLM-based query expansion.

**Architecture:**
- `QueryExpander` — Calls DeepSeek-Chat to generate N alternative phrasings of the query. Fail-open: returns `[original_query]` on error.
- `HybridSearchPipeline` — Runs BM25 (`BM25Search`) + vector (`KnowledgeBase.find_similar()`) in parallel, merges via RRF (`score = Σ(1/(k+rank))`, k=60), optionally re-ranks with `LLMReranker`.
- `Orchestrator.hybrid_search()` — Delegates to pipeline; accepts `use_query_expansion` and `use_reranking` flags.

**Files:**
- `orchestrator/query_expander.py`
- `orchestrator/hybrid_search_pipeline.py`
- Modified: `orchestrator/engine.py` (`hybrid_search()`, `__init__`)
- Tests: `tests/test_query_expander.py` (3), `tests/test_hybrid_search_pipeline.py` (5), `tests/test_hybrid_pipeline_engine_integration.py` (2)

**Usage:**
```python
results = await orch.hybrid_search(
    query="fibonacci recursion",
    project_id="proj1",
    use_query_expansion=True,  # expands to 3 variants
    use_reranking=True,        # LLM reranks final list
)
```

---

### Feature #3: TPM/RPM Rate Limiting per Tenant

**Status**: Complete

In-memory sliding-window rate limiter per `(tenant, model)` pair. Prevents burst abuse even when total budget is not exhausted.

**Algorithm:** 60-second deque of `(timestamp, tokens)` tuples. `check()` raises `RateLimitExceeded` (with `limit_type` and `retry_after`) if the projected window would exceed TPM or RPM. `record()` appends after a successful call.

**Files:**
- `orchestrator/rate_limiter.py` — `RateLimiter`, `RateLimitExceeded`
- Modified: `orchestrator/engine.py` (`_rate_limiter`, `configure_rate_limits()`)
- Tests: `tests/test_rate_limiter.py` (7), `tests/test_rate_limiter_engine_integration.py` (4)

**Usage:**
```python
orch.configure_rate_limits("acme", "deepseek-chat", tpm=50_000, rpm=100)
# Engine now enforces limits on every primary generation call for this tenant
```

---

### Feature #4: Hot/Warm/Cold Session Lifecycle

**Status**: Complete

Automatic session lifecycle management with LLM-based summarization on HOT→WARM transitions.

**Architecture:**
- `SessionLifecycleManager` wraps `MemoryTierManager`
- Before `migrate_tiers()` runs, calls `_summarize_due_entries()`: for each HOT entry with `age_days >= hot_ttl_days`, calls DeepSeek-Chat to write a 2-3 sentence summary into `entry.summary`. Fail-open: LLM errors let the entry migrate without a summary.
- `start()` / `stop()` manage an asyncio background scheduler (configurable interval, default 1h). `stop()` is called automatically in `Orchestrator.__aexit__`.
- `configure_session_lifecycle()` raises `RuntimeError` if called while scheduler is running.

**Files:**
- `orchestrator/session_lifecycle.py` — `SessionLifecycleManager`
- Modified: `orchestrator/engine.py` (`_lifecycle_manager`, `configure_session_lifecycle()`, `__aexit__`)
- Tests: `tests/test_session_lifecycle.py` (6), `tests/test_session_lifecycle_engine_integration.py` (4)

**Usage:**
```python
orch.configure_session_lifecycle(migration_interval_hours=2, llm_model="deepseek-chat")
await orch._lifecycle_manager.start()  # Begin automatic migrations
```

---

## Testing

**Test Coverage**: ~720 tests passing (baseline: 616)
- Surgical bug fixes: 4 tests
- Resilience fixes: 9 tests
- Codebase Enhancer feature: 16 tests
- Content Preflight Gate: ~14 tests
- Hybrid Search Pipeline: 10 tests (query_expander + pipeline + engine integration)
- Rate Limiter: 11 tests (unit + engine integration)
- Session Lifecycle: 10 tests (unit + engine integration)
- v5.1 SRE bug fixes round-1: 10 tests (BUG-001 budget leak, BUG-002 gather orphans, BUG-003 SearchResult mutation)
- v5.1 SRE bug fixes round-2: 10 tests (BUG-004 team remaining, BUG-005 rate limiter TOCTOU, OpenAI temperature)
- Pre-existing stress test failures: 4 (unchanged, documented)

**Key Test Files**:
- `tests/test_terminal_status_fix.py` — COMPLETED_DEGRADED status distinction
- `tests/test_critique_resilience_fix.py` — 3-strike circuit breaker behavior
- `tests/test_budget_hierarchy_integration.py` — BudgetHierarchy charging
- `tests/test_codebase_analyzer.py` — Static file scanning, language/project detection
- `tests/test_preflight_integration.py` — Content preflight gate (PASS/WARN/BLOCK)
- `tests/test_query_expander.py` — LLM-based query expansion
- `tests/test_hybrid_search_pipeline.py` — RRF fusion pipeline
- `tests/test_rate_limiter.py` — Sliding-window TPM/RPM limits
- `tests/test_session_lifecycle.py` — HOT/WARM migration with LLM compression
- `tests/test_bug_fixes_v5_1.py` — BUG-001/002/003 regression tests
- `tests/test_bug_fixes_v5_1_round2.py` — BUG-004/005/OpenAI-T regression tests

**Running Tests**:
```bash
python -m pytest tests/ -v -m "not slow"          # All tests (skip slow)
python -m pytest tests/test_rate_limiter.py -v --no-cov  # Single module
python -m pytest --tb=short -q                     # Summary
```

---

## Development Workflow

### Using Git Worktrees
This project uses git worktrees for isolated feature branches:

```bash
# Create new worktree for feature
git worktree add .claude/worktrees/feature-name -b feature-name

# Work in worktree, test thoroughly
cd .claude/worktrees/feature-name
pytest tests/
git commit -m "fix: description"
git push -u origin feature-name

# After merge, clean up
cd ../..
git worktree remove .claude/worktrees/feature-name
```

Note: `.claude/worktrees/` is in `.gitignore` for safety

### Test-Driven Development
All features follow TDD discipline:
1. Write failing test (RED phase)
2. Verify test fails with expected error
3. Implement minimal code to pass (GREEN phase)
4. Run full suite to verify no regressions
5. Commit with detailed message

---

## Next High-Value Additions

### 1. Command-Specific Token Compression (✅ Implemented)
**What**: 50+ domain-specific output filters (git log, pytest, eslint, docker ps) for 60-90% token reduction
**Why**: token_optimizer.py is generic; domain-specific strategies give targeted savings for agentic workflows
**ROI**: Low complexity, high cost reduction for CI/agent scenarios
**Status**: Implemented in `orchestrator/token_optimizer.py`

### 2. A2A External Agent Client (✅ Implemented)
**What**: Client-side A2A invocation of external agents (LangGraph, Vertex AI, Azure AI Foundry)
**Why**: a2a_protocol.py is server-side only; client-side unlocks ecosystem integration
**ROI**: Expands orchestrator's scope dramatically — tasks can be delegated to specialized external agents
**Status**: Implemented in `orchestrator/a2a_protocol.py`

### 3. Persona Modes (✅ Implemented)
**What**: Per-request behavioral modes (Strict for production, Creative for ideation)
**Why**: Policy system exists but not persona-based; mode switching without config changes
**ROI**: Improves UX, low complexity
**Status**: Implemented in `orchestrator/persona_modes.py`

### 4. Persistent Cross-Run Learning (✅ Implemented)
**Why**: Without persistent learning, users can switch to competitors cost-free
**What**: ModelProfile quality scores aggregated across ALL runs (not session-local)
- Task-type specific routing learned from historical success rates
- Auto-generated routing recommendations ("Save $1,200/month by routing reasoning to DeepSeek-R1")
**ROI**: 3-month payback; competitive moat after 6 months of usage
**Status**: Implemented in `orchestrator/learning_aggregator.py`

### 5. Multi-tenant API Gateway (✅ Implemented)
**What**: JWT/API-key auth layer, exposing orchestrator as hosted service
**Why**: Currently library/CLI only; gateway unlocks SaaS deployment model
**ROI**: High long-term value, high engineering cost
**Status**: Implemented in `orchestrator/multi_tenant_gateway.py`

---

## Known Limitations & TODOs

- Stress tests: 4 pre-existing failures in `tests/stress_test.py` (S2, S6, S7) — documented but not blocking
- Resume detection: Uses file modification time heuristic; could be more robust
- Policy system: Enforcement mode selection (HARD/SOFT/MONITOR) not yet fully integrated

---

## Code Quality Standards

- **Testing**: TDD required; all features must have failing test first
- **Commits**: Atomic, descriptive messages with context
- **Documentation**: Docstrings on all public methods
- **Code Review**: Via GitHub PR; resilience fixes approved before merge

---

## Contact & Questions

For questions about architecture, strategy, or development approach, refer to:
- PR #5: Resilience black-swan fixes (detailed rationale)
- Issue discussions: Architecture and design rationale documented there
- Commit messages: Detailed technical context in each commit

---

**Last Updated**: 2026-03-15 (v5.1 — Hybrid Search, Rate Limiting, Session Lifecycle, Preflight Gate)
