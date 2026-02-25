# Multi-LLM Orchestrator — CLAUDE.md

**Project**: Multi-LLM Orchestrator with cross-provider routing, budget control, and resilience
**Status**: Active development (v1.0 baseline stable, resilience hardening complete)
**Repository**: https://github.com/georgehadji/multi-llm-orchestrator

---

## Project Overview

A production-grade orchestrator for coordinating multiple LLM providers (Anthropic, OpenAI, Google, Kimi, DeepSeek) with intelligent routing, budget hierarchy enforcement, and circuit breaker resilience.

**Core Capabilities:**
- Multi-provider routing with quality-aware model selection
- Cross-run budget hierarchy with pre-flight checks
- Resume capability with auto-detection (prevents infinite loops)
- Policy-driven enforcement (compliance, latency, cost constraints)
- Deterministic validation + LLM-based evaluation scoring
- Telemetry collection and circuit breaker health tracking

---

## Recent Fixes (v1.0 Resilience Hardening)

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

## Testing

**Test Coverage**: 644 tests passing (baseline: 616)
- Surgical bug fixes: 4 tests (merged to master)
- Resilience fixes: 9 new tests (all passing)
- Pre-existing stress test failures: 4 (unchanged, documented)

**Key Test Files**:
- `tests/test_terminal_status_fix.py` — COMPLETED_DEGRADED status distinction
- `tests/test_critique_resilience_fix.py` — 3-strike circuit breaker behavior
- `tests/test_budget_hierarchy_integration.py` — BudgetHierarchy charging

**Running Tests**:
```bash
pytest tests/ -v                    # All tests
pytest tests/test_*_fix.py -v       # Resilience-specific
pytest --tb=short -q               # Summary
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

### 1. Persistent Cross-Run Learning (🔑 Nash Stability Feature)
**Why**: Without persistent learning, users can switch to competitors cost-free
**What**: ModelProfile quality scores aggregated across ALL runs (not session-local)
- Task-type specific routing learned from historical success rates
- Org/team-level learning hierarchies
- Auto-generated routing recommendations ("Save $1,200/month by routing reasoning to DeepSeek-R1")
- Creates lock-in through accumulated intelligence (switching cost = lost learning)
**ROI**: 3-month payback; competitive moat after 6 months of usage

### 2. Observability Dashboard
**What**: Real-time cost tracking, quality heatmaps, policy audit trail
**Why**: Fast to build, immediate stickiness, becomes part of customer's FinOps workflow
**ROI**: 2-week build; high adoption

### 3. Adaptive Fine-tuning Recommendations
**What**: Detect patterns models fail on, recommend fine-tuning with ROI analysis
**Why**: Creates feedback loop (recommend → user fine-tunes through platform → switching cost)
**ROI**: 4-week build; slower payback but high long-term value

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

**Last Updated**: 2026-02-25 (v1.0 Resilience Hardening)
