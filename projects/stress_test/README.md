# Orchestrator Stress-Test Suite — Website Projects

Eight website projects designed to stress-test every dimension of the AI Orchestrator
pipeline: task decomposition, parallel execution, model routing, budget enforcement,
revision cycles, and graceful degradation under pressure.

## Quick Start

```bash
# Run individual tests
python -m orchestrator --file projects/stress_test/01_smoke_landing_page.yaml

# Run all sequentially (recommended: start with #1, then #8 for quick feedback)
for f in projects/stress_test/*.yaml; do
  echo "=== Running: $f ==="
  python -m orchestrator --file "$f"
done
```

## Test Inventory

| # | File | Website | Budget | Time | Concurrency | Stress Dimension |
|---|---|---|---|---|---|---|
| 01 | `01_smoke_landing_page.yaml` | Static Landing Page | $0.80 | 15 min | 1 | **Baseline pipeline** — CODE_GEN + WRITING canary |
| 02 | `02_collaborative_whiteboard.yaml` | Collaborative Whiteboard | $5.00 | 40 min | 8 | **Concurrency** — parallel task scheduler at limit |
| 03 | `03_multitenant_saas_dashboard.yaml` | Multi-Tenant SaaS Dashboard | $6.00 | 90 min | 5 | **Security reasoning** — RBAC, tenant isolation, revision cycles |
| 04 | `04_component_library_30.yaml` | 30-Component Library + Docs | $8.00 | 120 min | 8 | **Scale** — ~100 files, parallel decomposition, token limits |
| 05 | `05_pwa_news_reader.yaml` | PWA Offline-First News Reader | $5.00 | 80 min | 4 | **Unusual tech** — Service Worker, IndexedDB, Background Sync |
| 06 | `06_microfrontend_shell.yaml` | Micro-Frontend Shell (5 teams) | $10.00 | 180 min | 6 | **Architecture** — module contracts, dependency graphs |
| 07 | `07_multilingual_docs_site.yaml` | Multi-Lingual Docs (5 locales) | $7.00 | 120 min | 6 | **WRITING + IMAGE_GEN** — RTL layout, 40 pages content |
| 08 | `08_budget_cascade_stress.yaml` | REST API on $1.50 Budget | $1.50 | 30 min | 2 | **Budget enforcement** — model cascading, graceful degradation |

## Suggested Run Order

### Phase 1: Quick baseline (15–30 min)
- **01** first — if this fails, the pipeline is broken; fix before continuing
- **08** second — quick feedback on budget enforcement

### Phase 2: Core stress (2–3 hours, can run in parallel if you have multiple terminals)
- **02** — concurrency stress
- **03** — security/architecture stress
- **05** — edge-case tech stress

### Phase 3: Heavy stress (4–6 hours, run overnight)
- **04** — scale stress (~100 files)
- **06** — architecture complexity (micro-frontends)
- **07** — multi-locale content + images

## What to Watch For

- **Circuit breaker trips**: provider failures triggering fallback chains
- **Budget exhaustion**: does #08 produce a working result before running out?
- **Concurrency saturation**: do #02 and #04 actually use all workers?
- **Revision loop count**: does #05 (PWA) trigger more critique→revise cycles?
- **Task decomposition quality**: does #06 properly plan architecture before coding?
- **Model selection logs**: which models are chosen under budget pressure (#08 vs #06)?
- **Resume behavior**: kill mid-run and resume — does state recover cleanly?

## Known Gaps This Suite Tests

From the project audit (`ARCHITECTURE_REMEDIATION_PLAN.md`, `PROJECT_SHUTDOWN_AUDIT_SUMMARY.md`):
- **U002 / TD-009**: No formal load/stress testing — these YAMLs fill that gap
- **Stress markers** (`pytest.mark.stress`, `pytest.mark.load`) were registered but unused
- **`tests/stress_test.py`** was documented but never committed — these are the replacement
- **Planned `tests/load/test_concurrent_runs.py`** for 10 concurrent `run_project()` calls — test #02 and #04 approximate this
