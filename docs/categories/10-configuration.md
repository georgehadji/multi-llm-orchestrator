# Category 10: Configuration & Infrastructure

> **Focus:** Deployment tools, workflow engines, isolated execution environments, and configuration synchronization.
> **Phases:** 5 | **Est. Days:** 12-16 | **Weight:** 7% of total roadmap

---

## Overview

This category provides the **execution infrastructure** for the orchestrator — interactive plan workflows, isolated sandbox task execution, configuration synchronization with backends, and local development servers.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | 2 | Replit | **Plan-then-Build Workflow** — extend `dry_run()` into interactive planning with approve/reject/refine before any code executes | 3-4 | — |
| 2 | 3 | Replit | **Self-Review Toggle** — configurable same-model pre-pass before expensive cross-model critique, 30% cost reduction target | 1-2 | — |
| 3 | 4 | Replit | **Isolated Sandbox Tasks** — temp workspace copies with diff → review → merge workflow, AI-assisted conflict resolution | 3-4 | 1 |
| 4 | B4 | Base44 | **Push/Pull Configuration Workflow** — sync entity/auth/agent config with Base44, Supabase, or custom backends | 2-3 | B1 |
| 5 | B6 | Base44 | **Local Development Server** — auto-detect project type, start appropriate dev server with hot reload | 3-4 | — |

---

## Capability Progression

```
Plan-then-Build Workflow (interactive plan execution)
    │
    ├── Self-Review Toggle (cheaper same-model pass)
    ├── Sandbox Tasks (isolated execution, diff → merge)
    │
    ├── Push/Pull Config (sync with backends)
    └── Local Dev Server (auto-detect, hot reload)
```

## Key Innovations

- **Plan-then-Build** changes the execution model from fire-and-forget to review-before-execute
- **Self-Review toggle** saves 30% on critique costs by catching obvious issues with the same model first
- **Sandbox tasks** execute in isolation, produce diffs, and require human approval before merging
- **Push/Pull config** enables a "configuration source of truth" workflow for generated projects
- **Local dev server** auto-detects project type and starts the right server
