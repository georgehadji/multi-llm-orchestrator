# Category 7: User Interface & Experience

> **Focus:** Dashboard panels, workflow builders, visual feedback, and real-time diagnostic displays.
> **Phases:** 6 | **Est. Days:** 12-17 | **Weight:** 9% of total roadmap

---

## Overview

This category provides the **visual interface** for the orchestrator — from plan review through live app preview to real-time generation progress and system diagnostics.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | U1 | Both (UI) | **Plan Review Panel** — interactive task list with approve/reject/refine/prioritize controls before execution | 3-4 | 2 |
| 2 | U2 | Both (UI) | **App Preview + Console + Code Editor** — iframe preview of running app, live stdout streaming, Monaco code editor | 4-5 | 5 |
| 3 | U3 | Both (UI) | **Diff View + Checkpoint Timeline** — side-by-side code diffs from sandboxes, checkpoint history with rollback | 2-3 | 1, 4 |
| 4 | U5 | Both (UI) | **Knowledge/Skills Sidebar** — collapsible panels showing active context, available skills, connected design systems | 2-3 | 7,8,9,10 |
| 5 | U6 | Both (UI) | **Generation Progress** — streaming token output with progress bar, per-call cost, model info, retry indicator | 1-2 | — |
| 6 | D6 | Dyad | **System Diagnostics Drawer** — real-time build/install/test/error status, collapsible panel at bottom of IDE | 2-3 | U2 |

---

## Capability Progression

```
Plan Review Panel (approve/reject before execution)
    │
    ├── App Preview + Console + Code Editor (split-pane workspace)
    ├── Diff View + Timeline (sandbox diffs, checkpoint history)
    │
    ├── Knowledge/Skills Sidebar (persistent context panels)
    ├── Generation Progress (streaming output, cost metrics)
    │
    └── System Diagnostics Drawer (build/install/test status)
```

## Key Innovations

- **Plan Review Panel** enables interactive workflow refinement before a single line of code is generated
- **App Preview + Console** creates a full development environment within the orchestrator
- **Sidebar panels** surface knowledge, skills, and design context persistently
- **Generation progress** shows streaming output in real-time with cost tracking
- **Diagnostics drawer** aggregates all system operations in one collapsible panel
