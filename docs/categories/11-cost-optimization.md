# Category 11: Cost & Token Optimization

> **Focus:** Budget efficiency, per-call cost visibility, and operational metrics.
> **Phases:** 2 | **Est. Days:** 2 | **Weight:** 2% of total roadmap

---

## Overview

This category provides **transparent cost visibility** — from per-call USD tracking to a persistent status bar showing model, cost, tokens, and latency for every operation.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | N7 | Newly | **Per-Call Cost Visibility** — `CallRecord` dataclass logged in `UnifiedClient`, exact USD per LLM call with token breakdown, GET /api/calls endpoint, expensive call warnings | 1 | — |
| 2 | U7 | Both (UI) | **Status Bar** — bottom bar showing active model, cumulative cost, tokens consumed, last operation latency | 1 | — |

---

## Capability Progression

```
Per-Call Cost Logging (every LLM call tracked)
    │
    └── Status Bar (persistent metrics display)
```

## Key Innovations

- **Per-call tracking** moves beyond aggregate budget tracking — every single LLM call is logged with cost, tokens, and event type
- **Expensive call warnings** flag unusually costly operations in real-time
- **Status bar** provides at-a-glance operational awareness without opening a dashboard
