# Category 12: Scoping & Safety

> **Focus:** File-level AI scope control, entity schema validation, and browser-based app testing.
> **Phases:** 3 | **Est. Days:** 9-12 | **Weight:** 6% of total roadmap

---

## Overview

This category provides **fine-grained control** over what the AI can and cannot modify, schema-level validation, and comprehensive browser testing infrastructure.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | W2 | Bolt.new | **Target/Lock Files** — scope AI attention to specific files ("Target"), prevent AI from modifying specific files ("Lock"), directory-level locking | 3-4 | U1 |
| 2 | B5 | Base44 | **Entity RLS Validation** — auto-generate row-level + field-level security rules from entity relationships, validate schema integrity | 2-3 | B1 |
| 3 | 5 | Replit | **Browser-Based App Testing** — Playwright test scenarios, video recording, auto-generated test scripts from app structure, auto-fix from failures | 4-5 | — |

---

## Capability Progression

```
Target/Lock Files (per-file AI scope control)
    │
    ├── Entity RLS Validation (schema integrity + auto-security rules)
    │
    └── Browser Testing (Playwright + video + auto-fix)
```

## Key Innovations

- **Target/Lock files** gives the user surgical control over AI scope — unprecedented granularity
- **Entity RLS** auto-generates row-level security rules based on entity relationships and auth patterns
- **Browser testing** goes beyond unit tests — real browser interaction with video recording and auto-generated test scripts
