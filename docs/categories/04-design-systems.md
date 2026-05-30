# Category 4: Design Systems & Visual Editing

> **Focus:** Design tokens, visual element controls, component registries, and design system propagation across projects.
> **Phases:** 6 | **Est. Days:** 17-23 | **Weight:** 13% of total roadmap

---

## Overview

This category ensures **visual consistency** across generated projects — from structured design tokens through visual editing tools to marketplace-distributed design systems. Phases progress from basic brand injection to fully interactive design panels.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | 6 | Replit | **Design System Injection** — `.design-system.yml` brand tokens (colors, typography, spacing, assets) injected into every AI prompt | 1-2 | — |
| 2 | V1 | v0 | **Design System Registry** — shadcn/ui compatible format: `registry.json`, CSS variables, Tailwind config, component + block specs | 3-4 | 6, 10 |
| 3 | 10 | Lovable | **Design System Projects** — dedicated design system project that propagates updates to all connected projects | 3-4 | 6, 9 |
| 4 | U4 | Both (UI) | **Element Picker** — click elements in app preview to select them, see source code location, edit properties inline | 4-5 | U2, 5 |
| 5 | V2 | v0 | **Visual Design Panel** — full visual controls: typography (font, size, weight, line height, alignment), color, layout (margin/padding), border, shadow, opacity, radius, content | 3-4 | U4, V1 |
| 6 | V7 | v0 | **Templates + Registry Marketplace** — ready-made components + full-page designs, importable via registry URLs | 3-4 | V1, 10 |

---

## Capability Progression

```
Design System Injection (brand .yml)
    │
    ├── Design Registry (shadcn/ui format, tokens, components, blocks)
    ├── Design System Projects (propagation to connected projects)
    │
    ├── Element Picker (click to select, see source)
    ├── Visual Design Panel (full typography/color/layout/border controls)
    │
    └── Templates + Marketplace (registry-based distribution)
```

## Key Innovations

- **shadcn/ui registry format** is the most structured design system approach across all 10 platforms
- **Design system as a project**: live connection means updates propagate automatically
- **Visual Design Panel** combines precise numeric controls with natural-language instructions on the same element
- **Element Picker** bridges the gap between visual design and source code
- **Registry URLs** enable design system distribution and reuse across teams
