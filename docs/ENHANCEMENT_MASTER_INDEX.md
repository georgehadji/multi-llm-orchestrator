# Enhancement Master Index — All 10 Platforms, Categorized

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Total:** 61 phases from 10 platforms | 132-190 estimated days | 79 new files, 156 modifications

---

## Category 1: Autonomous Execution & Agent Intelligence (12 phases)

Agents that can run independently, self-correct, and execute multi-step workflows.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | X1 | Create.xyz | **Autonomous Agent Loop** — build → test in browser → fix → repeat until goal achieved | 4-5 | V3, V5, 5, N3 |
| 2 | X2 | Create.xyz | **Parallel Autonomous Agents** — multiple Max instances, one codebase, conflict detection | 3-4 | X1, D1 |
| 3 | V3 | v0 | **Browser-Use Agent** — opens app, uses it, debugs, critiques, sends screenshots | 3-4 | 5, V1, N3 |
| 4 | X4 | Create.xyz | **Multi-Mode Selector** — 6 modes (Discuss/Plan/Fast/Thinking/Auto/Max) | 2-3 | W1, X1 |
| 5 | W1 | Bolt.new | **Agent Profiles** — Standard/Max/Creative/Conservative/Research profiles | 2-3 | — |
| 6 | N4 | Newly | **Brainstorming Mode** — AI asks 3-5 clarifying questions before planning | 2-3 | 2 |
| 7 | W3 | Bolt.new | **Quick Action Buttons** — contextual actions after Plan Mode (implement, refine, etc.) | 1-2 | N4, U1 |
| 8 | N1 | Newly | **Ask Mode** — query-only answers, no code gen, 10x cheaper | 1-2 | — |
| 9 | D4 | Dyad | **Undo + Retry** — revert change, switch to better model, retry same prompt | 2-3 | V6, D1 |
| 10 | V4 | v0 | **Permission Modes** — Ask/Auto/Full for agent tool execution safety | 1-2 | — |
| 11 | V5 | v0 | **Auto-Error Fix Button** — one-click fix from validation/test/build errors | 2-3 | — |
| 12 | N3 | Newly | **Screenshot-Based Debugging** — paste error screenshots, AI diagnoses + fixes | 2-3 | 5 |

---

## Category 2: Version Control & State Management (7 phases)

Checkpoints, versioning, releases, and project state management.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 13 | 1 | Replit | **Checkpoints** — named snapshots (state + files + context), bidirectional rollback | 2-3 | — |
| 14 | V6 | v0 | **Versions as First-Class** — auto-version every code change, diff, revert, chain nav | 3-4 | 1, U3 |
| 15 | R2 | Retool | **App Release Management** — semantic versioning, draft/published, diff, notes | 2-3 | V6 |
| 16 | N2 | Newly | **Chat-Level Restore Points** — restore to before any prompt in history | 1-2 | 1 |
| 17 | N5 | Newly | **Auto-Commit Messages** — descriptive git messages per task (fast/LLM) | 1 | 1, 4 |
| 18 | W4 | Bolt.new | **Project vs Site Separation** — dev workspace, explicit publish, unpublished changes | 2-3 | V6 |
| 19 | D5 | Dyad | **Copy Project** — duplicate project for safe experimentation, merge back | 1-2 | — |

---

## Category 3: Context & Knowledge Management (5 phases)

Persistent instructions, context windows, and knowledge reuse.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 20 | 7 | Lovable | **Knowledge System** — workspace + project `.md` files always injected into context | 2-3 | — |
| 21 | 8 | Lovable | **Skills System** — name playbooks with trigger descriptions, `/skill-name` invocation | 2-3 | 7 |
| 22 | D1 | Dyad | **Multiple Contexts per Project** — separate AI conversations, shared codebase + versions | 3-4 | V6 |
| 23 | D3 | Dyad | **Summarize into New Context** — LLM compresses full conversation into fresh context | 1-2 | D1 |
| 24 | 9 | Lovable | **Cross-Project Referencing** — `@ProjectName` to reuse code from other projects | 2-3 | 1, 7 |

---

## Category 4: Design Systems & Visual Editing (6 phases)

Design tokens, visual element controls, and component registries.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 25 | V1 | v0 | **Design System Registry** — shadcn/ui format, registry.json, CSS tokens, Tailwind config | 3-4 | 6, 10 |
| 26 | V2 | v0 | **Visual Design Panel** — typography, color, layout, border, shadow, opacity, radius controls | 3-4 | U4, V1 |
| 27 | U4 | Both (UI) | **Element Picker** — click elements in preview, see source code, edit properties | 4-5 | U2, 5 |
| 28 | 6 | Replit | **Design System Injection** — `.design-system.yml` brand config injected into prompts | 1-2 | — |
| 29 | 10 | Lovable | **Design System Projects** — dedicated design project propagating to connected projects | 3-4 | 6, 9 |
| 30 | V7 | v0 | **Templates + Registry Marketplace** — ready-made templates, marketplace for sellers | 3-4 | V1, 10 |

---

## Category 5: Project Output Quality (7 phases)

Rich components, modules, types, docs, and i18n for generated projects.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 31 | R1 | Retool | **Module System** — reusable component+query packages with inputs/outputs | 3-4 | — |
| 32 | R4 | Retool | **Component Library** — 100+ prebuilt UI components, 10 categories, Storybook | 4-5 | V1, B2 |
| 33 | B1 | Base44 | **Configuration-as-Code** — entity/auth/agent schemas as JSON/YAML output | 2-3 | — |
| 34 | B2 | Base44 | **Dynamic Type Generation** — TypeScript + Pydantic types from entity schemas | 1-2 | B1 |
| 35 | R7 | Retool | **App Documentation** — README, user guide, API reference, changelog, architecture doc | 1-2 | — |
| 36 | R6 | Retool | **Internationalization** — i18n, 16 locales, key extraction, LLM translation | 2-3 | R4 |
| 37 | B3 | Base44 | **Automations/Scheduling** — cron, simple schedules, entity events, webhooks | 2-3 | — |

---

## Category 6: Security & Validation (1 phase)

Structured security auditing and vulnerability management.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 38 | D2 | Dyad | **AI Security Review** — severity levels (Critical→Info), CWE IDs, SECURITY_RULES.md | 2-3 | N1 |

---

## Category 7: User Interface & Experience (6 phases)

Dashboard panels, workflow builders, and visual feedback.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 39 | U1 | Both (UI) | **Plan Review Panel** — interactive task approve/reject/refine before execution | 3-4 | 2 |
| 40 | U2 | Both (UI) | **App Preview + Console** — iframe preview, live stdout streaming, Monaco editor | 4-5 | 5 |
| 41 | U3 | Both (UI) | **Diff View + Checkpoint Timeline** — side-by-side diffs, checkpoint rollback | 2-3 | 1, 4 |
| 42 | U5 | Both (UI) | **Knowledge/Skills Sidebar** — persistent context, /commands, @references | 2-3 | 7,8,9,10 |
| 43 | U6 | Both (UI) | **Generation Progress** — streaming output, progress bar, cost/tokens per generation | 1-2 | — |
| 44 | D6 | Dyad | **System Diagnostics Drawer** — real-time build/install/test/error status | 2-3 | U2 |

---

## Category 8: Integration & Data Sources (4 phases)

Data source connections, query generation, and integration discovery.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 45 | R5 | Retool | **Data Source Integration** — 50+ templates, connection pools, health checks, Docker Compose | 2-3 | — |
| 46 | R3 | Retool | **AI Query Generation** — natural language → SQL/JS/GraphQL with schema validation | 2-3 | B1, B2 |
| 47 | X3 | Create.xyz | **Slash Command Integrations** — 100+ via `/chatgpt`, `/stripe`, `/google-maps` | 2-3 | R5 |
| 48 | N6 | Newly | **Two-Way Git Sync** — pull external changes back into AI context | 2-3 | 5 |

---

## Category 9: Collaboration & Team Features (3 phases)

Multi-user work, team templates, and marketplace.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 49 | W5 | Bolt.new | **Team Templates** — shareable project configs (entities, auth, modules, integrations) | 2-3 | V7 |
| 50 | X5 | Create.xyz | **Template Marketplace** — sell/buy templates for credits, review & rating system | 3-5 | V7 |
| 51 | B7 | Base44 | **Skills for External Agents** — SKILL.md files for Claude, Cursor, Copilot | 2-3 | 8, B1 |

---

## Category 10: Configuration & Infrastructure (5 phases)

Deployment tools, dev servers, and configuration sync.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 52 | 2 | Replit | **Plan-then-Build Workflow** — interactive plan refinement before execution | 3-4 | — |
| 53 | 3 | Replit | **Self-Review Toggle** — same-model pre-pass before cross-model critique | 1-2 | — |
| 54 | 4 | Replit | **Sandbox Tasks** — isolated task execution with diff → review → merge | 3-4 | 1 |
| 55 | B4 | Base44 | **Push/Pull Config** — sync config with Base44, Supabase, custom backends | 2-3 | B1 |
| 56 | B6 | Base44 | **Local Dev Server** — auto-detect project type, start appropriate server | 3-4 | — |

---

## Category 11: Cost & Token Optimization (2 phases)

Budget efficiency and per-call visibility.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 57 | N7 | Newly | **Per-Call Cost Visibility** — exact USD per LLM call, token breakdown | 1 | — |
| 58 | U7 | Both (UI) | **Status Bar** — model, cost, tokens, latency per operation | 1 | — |

---

## Category 12: Scoping & Safety (3 phases)

File-level AI scope control and entity validation.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 59 | W2 | Bolt.new | **Target/Lock Files** — AI focus on specific files, lock others from modification | 3-4 | U1 |
| 60 | B5 | Base44 | **Entity RLS Validation** — auto-generate row-level + field-level security rules | 2-3 | B1 |
| 61 | 5 | Replit | **Browser-Based App Testing** — Playwright tests, video recording, auto-fix | 4-5 | — |

---

## Summary by Category

| Category | Phases | Days | Weight |
|----------|--------|------|--------|
| 1. Autonomous Execution | 12 | 24-36 | 20% |
| 2. Version Control | 7 | 12-17 | 9% |
| 3. Context & Knowledge | 5 | 10-14 | 8% |
| 4. Design Systems | 6 | 17-23 | 13% |
| 5. Output Quality | 7 | 17-23 | 13% |
| 6. Security | 1 | 2-3 | 2% |
| 7. User Interface | 6 | 12-17 | 9% |
| 8. Integrations | 4 | 8-11 | 6% |
| 9. Collaboration | 3 | 7-10 | 5% |
| 10. Configuration | 5 | 12-16 | 7% |
| 11. Cost Optimization | 2 | 2 | 2% |
| 12. Scoping | 3 | 9-12 | 6% |
| **Total** | **61** | **132-190** | **100%** |

## Recommended Implementation Order

### Wave 1: Quick Wins (1-2 days each, low dependencies)
1. N1 — Ask Mode (cost saving)
2. N7 — Per-Call Cost Visibility (observability)
3. U7 — Status Bar (UI polish)
4. 3 — Self-Review Toggle (cost saving)
5. R7 — App Documentation (output quality)

### Wave 2: Foundation (2-3 days, moderate dependencies)
6. 1 — Checkpoints (state foundation)
7. V6 — Versions (version foundation)
8. 7 — Knowledge System (context foundation)
9. W1 — Agent Profiles (agent abstraction)
10. B1 — Configuration-as-Code (output format)

### Wave 3: Core Features (3-4 days, stronger dependencies)
11. 2 — Plan-then-Build Workflow
12. 4 — Sandbox Tasks
13. D1 — Multiple Contexts
14. V1 — Design System Registry
15. R1 — Module System
16. 5 — Browser Testing

### Wave 4: Advanced Capabilities (4-5 days, high dependencies)
17. X1 — Autonomous Agent Loop
18. V3 — Browser-Use Agent
19. U4 — Element Picker
20. R4 — Component Library

### Wave 5: Platform Features (3-5 days, marketplace-ready)
21. X2 — Parallel Agents
22. V2 — Visual Design Panel
23. X5 — Template Marketplace
