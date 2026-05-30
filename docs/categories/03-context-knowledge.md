# Category 3: Context & Knowledge Management

> **Focus:** Persistent instructions, context window optimization, and knowledge reuse across sessions and projects.
> **Phases:** 5 | **Est. Days:** 10-14 | **Weight:** 8% of total roadmap

---

## Overview

This category addresses **context rot** — the problem of AI agents forgetting architectural decisions, coding standards, and project requirements across sessions. Phases provide persistent knowledge injection, reusable skill playbooks, and cross-project knowledge sharing.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | 7 | Lovable | **Knowledge System** — workspace-level + project-level `.md` files always injected into every AI context | 2-3 | — |
| 2 | 8 | Lovable | **Skills System** — named, portable playbooks with trigger descriptions, `/skill-name` invocation, Claude SKILL.md compatible | 2-3 | 7 |
| 3 | D1 | Dyad | **Multiple Execution Contexts per Project** — separate AI conversations sharing the same codebase and version history | 3-4 | V6 |
| 4 | D3 | Dyad | **Summarize into New Context** — when context pressure exceeds 80%, LLM compresses full conversation into fresh context | 1-2 | D1 |
| 5 | 9 | Lovable | **Cross-Project Referencing** — `@ProjectName` references to read and reuse code, assets, and architecture from other projects | 2-3 | 1, 7 |

---

## Capability Progression

```
Knowledge System (persistent instructions)
    │
    ├── Skills System (on-demand playbooks)
    │
    ├── Multiple Contexts (isolated conversations, shared code)
    ├── Context Summarization (compression when full)
    │
    └── Cross-Project Referencing (@ProjectName reuse)
```

## Key Innovations

- **Two-level knowledge**: workspace rules apply to all projects, project-specific rules override
- **Skills are portable**: compatible with Anthropic Claude `SKILL.md` format for cross-tool reuse
- **Multiple contexts** is Dyad's most unique feature — one project, many independent AI conversations
- **Context summarization** automatically detects pressure and offers one-click compression
- **Cross-project references** via `@ProjectName` syntax enable code reuse without copy-paste
