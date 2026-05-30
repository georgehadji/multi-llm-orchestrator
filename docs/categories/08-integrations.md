# Category 8: Integration & Data Sources

> **Focus:** Data source connections, AI-powered query generation, integration discovery via slash commands, and two-way version control sync.
> **Phases:** 4 | **Est. Days:** 8-11 | **Weight:** 6% of total roadmap

---

## Overview

This category connects the orchestrator to the **external world** — databases, APIs, payment providers, and AI models. Phases range from templated data source connections to AI-powered query generation to slash-command integration discovery.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | R5 | Retool | **Data Source Integration Templates** — 50+ templates (PostgreSQL, MongoDB, Redis, S3, REST, GraphQL, etc.) with connection pooling, health checks, env config, Docker Compose | 2-3 | — |
| 2 | R3 | Retool | **AI-Assisted Query Generation** — natural language → SQL/JS/GraphQL with schema validation, explanation, optimization, and fix capabilities | 2-3 | B1, B2 |
| 3 | X3 | Create.xyz | **Slash Command Integration Discovery** — 100+ integrations via `/chatgpt`, `/stripe`, `/google-maps`, `/pdf-generation`, `/shadcn-ui`, etc. with autocomplete and suggestions | 2-3 | R5 |
| 4 | N6 | Newly | **Two-Way Git Sync** — pull external changes back into AI context when code is pushed from outside the orchestrator | 2-3 | 5 |

---

## Capability Progression

```
Data Source Integration (50+ templates)
    │
    ├── AI Query Generation (NL → SQL with validation)
    ├── Slash Command Integration (100+ discoverable via /)
    │
    └── Two-Way Git Sync (pull external changes into AI context)
```

## Key Innovations

- **50+ data source templates** generate production-ready connection code with pooling and retry logic
- **AI query generation** validates queries against the actual database schema
- **Slash command system** makes 100+ integrations discoverable via autocomplete — type `/stripe` to wire up payments
- **Two-way git sync** keeps the AI aware of changes made outside the orchestrator (e.g., human developer PRs)
