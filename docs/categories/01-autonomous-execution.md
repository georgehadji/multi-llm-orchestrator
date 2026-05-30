# Category 1: Autonomous Execution & Agent Intelligence

> **Focus:** Agents that can run independently, self-correct, and execute multi-step workflows without human intervention.
> **Phases:** 12 | **Est. Days:** 24-36 | **Weight:** 20% of total roadmap

---

## Overview

This category forms the **intelligence core** of the orchestrator — enabling the system to operate autonomously, from simple question-answering to fully autonomous end-to-end development loops. Phases progress from basic agent profiles through multi-mode selection to fully autonomous Max-mode agents running in parallel.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | W1 | Bolt.new | **Agent Profiles** — Standard/Max/Creative/Conservative/Research profiles that auto-select models | 2-3 | — |
| 2 | N1 | Newly | **Ask Mode** — query-only answers, no code generation, 10x cheaper than build mode | 1-2 | — |
| 3 | N4 | Newly | **Brainstorming Mode** — AI asks 3-5 clarifying questions before creating a plan | 2-3 | — |
| 4 | W3 | Bolt.new | **Quick Action Buttons** — contextual actions after Plan Mode (implement, refine, example, alternative) | 1-2 | N4, U1 |
| 5 | X4 | Create.xyz | **Multi-Mode Selector** — 6 modes: Discuss/Plan/Fast/Thinking/Auto/Max, each with different iteration + cost profiles | 2-3 | W1, X1 |
| 6 | V4 | v0 | **Permission Modes** — Ask/Auto/Full for agent tool execution safety | 1-2 | — |
| 7 | D4 | Dyad | **Undo + Retry** — revert bad change, switch to better model, retry same prompt | 2-3 | V6, D1 |
| 8 | V5 | v0 | **Auto-Error Fix Button** — one-click fix from validation/test/build errors | 2-3 | — |
| 9 | N3 | Newly | **Screenshot-Based Debugging** — paste error screenshots, AI diagnoses root cause, produces fix | 2-3 | — |
| 10 | V3 | v0 | **Browser-Use Agent** — opens app, uses it like a user, critiques design, debugs, sends screenshots | 3-4 | 5, V1, N3 |
| 11 | X1 | Create.xyz | **Autonomous Agent Loop (Max Mode)** — build → test in browser → find issues → fix → retest → repeat until goal achieved | 4-5 | V3, V5, 5, N3 |
| 12 | X2 | Create.xyz | **Parallel Autonomous Agents** — multiple Max instances running simultaneously, one codebase, conflict detection | 3-4 | X1, D1 |

---

## Capability Progression

```
Agent Profiles (pick strategy)
    │
    ├── Ask Mode (query only)
    ├── Brainstorming (plan with Q&A)
    ├── Quick Action Buttons (contextual post-plan actions)
    │
    ├── Multi-Mode Selector (6 modes)
    ├── Permission Modes (safety at tool level)
    ├── Undo + Retry (correct mistakes intelligently)
    │
    ├── Auto-Error Fix (one-click repair)
    ├── Screenshot Debugging (visual issue diagnosis)
    │
    ├── Browser-Use Agent (autonomous app interaction)
    ├── Autonomous Agent Loop (full Max mode)
    │
    └── Parallel Autonomous Agents (concurrent Max instances)
```

## Key Innovations

- **Agent Profiles** abstracts model selection behind named strategy profiles
- **Max Mode** is the most autonomous execution model across all 10 platforms analyzed
- **Parallel agents** enable concurrent autonomous execution with conflict detection
- **Browser-use agent** goes beyond testing — it actively uses the app as a user would
- **Multi-mode selector** gives 6 levels of autonomy with transparent cost profiles
