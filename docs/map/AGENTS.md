# Map — catalog

System map of `orchestrator/`: the nouns, and what else moves when one changes. The source tree stays the source of truth. Cards cite it; they never replace it.

Status: slice 1 (catalog only). No cards written yet. Every noun is a stub in [objects/_index.md](objects/_index.md).

## Where do I go for...
| Question | Go to |
|---|---|
| What is X? | [objects/_index.md](objects/_index.md), then the card |
| How do I walk this map, what do live/leftover/ghost mean | [CONTEXT.md](CONTEXT.md) |
| What node types exist, how are they named | [_meta/schema.md](_meta/schema.md) |
| Add a card | copy from `_templates/` |
| What else moves if I change X | not built yet (slice 4: `effects/`) |

## Clusters
`work`, `routing`, `money`, `ports`, `pipeline`. One folder each under `objects/` once cards exist.

## Product name vs code name
| Product word | Code |
|---|---|
| Per-run budget | `Budget` in `orchestrator/budget.py` |
| Cross-run budget | `BudgetHierarchy` in `orchestrator/cost.py` |
| State store | `StateManager` in `orchestrator/infrastructure/state.py` (root `state.py` is a shim) |
| LLM client | `UnifiedClient` in `orchestrator/infrastructure/llm_client.py` (root `api_clients.py` is a shim) |

Entry files: edit this `CLAUDE.md` only. `AGENTS.md` and `routing.md` here are byte-identical copies. Regenerate with `cp CLAUDE.md AGENTS.md && cp CLAUDE.md routing.md`.
