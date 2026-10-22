# Schema — the rules of this map

Closed set of node types and labels. When practice and this file disagree, reconcile the same day.

## Node types
| `type:` | Lives at | Carries |
|---|---|---|
| object | `objects/<cluster>/<Name>.md` | One noun: why, shape, connections, Hits / Does not hit |
| process | `processes/<slug>.md` | One real movement: Input → Movement → Output, steps with citations |

No other types. `processes/` and `effects/` are created only when verified content exists.

## Labels
`type`, `cluster` (work, routing, money, ports, pipeline), `universe` (live, leftover, ghost), `status` (stub, verified, stale), `entity` (owning file path). Processes add `consumes:` and `produces:` as links to object cards.

## Naming
- Object cards: the code name, Title Case, e.g. `ProjectState.md`.
- Process cards: kebab-case, e.g. `run-project.md`.
- Citations: `path:line`, repo-relative.
- `_index.md` is edited by hand until a generator exists. Rebuild it, do not patch it, once one does.
