# docs/ — documentation index

**Job:** one home for every plan, audit, decision, and guide. Link here, do not copy.

## Live reference (keep current)
| File | Holds |
|---|---|
| `CODEBASE_MINDMAP.md` | Master architecture map. Read before architectural decisions |
| `DEV_GUIDE.md` | Setup, commands, workflow, TDD, worktrees (former `CLAUDE.md`) |
| `CI_AND_CONTRACTS.md` | CI order, import contracts, style, gotchas (former `AGENTS.md`). Its contract count is stale: `.importlinter` is authoritative |
| `DEPENDENCY_POLICY.md`, `security/PRODUCTION_PROFILE.md` | Policy |
| Debugging | `docs/debugging/DEBUGGING_GUIDE.md` is cited by old docs but does not exist. Use skill `orchestrator-debugging-playbook` |

## Folders
| Dir | Holds |
|---|---|
| `adr/` | Architecture decision records |
| `plans/` | Dated design and implementation plans |
| `categories/` | Feature categories 01–12 |
| `hunts/` | Defect-hunt programme (`INVENTORY.md` is the running log) |
| `archive/` | Superseded material |
| `security/` | Security docs |

## Historical (do not treat as current)
- Root files: `*_PLAN.md`, `*_audit*.md`, `*REMEDIATION*.md`, `implementation_*.md`, `ARCH-AUDIT-*.md`, `standardmap_proposals.md`, `architecture_mindmap_v7.md`.
- Most top-level `docs/*PLAN*.md`, `docs/ARCHITECTURAL_AUDIT_*.md`, `docs/*_INSPIRED_ENHANCEMENTS.md`.
- 11 unreferenced root docs already moved to `archive/`. The rest stay at root because skills, README, tests or `scripts/check_new_root_files.py` reference them. Update those referrers before moving more.

## Human check
Before archiving or deleting a doc, search for links to it (docs, scripts, CI, skills, `.claude/`).
