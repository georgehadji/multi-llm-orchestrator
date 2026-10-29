---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# Verdict

An immutable pass/fail plus a 0..1 score returned by a verifier.

## Why this shape

Frozen so a verdict cannot change after a cascade reads it. `signals` records which sub-checks fired, for telemetry (`models.py:1062-1073`).

## Shape

- `passed: bool`, `score: float`, `signals: tuple[str, ...] = ()`, `detail: str = ""` (`models.py:1070-1073`).

Citations: `orchestrator/models.py:1061-1073`

## Connected to

- **owns:** nothing.
- **owned-by:** returned by the Verifier port (`verification/port.py:34`).
- **joins:** built in `verification/composite.py:80,101,118` and `verification/json_schema.py:113,130`.
- **looks-like-but-is-not:** `Verdict` Enum in `generators/wf100/report.py:43`, and `JudgeVerdict` in `services/completion_judge.py:27`. Neither is this class.

## If you change this

- **Hits:** every verifier in `verification/` and `commands/website_audit.py`; cascade logic comparing `min_score`.
- **Does not hit:** persistence. `Verdict` is not in `infrastructure/state.py`, and `ProjectState` does not hold one.

## Surfaces

| Surface | Role |
|---|---|
| `verification/*` | writes |
| engine and cascade code | reads |

## See

- Source: `orchestrator/models.py:1061`
