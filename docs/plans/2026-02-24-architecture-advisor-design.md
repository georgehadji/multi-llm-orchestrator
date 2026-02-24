# Design: Architecture Advisor — LLM-Powered Architecture Decision Before Decomposition

**Date:** 2026-02-24
**Status:** Approved
**Scope:** Replace `AppDetector` with `ArchitectureAdvisor` — a single LLM call that decides optimal software architecture (structural pattern, topology, API paradigm, data paradigm) before decomposition runs.

---

## Problem Statement

The current `AppDetector` classifies only the *app type* (FastAPI, Next.js, etc.) to pick a scaffold template. There is no reasoning about *how* the app should be structured: no decision about layered vs hexagonal vs CQRS, monolith vs microservices, REST vs GraphQL, relational vs document storage. The decomposition LLM must implicitly guess these choices, leading to inconsistent and suboptimal task graphs.

---

## Approved Design

### Approach: Single-call ArchitectureAdvisor (Approach A)

One LLM call replaces `AppDetector`. Returns `ArchitectureDecision` — a superset of `AppProfile` that includes all existing fields plus four new architecture fields. Prints a summary to terminal (inform mode — no user prompt). Injected into the decomposition prompt so all tasks follow the chosen architecture.

---

## Architecture

```
CLI._async_new_project()
  │
  ├── 1. Auto-resume check
  ├── 2. Project Enhancer        (spec improvement)
  ├── 3. ArchitectureAdvisor     ← NEW (replaces AppDetector)
  │         Auto-selects model: DeepSeek Reasoner (>50 words) | DeepSeek Chat (≤50 words)
  │         One LLM call → ArchitectureDecision
  │         Prints 🏗 summary block, proceeds automatically
  │
  └── 4. AppBuilder.build()
            Uses ArchitectureDecision for scaffold selection + decomposition
```

---

## Components

### `ArchitectureDecision` dataclass

```python
@dataclass
class ArchitectureDecision:
    # ── existing AppProfile fields (unchanged) ──────────────────────
    app_type: str            # "fastapi" | "nextjs" | "cli" | "library" |
                             #  "script" | "react-fastapi" | "flask" | "generic"
    tech_stack: list[str]    # ["python", "fastapi", "postgresql"]
    entry_point: str         # "src/main.py"
    test_command: str        # "pytest"
    run_command: str         # "uvicorn src.main:app --reload"
    requires_docker: bool    # False
    detected_from: str       # "advisor" | "yaml_override"

    # ── new architecture fields ──────────────────────────────────────
    structural_pattern: str  # "layered" | "hexagonal" | "cqrs" |
                             #  "event-driven" | "mvc" | "script"
    topology: str            # "monolith" | "microservices" |
                             #  "serverless" | "bff" | "library"
    data_paradigm: str       # "relational" | "document" | "time-series" |
                             #  "key-value" | "none"
    api_paradigm: str        # "rest" | "graphql" | "grpc" | "cli" | "none"
    rationale: str           # 2-3 sentences explaining all choices
```

**Backward compatibility:** `AppProfile` becomes a type alias in `app_detector.py`:
```python
AppProfile = ArchitectureDecision
```
All existing code that imports `AppProfile` continues to work with zero changes.

---

### Model auto-selection

```python
def _select_model(description: str) -> Model:
    return (
        Model.DEEPSEEK_REASONER   # multi-dimension reasoning for complex specs
        if len(description.split()) > 50
        else Model.DEEPSEEK_CHAT  # fast + cheap for simple descriptions
    )
```

Fallback chain if DeepSeek unavailable: `Kimi K2.5 → GPT-4o-mini → any available`.

---

### LLM system prompt

```
You are a senior software architect.
Given a project description and success criteria, select the optimal architecture.
Return only a JSON object — no markdown, no explanation.
```

### LLM user prompt

```
PROJECT: {description}
SUCCESS CRITERIA: {criteria}

Return a JSON object with these exact fields:
{
  "app_type": "fastapi|flask|cli|library|script|react-fastapi|nextjs|generic",
  "tech_stack": ["list", "of", "technologies"],
  "entry_point": "relative/path/to/main.py",
  "test_command": "pytest",
  "run_command": "command to start app",
  "requires_docker": false,
  "structural_pattern": "layered|hexagonal|cqrs|event-driven|mvc|script",
  "topology": "monolith|microservices|serverless|bff|library",
  "data_paradigm": "relational|document|time-series|key-value|none",
  "api_paradigm": "rest|graphql|grpc|cli|none",
  "rationale": "2-3 sentences explaining all architectural choices"
}

Rules:
- Choose the architecture that best fits the project's scale and requirements
- layered: routes → services → repositories (standard API services)
- hexagonal: ports & adapters (when testing or swappable infra matters)
- cqrs: separate read/write paths (high-read or event-sourced systems)
- event-driven: async message-passing (real-time, decoupled components)
- mvc: model-view-controller (web apps with server-side rendering)
- Return ONLY the JSON object, no markdown fences
```

---

### Terminal UX

```
🏗  Architecture Decision (DeepSeek Reasoner):
    Pattern:  Layered  │  Topology: Monolith  │  API: REST  │  Storage: Relational
    FastAPI is well-suited for RESTful services. A layered architecture
    (routes → services → repositories) keeps the codebase maintainable at this
    scale. PostgreSQL for persistence; no need for microservices at this scope.
──────────────────────────────────────────────────────────────────────────────
```

Prints automatically, no user prompt. Always runs unless `app_type_override` is set in YAML.

---

### Enriched decomposition prompt (engine.py)

The existing `app_context_block` is extended with architecture fields:

```
APP_TYPE: fastapi
TECH_STACK: python, fastapi, postgresql
SCAFFOLD_FILES (already exist — fill or extend these):
  - src/main.py
  - ...

ARCHITECTURE DECISION:
  Structural pattern: layered (routes → services → repositories)
  Topology:           monolith
  API paradigm:       REST
  Data paradigm:      relational (PostgreSQL/SQLite)
  Rationale:          FastAPI service at this scale benefits from a clear
                      layered structure...

Each task MUST follow this architecture — do not invent an alternative structure.
```

---

### AppBuilder change (one-line swap)

```python
# Before:
detector = AppDetector(client=self._client)
profile  = await detector.detect(description, app_type_override)

# After:
advisor  = ArchitectureAdvisor(client=self._client)
profile  = await advisor.analyze(description, criteria, app_type_override)
```

---

## New File

| File | Role |
|------|------|
| `orchestrator/architecture_advisor.py` | `ArchitectureDecision` dataclass, `ArchitectureAdvisor` class, `_select_model`, `_parse_response`, `_print_summary` |
| `tests/test_architecture_advisor.py` | 11 unit tests covering parsing, model selection, fallback, terminal output, decomposition prompt injection |

## Modified Files

| File | Change |
|------|--------|
| `orchestrator/app_detector.py` | Add `AppProfile = ArchitectureDecision` type alias at bottom; add import |
| `orchestrator/app_builder.py` | Swap `AppDetector.detect()` → `ArchitectureAdvisor.analyze()` |
| `orchestrator/engine.py` | Extend `app_context_block` with architecture fields when present |
| `orchestrator/__init__.py` | Export `ArchitectureDecision`, `ArchitectureAdvisor` |

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| LLM returns invalid JSON | Fall back to "script" defaults + empty arch fields; log warning |
| LLM call times out (>30s) | Same fallback |
| No provider available | Same fallback |
| Unknown `structural_pattern` value | Accept as-is (LLM may choose reasonable unlisted values) |
| YAML specifies `app_type_override` | Skip LLM — call `detect_from_yaml()`, fill arch fields with type-appropriate defaults |

---

## Default arch fields per app_type (used for YAML override fallback)

| app_type | structural_pattern | topology | data_paradigm | api_paradigm |
|----------|--------------------|----------|---------------|--------------|
| fastapi | layered | monolith | relational | rest |
| flask | mvc | monolith | relational | rest |
| nextjs | mvc | monolith | relational | rest |
| react-fastapi | layered | monolith | relational | rest |
| graphql | layered | monolith | relational | graphql |
| cli | script | library | none | cli |
| library | script | library | none | none |
| script | script | library | none | none |
| generic | layered | monolith | relational | rest |

---

## Testing Strategy

| Test | Description |
|------|-------------|
| `test_parse_valid_response` | Valid JSON → correct `ArchitectureDecision` fields |
| `test_parse_invalid_json` | Bad JSON → fallback defaults, no exception |
| `test_parse_missing_arch_fields` | JSON with only AppProfile fields → arch fields get sensible defaults |
| `test_select_model_short` | ≤50 words → DeepSeek Chat |
| `test_select_model_long` | >50 words → DeepSeek Reasoner |
| `test_analyze_calls_llm_once` | `analyze()` makes exactly one LLM call |
| `test_analyze_handles_exception` | LLM raises → returns fallback defaults, no crash |
| `test_analyze_prints_summary` | Terminal output contains pattern + topology + api + storage |
| `test_detect_from_yaml` | `app_type_override="fastapi"` → fills known defaults, skips LLM |
| `test_app_profile_alias` | `AppProfile is ArchitectureDecision` — type alias works |
| `test_decompose_prompt_includes_arch` | `engine._decompose()` prompt contains `structural_pattern` |

---

## Implementation Order

1. `architecture_advisor.py` — `ArchitectureDecision` dataclass + `_parse_response` + `_select_model` (pure, testable)
2. `architecture_advisor.py` — `ArchitectureAdvisor.analyze()` + `_print_summary` + `detect_from_yaml()`
3. `app_detector.py` — add `AppProfile = ArchitectureDecision` alias
4. `app_builder.py` — swap AppDetector → ArchitectureAdvisor
5. `engine.py` — enrich `app_context_block` with arch fields
6. `__init__.py` — exports
7. Tests — full coverage per table
