# Design: Project Enhancer — LLM-Powered Spec Improvement Before Execution

**Date:** 2026-02-24
**Status:** Approved
**Scope:** Before a project runs, use an LLM to suggest and interactively apply improvements to the project description and success criteria.

---

## Problem Statement

Users typically write minimal project descriptions like *"Build a FastAPI auth service"* with vague criteria like *"tests pass"*. The decomposition LLM must guess what's missing — leading to incomplete task graphs (no refresh tokens, no password hashing, no migration scripts, no coverage target). A one-time enhancement pass fixes the spec before decomposition, producing significantly better output for a small upfront cost.

---

## Approved Design

### Approach: Pre-run Enhancement Pass (Approach A)

One structured LLM call before `_decompose()`. Returns 3–7 concrete suggestions. User accepts/rejects each interactively. Accepted patches are injected into description + criteria. Decomposition runs with the enriched spec.

---

## Architecture

```
CLI._async_new_project()
  │
  ├── 1. ProjectEnhancer.analyze(description, criteria)
  │         Auto-selects model: DeepSeek Reasoner (>50 words) | DeepSeek Chat (≤50 words)
  │         Returns list[Enhancement]  (3–7 items)
  │         Budget cap: max $0.10 per enhancement call
  │
  ├── 2. _present_enhancements(enhancements) → accepted: list[Enhancement]
  │         Prints each suggestion with [Y/n] prompt
  │         User accepts or rejects individually
  │
  ├── 3. _apply_enhancements(description, criteria, accepted)
  │         Appends patch_description to description
  │         Appends patch_criteria to criteria (if non-empty)
  │         Returns (enhanced_description, enhanced_criteria)
  │
  └── 4. _decompose(enhanced_description, enhanced_criteria)   ← unchanged
```

---

## Components

### Enhancement dataclass

```python
@dataclass
class Enhancement:
    type: str             # "completeness" | "criteria" | "risk"
    title: str            # e.g. "Missing: refresh tokens"
    description: str      # Why it matters (1–2 sentences)
    patch_description: str  # Clause to append to project description
    patch_criteria: str   # Clause to append to success criteria ("" if none)
```

### Model auto-selection

```python
def _select_enhance_model(description: str) -> Model:
    return (
        Model.DEEPSEEK_REASONER   # o1-class, better for complex analysis
        if len(description.split()) > 50
        else Model.DEEPSEEK_CHAT  # fast + cheap for short descriptions
    )
```

### LLM system prompt

```
You are a senior software architect reviewing a project spec before implementation.
Your job: identify real omissions, vague success criteria, and architectural risks.
Be concrete and brief. No padding.
```

### LLM user prompt

```
PROJECT: {description}
SUCCESS CRITERIA: {criteria}

Return a JSON array of 3–7 improvements. Each item:
{{
  "type": "completeness" | "criteria" | "risk",
  "title": "short label (≤8 words)",
  "description": "why this matters (1–2 sentences)",
  "patch_description": "concise clause to append to project description",
  "patch_criteria": "concise clause to append to success criteria, or empty string"
}}

Rules:
- Only real omissions and risks — no generic advice
- patch_description must be a clause, not a full sentence
- Examples of good patch_description: "with bcrypt password hashing (cost 12)",
  "including Alembic database migrations", "with rate limiting (100 req/min per IP)"
- Examples of bad patch_description: "make sure to add security", "improve the code"
- Return ONLY the JSON array, no markdown fences
```

### Terminal UX

```
⚡ Analyzing project spec (DeepSeek Chat)...

  📋 3 improvements found:

  [1/3] completeness — Missing: refresh tokens
        JWT auth without refresh tokens forces users to re-login every 24h.
        Adds: "with JWT access tokens (24h expiry) and refresh tokens (7d)"
        Apply? [Y/n]: y

  [2/3] criteria — Vague success criteria
        "tests pass" doesn't specify coverage or endpoint behaviour.
        Adds criteria: "≥80% test coverage, all endpoints return correct HTTP status codes"
        Apply? [Y/n]: y

  [3/3] risk — Missing: password hashing
        Storing plain-text passwords is a critical security flaw.
        Adds: "bcrypt password hashing (cost factor 12)"
        Apply? [Y/n]: n

✓ Applied 2/3 enhancements. Running enhanced project...
──────────────────────────────────────────────────────
```

### `--no-enhance` flag

```bash
python -m orchestrator --project "..." --no-enhance
# Skips enhancement, runs original description immediately
```

---

## New File

| File | Role |
|------|------|
| `orchestrator/enhancer.py` | `Enhancement` dataclass, `ProjectEnhancer` class, `_select_enhance_model`, `_present_enhancements`, `_apply_enhancements` |
| `tests/test_enhancer.py` | Unit tests for parsing, model selection, apply logic; integration test with mocked LLM |

## Modified Files

| File | Change |
|------|--------|
| `orchestrator/cli.py` | Call `ProjectEnhancer` at start of `_async_new_project()`; add `--no-enhance` flag |
| `orchestrator/__init__.py` | Export `Enhancement`, `ProjectEnhancer` |

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| LLM returns invalid JSON | Silently skip enhancement, log warning, proceed with original spec |
| LLM call exceeds $0.10 cap | Abort enhancement, proceed with original spec |
| LLM call times out (>10s) | Same as invalid JSON — skip silently |
| User presses Ctrl-C at prompt | Treat as "n" for all remaining, proceed |
| 0 suggestions returned | Print "✓ Spec looks complete — no suggestions." and continue |

---

## Budget

The enhancement call is capped at **$0.10 USD** — enforced via a mini `Budget` object passed to `UnifiedClient`. For reference:
- DeepSeek Chat: ~$0.001–0.003 per call on typical descriptions
- DeepSeek Reasoner: ~$0.005–0.015 per call

Well within cap for any realistic project description.

---

## Testing Strategy

| Test | Description |
|------|-------------|
| `test_select_model_short` | ≤50 words → DeepSeek Chat |
| `test_select_model_long` | >50 words → DeepSeek Reasoner |
| `test_parse_enhancements_valid` | Valid JSON → list[Enhancement] |
| `test_parse_enhancements_invalid_json` | Invalid JSON → empty list (no crash) |
| `test_parse_enhancements_empty_array` | `[]` → empty list |
| `test_apply_patches_description` | patch_description appended correctly |
| `test_apply_patches_criteria` | patch_criteria appended when non-empty |
| `test_apply_patches_empty_criteria` | empty patch_criteria → criteria unchanged |
| `test_apply_no_accepted` | No accepted → description/criteria unchanged |
| `test_present_user_accepts_all` | All [Y] → all returned |
| `test_present_user_rejects_all` | All [n] → empty list returned |
| `test_present_user_mixed` | Mixed → only accepted returned |
| `test_cli_no_enhance_flag` | `--no-enhance` skips enhancement entirely |

---

## Implementation Order

1. `enhancer.py` — dataclass + pure functions (testable without LLM)
2. `enhancer.py` — `ProjectEnhancer.analyze()` with mocked LLM in tests
3. `cli.py` — wire in + `--no-enhance` flag
4. `__init__.py` — exports
5. Tests — full coverage per table
