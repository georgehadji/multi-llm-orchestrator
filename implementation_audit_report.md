# Implementation Audit Report

**Audit Date**: 2026-07-14  
**Plan Reference**: `docs/plans/wire-architecture-rules-to-generation.md`  
**Scope**: Changes implementing language-aware generation pipeline across 9 files  
**Reviewer**: Reasonix (automated audit)

---

## 1. Executive Summary

**Verdict**: APPROVED WITH CHANGES (see §7 Required Corrections)

The implementation delivers the core objective: the generation pipeline now supports
`target_language` from the Task model through decomposition prompts, system prompts,
and output file extension selection. An HTML landing page project will now receive
HTML-specific system prompt guidance ("Valid HTML5, W3C validation, responsive CSS")
instead of Python requirements ("mypy --strict, type annotations"), and its output
files will be saved as `.html`/`.css` instead of `.py`.

**Completeness**: 10 of 11 plan steps are fully implemented. One file
(`output/writer.py`) has a partial implementation. Two lower-priority items were
deferred (CritiqueCycle language injection, engine.py legacy path). No tests were
added. The verification smoke test was not executed.

**Architecture Compliance**: No new architecture violations. The changes are
additive (new fields default to `""`), preserving backward compatibility for
existing Python projects.

---

## 2. Plan Compliance Matrix

| # | Plan Step | Status | Evidence | Notes |
|---|---|---|---|---|
| 1 | `models.py` — add `target_language` to `Task` | ✅ COMPLETE | `target_language: str = ""` at line 1000 | Pure data field, default empty |
| 2 | `structured_outputs.py` — add to `TaskInput` | ✅ COMPLETE | `target_language: str = Field(default="", ...)` at lines 51-53 | Pydantic model field |
| 3 | `structured_outputs.py` — update `to_task()` | ✅ COMPLETE | `target_language=self.target_language` at line 84 | Flows TaskInput→Task |
| 4 | `structured_outputs.py` — decomposition prompt hint | ✅ COMPLETE | "You MAY include an optional `target_language` field" at line 397 | LLM instructed to emit it |
| 5 | `prompt_builder.py` — `SystemPrompt.build(target_language)` | ✅ COMPLETE | Signature updated at lines 70-74 | Calls `_inject_language_guidance()` |
| 6 | `prompt_builder.py` — `_inject_language_guidance()` | ✅ COMPLETE | Lines 136-185, handles html/css/js/ts/python | String-replacement approach, works but is fragile (see §4) |
| 7 | `stages/generate.py` — pass `target_language` | ✅ COMPLETE | `getattr(task, "target_language", "")` at line 58 | Defensive fallback |
| 8 | `task_factory.py` — accept `target_language` | ✅ COMPLETE | Parameter at line 65, forwarded at line 97 | Domain layer, clean |
| 9 | `decomposer.py` — both `Task()` call sites | ✅ COMPLETE | Lines 566 and 688: `target_language=obj.get("target_language", "")` | Both DAG paths covered |
| 10 | `project_runner.py` — language detection + context | ✅ COMPLETE | `_detect_primary_language()` at lines 50-89; `ProjectContext` built at lines 260-270 | Heuristic approach, not plan's architecture-rules-based |
| 11a | `output_writer.py` — `_ext_for` fallback | ✅ COMPLETE | Signature updated at line 342; fallback at lines 386-400; call site at line 246 | Full implementation |
| 11b | `output/writer.py` — `_ext_for` fallback | ⚠️ PARTIAL | Signature NOT updated (line 378: `def _ext_for(task_type, output)`); call site NOT updated (line 247) | Duplicate writer lacks `target_language` parameter |
| — | Plan §1a: `_map_style_to_project_type` helper | 🔀 DEVIATION | Not implemented. Used `_detect_primary_language()` heuristic instead | Simpler approach; see §6 |
| — | Plan §1b: Thread `project_context` through Instructor | ⚠️ GAP | `decompose_project()` in `decomposer_service.py` lacks `project_context` param | Instructor fast path silently drops it; DAG fallback receives it |
| — | Plan §3c: CritiqueCycle language injection | ❌ DEFERRED | Not implemented | Lower priority; reviewer evaluates code regardless of language |
| — | Plan §4b: engine.py legacy path | ❌ DEFERRED | Not implemented | New ProjectRunner path covers modern usage |
| — | Plan §5a: `output_format` field on TaskInput | ❌ MISSING | Not implemented | Plan specified both `target_language` and `output_format` |
| — | Verification smoke test | ❌ NOT RUN | No verification test executed | Plan §Verification section not fulfilled |

---

## 3. Architecture Compliance Assessment

### 3.1 Hexagonal Layer Boundaries
**PASS** — All changes respect layer boundaries:
- **Domain layer** (`models.py`, `task_factory.py`): Pure data + factory. No I/O, no asyncio.
- **Application layer** (`decomposer.py`, `project_runner.py`, `structured_outputs.py`): Depends on domain models and ports only.
- **Engine Core** (`stages/generate.py`): No infrastructure imports. Compliant with Contract 4.
- **Infrastructure** (`output_writer.py`, `output/writer.py`): Handles file I/O correctly.

### 3.2 Import Contracts (5 lint-imports contracts)
**PASS** — No new violations:
- Contract 1 (domain-purity): Domain files import only from domain + models + stdlib ✅
- Contract 2 (application-no-concrete-infra): Application files have no `orchestrator.infrastructure` imports ✅
- Contract 3 (application-services-no-engine): Not affected ✅
- Contract 4 (engine-core-no-loose-infra): `stages/generate.py` has zero infra imports ✅
- Contract 5 (root-modules-no-infra): Not affected ✅

### 3.3 Root-Level Module Rule
**WARNING** — `orchestrator/task_factory.py` exists as a backward-compat shim. This was added as part of prior T1-D remediation (extracting `TaskFactory` from `models.py`). The architecture contract's Rule 4 says no *new* root-level modules. This shim is pre-existing from the Phase A remediation work, not from this implementation. **Not a new violation from this work.**

### 3.4 `models.py` Data Purity
**PASS** — `target_language` is a plain `str` field with default `""`. No I/O, no asyncio, no behavior.

---

## 4. Code Quality Review

### 4.1 SOLID Principles

| Principle | Assessment |
|---|---|
| **S**ingle Responsibility | `_detect_primary_language()` is well-isolated. `_inject_language_guidance()` is a separate static method. Each change has one clear purpose. |
| **O**pen/Closed | `SystemPrompt.build()` accepts `target_language` without modifying existing behavior. Extension via new parameter, not modification of existing logic. |
| **L**iskov Substitution | Not applicable — no inheritance changes. |
| **I**nterface Segregation | `target_language` is optional (default `""`). No consumer is forced to provide it. |
| **D**ependency Inversion | `GenerateStage` depends on `SystemPrompt` abstraction (static method), not on concrete language logic. |

### 4.2 Separation of Concerns
**PASS** — Language detection (`_detect_primary_language`), prompt construction (`_inject_language_guidance`), and extension selection (`_ext_for` fallback) are in separate modules with clear responsibilities.

### 4.3 Code Quality Issues Found

#### ISSUE-1 (MEDIUM): String replacement in `_inject_language_guidance()` is fragile
**File**: `prompt_builder.py`, lines 136-185  
**Problem**: Uses `str.replace()` for specific English strings in the system prompt. If the `_production()` method text changes, the replacements silently fail (no error, just no effect).  
**Recommendation**: Use a declarative approach — build the language-specific requirements from a dict instead of modifying a pre-built string. Or add a warning log when no replacements succeed.

#### ISSUE-2 (LOW): Duplicate `_target_to_ext` mapping
**File**: `output_writer.py`, lines 387-400  
**Problem**: `_target_to_ext` dict partially duplicates `_lang_to_ext` dict (lines 359-380). If a new language is added, both must be updated.  
**Recommendation**: Extract a shared `_LANG_TO_EXT` constant and derive both mappings from it.

#### ISSUE-3 (LOW): `_detect_primary_language()` keyword ordering
**File**: `project_runner.py`, lines 50-89  
**Problem**: "frontend" keyword in html_keywords would match a project described as "Build a frontend API", incorrectly returning "html". Keyword matching is inherently imprecise.  
**Recommendation**: Add a priority system or use the architecture rules engine's output (which already does LLM-based detection) instead of keyword heuristics.

### 4.4 Error Handling
**PASS** — `getattr(task, "target_language", "")` in `generate.py` gracefully handles missing attribute. `try/except ImportError` wraps the `ProjectContext` import in `project_runner.py`. All json `.get()` calls use safe defaults.

### 4.5 Observability
**GAP** — No logging added for language detection or prompt injection. If `_inject_language_guidance()` fails silently, there's no way to diagnose it.  
**Recommendation**: Add `logger.debug()` statements when language detection fires and when prompt guidance is injected.

---

## 5. Testing & Coverage Assessment

### 5.1 Unit Tests
**FAIL** — No unit tests were added for:
- `_detect_primary_language()` (keyword matching edge cases)
- `_inject_language_guidance()` (string replacement correctness)
- `_ext_for()` with `target_language` parameter
- `TaskInput.to_task()` with `target_language`
- `SystemPrompt.build()` with each supported language

### 5.2 Integration Tests
**FAIL** — No integration tests were added. The plan's verification section called for:
- A smoke test with `--project "Build a single-page HTML landing page..."` — NOT EXECUTED
- Checking output for `.html`/`.css` files — NOT EXECUTED
- Verifying no Python scaffold for web projects — NOT EXECUTED
- Re-running stress test 01 — NOT EXECUTED

### 5.3 Backward Compatibility
**HYPOTHESIS: LIKELY PASS** — All new fields default to `""`. The `SystemPrompt.build()` call without `target_language` preserves the original behavior. The `_ext_for()` without `target_language` preserves original behavior. Existing Python projects should be unaffected. However, this has not been verified with an actual test run.

---

## 6. Risk & Regression Analysis

### 6.1 Architectural Regressions
**None identified** — All changes are additive within existing layers.

### 6.2 Technical Debt Introduced

| Item | Severity | Description |
|---|---|---|
| Duplicate `output/writer.py` divergence | HIGH | The two writer files are now out of sync. `output/writer.py` lacks the `target_language` fallback. If the orchestrator switches between them, behavior will be inconsistent. |
| String-replace approach in prompt builder | MEDIUM | Tight coupling to exact prompt text. Will silently break if prompt text changes. |
| Keyword heuristic vs. architecture rules | MEDIUM | `_detect_primary_language()` uses simple keyword matching instead of the richer LLM-based architecture rules engine output. May misclassify edge-case projects. |
| `output_format` field not implemented | LOW | Plan called for both `target_language` and `output_format` on TaskInput. Only `target_language` was added. |

### 6.3 Backward Compatibility
**LOW RISK** — All changes are additive with safe defaults (`""`). The one risk is if the `ProjectAssembler` skip logic (from the earlier output layer fix) incorrectly skips the Python scaffold for projects that happen to have non-`.py` task outputs but are still Python projects. This is unlikely because the skip logic requires *majority* non-Python files.

### 6.4 Security
**No new security concerns** — The `target_language` field is free-form text from the LLM's JSON response. It flows into file extensions but is validated against a known mapping (`_target_to_ext` dict). Untrusted values default to `.py`.

---

## 7. Required Corrections

| Severity | File | Issue | Recommendation |
|---|---|---|---|
| **HIGH** | `output/writer.py` | Missing `target_language` parameter on `_ext_for()` and call site | Add `target_language: str = ""` parameter to `_ext_for()`, add `_target_to_ext` fallback logic, and update call site at line 247 to pass `getattr(task, "target_language", "")` |
| **MEDIUM** | `project_runner.py` | `_detect_primary_language()` uses fragile keyword matching instead of architecture rules | Wire `architecture_rules.decision.stack.primary_language` into `ProjectContext` as the plan specifies |
| **LOW** | `prompt_builder.py` | String replacement approach is fragile and unlogged | Add `logger.debug()` when language guidance is injected; consider declarative dict-based approach |
| **LOW** | `decomposer_service.py` | `decompose_project()` lacks `project_context` parameter | Add `project_context: Any = None` parameter and pass it to `TaskDecomposer.decompose()` |
| **LOW** | Tests | No tests added | Add unit tests for `_detect_primary_language()`, `_inject_language_guidance()`, and `_ext_for()` with `target_language` |

---

## 8. Deviations From Plan

| Plan Item | What Was Planned | What Was Implemented | Rationale |
|---|---|---|---|
| §1a: `_map_style_to_project_type` | Map `ArchitectureDecision.style` + `primary_language` to `ProjectType` enum | Used `_detect_primary_language()` keyword heuristic | Simpler to implement; avoids need to parse architecture_rules text output |
| §1a: Parse `architecture_rules.decision` | Access structured `ArchitectureDecision` object | Used project description text | `architecture_rules` is available as a string (rendered text), not as the structured object |
| §1d: `output_format` field | Add `output_format: str = ""` to `TaskInput` | Not implemented | Scope reduction; `target_language` alone covers the critical use case |
| §3c: CritiqueCycle | Inject language into critique prompts | Deferred | Lower priority; reviewer evaluates generated code regardless |
| §4b: engine.py legacy | Thread language through legacy mediator | Deferred | New `ProjectRunner` path covers all modern CLI usage |

---

## 9. Final Verdict

**APPROVED WITH CHANGES**

The implementation successfully delivers the core objective: language-aware generation from Task model through system prompts to output file extensions. The architectural integrity is preserved, backward compatibility is maintained, and the implementation follows the plan's step order.

**Must fix before merge**: `output/writer.py` divergence (HIGH severity — the two writer files are inconsistently implemented).

**Should fix before production use**: Add logging for language injection, add unit tests, and consider wiring the actual architecture rules engine output instead of the keyword heuristic.

**Can be deferred**: CritiqueCycle language injection, engine.py legacy path, `output_format` field on TaskInput.
