# Karpathy Guidelines — Integration Analysis for AI Orchestrator

> **Source:** [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) (154k ⭐, 15.7k forks)
> **Type:** System prompt enhancement — behavioral guardrails for all agents
> **Author:** Georgios-Chrysovalantis Chatzivantsidis | **Date:** 2026-05-25

---

## Overview

Andrej Karpathy's observations on LLM coding pitfalls distilled into 4 principles in a single `CLAUDE.md` file. These are not new features — they are **behavioral guidelines** that improve the quality of AI-generated code. The orchestrator can inject these into all agent system prompts immediately.

## The 4 Principles → Orchestrator Mapping

### 1. Think Before Coding → Strengthen Brainstorming Mode (Phase N4)

| Karpathy Principle | Current Orchestrator State | Enhancement |
|-------------------|---------------------------|-------------|
| State assumptions explicitly | `ArchitectureAnalyzer` detects keywords but doesn't surface ambiguity | Inject assumption-surfacing prompt into Architect agent |
| Present multiple interpretations | `dry_run()` generates one plan | Add "multiple interpretations" pass in Plan Mode quick actions |
| Push back when warranted | No pushback mechanism | Add "simpler alternative exists" check in architecture analysis |
| Stop when confused | Agents continue with assumptions | Add confusion-detection gate: if confidence < 0.7, ask |

**Integration:** Inject into `AgentOrchestrator` system prompt and `ArchitectureAnalyzer._prompt_llm_optimization()`.

### 2. Simplicity First → Add Simplicity Audit Stage

| Karpathy Principle | Current Orchestrator State | Enhancement |
|-------------------|---------------------------|-------------|
| No features beyond what was asked | `PreflightValidator` checks completeness but not overcomplication | Add `SimplicityAudit` validator |
| No abstractions for single-use code | Not checked | AST analysis: detect Strategy/Factory pattern for < 2 implementations |
| No speculative flexibility | Not checked | Regex check for `configurable`, `flexible`, `extensible` in generated comments |
| 200 lines → 50 lines rewrite | Not checked | Add line-count-to-feature-complexity ratio gate |

**Integration:** New `validate_simplicity()` validator in `orchestrator/validators.py` — runs after code generation, warns on over-engineering patterns.

### 3. Surgical Changes → Reinforces Target/Lock Files (Phase W2)

| Karpathy Principle | Current Orchestrator State | Enhancement |
|-------------------|---------------------------|-------------|
| Don't "improve" adjacent code | No mechanism to prevent scope creep | FileScope `lock()` prevents modification |
| Match existing style | `validate_ruff()` enforces project style but doesn't check drift | Add style-consistency check: changed lines must match existing patterns |
| Don't refactor unbroken things | Not enforced | `validate_surgical()`: detect changes in files not targeted by task |
| Every changed line → user request | Not tracked | Requirement trace: map each changed line to a task requirement |

**Integration:** `FileScope` (Phase W2) + new `validate_surgical_changes()` that checks diff scope against task description.

### 4. Goal-Driven Execution → Already in Max Mode (Phase X1)

| Karpathy Principle | Current Orchestrator State | Enhancement |
|-------------------|---------------------------|-------------|
| Transform tasks into verifiable goals | `CritiqueCycle` has plateau detection | Already matches — inject explicit success criteria in task descriptions |
| "Add validation" → "Write tests, make them pass" | `ValidatePytest` runs tests | Extend task decomposition to auto-generate verification criteria |
| Loop until verified | Max Mode (X1): test → fix → retest | Already matches — Karpathy's insight validates the Max Mode architecture |
| Multi-step with verification | `Orchestrator._execute_all()` has dependency order | Add per-step verification gates in execution plan |

**Integration:** The Max Mode (Phase X1) already implements this principle. The Karpathy guidelines validate that architecture as correct. No changes needed — inject guidelines into Max Mode system prompt.

---

## Immediate Integration: System Prompt Injection

The simplest integration: inject the 4 principles into every agent's system prompt. No code changes needed — just update `AgentBase` or `SystemPrompt.build()`.

```python
# orchestrator/prompt_builder.py — add to SystemPrompt.build()

KARPATHY_GUIDELINES = """
## Behavioral Guidelines (Karpathy Principles)

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ASK.
- If multiple interpretations exist, present ALL of them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, STOP. Name what's confusing. Ask for clarification.

### 2. Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked. No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- If 200 lines could be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?"

### 3. Surgical Changes
- Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
- Define success criteria. Loop until verified.
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- For multi-step tasks, state a brief plan with verification for each step.
"""
```

---

## Feature Enhancements (3 New Validators)

### 1. SimplicityAudit Validator

```python
def validate_simplicity(code: str, task_description: str) -> ValidationResult:
    """Check generated code for over-engineering patterns.
    
    Detects:
    - Strategy/Factory patterns with < 2 implementations
    - Abstract base classes with single subclass
    - Empty error handlers (except: pass)
    - Comments mentioning "flexible", "configurable", "extensible"
    - Lines-to-features ratio > 50:1 (warn if > 200 lines for a single feature)
    """
```

### 2. Surgical Change Validator

```python
def validate_surgical_changes(
    diff: str, task_id: str, locked_files: set[str]
) -> ValidationResult:
    """Check that changes are scoped to the task.
    
    Detects:
    - Changes to files not in task scope
    - Style changes in untouched code sections
    - Refactoring of unrelated code
    - Comment changes in unchanged functions
    """
```

### 3. Assumption Surface Gate

```python
async def surface_assumptions(
    task_description: str, client: UnifiedClient
) -> list[str]:
    """Ask the LLM to surface hidden assumptions before implementation.
    
    Returns list of assumptions that should be confirmed with the user.
    Triggered when task_description contains ambiguous terms.
    """
```

---

## Effort Estimate

| Item | Type | Effort |
|------|------|--------|
| Inject Karpathy guidelines into system prompts | Prompt update | 30 min |
| `validate_simplicity()` | New validator | 2-3 hours |
| `validate_surgical_changes()` | New validator | 2-3 hours |
| `surface_assumptions()` | New gate | 2-3 hours |
| **Total** | | **1-2 days** |

---

## Impact Assessment

The Karpathy guidelines directly address the **quality of generated code** — not adding new capabilities, but making existing capabilities produce better output. Key benefits:

- **Fewer rewrites**: Simplicity-first reduces the critique→revise cycle
- **Smaller diffs**: Surgical changes reduce merge conflicts
- **Better requirements**: Assumption-surfacing catches ambiguity before code is written
- **Verifiable quality**: Goal-driven execution makes validation deterministic
