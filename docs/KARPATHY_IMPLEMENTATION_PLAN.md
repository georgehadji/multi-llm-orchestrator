# Implementation Plan: Karpathy Guidelines Integration

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) (154k ⭐)  
> **Estimated Effort:** 1-2 days

---

## Overview

Four behavioral principles from Andrej Karpathy's observations on LLM coding pitfalls, injected into the orchestrator's agent prompts and reinforced by 3 new deterministic validators.

```
Karpathy Guidelines
    │
    ├── System Prompt Injection (30 min)
    │   └── Modify: orchestrator/prompt_builder.py
    │       └── SystemPrompt.build() — inject 4 principles into every agent prompt
    │
    ├── Assumption-Surfacing Gate (2-3 hours)
    │   └── New: orchestrator/assumption_gate.py
    │       └── Pre-generation gate: ask LLM to surface hidden assumptions
    │
    ├── Simplicity Auditor (2-3 hours)
    │   └── New validator in: orchestrator/validators.py
    │       └── validate_simplicity() — detect over-engineering patterns
    │
    └── Surgical Change Auditor (2-3 hours)
        └── New validator in: orchestrator/validators.py
            └── validate_surgical_changes() — detect scope creep in diffs
```

---

## Step 1: System Prompt Injection (30 min)

### 1.1 — Extend `SystemPrompt.build()` to include Karpathy guidelines

**File:** `orchestrator/prompt_builder.py`

Currently `SystemPrompt._standard()` returns a bare one-liner:
```python
"You are an expert software engineer executing a task. Produce high-quality, complete output."
```

`SystemPrompt._production(task_type)` has 8 quality requirements but no behavioral guidelines.

**Change:** Add `_karpathy_guidelines()` method and inject into both `_standard()` and `_production()`.

```python
class SystemPrompt:
    @staticmethod
    def _karpathy_guidelines() -> str:
        """Karpathy behavioral principles for better LLM code quality."""
        return (
            "\n\n## Behavioral Guidelines\n\n"
            "### 1. Think Before Coding\n"
            "- State assumptions explicitly. If uncertain, ASK before implementing.\n"
            "- If multiple interpretations exist, present ALL of them — don't pick silently.\n"
            "- If a simpler approach exists, say so. Push back when warranted.\n"
            "- If something is unclear, STOP. Name what's confusing. Ask for clarification.\n\n"
            "### 2. Simplicity First\n"
            "- Minimum code that solves the problem. Nothing speculative.\n"
            "- No features beyond what was asked. No abstractions for single-use code.\n"
            '- No "flexibility" or "configurability" that wasn\'t requested.\n'
            "- No error handling for impossible scenarios.\n"
            "- If 200 lines could be 50, rewrite it.\n"
            '- Ask yourself: "Would a senior engineer say this is overcomplicated?"\n\n'
            "### 3. Surgical Changes\n"
            "- Touch only what you must. Clean up only your own mess.\n"
            '- Don\'t "improve" adjacent code, comments, or formatting.\n'
            "- Don't refactor things that aren't broken.\n"
            "- Match existing style, even if you'd do it differently.\n"
            "- If you notice unrelated dead code, mention it — don't delete it.\n"
            "- When your changes create orphans (unused imports/variables), remove ONLY those.\n"
            "- The test: Every changed line should trace directly to the user's request.\n\n"
            "### 4. Goal-Driven Execution\n"
            "- Define success criteria. Loop until verified.\n"
            '- "Add validation" → "Write tests for invalid inputs, then make them pass"\n'
            '- "Fix the bug" → "Write a test that reproduces it, then make it pass"\n'
            '- "Refactor X" → "Ensure tests pass before and after"\n'
            "- For multi-step tasks, state a brief plan with verification for each step."
        )

    @staticmethod
    def _standard() -> str:
        return (
            "You are an expert software engineer executing a task. "
            "Produce high-quality, complete output. "
            "Follow best practices and ensure all code is valid and runnable."
            + SystemPrompt._karpathy_guidelines()
        )

    @staticmethod
    def _production(task_type: str = "") -> str:
        base = (
            "You are a senior software engineer delivering production-grade output. "
            "Requirements:\n"
            "1. Full type annotations on every function and class.\n"
            "2. Comprehensive error handling and input validation.\n"
            "3. Unit tests for every public function (pytest style).\n"
            "4. Docstrings on every module, class, and public function.\n"
            "5. Logging via the standard library logger (not print).\n"
            "6. No TODOs, no placeholder implementations.\n"
            "7. Follow SOLID principles and keep cyclomatic complexity \u2264 10.\n"
            "8. Include a brief inline comment for any non-obvious logic.\n"
        )
        if task_type in ("code_gen", "code_generation"):
            base += (
                "9. Return ONLY raw code \u2014 no markdown fences, no prose outside code.\n"
                "10. Code must pass mypy --strict.\n"
            )
        return base + SystemPrompt._karpathy_guidelines()
```

**Verification:** Run existing tests — `python -m pytest tests/test_god_file_refactoring.py -x --no-cov -q`. All prompts now include Karpathy guidelines. No behavior change expected (guidelines are additive).

### 1.2 — Inject into Critique/Synthesis system prompts

**File:** `orchestrator/prompt_builder.py`

The `CritiquePrompt` and `RevisionPrompt` classes build system prompts for reviewers. These should also follow the guidelines.

```python
class CritiquePrompt:
    @staticmethod
    def build(task_prompt, output):
        system_prompt = (
            "You are a critical reviewer. Find flaws, be specific. "
            "Apply the Simplicity First principle — flag over-complication. "
            "Apply the Surgical Changes principle — flag unrelated changes."
        )
        # ... existing logic
```

---

## Step 2: Assumption-Surfacing Gate (2-3 hours)

### 2.1 — Create `orchestrator/assumption_gate.py`

**Objective:** Before a task is decomposed or executed, ask the LLM to surface hidden assumptions in the task description. If ambiguity is detected, present it to the user before proceeding.

```python
"""
Assumption Gate — surface hidden assumptions before generation.
=================================================================

Karpathy Principle 1: "Think Before Coding"
- State assumptions explicitly. If uncertain, ASK.
- If multiple interpretations exist, present ALL of them.

This gate runs BEFORE task decomposition to catch ambiguity early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api_clients import UnifiedClient


@dataclass
class AssumptionReport:
    """Report of hidden assumptions detected in a task description."""
    has_ambiguity: bool
    assumptions: list[dict] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_prompt_context(self) -> str:
        """Render as context for the user to review."""
        if not self.has_ambiguity and not self.assumptions:
            return ""

        lines = ["## Assumptions & Clarifications"]
        if self.assumptions:
            lines.append("\n### Assumptions Made")
            for a in self.assumptions:
                lines.append(f"- {a['statement']} (confidence: {a.get('confidence', 'medium')})")

        if self.interpretations:
            lines.append("\n### Multiple Interpretations")
            for i, interp in enumerate(self.interpretations, 1):
                lines.append(f"{i}. {interp}")

        if self.clarification_questions:
            lines.append("\n### Questions to Clarify")
            for q in self.clarification_questions:
                lines.append(f"- {q}")

        return "\n".join(lines)


async def surface_assumptions(
    task_description: str,
    client: UnifiedClient,
    threshold: float = 0.7,
) -> AssumptionReport:
    """Ask the LLM to surface hidden assumptions before implementation.

    Only triggers the LLM call if the task description contains
    ambiguous language or appears to warrant clarification.
    Cost: ~$0.001 per call (cheapest model).

    Args:
        task_description: The task description to analyze
        client: API client for LLM calls
        threshold: Confidence threshold below which ambiguity is flagged

    Returns:
        AssumptionReport with surfaced assumptions, or empty report
    """
    # Quick pre-check: skip LLM call for clearly unambiguous descriptions
    if _is_unambiguous(task_description):
        return AssumptionReport(has_ambiguity=False)

    prompt = (
        "Analyze this task description for hidden assumptions and ambiguity.\n\n"
        f"TASK: {task_description}\n\n"
        "Return JSON:\n"
        '{\n'
        '  "has_ambiguity": true/false,\n'
        '  "assumptions": [{"statement": "...", "confidence": "high|medium|low"}],\n'
        '  "interpretations": ["interpretation 1", "interpretation 2"],\n'
        '  "clarification_questions": ["question 1", "question 2"],\n'
        '  "confidence": 0.0-1.0\n'
        '}\n\n'
        "Only flag actual ambiguity. Don't fabricate issues for clear descriptions."
    )

    try:
        response = await client.call(
            model=_get_cheapest_model(client),
            prompt=prompt,
            max_tokens=300,
            temperature=0.1,
        )

        data = json.loads(response.text)
        return AssumptionReport(
            has_ambiguity=data.get("has_ambiguity", False),
            assumptions=data.get("assumptions", []),
            interpretations=data.get("interpretations", []),
            clarification_questions=data.get("clarification_questions", []),
            confidence=data.get("confidence", 1.0),
        )
    except Exception:
        logger.warning("Assumption surfacing failed, proceeding without check")
        return AssumptionReport(has_ambiguity=False)


def _is_unambiguous(description: str) -> bool:
    """Quick heuristic check for clearly unambiguous descriptions.
    
    Unambiguous signals:
    - Contains specific file paths: "in src/auth.py"
    - Contains exact values: "set timeout to 30s"
    - Contains test-first language: "write a test that..."
    
    Ambiguous signals:
    - Contains vague verbs: "make it better", "fix it", "improve"
    - Contains ambiguous nouns: "the system", "the thing"
    - Contains no specific targets
    """
    # Ambiguous patterns
    vague_patterns = [
        r"\bmake it\b", r"\bfix it\b", r"\bimprove\b",
        r"\bthe system\b", r"\bthe thing\b", r"\bthe app\b",
    ]
    for pattern in vague_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return False

    # Specific patterns
    specific_patterns = [
        r"\bin \w+\.\w+\b",  # "in auth.py"
        r"\bset \w+ to \w+\b",  # "set timeout to 30s"
        r"\bwrite a test\b",  # test-first
        r"\badd a \w+\.\w+\b",  # "add a Button.tsx"
    ]
    for pattern in specific_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return True

    # Default: assume ambiguous for safety
    return False
```

### 2.2 — Wire into execution pipeline

**File:** `orchestrator/engine.py`

In `_decompose()` and `_execute_task()`, call `surface_assumptions()` before proceeding:

```python
async def _decompose(self, project_description, success_criteria):
    # Surface assumptions before decomposition
    report = await surface_assumptions(
        project_description, self.client
    )
    if report.has_ambiguity and report.clarification_questions:
        # In interactive mode: present to user
        # In non-interactive mode: log and proceed with assumptions stated
        logger.info(f"Assumptions surfaced: {len(report.clarification_questions)} questions")
        for q in report.clarification_questions:
            logger.info(f"  Q: {q}")
    
    # Inject surfaced assumptions into decomposition prompt
    assumption_context = report.to_prompt_context()
    if assumption_context:
        project_description = f"{project_description}\n\n{assumption_context}"
    
    # ... existing decomposition logic
```

### 2.3 — Verification Gate

```bash
# Test with ambiguous description
python -c "
from orchestrator.assumption_gate import surface_assumptions
import asyncio

report = asyncio.run(surface_assumptions(
    'Make the search faster', mock_client
))
assert report.has_ambiguity
assert len(report.interpretations) >= 2
print('Ambiguity detected:', report.interpretations)
"

# Test with specific description
python -c "
report = asyncio.run(surface_assumptions(
    'Add input validation to src/auth.py with minimum 8 char password', mock_client
))
assert not report.has_ambiguity
print('No ambiguity detected (correct)')
"
```

---

## Step 3: Simplicity Auditor (2-3 hours)

### 3.1 — Add `validate_simplicity()` to `orchestrator/validators.py`

```python
def validate_simplicity(output: str, task_description: str = "") -> ValidationResult:
    """Karpathy Principle 2: Simplicity First — detect over-engineering.

    Detects:
    1. Strategy/Factory/Abstract patterns with < 2 implementations
    2. Comments mentioning speculative flexibility
    3. Lines-to-features ratio warnings (>200 lines for a single feature)
    4. Empty error handlers (except: pass)

    These are warnings (not hard failures) — they surface complexity
    but don't block execution.
    """

    code = _extract_code_block(output, "python")
    issues = []

    # --- Check 1: Single-implementation abstractions ---
    # Detect abstract base classes with only one concrete subclass
    class_pattern = re.compile(r"class (\w+)\((?:ABC|metaclass=ABCMeta)\)", re.MULTILINE)
    subclasses = {}
    for m in class_pattern.finditer(code):
        base = m.group(1)
        # Find classes that inherit from this base
        sub_pat = re.compile(rf"class \w+\({base}\)")
        subclass_count = len(sub_pat.findall(code))
        if subclass_count <= 1:
            issues.append(
                f"Abstract class '{base}' has only {subclass_count} concrete subclass(es) — "
                "consider simplifying into a single class"
            )

    # --- Check 2: Speculative flexibility comments ---
    speculative_patterns = [
        (r"#.*?\b(configurable|extensible|flexible)\b", "speculative flexibility comment"),
        (r"#.*?\b(TODO|FIXME|HACK).*?\b(later|someday|eventually)\b", "speculative TODO comment"),
        (r"#.*?\b(might|could|may)\s+need\b", "speculative future-need comment"),
    ]
    for pattern, desc in speculative_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            issues.append(f"{desc}: found {len(matches)} instance(s)")

    # --- Check 3: Lines-to-features ratio ---
    code_lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) > 200 and task_description:
        # Simple heuristic: if task doesn't mention multiple files/modules,
        # 200+ lines is likely over-engineered
        multi_file_signals = ["multiple", "several", "database", "migration", "full-stack"]
        if not any(s in task_description.lower() for s in multi_file_signals):
            issues.append(
                f"Output is {len(code_lines)} lines for a single feature — "
                "consider if this could be simplified to < 100 lines"
            )

    # --- Check 4: Empty error handlers ---
    # except Exception: pass  (or silent logging only)
    empty_handler = re.compile(
        r"except[^:]*:\s*\n\s*(pass|logger\.(info|debug)\(['\"].*?['\"]\))"
    )
    if empty_handler.search(code):
        issues.append(
            "Empty error handler detected (bare 'pass' or silent log). "
            "Either handle the error properly or let it propagate."
        )

    if issues:
        return ValidationResult(
            False,
            f"Simplicity concerns ({len(issues)}):\n" + "\n".join(f"  - {i}" for i in issues),
            "simplicity",
        )

    return ValidationResult(True, "Simplicity check passed", "simplicity")
```

### 3.2 — Register in VALIDATORS dict

```python
VALIDATORS = {
    # ... existing validators ...
    "simplicity": validate_simplicity,  # NEW: Karpathy Principle 2
}
```

### 3.3 — Add to task decomposition (suggest, don't enforce)

The `simplicity` validator is a **soft check** — it flags issues but doesn't block execution. The orchestrator's `_decompose()` method should suggest it for code generation tasks:

```python
# In DecompositionPrompt.build():
"Suggested validators for code_generation tasks:\n"
'  - "python_syntax" (required)\n'
'  - "pytest" (recommended)\n'
'  - "simplicity" (optional) — flags over-engineered code\n'
```

### 3.4 — Verification Gate

```bash
# Test: over-engineered code with strategy pattern for single implementation
python -c "
from orchestrator.validators import validate_simplicity

# Over-engineered code
code = '''
from abc import ABC, abstractmethod
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount): pass
class PercentageDiscount(DiscountStrategy):
    def calculate(self, amount): return amount * 0.1
'''

result = validate_simplicity(code, 'Add discount calculation')
assert not result.passed, f'Expected failure, got: {result.details}'
print('Over-engineering detected:', result.details)

# Simple code
code = '''
def calculate_discount(amount, percent):
    return amount * (percent / 100)
'''
result = validate_simplicity(code, 'Add discount calculation')
assert result.passed, f'Expected pass, got: {result.details}'
print('Simple code passed:', result.details)
"
```

---

## Step 4: Surgical Change Auditor (2-3 hours)

### 4.1 — Add `validate_surgical_changes()` to `orchestrator/validators.py`

```python
def validate_surgical_changes(
    output: str,
    diff: str = "",
    task_description: str = "",
    locked_files: list[str] | None = None,
) -> ValidationResult:
    """Karpathy Principle 3: Surgical Changes — detect scope creep.

    Detects:
    1. Changes to files not in the task scope
    2. Style/formatting changes in untouched code sections
    3. Removal of pre-existing code unrelated to the task
    4. New imports not used in the changed code

    Args:
        output: The generated code
        diff: The diff from the change (if modifying existing code)
        task_description: What the task was supposed to do
        locked_files: Files that should NOT be modified
    """

    if not diff:
        return ValidationResult(True, "No diff provided, skipping", "surgical_changes")

    issues = []
    locked = set(locked_files or [])

    # --- Check 1: Locked files modified ---
    if locked:
        # Parse diff for file paths
        file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)", re.MULTILINE)
        changed_files = set()
        for m in file_pattern.finditer(diff):
            changed_files.add(m.group(1))
            changed_files.add(m.group(2))

        for locked_file in locked:
            if any(locked_file in f for f in changed_files):
                issues.append(f"Modified locked file: {locked_file}")

    # --- Check 2: Style-only changes ---
    # Detect lines that changed only style (quote style, whitespace, comment format)
    diff_lines = diff.split("\n")
    style_change_count = 0
    for line in diff_lines:
        if not line.startswith(("+", "-")):
            continue
        stripped = line[1:].strip()
        # Skip actual code changes
        if stripped.startswith(("import ", "from ", "def ", "class ", "return ", "if ", "for ")):
            continue
        # Detect style-only changes: line same except quotes/spacing/comments
        if line.startswith("+") and line[1:].lstrip().startswith(("#", "'''", '"""')):
            style_change_count += 1
        if line.startswith("-") and line[1:].lstrip().startswith(("#", "'''", '"""')):
            style_change_count += 1

    if style_change_count > 3:
        issues.append(
            f"Detected {style_change_count} style-only changes (comments, formatting) — "
            "remove style changes unrelated to the task"
        )

    # --- Check 3: Pre-existing code removed ---
    # Detect removal of functions/classes not mentioned in task
    removed_defs = re.findall(r"^-\s*(?:def |class )(\w+)", diff, re.MULTILINE)
    if removed_defs and task_description:
        for removed in removed_defs:
            if removed.lower() not in task_description.lower():
                issues.append(
                    f"Removed pre-existing definition '{removed}' — "
                    "not mentioned in task description"
                )

    # --- Check 4: New imports not used ---
    new_imports = re.findall(r"^\+import (\w+)", diff, re.MULTILINE)
    new_from_imports = re.findall(r"^\+from (\w+) import (\w+)", diff, re.MULTILINE)
    if new_imports or new_from_imports:
        # Check if these new imports are actually used in the added code
        added_lines = [l for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
        added_code = "\n".join(l[1:] for l in added_lines)

        for imp in new_imports:
            if imp not in added_code:
                issues.append(f"New import '{imp}' not used in added code — remove")

    if issues:
        return ValidationResult(
            False,
            f"Surgical change violations ({len(issues)}):\n" + "\n".join(f"  - {i}" for i in issues),
            "surgical_changes",
        )

    return ValidationResult(True, "Surgical change check passed", "surgical_changes")
```

### 4.2 — Register in VALIDATORS dict

```python
VALIDATORS = {
    # ... existing validators ...
    "simplicity": validate_simplicity,
    "surgical_changes": validate_surgical_changes,  # NEW: Karpathy Principle 3
}
```

### 4.3 — Verification Gate

```bash
# Test: diff that modifies a locked file
python -c "
from orchestrator.validators import validate_surgical_changes

diff = '''diff --git a/src/config.py b/src/config.py
+DEBUG = True  # Unrelated style change
- SECRET_KEY = 'abc123'
'''
result = validate_surgical_changes(
    '', diff=diff, task_description='Fix login bug',
    locked_files=['src/config.py']
)
assert not result.passed
print('Locked file modified (correct):', result.details)
"
```

---

## Step 5: Integration Summary

### Files Modified

| File | Change | Effort |
|------|--------|--------|
| `orchestrator/prompt_builder.py` | Add `_karpathy_guidelines()` to `SystemPrompt`; inject into `_standard()`, `_production()`, `CritiquePrompt` | 30 min |
| `orchestrator/validators.py` | Add `validate_simplicity()`, `validate_surgical_changes()`, register in VALIDATORS | 4-6 hours |
| `orchestrator/assumption_gate.py` | **New file** — `surface_assumptions()` gate | 2-3 hours |
| `orchestrator/engine.py` | Wire `surface_assumptions()` into `_decompose()`; suggest simplicity/surgical validators for code tasks | 1 hour |

### Test Verification

```bash
# All existing tests must still pass
pytest tests/test_god_file_refactoring.py tests/test_decomposer.py \
       tests/test_validator.py tests/test_pipeline.py -x --no-cov -q

# New validator tests
pytest tests/test_karpathy_validators.py -v  # Create this file

# Assumption gate integration test
python -m pytest tests/test_assumption_gate.py -v
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Simplicity validator catches Strategy pattern with 1 subclass | ✅ flag |
| Surgical validator detects changes to locked files | ✅ flag |
| Assumption gate detects ambiguity in vague descriptions | ✅ surfacing |
| Existing tests still pass | 100% |
| No prompt-based regressions (code quality maintained or improved) | ✅ |
