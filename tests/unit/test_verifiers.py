"""
Unit tests for the objective verifier modules.

Tests cover:
- PythonASTVerifier: valid/invalid code, code-block extraction, non-code task types
- JSONSchemaVerifier: valid/invalid JSON, schema validation, missing field detection
- RegexAssertVerifier: must_match / must_not_match rules, per-task-type defaults
- CompositeVerifier: weighted scoring, hard-gate fail, empty verifier list
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import TaskType, Verdict
from orchestrator.verification.python_ast import PythonASTVerifier
from orchestrator.verification.json_schema import JSONSchemaVerifier
from orchestrator.verification.regex_assert import RegexAssertVerifier, AssertionRule
from orchestrator.verification.composite import CompositeVerifier

# ═══════════════════════════════════════════════════════════════════════════════
# PythonASTVerifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestPythonASTVerifier:
    """Tests for PythonASTVerifier — AST-based syntactic verification."""

    @pytest.mark.asyncio
    async def test_valid_python_passes(self):
        verifier = PythonASTVerifier(extract_blocks=False)
        verdict = await verifier.verify(
            prompt="Write a function",
            response="def foo():\n    return 42",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5
        assert "ast_valid" in verdict.signals

    @pytest.mark.asyncio
    async def test_syntax_error_fails(self):
        verifier = PythonASTVerifier(extract_blocks=False)
        verdict = await verifier.verify(
            prompt="Write a function",
            response="def foo(:",  # invalid syntax
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is False
        assert verdict.score == 0.0
        assert "syntax_error" in verdict.signals

    @pytest.mark.asyncio
    async def test_extract_code_blocks(self):
        """Fenced code blocks are extracted and verified."""
        verifier = PythonASTVerifier(extract_blocks=True)
        response = """Here's the code:
```python
def hello():
    print("world")
```
"""
        verdict = await verifier.verify(
            prompt="Write hello world",
            response=response,
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5

    @pytest.mark.asyncio
    async def test_non_code_task_returns_not_applicable(self):
        verifier = PythonASTVerifier(extract_blocks=False)
        verdict = await verifier.verify(
            prompt="Write a story",
            response="Once upon a time...",
            task_type=TaskType.WRITING,
        )
        assert verdict.passed is True
        assert verdict.score == 0.5
        assert "not_applicable" in verdict.signals

    @pytest.mark.asyncio
    async def test_class_and_function_get_bonus(self):
        verifier = PythonASTVerifier(extract_blocks=False)
        verdict = await verifier.verify(
            prompt="Write a class",
            response="class Foo:\n    def bar(self):\n        pass",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.6  # class + func bonus
        assert "class_def" in verdict.signals
        assert "func_def" in verdict.signals

    @pytest.mark.asyncio
    async def test_empty_response_returns_minimal_score(self):
        """Empty string is valid AST (empty module), so it passes with base score."""
        verifier = PythonASTVerifier(extract_blocks=False)
        verdict = await verifier.verify(
            prompt="Write code",
            response="",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score == 0.5
        assert "ast_valid" in verdict.signals


# ═══════════════════════════════════════════════════════════════════════════════
# JSONSchemaVerifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestJSONSchemaVerifier:
    """Tests for JSONSchemaVerifier — JSON validity and schema compliance."""

    @pytest.mark.asyncio
    async def test_valid_json_passes(self):
        verifier = JSONSchemaVerifier()
        verdict = await verifier.verify(
            prompt="Return JSON",
            response='{"name": "Alice", "age": 30}',
            task_type=TaskType.DATA_EXTRACT,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5
        assert "json_valid" in verdict.signals

    @pytest.mark.asyncio
    async def test_invalid_json_fails(self):
        verifier = JSONSchemaVerifier()
        verdict = await verifier.verify(
            prompt="Return JSON",
            response="not json at all",
            task_type=TaskType.DATA_EXTRACT,
        )
        assert verdict.passed is False
        assert verdict.score == 0.0
        assert "json_parse_failed" in verdict.signals

    @pytest.mark.asyncio
    async def test_schema_validation_required_fields(self):
        """Detect missing required fields without jsonschema library."""
        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        verifier = JSONSchemaVerifier(schema=schema)
        verdict = await verifier.verify(
            prompt="Return user JSON",
            response='{"name": "Alice"}',  # missing email
            task_type=TaskType.DATA_EXTRACT,
        )
        assert verdict.passed is False
        assert verdict.score == 0.0
        assert "json_schema_failed" in verdict.signals

    @pytest.mark.asyncio
    async def test_schema_validation_passes(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
            },
        }
        verifier = JSONSchemaVerifier(schema=schema)
        verdict = await verifier.verify(
            prompt="Return user JSON",
            response='{"name": "Alice"}',
            task_type=TaskType.DATA_EXTRACT,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# RegexAssertVerifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegexAssertVerifier:
    """Tests for RegexAssertVerifier — pattern-based contract checks."""

    @pytest.mark.asyncio
    async def test_code_gen_with_function_passes(self):
        verifier = RegexAssertVerifier()
        verdict = await verifier.verify(
            prompt="Write a function",
            response="def add(a, b):\n    return a + b",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5

    @pytest.mark.asyncio
    async def test_code_gen_with_todo_fails(self):
        verifier = RegexAssertVerifier()
        verdict = await verifier.verify(
            prompt="Write a function",
            response="def add(a, b):\n    # TODO implement\n    pass",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is False
        assert verdict.score < 0.7

    @pytest.mark.asyncio
    async def test_code_review_terms(self):
        verifier = RegexAssertVerifier()
        verdict = await verifier.verify(
            prompt="Review this code",
            response="```\nprint('hello')\n```\nI found a security risk in the input validation.",
            task_type=TaskType.CODE_REVIEW,
        )
        assert verdict.passed is True

    @pytest.mark.asyncio
    async def test_non_relevant_task_returns_not_applicable(self):
        verifier = RegexAssertVerifier()
        verdict = await verifier.verify(
            prompt="Generate an image",
            response="some image data",
            task_type=TaskType.IMAGE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score == 0.5
        assert "not_applicable" in verdict.signals

    @pytest.mark.asyncio
    async def test_custom_rules(self):
        custom_rules = {
            "code_generation": [
                AssertionRule(r"async def", "has_async", kind="must_match"),
            ],
        }
        verifier = RegexAssertVerifier(rules=custom_rules)

        # Async function: should pass
        verdict = await verifier.verify(
            prompt="Write async function",
            response="async def fetch_data():\n    pass",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True

        # Sync only: should fail custom rule
        verdict = await verifier.verify(
            prompt="Write function",
            response="def fetch_data():\n    pass",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# CompositeVerifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompositeVerifier:
    """Tests for CompositeVerifier — weighted combination of sub-verifiers."""

    @pytest.mark.asyncio
    async def test_all_pass(self):
        ast_v = PythonASTVerifier(extract_blocks=False)
        regex_v = RegexAssertVerifier()
        composite = CompositeVerifier([(ast_v, 0.6), (regex_v, 0.4)])

        verdict = await composite.verify(
            prompt="Write a function",
            response="def foo():\n    return 42",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score >= 0.5

    @pytest.mark.asyncio
    async def test_hard_gate_failure(self):
        """A failing verifier with weight >= hard_gate_weight fails composite."""
        ast_v = PythonASTVerifier(extract_blocks=False)
        regex_v = RegexAssertVerifier()
        composite = CompositeVerifier(
            [(ast_v, 0.6), (regex_v, 0.4)],
            hard_gate_weight=0.3,
        )

        # Bad code with TODO should trigger regex failure
        verdict = await composite.verify(
            prompt="Write a function",
            response="def foo():\n    # TODO fix this\n    pass",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_empty_verifiers_returns_default(self):
        composite = CompositeVerifier()
        verdict = await composite.verify(
            prompt="Anything",
            response="Hello",
            task_type=TaskType.CODE_GEN,
        )
        assert verdict.passed is True
        assert verdict.score == 0.5
        assert "composite_empty" in verdict.signals

    @pytest.mark.asyncio
    async def test_failing_verifier_exception_caught(self):
        """A sub-verifier that raises should not crash the composite."""

        class FailingVerifier:
            async def verify(self, **kwargs) -> Verdict:
                raise RuntimeError("Intentional failure")

        composite = CompositeVerifier(
            [(FailingVerifier(), 1.0)],
            hard_gate_weight=0.0,
        )
        verdict = await composite.verify(
            prompt="test",
            response="test",
            task_type=TaskType.CODE_GEN,
        )
        # Failing verifier returns passed=False with score=0.0
        assert verdict.passed is False

    @pytest.mark.asyncio
    async def test_verifier_count_property(self):
        ast_v = PythonASTVerifier(extract_blocks=False)
        regex_v = RegexAssertVerifier()
        composite = CompositeVerifier([(ast_v, 0.5), (regex_v, 0.5)])
        assert composite.verifier_count == 2

        empty = CompositeVerifier()
        assert empty.verifier_count == 0
