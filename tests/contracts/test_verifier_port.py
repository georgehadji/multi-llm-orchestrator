"""
Contract tests for Verifier protocol.

Every concrete verifier must:
1. Satisfy the ``Verifier`` Protocol (runtime-checkable)
2. Accept ``prompt``, ``response``, ``task_type`` keyword arguments
3. Return a ``Verdict`` dataclass with ``passed``, ``score``, ``signals``, ``detail``
4. Never raise on any input — fail-closed to ``Verdict(passed=False, score=0.0)``
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from orchestrator.models import TaskType, Verdict
from orchestrator.verification.port import Verifier
from orchestrator.verification.python_ast import PythonASTVerifier
from orchestrator.verification.json_schema import JSONSchemaVerifier
from orchestrator.verification.regex_assert import RegexAssertVerifier

# Every concrete verifier class to test
CONCRETE_VERIFIERS = [
    PythonASTVerifier(extract_blocks=False),
    PythonASTVerifier(extract_blocks=True),
    JSONSchemaVerifier(),
    JSONSchemaVerifier(schema={"type": "object", "required": ["name"]}),
    RegexAssertVerifier(),
    RegexAssertVerifier(rules={"code_generation": []}),
]


class TestVerifierProtocolConformance:
    """Every concrete verifier must satisfy the Verifier protocol."""

    @pytest.mark.parametrize("verifier", CONCRETE_VERIFIERS, ids=lambda v: type(v).__name__)
    def test_is_verifier_protocol(self, verifier):
        """Instance must satisfy the Verifier runtime-checkable protocol."""
        assert isinstance(
            verifier, Verifier
        ), f"{type(verifier).__name__} does not satisfy Verifier protocol"

    @pytest.mark.parametrize("verifier", CONCRETE_VERIFIERS, ids=lambda v: type(v).__name__)
    @pytest.mark.asyncio
    async def test_verify_returns_verdict(self, verifier):
        """verify() must return a Verdict dataclass."""
        result = await verifier.verify(
            prompt="test prompt",
            response="test response",
            task_type=TaskType.CODE_GEN,
        )
        assert isinstance(result, Verdict), f"Expected Verdict, got {type(result).__name__}"

    @pytest.mark.parametrize("verifier", CONCRETE_VERIFIERS, ids=lambda v: type(v).__name__)
    @pytest.mark.asyncio
    async def test_verdict_has_required_fields(self, verifier):
        """Verdict must have passed: bool, score: float, signals: tuple, detail: str."""
        result = await verifier.verify(
            prompt="test prompt",
            response="test response",
            task_type=TaskType.CODE_GEN,
        )
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)
        assert isinstance(result.signals, tuple)
        assert isinstance(result.detail, str)
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of [0, 1] range"

    @pytest.mark.parametrize("verifier", CONCRETE_VERIFIERS, ids=lambda v: type(v).__name__)
    @pytest.mark.asyncio
    async def test_verify_never_raises(self, verifier):
        """verify() must never raise — fail-closed behavior."""
        # Various edge-case inputs
        inputs = [
            {"prompt": "", "response": "", "task_type": TaskType.CODE_GEN},
            {"prompt": "\x00null bytes", "response": "\x00", "task_type": TaskType.WRITING},
            {"prompt": "a" * 10000, "response": "b" * 10000, "task_type": TaskType.CODE_REVIEW},
            {
                "prompt": "```\nbad\n```",
                "response": "```\ncode\n```",
                "task_type": TaskType.DATA_EXTRACT,
            },
        ]
        for kwargs in inputs:
            try:
                result = await verifier.verify(**kwargs)
                assert isinstance(result, Verdict)
            except Exception as exc:
                pytest.fail(f"{type(verifier).__name__} raised {type(exc).__name__}: {exc}")


class TestNonVerifierRejected:
    """Objects that are not Verifier protocol should not pass isinstance check."""

    def test_plain_object_not_verifier(self):
        assert not isinstance(object(), Verifier)

    def test_wrong_signature_not_verifier(self):
        class NotAVerifier:
            async def not_verify(self, foo, bar):
                return None

        assert not isinstance(NotAVerifier(), Verifier)
