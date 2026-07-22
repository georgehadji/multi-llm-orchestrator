"""
Tests for WBS-1: concrete verification check adapters.
These are infrastructure-layer tests that check adapter factory functions.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# We test the check adapters via their async run functions.
# We import lazily to avoid heavy deps during collection.
pytest.importorskip("orchestrator")


class TestSyntaxCheck:
    """Syntax check adapter — pure Python compile()."""

    @pytest.mark.asyncio
    async def test_passes_valid_python(self):
        from orchestrator.infrastructure.verification_checks import _make_syntax_check

        check = _make_syntax_check()
        ok, reason = await check("x = 1")
        assert ok is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fails_invalid_python(self):
        from orchestrator.infrastructure.verification_checks import _make_syntax_check

        check = _make_syntax_check()
        ok, reason = await check("x = ")
        assert ok is False
        assert "SyntaxError" in reason

    @pytest.mark.asyncio
    async def test_passes_empty_string(self):
        from orchestrator.infrastructure.verification_checks import _make_syntax_check

        check = _make_syntax_check()
        ok, reason = await check("")
        assert ok is True


class TestBuildCheck:
    """Build check — exec() in isolated namespace."""

    @pytest.mark.asyncio
    async def test_passes_importable_code(self):
        from orchestrator.infrastructure.verification_checks import _make_build_check

        check = _make_build_check()
        ok, reason = await check("def foo(): return 42")
        assert ok is True

    @pytest.mark.asyncio
    async def test_fails_runtime_error(self):
        from orchestrator.infrastructure.verification_checks import _make_build_check

        check = _make_build_check()
        ok, reason = await check("raise RuntimeError('boom')")
        assert ok is False
        assert "RuntimeError" in reason


class TestSecurityCheck:
    """Security check — static pattern detection."""

    @pytest.mark.asyncio
    async def test_passes_clean_code(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check("def add(a, b): return a + b")
        assert ok is True

    @pytest.mark.asyncio
    async def test_detects_eval(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check("eval('print(1)')")
        assert ok is False
        assert "eval" in reason

    @pytest.mark.asyncio
    async def test_detects_subprocess(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check("import subprocess; subprocess.call('rm -rf /')")
        assert ok is False
        assert "subprocess" in reason

    @pytest.mark.asyncio
    async def test_ignores_comments(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check("# eval is dangerous but this is a comment")
        assert ok is True  # comment-only lines should pass


class TestDefaultChecks:
    """Default check set."""

    @pytest.mark.asyncio
    async def test_default_checks_always_includes_syntax_and_build(self):
        from orchestrator.infrastructure.verification_checks import default_checks

        checks = default_checks()
        names = {c.name for c in checks}
        assert "syntax" in names
        assert "build" in names
        assert "security" in names
