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
        ok, reason = await check.run("x = 1")
        assert ok is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_fails_invalid_python(self):
        from orchestrator.infrastructure.verification_checks import _make_syntax_check

        check = _make_syntax_check()
        ok, reason = await check.run("x = ")
        assert ok is False
        assert "SyntaxError" in reason

    @pytest.mark.asyncio
    async def test_passes_empty_string(self):
        from orchestrator.infrastructure.verification_checks import _make_syntax_check

        check = _make_syntax_check()
        ok, reason = await check.run("")
        assert ok is True

    @pytest.mark.asyncio
    async def test_ambiguous_assignment_is_checked(self):
        """is_python_code fix: 'x = ' is treated as Python, so the syntax
        error surfaces instead of silently passing (pre-existing bug)."""
        from orchestrator.infrastructure.verification_checks import (
            _make_syntax_check,
            is_python_code,
        )

        assert is_python_code("x = ") is True
        check = _make_syntax_check()
        ok, reason = await check.run("x = ")
        assert ok is False
        assert "SyntaxError" in reason


class TestWorkspaceMaterializerSecurity:
    """Audit fix #4: path traversal guard on workspace file names."""

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self):
        from orchestrator.infrastructure.workspace_materializer import (
            WorkspaceMaterializer,
        )

        materializer = WorkspaceMaterializer()
        with pytest.raises(ValueError, match="Path traversal"):
            await materializer.materialize(
                framework="pytest",
                extra_files={"../../etc/passwd": "pwned"},
            )

    @pytest.mark.asyncio
    async def test_source_path_traversal_rejected(self):
        from orchestrator.infrastructure.workspace_materializer import (
            WorkspaceMaterializer,
        )

        materializer = WorkspaceMaterializer()
        with pytest.raises(ValueError, match="Path traversal"):
            await materializer.materialize(
                framework="pytest",
                source_files={"../outside.py": "print(1)"},
            )


class TestBuildCheck:
    """Build check — exec() in isolated namespace."""

    @pytest.mark.asyncio
    async def test_passes_importable_code(self):
        from orchestrator.infrastructure.verification_checks import _make_build_check

        check = _make_build_check()
        ok, reason = await check.run("def foo(): return 42")
        assert ok is True

    @pytest.mark.asyncio
    async def test_fails_runtime_error(self):
        from orchestrator.infrastructure.verification_checks import _make_build_check

        check = _make_build_check()
        ok, reason = await check.run("raise RuntimeError('boom')")
        assert ok is False
        assert "RuntimeError" in reason


class TestSecurityCheck:
    """Security check — static pattern detection."""

    @pytest.mark.asyncio
    async def test_passes_clean_code(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check.run("def add(a, b): return a + b")
        assert ok is True

    @pytest.mark.asyncio
    async def test_detects_eval(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check.run("eval('print(1)')")
        assert ok is False
        assert "eval" in reason

    @pytest.mark.asyncio
    async def test_detects_subprocess(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check.run("import subprocess; subprocess.call('rm -rf /')")
        assert ok is False
        assert "subprocess" in reason

    @pytest.mark.asyncio
    async def test_ignores_comments(self):
        from orchestrator.infrastructure.verification_checks import _make_security_check

        check = _make_security_check()
        ok, reason = await check.run("# eval is dangerous but this is a comment")
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
