"""T10 — the fixed findings must not come back quietly.

Each rule here corresponds to a finding that was actually fixed. A rule with no
history behind it is a rule someone eventually adds a blanket exemption to, so
this file stays limited to regressions we have really had.

Companion guards, kept separate because they carry their own explanations:
`test_no_pickle_deserialization.py` (SEC-002) and `test_cors_and_bind_policy.py`
(SEC-001).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "orchestrator"


def _modules() -> list[tuple[Path, ast.Module]]:
    parsed: list[tuple[Path, ast.Module]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            parsed.append(
                (path.relative_to(REPO_ROOT), ast.parse(path.read_text(encoding="utf-8")))
            )
        except (SyntaxError, UnicodeDecodeError):
            continue
    return parsed


def _attr_calls(tree: ast.Module, name: str) -> list[int]:
    """Line numbers of `<anything>.<name>(...)` calls."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


class TestNoShellExecution:
    """SEC-003: a host shell turns one argument into several commands."""

    #: dev_server runs a project's own build commands. They are literals in
    #: that module's ProjectType table, and the only interpolated value is a
    #: port already validated as an int in 1..65535.
    ALLOWED = {Path("orchestrator/dev_server.py")}

    def test_no_create_subprocess_shell(self) -> None:
        offenders = [
            f"{rel}:{line}"
            for rel, tree in _modules()
            if rel not in self.ALLOWED
            for line in _attr_calls(tree, "create_subprocess_shell")
        ]
        assert not offenders, (
            "asyncio.create_subprocess_shell is banned (SEC-003) — build an "
            "argv list and use create_subprocess_exec:\n  " + "\n  ".join(offenders)
        )

    def test_no_shell_true_keyword(self) -> None:
        offenders: list[str] = []
        for rel, tree in _modules():
            if rel in self.ALLOWED:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        offenders.append(f"{rel}:{node.lineno}")

        assert (
            not offenders
        ), "shell=True is banned (SEC-003) — pass an argv list instead:\n  " + "\n  ".join(
            offenders
        )


class TestOutboundFetchesGoThroughThePolicy:
    """SEC-004: an unguarded fetch reaches loopback and cloud metadata."""

    #: The policy module itself performs the one guarded request, and the
    #: readiness probes only ever dial a localhost URL they constructed.
    ALLOWED = {
        Path("orchestrator/safety/outbound.py"),
        Path("orchestrator/infrastructure/readiness_probes/_boot.py"),
        Path("orchestrator/infrastructure/model_capabilities.py"),
        Path("orchestrator/generators/wf100/evidence.py"),
    }

    def test_no_direct_urlopen(self) -> None:
        offenders = [
            f"{rel}:{line}"
            for rel, tree in _modules()
            if rel not in self.ALLOWED
            for line in _attr_calls(tree, "urlopen")
        ]
        assert not offenders, (
            "urllib.request.urlopen bypasses the SSRF policy (SEC-004) — use "
            "orchestrator.safety.outbound.fetch_json or check_destination "
            "first:\n  " + "\n  ".join(offenders)
        )


class TestSecurityModulesStayWired:
    """A policy nothing calls is documentation, not a control."""

    @pytest.mark.parametrize(
        ("module", "symbol"),
        [
            ("orchestrator.safety.outbound", "check_destination"),
            ("orchestrator.safety.api_keys", "KeyStore"),
            ("orchestrator.safety.secure_execution", "SecurePath"),
            ("orchestrator.domain.security", "Principal"),
            ("orchestrator.ide_backend.auth", "require_owner"),
            ("orchestrator.ide_backend.security", "validate_bind_target"),
        ],
    )
    def test_policy_symbol_is_importable(self, module: str, symbol: str) -> None:
        import importlib

        assert hasattr(importlib.import_module(module), symbol)

    def test_api_server_verifies_through_the_key_store(self) -> None:
        source = (PACKAGE_ROOT / "api_server.py").read_text(encoding="utf-8")
        assert "self.key_store.verify" in source, (
            "api_server must resolve keys through the KeyStore so revocation "
            "and expiry apply (T8)"
        )

    def test_ide_routes_check_ownership(self) -> None:
        source = (PACKAGE_ROOT / "ide_backend" / "api" / "routes.py").read_text(encoding="utf-8")
        # Every session-scoped handler goes through the one helper.
        assert "_owned_session" in source
        assert source.count("_owned_session(") >= 8
