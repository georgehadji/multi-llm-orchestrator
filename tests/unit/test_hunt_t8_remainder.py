"""
T8 (remainder, coverage-ordered) proof-of-defect and no-regression tests.

Five VERIFIED DEFECTs from a background-agent survey of 8 residual candidates
carried over from T2/T5/T6/T7's own triage notes — see
docs/hunts/t8-remainder/inventory.md for full detail per candidate.

C1 — generators/secrets_manager.py's SecretsFilter was fully built but never
     attached to a real logger anywhere in the app; the one real logging-setup
     function, log_config.py::configure_logging(), only ever attached
     CorrelationIdFilter.
C4 — plugin/plugin_isolation_secure.py's seccomp network-syscall-blocking loop
     swallowed every add_rule() failure with a bare `except Exception: pass`
     and no log, unlike its sibling "dangerous syscalls" loop (which at least
     documents the same silence as an accepted arch-dependent tradeoff) — a
     silent, selective fail-open of the network-egress sandbox boundary.
C5 — generators/website_validator.py's hardcoded-secret scanner silently
     skipped any frontend file it couldn't read and then reported a clean
     "No secrets found" result — a false-clean scan gating the real,
     documented `orchestrator website --min-quality`/`--require-all-checks`
     CLI flags.
C6 — operations/diagnostics.py's environment check required OPENAI_API_KEY/
     GOOGLE_API_KEY/ANTHROPIC_API_KEY/MINIMAX_API_KEY, none of which
     infrastructure/llm_client.py::UnifiedClient (the live client) ever reads
     — it reads OPENROUTER_API_KEY/DEEPSEEK_API_KEY/XAI_API_KEY. A correctly
     configured OPENROUTER_API_KEY-only setup was reported CRITICAL-broken.
C7 — adaptive_router.py::AdaptiveRouter.is_available() returned True
     unconditionally whenever a concurrent writer held the lock, ignoring
     already-committed DISABLED/DEGRADED state for every model.
"""

from __future__ import annotations

import logging
import sys

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_configure_logging_installs_secrets_filter(capsys):
    from orchestrator.log_config import configure_logging

    orchestrator_logger = logging.getLogger("orchestrator")
    original_handlers = orchestrator_logger.handlers[:]
    original_level = orchestrator_logger.level
    try:
        configure_logging(level="INFO", format="text")
        child = logging.getLogger("orchestrator.test_c1_secrets_filter")
        child.info("using key sk-abcdefghij1234567890klmno for auth")

        captured = capsys.readouterr()
        assert (
            "sk-abcdefghij1234567890klmno" not in captured.out
        ), "raw secret leaked into log output unmasked"
        assert "REDACTED" in captured.out
    finally:
        orchestrator_logger.handlers.clear()
        orchestrator_logger.handlers.extend(original_handlers)
        orchestrator_logger.setLevel(original_level)


# --- C4 -----------------------------------------------------------------


class _FakeSeccompFilter:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def add_rule(self, _action, syscall) -> None:
        if syscall == "socket":
            raise OSError("simulated: syscall not supported on this arch")

    def load(self) -> None:
        pass


class _FakeSeccompModule:
    ALLOW = object()

    def ERRNO(self, code):  # noqa: N802 - matches real seccomp module's API
        return code

    SyscallFilter = _FakeSeccompFilter


def test_c4_seccomp_network_rule_failure_is_logged(monkeypatch, caplog):
    from orchestrator.plugin.plugin_isolation_secure import (
        SecureIsolatedRuntime,
        SecureIsolationConfig,
    )

    monkeypatch.setitem(sys.modules, "seccomp", _FakeSeccompModule())
    runtime = SecureIsolatedRuntime(SecureIsolationConfig(allow_network=False))

    with caplog.at_level(logging.WARNING):
        runtime._apply_seccomp(runtime.config)

    assert any(
        "socket" in rec.message for rec in caplog.records
    ), f"expected a warning naming the failed network syscall rule, got: {[r.message for r in caplog.records]}"


# --- C5 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c5_website_validator_flags_unreadable_frontend_file(tmp_path, caplog):
    from orchestrator.generators.website_validator import WebsiteQualityValidator

    (tmp_path / "good.js").write_text("console.log('hello');", encoding="utf-8")
    # A directory named *.js can't be read as text — a real, deterministic
    # read failure that doesn't depend on OS permission bits.
    (tmp_path / "bad.js").mkdir()

    validator = WebsiteQualityValidator()
    with caplog.at_level(logging.WARNING):
        check = await validator._check_secret_exposure(tmp_path)

    assert check.passed is False, "an unreadable file must not roll into a passing scan"
    assert any(
        "bad.js" in rec.message for rec in caplog.records
    ), f"expected a warning naming the unreadable file, got: {[r.message for r in caplog.records]}"


# --- C6 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c6_diagnostics_accepts_openrouter_key_alone(monkeypatch):
    from orchestrator.operations.diagnostics import Severity, SystemDiagnostic

    for var in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "DEEPSEEK_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-key-value")

    diag = SystemDiagnostic()
    await diag._check_environment()

    assert not any(
        issue.error_code == "ENV001" and issue.severity == Severity.CRITICAL
        for issue in diag.issues
    ), (
        "a real OPENROUTER_API_KEY-only setup (the live client's actual "
        f"requirement) was reported CRITICAL: {[i.description for i in diag.issues]}"
    )


# --- C7 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c7_is_available_reflects_disabled_state_during_concurrent_write():
    from orchestrator.adaptive_router import AdaptiveRouter
    from orchestrator.models import Model

    router = AdaptiveRouter()
    model = next(iter(Model))
    router._disabled.add(model)

    # Simulate a concurrent writer (e.g. recording another model's timeout)
    # holding the lock while is_available() is called for this model.
    await router._lock.acquire()
    try:
        assert router.is_available(model) is False, (
            "a permanently-DISABLED model must never report available, even "
            "while a concurrent writer holds the lock for an unrelated update"
        )
    finally:
        router._lock.release()
