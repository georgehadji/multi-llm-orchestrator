"""
Tests for the sandbox tiers (F-3) — isolation properties.
==========================================================
Unit tests exercise the SubprocessSandbox's isolation guarantees:
env scrubbing, determinism pins, temp HOME, timeout kill, cleanup.
Docker tests are integration-marked and skipped when docker is absent.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from orchestrator.domain.testing_models import IsolationLevel
from orchestrator.infrastructure.sandboxes import SubprocessSandbox, resolve_sandbox
from orchestrator.infrastructure.sandboxes.subprocess_sandbox import _scrub_env


@pytest.mark.unit
class TestEnvScrubbing:
    """Sensitive env never reaches model-authored code (D-3/F-3)."""

    def test_scrub_removes_sensitive_keys(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-secret",
            "DB_PASSWORD": "p4ss",
            "DATABASE_URL": "postgres://u:p@h/db",
            "SAFE": "keep-me",
        }
        clean = _scrub_env(env)
        assert "OPENAI_API_KEY" not in clean
        assert "DB_PASSWORD" not in clean
        assert "DATABASE_URL" not in clean
        assert clean["SAFE"] == "keep-me"

    def test_scrub_adds_determinism_pins(self) -> None:
        clean = _scrub_env({})
        assert clean["PYTHONHASHSEED"] == "0"
        assert clean["TZ"] == "UTC"
        assert clean["SOURCE_DATE_EPOCH"] == "0"


@pytest.mark.unit
class TestSubprocessSandboxIsolation:
    """Isolation guarantees of the subprocess tier (no Docker needed)."""

    def test_api_key_not_visible_to_child(self) -> None:
        async def run() -> str:
            sandbox = SubprocessSandbox()
            code = (
                "import os, json; " "print(json.dumps({'key': os.environ.get('OPENAI_API_KEY')}))"
            )
            rc, out, err = await sandbox.exec(
                ["python", "-c", code], cwd=Path("."), env={}, timeout_s=20
            )
            assert rc == 0, err
            return json.loads(out.strip())["key"]

        os.environ["OPENAI_API_KEY"] = "sk-test-scrub"
        try:
            assert asyncio.run(run()) is None
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_temp_home_used_and_cleaned(self) -> None:
        async def run() -> tuple[str, str]:
            sandbox = SubprocessSandbox()
            code = "import os; print(os.environ.get('HOME', os.environ.get('USERPROFILE','')))"
            rc, out, err = await sandbox.exec(
                ["python", "-c", code], cwd=Path("."), env={}, timeout_s=20
            )
            assert rc == 0, err
            home = out.strip()
            return home, home

        home, _ = asyncio.run(run())
        assert "orch-sandbox-home-" in home
        assert not Path(home).exists(), "temp HOME leaked after run"

    def test_determinism_pins_in_child(self) -> None:
        async def run() -> dict:
            sandbox = SubprocessSandbox()
            code = (
                "import os, json; print(json.dumps({"
                "'seed': os.environ.get('PYTHONHASHSEED'),"
                "'tz': os.environ.get('TZ')}))"
            )
            rc, out, err = await sandbox.exec(
                ["python", "-c", code], cwd=Path("."), env={}, timeout_s=20
            )
            assert rc == 0, err
            return json.loads(out.strip())

        data = asyncio.run(run())
        assert data["seed"] == "0"
        assert data["tz"] == "UTC"

    def test_timeout_kills_and_returns_minus_one(self) -> None:
        async def run() -> tuple[int, str]:
            sandbox = SubprocessSandbox()
            rc, out, err = await sandbox.exec(
                ["python", "-c", "import time; time.sleep(60)"],
                cwd=Path("."),
                env={},
                timeout_s=2,
            )
            return rc, err

        rc, err = asyncio.run(run())
        assert rc == -1
        assert "Timeout" in err

    def test_level_is_subprocess(self) -> None:
        assert SubprocessSandbox().level is IsolationLevel.SUBPROCESS


@pytest.mark.unit
class TestTierResolver:
    """Tier selection never returns NONE (F-3)."""

    def test_subprocess_tier_forced(self) -> None:
        sandbox = resolve_sandbox(tier="subprocess")
        assert isinstance(sandbox, SubprocessSandbox)
        assert sandbox.level is IsolationLevel.SUBPROCESS

    def test_resolver_never_returns_none_tier(self) -> None:
        sandbox = resolve_sandbox(tier="auto")
        assert sandbox.level in (IsolationLevel.SUBPROCESS, IsolationLevel.DOCKER)

    def test_docker_tier_raises_when_unavailable(self) -> None:
        from orchestrator.infrastructure.sandboxes.docker_sandbox import _docker_available

        if _docker_available():
            pytest.skip("docker available on this machine")
        with pytest.raises(RuntimeError):
            resolve_sandbox(tier="docker")
