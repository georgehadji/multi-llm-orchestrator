"""Tests for the runtime-operability artifact emitter (Phase 7, P-6)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest

from orchestrator.domain.readiness import AppArchetype
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.artifact_emitters.observability_emitter import ObservabilityEmitter


@dataclass
class _Profile:
    project_name: str = "sample_app"
    archetype: AppArchetype = AppArchetype.PYTHON_SERVICE


def _ws(root):
    return Workspace(root=root, framework="python")


@pytest.mark.unit
class TestApplicability:
    def test_applies_to_service_cli_and_fullstack(self):
        emitter = ObservabilityEmitter()
        assert emitter.applies_to(AppArchetype.PYTHON_SERVICE)
        assert emitter.applies_to(AppArchetype.PYTHON_CLI)
        assert emitter.applies_to(AppArchetype.FULLSTACK)

    def test_does_not_apply_to_library_or_web_static(self):
        emitter = ObservabilityEmitter()
        assert not emitter.applies_to(AppArchetype.LIBRARY)
        assert not emitter.applies_to(AppArchetype.WEB_STATIC)


@pytest.mark.unit
class TestEmit:
    async def test_service_archetype_writes_full_http_stack(self, tmp_path):
        written = await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        assert set(written) == {
            "app/__init__.py",
            "app/config.py",
            "app/logging_config.py",
            "app/observability.py",
            "app/health.py",
            "main.py",
        }
        for rel in written:
            assert (tmp_path / rel).is_file()

    async def test_fullstack_archetype_writes_full_http_stack(self, tmp_path):
        profile = _Profile(archetype=AppArchetype.FULLSTACK)
        written = await ObservabilityEmitter().emit(_ws(tmp_path), profile)
        assert "main.py" in written
        assert "app/health.py" in written

    async def test_cli_archetype_receives_no_http_artifacts(self, tmp_path):
        profile = _Profile(archetype=AppArchetype.PYTHON_CLI)
        written = await ObservabilityEmitter().emit(_ws(tmp_path), profile)
        assert set(written) == {"app/__init__.py", "app/config.py", "app/logging_config.py"}
        assert not (tmp_path / "main.py").exists()
        assert not (tmp_path / "app" / "health.py").exists()
        assert not (tmp_path / "app" / "observability.py").exists()

    async def test_service_name_threaded_into_config(self, tmp_path):
        profile = _Profile(project_name="my_special_service")
        await ObservabilityEmitter().emit(_ws(tmp_path), profile)
        config_text = (tmp_path / "app" / "config.py").read_text(encoding="utf-8")
        assert "my_special_service" in config_text

    async def test_config_has_no_default_secret_key(self, tmp_path):
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        config_text = (tmp_path / "app" / "config.py").read_text(encoding="utf-8")
        assert "secret_key: str = Field(" in config_text

    async def test_health_route_distinct_from_ready_route_in_source(self, tmp_path):
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        health_text = (tmp_path / "app" / "health.py").read_text(encoding="utf-8")
        assert '"/health"' in health_text
        assert '"/ready"' in health_text

    async def test_main_py_relies_on_uvicorn_signal_handling_not_a_custom_handler(self, tmp_path):
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        main_text = (tmp_path / "main.py").read_text(encoding="utf-8")
        assert "signal.signal" not in main_text
        assert "uvicorn.run" in main_text


def _run_py(tmp_path, code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    driver_path = tmp_path / "_driver.py"
    driver_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(driver_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=30,
        env=env,
    )


@pytest.mark.unit
class TestFailFastConfig:
    """Executes the emitted config module for real — a keyword grep on
    generated text cannot prove it actually raises, and doing so only at
    import time (not first request) is exactly the behavior P-6 requires.
    """

    async def test_missing_secret_key_raises_at_import(self, tmp_path):
        pytest.importorskip("pydantic_settings")
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        import os

        env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
        result = _run_py(tmp_path, "import app.config", env=env)
        assert result.returncode != 0
        assert "secret_key" in (result.stderr or "").lower()

    async def test_present_secret_key_imports_cleanly(self, tmp_path):
        pytest.importorskip("pydantic_settings")
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        import os

        env = {**os.environ, "SECRET_KEY": "test-secret"}
        result = _run_py(tmp_path, "import app.config; print('ok')", env=env)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


@pytest.mark.unit
class TestEmittedLoggingBehavior:
    async def test_correlation_id_present_on_emitted_logger_output(self, tmp_path):
        await ObservabilityEmitter().emit(_ws(tmp_path), _Profile())
        result = _run_py(
            tmp_path,
            """
            import app.logging_config as lc
            logger = lc.setup_logger("test")
            token = lc.correlation_id_var.set("corr-xyz-789")
            try:
                logger.info("hello from emitted app")
            finally:
                lc.correlation_id_var.reset(token)
            """,
        )
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert lines, f"no log output; stderr={result.stderr}"
        payload = json.loads(lines[-1])
        assert payload["correlation_id"] == "corr-xyz-789"
        assert payload["message"] == "hello from emitted app"
