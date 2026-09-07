"""Tests for JSON structured logging + correlation id in logging_generator.py
(Phase 7, P-6). Extends the pre-existing (previously orphan) LoggingConfigBuilder
rather than reimplementing it — mirrors P-5's extension of cicd_generator.py.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from orchestrator.generators.logging_generator import LoggingConfigBuilder, LogFormat


def _build_json_source(*, correlation_id: bool = True) -> str:
    builder = (
        LoggingConfigBuilder().for_python().with_standard_logging().with_format(LogFormat.JSON)
    )
    if correlation_id:
        builder = builder.with_correlation_id()
    return builder.build()


@pytest.mark.unit
class TestJsonLoggingGeneration:
    def test_json_format_emits_a_json_formatter_class(self):
        source = _build_json_source()
        assert "class JsonFormatter" in source
        assert "json.dumps" in source

    def test_text_format_is_unaffected(self):
        source = (
            LoggingConfigBuilder().for_python().with_standard_logging().with_format(LogFormat.TEXT)
        ).build()
        assert "class JsonFormatter" not in source

    def test_correlation_id_context_var_present_when_requested(self):
        source = _build_json_source(correlation_id=True)
        assert "correlation_id_var" in source
        assert "ContextVar" in source

    def test_correlation_id_omitted_when_not_requested(self):
        source = _build_json_source(correlation_id=False)
        assert "correlation_id_var" not in source


@pytest.mark.unit
class TestJsonFormatterBehavior:
    """Executes the generated source for real — a keyword grep on generated
    text cannot prove the formatter actually emits parseable, correlated
    JSON log lines.
    """

    def test_generated_formatter_emits_correlated_json(self, tmp_path):
        source = _build_json_source(correlation_id=True)
        module_path = tmp_path / "logging_config.py"
        module_path.write_text(source, encoding="utf-8")

        driver = textwrap.dedent("""
            import json
            import logging
            import sys

            sys.path.insert(0, r"{tmp_path}")
            import logging_config

            logger = logging_config.setup_logger("test")
            token = logging_config.correlation_id_var.set("corr-abc-123")
            try:
                logger.info("hello world")
            finally:
                logging_config.correlation_id_var.reset(token)
            """).format(tmp_path=str(tmp_path))
        driver_path = tmp_path / "driver.py"
        driver_path.write_text(driver, encoding="utf-8")

        import subprocess

        result = subprocess.run(
            [sys.executable, str(driver_path)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert lines, f"no log output captured; stderr={result.stderr}"
        payload = json.loads(lines[-1])
        assert payload["correlation_id"] == "corr-abc-123"
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
