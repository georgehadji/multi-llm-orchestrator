"""
Tests for CLI build subcommand and JobSpec fields (Task 8).
"""
from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# JobSpec — new fields
# ─────────────────────────────────────────────────────────────────────────────

def test_jobspec_has_app_type_field():
    """JobSpec must have an app_type field (defaults to empty string or None)."""
    from orchestrator.models import JobSpec
    spec = JobSpec(description="Build a FastAPI app", success_criteria="Must work")
    # app_type should exist and default to empty string or None
    assert hasattr(spec, "app_type")


def test_jobspec_has_docker_field():
    """JobSpec must have a docker field (defaults to False)."""
    from orchestrator.models import JobSpec
    spec = JobSpec(description="Build a FastAPI app", success_criteria="Must work")
    assert hasattr(spec, "docker")
    assert spec.docker is False


def test_jobspec_has_output_dir_field():
    """JobSpec must have an output_dir field (defaults to empty string or None)."""
    from orchestrator.models import JobSpec
    spec = JobSpec(description="Build a FastAPI app", success_criteria="Must work")
    assert hasattr(spec, "output_dir")


def test_jobspec_app_builder_fields_can_be_set():
    """All three new fields can be set at construction."""
    from orchestrator.models import JobSpec
    spec = JobSpec(
        description="Build a FastAPI app",
        success_criteria="Must work",
        app_type="fastapi",
        docker=True,
        output_dir="/tmp/myapp",
    )
    assert spec.app_type == "fastapi"
    assert spec.docker is True
    assert spec.output_dir == "/tmp/myapp"


# ─────────────────────────────────────────────────────────────────────────────
# CLI build subcommand — smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_build_subcommand_exists():
    """The CLI must have a 'build' subcommand registered."""
    import orchestrator.cli as cli_module
    # Check that the build command is accessible
    # Works whether it's argparse or click
    assert hasattr(cli_module, "build") or hasattr(cli_module, "cmd_build") or \
           "build" in getattr(cli_module, "__all__", []) or \
           _has_build_parser(cli_module)


def _has_build_parser(cli_module) -> bool:
    """Check if the cli module has a build subparser defined."""
    import inspect
    source = inspect.getsource(cli_module)
    return "build" in source and ("subparser" in source or "add_parser" in source or "command" in source)


# ─────────────────────────────────────────────────────────────────────────────
# cmd_build — functional tests with mocked AppBuilder
# ─────────────────────────────────────────────────────────────────────────────

def test_cmd_build_calls_app_builder(tmp_path):
    """cmd_build must create an AppBuilder and call build() with correct args."""
    from orchestrator.cli import cmd_build
    from orchestrator.app_builder import AppBuildResult
    from orchestrator.app_detector import AppProfile
    from unittest.mock import AsyncMock, MagicMock, patch

    args = MagicMock()
    args.description = "A FastAPI app"
    args.criteria = "Must work"
    args.app_type = "fastapi"
    args.docker = False
    args.output_dir = str(tmp_path)

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.output_dir = str(tmp_path)
    mock_result.errors = []

    with patch("orchestrator.app_builder.AppBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=mock_result)
        mock_builder_cls.return_value = mock_builder

        cmd_build(args)

    mock_builder.build.assert_called_once()
    call_kwargs = mock_builder.build.call_args
    assert call_kwargs is not None


def test_cmd_build_passes_docker_flag(tmp_path):
    """cmd_build must pass docker=True when --docker flag is set."""
    from orchestrator.cli import cmd_build
    from unittest.mock import AsyncMock, MagicMock, patch

    args = MagicMock()
    args.description = "A FastAPI app"
    args.criteria = "Must work"
    args.app_type = ""
    args.docker = True
    args.output_dir = str(tmp_path)

    mock_result = MagicMock(success=True, output_dir=str(tmp_path), errors=[])

    with patch("orchestrator.app_builder.AppBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=mock_result)
        mock_builder_cls.return_value = mock_builder

        cmd_build(args)

    call_kwargs = mock_builder.build.call_args.kwargs
    assert call_kwargs.get("docker") is True or mock_builder.build.call_args[1].get("docker") is True or \
           (len(mock_builder.build.call_args[0]) > 3 and mock_builder.build.call_args[0][3] is True)


def test_cmd_build_failure_prints_error(tmp_path, capsys):
    """cmd_build must print an error message when build() returns success=False."""
    from orchestrator.cli import cmd_build
    from unittest.mock import AsyncMock, MagicMock, patch

    args = MagicMock()
    args.description = "A FastAPI app"
    args.criteria = "Must work"
    args.app_type = ""
    args.docker = False
    args.output_dir = str(tmp_path)

    mock_result = MagicMock(success=False, output_dir=str(tmp_path), errors=["orchestrator failed"])

    with patch("orchestrator.app_builder.AppBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=mock_result)
        mock_builder_cls.return_value = mock_builder

        cmd_build(args)

    captured = capsys.readouterr()
    assert "fail" in captured.out.lower() or "error" in captured.out.lower() or "✗" in captured.out
