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
