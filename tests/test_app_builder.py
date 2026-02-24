"""
Tests for AppBuilder top-level pipeline (Task 7).
All Orchestrator calls and subprocess calls are mocked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.app_builder import AppBuildResult, AppBuilder
from orchestrator.app_detector import AppProfile


# ─────────────────────────────────────────────────────────────────────────────
# AppBuildResult — dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_app_build_result_fields():
    """AppBuildResult must have required fields."""
    result = AppBuildResult(
        success=True,
        output_dir="/tmp/myapp",
        profile=AppProfile(app_type="fastapi"),
        errors=[],
    )
    assert result.success is True
    assert result.output_dir == "/tmp/myapp"
    assert result.profile.app_type == "fastapi"
    assert result.errors == []


# ─────────────────────────────────────────────────────────────────────────────
# AppBuilder.build — full pipeline (mocked)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_returns_app_build_result(tmp_path):
    """build() must return an AppBuildResult instance."""
    import asyncio

    builder = AppBuilder()
    with _mock_pipeline(tmp_path):
        result = asyncio.run(builder.build(
            description="Build a FastAPI app",
            criteria="Must have health endpoint",
            output_dir=tmp_path,
        ))

    assert isinstance(result, AppBuildResult)


def test_build_success_when_all_steps_pass(tmp_path):
    """build() must return success=True when all pipeline steps succeed."""
    import asyncio

    builder = AppBuilder()
    with _mock_pipeline(tmp_path):
        result = asyncio.run(builder.build(
            description="Build a FastAPI app",
            criteria="Must have health endpoint",
            output_dir=tmp_path,
        ))

    assert result.success is True
    assert result.errors == []


def test_build_with_app_type_override(tmp_path):
    """build() with app_type_override must pass the override to ArchitectureAdvisor.analyze."""
    import asyncio

    builder = AppBuilder()
    cli_profile = AppProfile(app_type="cli")
    analyze_mock = AsyncMock(return_value=cli_profile)

    with _mock_pipeline(tmp_path) as mocks, \
         patch("orchestrator.app_builder.ArchitectureAdvisor.analyze", analyze_mock):
        result = asyncio.run(builder.build(
            description="Build a CLI tool",
            criteria="Must parse args",
            output_dir=tmp_path,
            app_type_override="cli",
        ))

    analyze_mock.assert_called_once()
    call_args = analyze_mock.call_args
    assert call_args[0][2] == "cli"
    assert result.profile.app_type == "cli"
def test_build_with_docker_flag(tmp_path):
    """build() with docker=True must call verify_docker."""
    import asyncio

    builder = AppBuilder()
    with _mock_pipeline(tmp_path, docker=True) as mocks:
        result = asyncio.run(builder.build(
            description="Build a FastAPI app",
            criteria="Must have health endpoint",
            output_dir=tmp_path,
            docker=True,
        ))

    mocks["verify_docker"].assert_called_once()


def test_build_handles_orchestrator_failure(tmp_path):
    """If the Orchestrator raises, build() must return success=False with error info."""
    import asyncio

    builder = AppBuilder()
    with _mock_pipeline(tmp_path, orchestrator_fails=True):
        result = asyncio.run(builder.build(
            description="Build something",
            criteria="Works",
            output_dir=tmp_path,
        ))

    assert result.success is False
    assert len(result.errors) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def _mock_pipeline(tmp_path: Path, docker: bool = False, orchestrator_fails: bool = False):
    """Context manager that mocks all external calls in the AppBuilder pipeline."""
    from orchestrator.app_assembler import AssemblyReport
    from orchestrator.app_detector import AppProfile
    from orchestrator.app_verifier import VerifyReport
    from orchestrator.dep_resolver import ResolveReport

    mock_profile = AppProfile(app_type="fastapi")
    mock_assembly = AssemblyReport(files_written=["src/main.py"])
    mock_deps = ResolveReport(packages=["fastapi"])
    mock_verify = VerifyReport(local_install_ok=True, tests_passed=True, startup_ok=True)
    mock_docker_verify = VerifyReport(docker_build_ok=True, docker_run_ok=True)

    mocks = {}

    detect_mock = AsyncMock(return_value=mock_profile)
    scaffold_mock = MagicMock(return_value={"src/main.py": "# code"})
    assemble_mock = MagicMock(return_value=mock_assembly)
    resolve_mock = MagicMock(return_value=mock_deps)
    verify_local_mock = MagicMock(return_value=mock_verify)
    verify_docker_mock = MagicMock(return_value=mock_docker_verify)

    if orchestrator_fails:
        orchestrator_mock = AsyncMock(side_effect=RuntimeError("Orchestrator failed"))
    else:
        # Return a mock ProjectState-like object
        mock_state = MagicMock()
        mock_state.status.name = "SUCCESS"
        mock_state.results = {}
        orchestrator_mock = AsyncMock(return_value=mock_state)

    mocks["verify_docker"] = verify_docker_mock

    with patch("orchestrator.app_builder.ArchitectureAdvisor.analyze", detect_mock), \
         patch("orchestrator.app_builder.ScaffoldEngine.scaffold", scaffold_mock), \
         patch("orchestrator.app_builder.AppAssembler.assemble", assemble_mock), \
         patch("orchestrator.app_builder.DependencyResolver.resolve", resolve_mock), \
         patch("orchestrator.app_builder.AppVerifier.verify_local", verify_local_mock), \
         patch("orchestrator.app_builder.AppVerifier.verify_docker", verify_docker_mock), \
         patch("orchestrator.app_builder.AppBuilder._run_orchestrator", orchestrator_mock):
        yield mocks
