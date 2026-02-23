"""
AppBuilder — top-level class that wires all App Builder pipeline components.
Author: Georgios-Chrysovalantis Chatzivantsidis

Pipeline:
  1. AppDetector.detect()        -> AppProfile
  2. ScaffoldEngine.scaffold()   -> dict[rel_path, content]
  3. _run_orchestrator()         -> ProjectState (with TaskResults)
  4. AppAssembler.assemble()     -> AssemblyReport
  5. DependencyResolver.resolve() -> ResolveReport
  6. AppVerifier.verify_local()  -> VerifyReport
  7. [optional] AppVerifier.verify_docker() -> VerifyReport

Returns AppBuildResult.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from orchestrator.app_assembler import AppAssembler, AssemblyReport
from orchestrator.app_detector import AppDetector, AppProfile
from orchestrator.app_verifier import AppVerifier, VerifyReport
from orchestrator.dep_resolver import DependencyResolver, ResolveReport
from orchestrator.scaffold import ScaffoldEngine

logger = logging.getLogger(__name__)


@dataclass
class AppBuildResult:
    """Result of a full app build pipeline run."""

    success: bool = False
    output_dir: str = ""
    profile: Optional[AppProfile] = None
    assembly: Optional[AssemblyReport] = None
    dependencies: Optional[ResolveReport] = None
    local_verify: Optional[VerifyReport] = None
    docker_verify: Optional[VerifyReport] = None
    errors: list[str] = field(default_factory=list)


class AppBuilder:
    """
    Top-level pipeline class that builds a complete app from a description.

    Usage:
        builder = AppBuilder()
        result = await builder.build(
            description="Build a FastAPI REST API",
            criteria="Must have health endpoint and tests",
            output_dir=Path("/tmp/my-app"),
            app_type_override="fastapi",  # optional override
            docker=False,
        )
    """

    def __init__(self) -> None:
        self._detector = AppDetector()
        self._scaffold = ScaffoldEngine()
        self._assembler = AppAssembler()
        self._resolver = DependencyResolver()
        self._verifier = AppVerifier()

    async def build(
        self,
        description: str,
        criteria: str,
        output_dir: Path,
        app_type_override: Optional[str] = None,
        docker: bool = False,
    ) -> AppBuildResult:
        """
        Run the full app build pipeline.

        Parameters
        ----------
        description: Plain-text description of the app to build
        criteria:    Success criteria / acceptance test description
        output_dir:  Directory where the app will be written
        app_type_override: If set, skip LLM detection and use this app type
        docker:      If True, also run Docker verification
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = AppBuildResult(output_dir=str(output_dir))

        try:
            # -- Step 1: Detect app type --
            logger.info("AppBuilder: detecting app type...")
            if app_type_override:
                # Skip LLM detection; use the YAML override directly.
                profile = self._detector.detect_from_yaml(app_type_override)
            else:
                profile = await self._detector.detect(description)
            result.profile = profile
            logger.info(
                "AppBuilder: app_type=%s detected_from=%s",
                profile.app_type,
                profile.detected_from,
            )

            # -- Step 2: Scaffold --
            logger.info("AppBuilder: scaffolding...")
            scaffold = self._scaffold.scaffold(profile, output_dir)

            # -- Step 3: Run orchestrator to generate code --
            logger.info("AppBuilder: running orchestrator...")
            project_state = await self._run_orchestrator(
                description, criteria, output_dir, profile
            )

            # -- Step 4: Assemble --
            logger.info("AppBuilder: assembling...")
            tasks = getattr(project_state, "tasks", {})
            results = getattr(project_state, "results", {})
            assembly = self._assembler.assemble(results, tasks, scaffold, output_dir)
            result.assembly = assembly

            # -- Step 5: Resolve dependencies --
            logger.info("AppBuilder: resolving dependencies...")
            deps = self._resolver.resolve(output_dir)
            result.dependencies = deps

            # -- Step 6: Local verification --
            logger.info("AppBuilder: running local verification...")
            local_verify = self._verifier.verify_local(output_dir, profile)
            result.local_verify = local_verify

            # -- Step 7: Docker verification (optional) --
            if docker:
                logger.info("AppBuilder: running Docker verification...")
                docker_verify = self._verifier.verify_docker(output_dir, profile)
                result.docker_verify = docker_verify

            # -- Determine overall success --
            result.success = local_verify.success
            if docker and result.docker_verify is not None:
                result.success = result.success and result.docker_verify.success
            if result.errors:
                result.success = False

            logger.info(
                "AppBuilder: done -- success=%s files_written=%d",
                result.success,
                len(assembly.files_written),
            )

        except Exception as exc:
            logger.exception("AppBuilder pipeline failed: %s", exc)
            result.success = False
            result.errors.append(str(exc))

        return result

    async def _run_orchestrator(
        self,
        description: str,
        criteria: str,
        output_dir: Path,
        profile: AppProfile,
    ):
        """
        Isolated method to run the Orchestrator — allows test mocking.

        In production, this initializes and runs the existing Orchestrator.
        The Orchestrator class lives in orchestrator.engine and its main entry
        point is run_project(project_description, success_criteria).
        Returns the ProjectState.
        """
        # Import here to avoid circular imports at module load time.
        from orchestrator.engine import Orchestrator  # noqa: PLC0415

        orchestrator = Orchestrator()
        state = await orchestrator.run_project(
            project_description=description,
            success_criteria=criteria,
            app_profile=profile,
        )
        return state
