"""
Deployment Feedback Loop (Autonomous Software Maintainer)
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Deploy → Monitor → Auto-fix → Redeploy

Current State: Generate code → Stop
Future State: Continuous monitoring with auto-repair

Benefits:
- Transforms orchestrator from "code generator" to "autonomous maintainer"
- Base44 Superagent territory, but with verified code quality
- Continuous improvement loop

Usage:
    from orchestrator.deployment_feedback import DeploymentFeedbackLoop

    loop = DeploymentFeedbackLoop(orchestrator)
    await loop.monitor_and_fix(deployment_url="https://my-app.com")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .log_config import get_logger
from .models import TaskType

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class EscalationLevel(str, Enum):
    """Escalation level for auto-fix decisions."""

    AUTO = "auto"  # Auto-fix without human review
    REVIEW = "review"  # Human review required
    HUMAN_REQUIRED = "human_required"  # Human must fix


@dataclass
class HealthCheck:
    """Result of health check."""

    status: HealthStatus
    errors: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class Diagnosis:
    """Diagnosis of deployment issue."""

    summary: str
    root_cause: str
    severity: str  # critical, high, medium, low
    affected_components: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class AutoFix:
    """Auto-generated fix."""

    description: str
    code_changes: dict[str, str] = field(default_factory=dict)
    config_changes: dict[str, Any] = field(default_factory=dict)
    requires_restart: bool = False
    rollback_plan: str = ""


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""

    health_check_interval: int = 300  # 5 minutes
    max_auto_fix_attempts: int = 3
    escalation_threshold: float = 0.7  # Confidence threshold for auto-fix
    health_endpoint: str = "/health"
    timeout: int = 30


class DeploymentFeedbackLoop:
    """
    Monitor deployed apps, auto-fix issues, redeploy.

    Flow:
    1. Health check (every 5 minutes)
    2. If unhealthy → Diagnose
    3. Generate fix
    4. Verify fix locally
    5. Deploy fix
    6. Record in memory bank
    7. Repeat
    """

    def __init__(
        self,
        orchestrator,
        config: MonitoringConfig | None = None,
    ):
        """
        Initialize deployment feedback loop.

        Args:
            orchestrator: Orchestrator instance for generating fixes
            config: Monitoring configuration
        """
        self.orchestrator = orchestrator
        self.config = config or MonitoringConfig()

        self.is_monitoring = False
        self.monitoring_task: asyncio.Task | None = None

        # Statistics
        self.health_checks_run = 0
        self.issues_detected = 0
        self.fixes_applied = 0
        self.fixes_successful = 0

        logger.info("Deployment feedback loop initialized")

    async def start_monitoring(
        self,
        deployment_url: str,
        project_id: str,
    ) -> None:
        """
        Start continuous monitoring.

        Args:
            deployment_url: URL of deployed application
            project_id: Project identifier
        """
        if self.is_monitoring:
            logger.warning("Already monitoring")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(deployment_url, project_id)
        )

        logger.info(
            f"Started monitoring {deployment_url} (interval={self.config.health_check_interval}s)"
        )

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped monitoring")

    async def monitor_and_fix(
        self,
        deployment_url: str,
        project_id: str,
    ) -> None:
        """
        Run single monitoring cycle with auto-fix.

        Args:
            deployment_url: URL of deployed application
            project_id: Project identifier
        """
        logger.info(f"Running monitoring cycle for {project_id}")

        # 1. Health check
        health = await self._check_health(deployment_url)
        self.health_checks_run += 1

        if health.status == HealthStatus.HEALTHY:
            logger.info(f"  {project_id}: Healthy")
            return

        logger.warning(f"  {project_id}: {health.status.value} - {health.errors}")
        self.issues_detected += 1

        # 2. Diagnose
        diagnosis = await self._diagnose(health, project_id)
        logger.info(f"  Diagnosis: {diagnosis.summary} (confidence={diagnosis.confidence:.2f})")

        # 3. Determine escalation level
        escalation = self._determine_escalation(diagnosis)

        if escalation == EscalationLevel.HUMAN_REQUIRED:
            logger.warning(f"  {project_id}: Human intervention required")
            await self._escalate(diagnosis, project_id)
            return

        # 4. Generate fix
        fix = await self._generate_fix(diagnosis, project_id)

        # 5. Verify fix locally
        verified = await self._verify_fix(fix, project_id)

        if not verified:
            logger.warning(f"  {project_id}: Fix verification failed")
            await self._escalate(diagnosis, project_id)
            return

        # 6. Deploy fix
        if escalation == EscalationLevel.AUTO:
            deployed = await self._deploy_fix(fix, project_id, deployment_url)

            if deployed:
                self.fixes_applied += 1
                self.fixes_successful += 1
                logger.info(f"  {project_id}: Fix deployed successfully")

                # 7. Record in memory bank
                await self._record_fix(diagnosis, fix, project_id)
            else:
                logger.error(f"  {project_id}: Fix deployment failed")
                self.fixes_applied += 1
                # Don't increment fixes_successful
        else:
            # REVIEW level - queue for human review
            logger.info(f"  {project_id}: Fix queued for human review")
            await self._queue_for_review(diagnosis, fix, project_id)

    async def _monitoring_loop(
        self,
        deployment_url: str,
        project_id: str,
    ) -> None:
        """Continuous monitoring loop."""
        while self.is_monitoring:
            try:
                await self.monitor_and_fix(deployment_url, project_id)
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")

            await asyncio.sleep(self.config.health_check_interval)

    async def _check_health(self, deployment_url: str) -> HealthCheck:
        """
        Check deployment health.

        Args:
            deployment_url: Deployment URL

        Returns:
            HealthCheck result
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                # Health endpoint check
                health_url = f"{deployment_url.rstrip('/')}{self.config.health_endpoint}"

                try:
                    response = await client.get(health_url)

                    if response.status_code == 200:
                        return HealthCheck(
                            status=HealthStatus.HEALTHY,
                            timestamp=asyncio.get_event_loop().time(),
                        )
                    elif response.status_code < 500:
                        return HealthCheck(
                            status=HealthStatus.DEGRADED,
                            errors=[f"Health endpoint returned {response.status_code}"],
                            timestamp=asyncio.get_event_loop().time(),
                        )
                    else:
                        return HealthCheck(
                            status=HealthStatus.UNHEALTHY,
                            errors=[f"Health endpoint returned {response.status_code}"],
                            timestamp=asyncio.get_event_loop().time(),
                        )

                except httpx.HTTPError as e:
                    return HealthCheck(
                        status=HealthStatus.UNHEALTHY,
                        errors=[f"Health check failed: {str(e)}"],
                        timestamp=asyncio.get_event_loop().time(),
                    )

        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNKNOWN,
                errors=[f"Health check error: {str(e)}"],
                timestamp=asyncio.get_event_loop().time(),
            )

    async def _diagnose(
        self,
        health: HealthCheck,
        project_id: str,
    ) -> Diagnosis:
        """
        Diagnose deployment issue.

        Args:
            health: Health check result
            project_id: Project identifier

        Returns:
            Diagnosis with root cause analysis
        """
        # Build diagnosis prompt
        prompt = (
            f"Diagnose this deployment issue:\n\n"
            f"Project: {project_id}\n"
            f"Health Status: {health.status.value}\n"
            f"Errors: {', '.join(health.errors)}\n"
            f"Logs: {'; '.join(health.logs[:10]) if health.logs else 'None'}\n\n"
            f"Provide:\n"
            f"1. Summary of the issue\n"
            f"2. Root cause\n"
            f"3. Severity (critical/high/medium/low)\n"
            f"4. Affected components\n"
            f"5. Suggested fix\n"
            f"6. Confidence score (0.0-1.0)"
        )

        try:
            from .api_clients import UnifiedClient

            client = UnifiedClient()

            response = await client.call(
                model=self.orchestrator._get_available_models(TaskType.CODE_REVIEW)[0],
                prompt=prompt,
                system="You are an expert DevOps engineer diagnosing deployment issues. "
                "Be specific about root causes and provide actionable fixes.",
                max_tokens=1000,
                temperature=0.2,
                timeout=60,
            )

            # Parse diagnosis from response
            # In production, would use structured output
            return Diagnosis(
                summary=response.text[:200],
                root_cause=response.text[:200],
                severity="medium",
                affected_components=[],
                suggested_fix=response.text,
                confidence=0.7,
            )

        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return Diagnosis(
                summary=f"Failed to diagnose: {e}",
                root_cause="Unknown",
                severity="medium",
                confidence=0.5,
            )

    def _determine_escalation(self, diagnosis: Diagnosis) -> EscalationLevel:
        """
        Determine escalation level based on diagnosis.

        FIX-PS-004a: Require human review for ALL auto-deploys.

        Rationale: LLM confidence scores are not reliable security guarantees.
        Attacker can craft issues that trigger high-confidence malicious fixes.
        Auto-deploy disabled until code signing + verification implemented.

        Args:
            diagnosis: Diagnosis result

        Returns:
            EscalationLevel
        """
        # ═══════════════════════════════════════════════════════
        # FIX-PS-004a: Disable auto-deploy for security
        # ═══════════════════════════════════════════════════════

        # CRITICAL: Never auto-deploy without human review
        # Auto-deploy is too risky without mature verification

        if diagnosis.severity in ["critical", "high"]:
            # Critical/high severity always requires human review
            logger.info(f"Escalation: {diagnosis.severity} severity -> HUMAN_REQUIRED")
            return EscalationLevel.HUMAN_REQUIRED

        elif diagnosis.confidence >= 0.95:
            # Very high confidence -> queue for review (not auto)
            logger.info(f"Escalation: {diagnosis.confidence:.2f} confidence -> REVIEW")
            return EscalationLevel.REVIEW

        else:
            # All other cases require human intervention
            logger.info("Escalation: default -> HUMAN_REQUIRED")
            return EscalationLevel.HUMAN_REQUIRED

    async def _generate_fix(
        self,
        diagnosis: Diagnosis,
        project_id: str,
    ) -> AutoFix:
        """
        Generate fix for diagnosed issue.

        Args:
            diagnosis: Diagnosis
            project_id: Project identifier

        Returns:
            AutoFix with code/config changes
        """
        # Build fix generation prompt
        prompt = (
            f"Generate a fix for this issue:\n\n"
            f"Problem: {diagnosis.summary}\n"
            f"Root Cause: {diagnosis.root_cause}\n"
            f"Affected: {', '.join(diagnosis.affected_components)}\n\n"
            f"Provide:\n"
            f"1. Description of the fix\n"
            f"2. Code changes (file paths and new content)\n"
            f"3. Configuration changes if any\n"
            f"4. Whether restart is required\n"
            f"5. Rollback plan"
        )

        try:
            from .api_clients import UnifiedClient

            client = UnifiedClient()

            response = await client.call(
                model=self.orchestrator._get_available_models(TaskType.CODE_GEN)[0],
                prompt=prompt,
                system="You are an expert software engineer generating production fixes. "
                "Provide minimal, targeted changes that address the root cause.",
                max_tokens=4000,
                temperature=0.1,
                timeout=120,
            )

            return AutoFix(
                description=response.text[:500],
                code_changes={},  # Would parse from response
                config_changes={},
                requires_restart=False,
                rollback_plan="Revert to previous deployment",
            )

        except Exception as e:
            logger.error(f"Fix generation failed: {e}")
            return AutoFix(
                description=f"Failed to generate fix: {e}",
            )

    async def _verify_fix(
        self,
        fix: AutoFix,
        project_id: str,
    ) -> bool:
        """
        Verify fix locally before deployment.

        Args:
            fix: AutoFix to verify
            project_id: Project identifier

        Returns:
            True if verification passed
        """
        logger.info(f"  {project_id}: Verifying fix...")

        # Run tests if available
        # In production, would run full test suite

        # For now, basic validation
        if not fix.description or fix.description.startswith("Failed"):
            return False

        # Would run:
        # - Syntax validation
        # - Unit tests
        # - Integration tests
        # - Security scanning

        return True

    async def _deploy_fix(
        self,
        fix: AutoFix,
        project_id: str,
        deployment_url: str,
    ) -> bool:
        """
        Deploy fix to production.

        Args:
            fix: AutoFix to deploy
            project_id: Project identifier
            deployment_url: Deployment URL

        Returns:
            True if deployment successful
        """
        logger.info(f"  {project_id}: Deploying fix...")

        # In production, would integrate with:
        # - GitHub Actions
        # - AWS CodeDeploy
        # - Vercel/Netlify API
        # - Kubernetes

        # Placeholder - would call actual deployment API
        await asyncio.sleep(1)  # Simulate deployment

        return True

    async def _record_fix(
        self,
        diagnosis: Diagnosis,
        fix: AutoFix,
        project_id: str,
    ) -> None:
        """
        Record fix in memory bank for learning.

        Args:
            diagnosis: Original diagnosis
            fix: Applied fix
            project_id: Project identifier
        """
        logger.info(f"  {project_id}: Recording fix in memory bank")

        # Would save to memory bank:
        # - Issue pattern
        # - Root cause
        # - Fix applied
        # - Outcome

        pass

    async def _escalate(
        self,
        diagnosis: Diagnosis,
        project_id: str,
    ) -> None:
        """
        Escalate issue to human.

        Args:
            diagnosis: Diagnosis
            project_id: Project identifier
        """
        logger.warning(f"  {project_id}: Escalating to human review")

        # Would send notification via:
        # - Email
        # - Slack
        # - PagerDuty

        pass

    async def _queue_for_review(
        self,
        diagnosis: Diagnosis,
        fix: AutoFix,
        project_id: str,
    ) -> None:
        """
        Queue fix for human review.

        Args:
            diagnosis: Diagnosis
            fix: Generated fix
            project_id: Project identifier
        """
        logger.info(f"  {project_id}: Queued for human review")

        # Would add to review queue

        pass

    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "health_checks_run": self.health_checks_run,
            "issues_detected": self.issues_detected,
            "fixes_applied": self.fixes_applied,
            "fixes_successful": self.fixes_successful,
            "success_rate": (self.fixes_successful / max(1, self.fixes_applied)),
        }


__all__ = [
    "DeploymentFeedbackLoop",
    "HealthStatus",
    "EscalationLevel",
    "HealthCheck",
    "Diagnosis",
    "AutoFix",
    "MonitoringConfig",
]
