"""
Fallback Handler — Circuit Breaker & Model Health
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles model selection, circuit breaker logic, fallback chains, and health tracking.

Part of Engine Decomposition (Phase 1) - Extracted from engine.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models import Model

if TYPE_CHECKING:
    from ..adaptive_router import AdaptiveRouter
    from ..models import TaskType

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Manages model health, circuit breaker, and fallback routing.

    Responsibilities:
    1. Track model health status
    2. Implement circuit breaker pattern
    3. Select optimal model for task type
    4. Select cross-provider reviewer
    5. Handle fallback chains on failures

    Circuit Breaker Pattern:
    - Model marked unhealthy after 3 consecutive failures
    - Cooldown period before retry
    - Success resets failure counter
    """

    # Circuit breaker: model is marked unhealthy after this many consecutive errors
    CIRCUIT_BREAKER_THRESHOLD: int = 3

    # Cooldown period for degraded models (seconds)
    COOLDOWN_PERIOD: int = 60

    def __init__(self, adaptive_router: AdaptiveRouter | None = None):
        """
        Initialize fallback handler.

        Args:
            adaptive_router: Optional adaptive router for advanced routing
        """
        # Model health tracking
        self.api_health: dict[Model, bool] = dict.fromkeys(Model, True)

        # Circuit breaker counters — consecutive failures per model
        self._consecutive_failures: dict[Model, int] = dict.fromkeys(Model, 0)

        # Cooldown tracking
        self._cooldown_until: dict[Model, float] = {}

        # Adaptive router integration
        self._adaptive_router = adaptive_router

        logger.info("Fallback handler initialized")

    def get_available_models(
        self,
        task_type: TaskType,
        exclude: list[Model] | None = None,
    ) -> list[Model]:
        """
        Get list of healthy models suitable for task type.

        Uses routing table to determine appropriate models for task type,
        then filters by health status.

        Args:
            task_type: Type of task (CODE_GEN, CODE_REVIEW, etc.)
            exclude: Optional list of models to exclude

        Returns:
            List of available models ordered by preference
        """
        from .models import ROUTING_TABLE

        # Get preferred models for this task type
        preferred = ROUTING_TABLE.get(task_type, [])

        # Filter by health and exclusions
        exclude_set = set(exclude) if exclude else set()
        available = []

        for model in preferred:
            # Skip excluded models
            if model in exclude_set:
                continue

            # Skip unhealthy models
            if not self.is_model_healthy(model):
                logger.debug(f"Excluding unhealthy model: {model.value}")
                continue

            # Skip models in cooldown
            if self._is_in_cooldown(model):
                logger.debug(f"Excluding model in cooldown: {model.value}")
                continue

            available.append(model)

        # Fallback to any healthy model if preferred list exhausted
        if not available:
            logger.warning(f"No preferred models available for {task_type}, using fallback")
            available = [m for m in Model if self.is_model_healthy(m) and m not in exclude_set]

        return available

    def select_model(
        self,
        task_type: TaskType,
        exclude: list[Model] | None = None,
    ) -> Model | None:
        """
        Select best model for task type.

        Args:
            task_type: Type of task
            exclude: Models to exclude

        Returns:
            Selected model or None if no models available
        """
        available = self.get_available_models(task_type, exclude)

        if not available:
            logger.error(f"No models available for task type: {task_type}")
            return None

        # Use adaptive router if available for intelligent selection
        if self._adaptive_router:
            selected = self._adaptive_router.select_preferred(available)
            if selected:
                logger.debug(f"Adaptive router selected: {selected.value}")
                return selected

        # Default: use first (highest priority) available model
        selected = available[0]
        logger.debug(f"Selected model: {selected.value}")
        return selected

    def select_reviewer(
        self,
        generator: Model,
        task_type: TaskType,
    ) -> Model | None:
        """
        Select a reviewer model for cross-model review.

        Invariants:
        1. Cross-review always uses different provider than generator
        2. Reviewer must be healthy

        Args:
            generator: Model used for generation
            task_type: Type of task being reviewed

        Returns:
            Selected reviewer model or None
        """
        from .models import get_provider

        # Get generator's provider
        generator_provider = get_provider(generator)

        # Get available models excluding generator
        available = self.get_available_models(task_type, exclude=[generator])

        # Filter for different provider
        cross_provider = [m for m in available if get_provider(m) != generator_provider]

        if not cross_provider:
            # Fallback: any model except generator
            logger.warning(
                f"No cross-provider reviewers available for {generator.value}, "
                "using same provider"
            )
            cross_provider = available

        if not cross_provider:
            logger.error(f"No reviewer available for generator: {generator.value}")
            return None

        # Select first available cross-provider model
        reviewer = cross_provider[0]
        logger.debug(f"Selected reviewer: {reviewer.value} (generator: {generator.value})")
        return reviewer

    def record_model_success(self, model: Model) -> None:
        """
        Record successful model execution.

        Resets failure counter and removes from cooldown.

        Args:
            model: Model that succeeded
        """
        # Reset consecutive failures
        self._consecutive_failures[model] = 0

        # Remove from cooldown
        if model in self._cooldown_until:
            del self._cooldown_until[model]

        # Ensure marked healthy
        self.api_health[model] = True

        # Update adaptive router
        if self._adaptive_router:
            self._adaptive_router.record_success(model)

        logger.debug(f"Recorded success for model: {model.value}")

    def record_model_failure(
        self,
        model: Model,
        error: Exception | None = None,
    ) -> None:
        """
        Record model failure.

        Implements circuit breaker pattern:
        - Increment failure counter
        - Mark unhealthy after threshold
        - Start cooldown period

        Args:
            model: Model that failed
            error: Optional exception for logging
        """
        # Increment failure counter
        failures = self._consecutive_failures[model] + 1
        self._consecutive_failures[model] = failures

        # Log error details
        if error:
            logger.debug(f"Model {model.value} failed: {error}")

        # Check if circuit breaker should trip
        if failures >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._trip_circuit_breaker(model)
        else:
            logger.warning(
                f"Model {model.value} failure {failures}/" f"{self.CIRCUIT_BREAKER_THRESHOLD}"
            )

        # Update adaptive router
        if self._adaptive_router:
            self._adaptive_router.record_failure(model)

    def _trip_circuit_breaker(self, model: Model) -> None:
        """
        Trip circuit breaker for model.

        Args:
            model: Model to mark unhealthy
        """
        self.api_health[model] = False

        # Set cooldown period
        import time

        self._cooldown_until[model] = time.time() + self.COOLDOWN_PERIOD

        logger.error(
            f"Circuit breaker tripped for {model.value} "
            f"after {self._consecutive_failures[model]} failures. "
            f"Cooldown: {self.COOLDOWN_PERIOD}s"
        )

    def _is_in_cooldown(self, model: Model) -> bool:
        """
        Check if model is in cooldown period.

        Args:
            model: Model to check

        Returns:
            True if in cooldown
        """
        if model not in self._cooldown_until:
            return False

        import time

        if time.time() >= self._cooldown_until[model]:
            # Cooldown expired, remove
            del self._cooldown_until[model]
            return False

        return True

    def is_model_healthy(self, model: Model) -> bool:
        """
        Check if model is healthy and available.

        Args:
            model: Model to check

        Returns:
            True if healthy
        """
        # Check if explicitly marked unhealthy
        if not self.api_health.get(model, False):
            return False

        # Check if in cooldown
        return not self._is_in_cooldown(model)

    def get_model_status(self, model: Model) -> dict:
        """
        Get detailed status for a model.

        Args:
            model: Model to check

        Returns:
            Status dictionary
        """
        import time

        cooldown_remaining = 0
        if model in self._cooldown_until:
            remaining = self._cooldown_until[model] - time.time()
            cooldown_remaining = max(0, remaining)

        return {
            "healthy": self.is_model_healthy(model),
            "api_available": self.api_health.get(model, False),
            "consecutive_failures": self._consecutive_failures[model],
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining_seconds": cooldown_remaining,
        }

    def reset_model(self, model: Model) -> None:
        """
        Manually reset model state (for recovery).

        Args:
            model: Model to reset
        """
        self._consecutive_failures[model] = 0
        self.api_health[model] = True

        if model in self._cooldown_until:
            del self._cooldown_until[model]

        logger.info(f"Manually reset model: {model.value}")

    def reset_all(self) -> None:
        """Reset all model state (for testing/recovery)."""
        self._consecutive_failures = dict.fromkeys(Model, 0)
        self.api_health = dict.fromkeys(Model, True)
        self._cooldown_until.clear()

        if self._adaptive_router:
            self._adaptive_router.reset()

        logger.info("Fallback handler reset")

    def get_health_summary(self) -> dict[str, int]:
        """
        Get summary of model health.

        Returns:
            Dictionary with health statistics
        """
        healthy = sum(1 for m in Model if self.is_model_healthy(m))
        unhealthy = len(Model) - healthy
        in_cooldown = sum(1 for m in Model if self._is_in_cooldown(m))

        return {
            "total_models": len(Model),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "in_cooldown": in_cooldown,
        }
