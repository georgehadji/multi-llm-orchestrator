"""
Token Budget Module
====================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements output token budget control for 15-25% cost reduction.

Features:
- Phase-specific output token limits
- Automatic max_tokens enforcement
- Token usage tracking
- Cost estimation

Usage:
    from orchestrator.optimization import TokenBudget, OptimizationPhase

    budget = TokenBudget()
    
    # Get phase-specific limit
    max_tokens = budget.get_limit(OptimizationPhase.GENERATION)
    
    # Enforce in API call
    response = await client.call(
        model="claude-sonnet-4.6",
        prompt=prompt,
        max_tokens=budget.get_limit(phase),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..log_config import get_logger
from . import OptimizationPhase

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    requests: int = 0

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return self.input_cost + self.output_cost

    @property
    def avg_output_tokens(self) -> float:
        """Calculate average output tokens per request."""
        if self.requests == 0:
            return 0.0
        return self.output_tokens / self.requests


@dataclass
class TokenBudgetMetrics:
    """Metrics for token budget enforcement."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    tokens_saved: int = 0
    estimated_savings: float = 0.0
    limit_enforced_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_input_cost + self.total_output_cost,
            "tokens_saved": self.tokens_saved,
            "estimated_savings": self.estimated_savings,
            "limit_enforced_count": self.limit_enforced_count,
        }


class TokenBudget:
    """
    Manage output token budgets per phase.

    Usage:
        budget = TokenBudget()
        max_tokens = budget.get_limit(phase)
    """

    # Phase-specific output token limits
    # Based on typical output requirements
    DEFAULT_LIMITS = {
        OptimizationPhase.DECOMPOSITION: 2000,      # Task list, structured JSON
        OptimizationPhase.GENERATION: 4000,          # Code output
        OptimizationPhase.CRITIQUE: 800,             # Score + brief reasoning
        OptimizationPhase.EVALUATION: 500,           # Score only
        OptimizationPhase.PROMPT_ENHANCEMENT: 500,   # Enhanced prompt text
        OptimizationPhase.CONDENSING: 1000,          # Summary
    }

    # Cost per 1K tokens (USD) - approximate
    COST_PER_1K = {
        "input": {
            "claude-opus": 15.0,
            "claude-sonnet": 3.0,
            "claude-haiku": 0.25,
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "gpt-4o": 5.0,
            "deepseek": 1.0,
        },
        "output": {
            "claude-opus": 75.0,
            "claude-sonnet": 15.0,
            "claude-haiku": 1.25,
            "gpt-4": 60.0,
            "gpt-4-turbo": 30.0,
            "gpt-4o": 15.0,
            "deepseek": 4.0,
        },
    }

    def __init__(self, custom_limits: Optional[Dict[OptimizationPhase, int]] = None):
        """
        Initialize token budget.

        Args:
            custom_limits: Optional custom token limits per phase
        """
        self.metrics = TokenBudgetMetrics()
        self._limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
        self._usage: Dict[str, TokenUsage] = {}

    def get_limit(self, phase: OptimizationPhase) -> int:
        """
        Get output token limit for phase.

        Args:
            phase: Optimization phase

        Returns:
            Maximum output tokens
        """
        limit = self._limits.get(phase, 1000)
        logger.debug(f"Token limit for {phase.value}: {limit}")
        return limit

    def get_limit_by_name(self, phase_name: str) -> int:
        """
        Get output token limit by phase name string.

        Args:
            phase_name: Phase name (e.g., "generation")

        Returns:
            Maximum output tokens
        """
        try:
            phase = OptimizationPhase(phase_name)
            return self.get_limit(phase)
        except ValueError:
            logger.warning(f"Unknown phase: {phase_name}, using default limit")
            return 1000

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        phase: OptimizationPhase,
    ) -> None:
        """
        Record token usage for tracking.

        Args:
            model: Model used
            input_tokens: Input tokens consumed
            output_tokens: Output tokens consumed
            phase: Optimization phase
        """
        # Track usage
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens

        # Calculate costs
        input_cost = self._calculate_cost(model, input_tokens, "input")
        output_cost = self._calculate_cost(model, output_tokens, "output")

        self.metrics.total_input_cost += input_cost
        self.metrics.total_output_cost += output_cost

        # Track per-model usage
        if model not in self._usage:
            self._usage[model] = TokenUsage()

        usage = self._usage[model]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.input_cost += input_cost
        usage.output_cost += output_cost
        usage.requests += 1

        logger.debug(
            f"Token usage: model={model}, input={input_tokens}, "
            f"output={output_tokens}, cost=${output_cost:.4f}"
        )

    def enforce_limit(
        self,
        model: str,
        prompt: str,
        phase: OptimizationPhase,
        estimated_output: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Enforce token limit and return API parameters.

        Args:
            model: Model to use
            prompt: Prompt text
            phase: Optimization phase
            estimated_output: Optional estimated output tokens

        Returns:
            Dictionary with max_tokens and other parameters
        """
        limit = self.get_limit(phase)

        # Estimate output if not provided
        if estimated_output is None:
            # Rough estimate: 20% of input length
            estimated_output = int(len(prompt) / 4 * 0.2)

        # If estimated exceeds limit, log warning
        if estimated_output > limit:
            logger.warning(
                f"Estimated output ({estimated_output}) exceeds limit ({limit}) "
                f"for phase {phase.value}. Enforcing limit."
            )
            self.metrics.limit_enforced_count += 1

        # Estimate savings from limit enforcement
        potential_waste = max(0, estimated_output - limit)
        if potential_waste > 0:
            savings = self._calculate_cost(model, potential_waste, "output")
            self.metrics.tokens_saved += potential_waste
            self.metrics.estimated_savings += savings

        return {
            "max_tokens": limit,
            "phase": phase.value,
            "estimated_output": estimated_output,
        }

    def _calculate_cost(
        self,
        model: str,
        tokens: int,
        token_type: str,
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            model: Model name
            tokens: Number of tokens
            token_type: "input" or "output"

        Returns:
            Cost in USD
        """
        model_key = model.lower()
        cost_dict = self.COST_PER_1K.get(token_type, {})

        # Find matching model cost
        cost_per_1k = 1.0  # Default
        for key, cost in cost_dict.items():
            if key in model_key:
                cost_per_1k = cost
                break

        return (tokens / 1000) * cost_per_1k

    def get_usage(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get token usage statistics.

        Args:
            model: Optional specific model

        Returns:
            Usage statistics dictionary
        """
        if model:
            if model in self._usage:
                usage = self._usage[model]
                return {
                    "model": model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "input_cost": usage.input_cost,
                    "output_cost": usage.output_cost,
                    "total_cost": usage.total_cost,
                    "requests": usage.requests,
                    "avg_output_tokens": usage.avg_output_tokens,
                }
            return {"model": model, "error": "No usage data"}

        # All models
        return {
            "total_input_tokens": self.metrics.total_input_tokens,
            "total_output_tokens": self.metrics.total_output_tokens,
            "total_cost": self.metrics.total_input_cost + self.metrics.total_output_cost,
            "tokens_saved": self.metrics.tokens_saved,
            "estimated_savings": self.metrics.estimated_savings,
            "limit_enforced_count": self.metrics.limit_enforced_count,
            "models": {
                m: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "total_cost": u.total_cost,
                    "requests": u.requests,
                }
                for m, u in self._usage.items()
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive token budget metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = TokenBudgetMetrics()
        self._usage.clear()
        logger.info("Token budget metrics reset")


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def get_token_limit(phase: str) -> int:
    """
    Convenience function to get token limit.

    Args:
        phase: Phase name

    Returns:
        Token limit
    """
    budget = TokenBudget()
    return budget.get_limit_by_name(phase)


__all__ = [
    "TokenBudget",
    "TokenUsage",
    "TokenBudgetMetrics",
    "get_token_limit",
]
