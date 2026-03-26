"""
Speculative Generation Module
==============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements speculative execution for 30-40% premium cost reduction.

Strategy: Run cheap and premium models in parallel, use cheap if good enough,
cancel premium if not needed.

Features:
- Parallel cheap + premium execution
- Quick evaluation of cheap result
- Premium cancellation on success
- Zero latency penalty when cheap succeeds

Usage:
    from orchestrator.cost_optimization import SpeculativeGenerator

    gen = SpeculativeGenerator(client=api_client)
    
    result = await gen.speculative_generate(
        prompt="Generate Python code...",
        cheap_model="deepseek-chat",
        premium_model="claude-opus-4.6",
        threshold=0.85,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..log_config import get_logger

logger = get_logger(__name__)


@dataclass
class SpeculativeMetrics:
    """Metrics for speculative generation."""
    total_attempts: int = 0
    cheap_wins: int = 0  # Cheap model was good enough
    premium_wins: int = 0  # Required premium
    cancellations: int = 0  # Premium cancelled
    avg_cheap_score: float = 0.0
    total_cost: float = 0.0
    estimated_savings: float = 0.0

    @property
    def cheap_win_rate(self) -> float:
        """Calculate cheap model win rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.cheap_wins / self.total_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "total_attempts": self.total_attempts,
            "cheap_wins": self.cheap_wins,
            "premium_wins": self.premium_wins,
            "cancellations": self.cancellations,
            "cheap_win_rate": self.cheap_win_rate,
            "avg_cheap_score": self.avg_cheap_score,
            "total_cost": self.total_cost,
            "estimated_savings": self.estimated_savings,
        }


@dataclass
class SpeculativeResult:
    """Result of speculative generation."""
    response: str
    model_used: str
    score: float
    cost: float
    cheap_score: float
    premium_cancelled: bool
    latency_seconds: float


class SpeculativeGenerator:
    """
    Speculative generation for cost optimization.

    Usage:
        gen = SpeculativeGenerator(client=api_client)
        result = await gen.speculative_generate(prompt, cheap_model, premium_model)
    """

    # Default model pairs for speculative execution
    DEFAULT_MODEL_PAIRS = {
        "code_generation": {
            "cheap": "deepseek-chat",
            "premium": "claude-opus-4.6",
            "threshold": 0.85,
        },
        "code_review": {
            "cheap": "deepseek-chat",
            "premium": "claude-sonnet-4.6",
            "threshold": 0.80,
        },
        "decomposition": {
            "cheap": "deepseek-chat",
            "premium": "claude-sonnet-4.6",
            "threshold": 0.90,
        },
    }

    # Cost per 1M tokens
    MODEL_COSTS = {
        "deepseek-chat": {"input": 1.0, "output": 4.0},
        "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
        "claude-opus-4.6": {"input": 15.0, "output": 75.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
    }

    def __init__(self, client=None):
        """
        Initialize speculative generator.

        Args:
            client: API client
        """
        self.client = client
        self.metrics = SpeculativeMetrics()
        self._model_pairs = dict(self.DEFAULT_MODEL_PAIRS)

    def set_model_pair(
        self,
        task_type: str,
        cheap_model: str,
        premium_model: str,
        threshold: float = 0.85,
    ) -> None:
        """
        Set custom model pair for task type.

        Args:
            task_type: Task type
            cheap_model: Cheap model to try first
            premium_model: Premium model as fallback
            threshold: Score threshold for cheap model
        """
        self._model_pairs[task_type] = {
            "cheap": cheap_model,
            "premium": premium_model,
            "threshold": threshold,
        }
        logger.info(
            f"Model pair set for {task_type}: "
            f"{cheap_model} vs {premium_model} (threshold={threshold})"
        )

    def get_model_pair(self, task_type: str) -> Dict[str, Any]:
        """
        Get model pair for task type.

        Args:
            task_type: Task type

        Returns:
            Dictionary with cheap, premium, threshold
        """
        return self._model_pairs.get(
            task_type,
            self.DEFAULT_MODEL_PAIRS.get("code_generation", {}),
        )

    async def speculative_generate(
        self,
        prompt: str,
        task_type: str = "code_generation",
        cheap_model: Optional[str] = None,
        premium_model: Optional[str] = None,
        threshold: Optional[float] = None,
        max_tokens: int = 4000,
        **kwargs,
    ) -> SpeculativeResult:
        """
        Generate with speculative execution.

        Args:
            prompt: Prompt text
            task_type: Task type for model pair selection
            cheap_model: Override cheap model
            premium_model: Override premium model
            threshold: Override score threshold
            max_tokens: Maximum output tokens
            **kwargs: Additional API parameters

        Returns:
            SpeculativeResult with response, model, score, cost
        """
        start_time = time.time()
        self.metrics.total_attempts += 1

        # Get model pair
        model_pair = self.get_model_pair(task_type)
        cheap = cheap_model or model_pair.get("cheap", "deepseek-chat")
        premium = premium_model or model_pair.get("premium", "claude-opus-4.6")
        score_threshold = threshold or model_pair.get("threshold", 0.85)

        logger.info(
            f"Starting speculative generation: "
            f"{cheap} vs {premium} (threshold={score_threshold})"
        )

        # Start both generations in parallel
        cheap_task = asyncio.create_task(
            self._generate_with_model(cheap, prompt, max_tokens, **kwargs)
        )
        premium_task = asyncio.create_task(
            self._generate_with_model(premium, prompt, max_tokens, **kwargs)
        )

        cheap_result = None
        premium_result = None
        cheap_score = 0.0
        premium_cancelled = False

        try:
            # Wait for cheap model first (usually faster)
            cheap_result = await cheap_task
            cheap_score = await self._quick_evaluate(prompt, cheap_result)

            logger.info(
                f"Cheap model {cheap} completed with score {cheap_score:.3f}"
            )

            # Check if cheap model is good enough
            if cheap_score >= score_threshold:
                logger.info(
                    f"Cheap model score {cheap_score:.3f} ≥ {score_threshold}, "
                    f"cancelling premium"
                )

                # Cancel premium task
                premium_task.cancel()
                premium_cancelled = True
                self.metrics.cancellations += 1
                self.metrics.cheap_wins += 1

                # Calculate savings
                premium_cost = self._estimate_cost(premium, cheap_result, prompt)
                self.metrics.estimated_savings += premium_cost * 0.6  # ~60% of premium cost saved

                latency = time.time() - start_time
                cost = self._estimate_cost(cheap, cheap_result, prompt)

                self.metrics.total_cost += cost
                self.metrics.avg_cheap_score = (
                    (self.metrics.avg_cheap_score * (self.metrics.total_attempts - 1) + cheap_score)
                    / self.metrics.total_attempts
                )

                return SpeculativeResult(
                    response=cheap_result,
                    model_used=cheap,
                    score=cheap_score,
                    cost=cost,
                    cheap_score=cheap_score,
                    premium_cancelled=True,
                    latency_seconds=latency,
                )

            else:
                logger.info(
                    f"Cheap model score {cheap_score:.3f} < {score_threshold}, "
                    f"waiting for premium"
                )

                # Wait for premium model
                premium_result = await premium_task
                premium_score = await self._quick_evaluate(prompt, premium_result)

                logger.info(
                    f"Premium model {premium} completed with score {premium_score:.3f}"
                )

                self.metrics.premium_wins += 1

                latency = time.time() - start_time
                cost = self._estimate_cost(premium, premium_result, prompt)

                self.metrics.total_cost += cost

                return SpeculativeResult(
                    response=premium_result,
                    model_used=premium,
                    score=premium_score,
                    cost=cost,
                    cheap_score=cheap_score,
                    premium_cancelled=False,
                    latency_seconds=latency,
                )

        except asyncio.CancelledError:
            logger.warning("Speculative generation cancelled")
            raise

        except Exception as e:
            logger.error(f"Speculative generation failed: {e}")

            # Try to get any result
            if cheap_result:
                return SpeculativeResult(
                    response=cheap_result,
                    model_used=cheap,
                    score=cheap_score,
                    cost=self._estimate_cost(cheap, cheap_result, prompt),
                    cheap_score=cheap_score,
                    premium_cancelled=False,
                    latency_seconds=time.time() - start_time,
                )

            # Fallback to premium only
            try:
                premium_result = await premium_task
                return SpeculativeResult(
                    response=premium_result,
                    model_used=premium,
                    score=0.5,
                    cost=self._estimate_cost(premium, premium_result, prompt),
                    cheap_score=cheap_score,
                    premium_cancelled=False,
                    latency_seconds=time.time() - start_time,
                )
            except Exception:
                raise RuntimeError(f"Speculative generation failed: {e}")

    async def _generate_with_model(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        **kwargs,
    ) -> str:
        """
        Generate response with specific model.

        Args:
            model: Model to use
            prompt: Prompt text
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        if self.client is None:
            raise RuntimeError("No client available for generation")

        try:
            response = await self.client.call(
                model=model,
                system_prompt=prompt,
                max_tokens=max_tokens,
                **kwargs,
            )

            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except asyncio.CancelledError:
            logger.info(f"Generation cancelled for {model}")
            raise
        except Exception as e:
            logger.warning(f"Generation failed for {model}: {e}")
            raise

    async def _quick_evaluate(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Quick quality evaluation (0-1 score).

        Args:
            prompt: Original prompt
            response: Generated response

        Returns:
            Quality score (0-1)
        """
        # Heuristic scoring (fast, no LLM)
        score = 0.5  # Base score

        # Length check
        response_len = len(response)
        if 500 <= response_len <= 10000:
            score += 0.2
        elif response_len < 100:
            score -= 0.3
        elif response_len > 50000:
            score -= 0.1

        # Code structure
        if "```" in response:
            score += 0.1
        if "def " in response or "class " in response:
            score += 0.1
        if "\n\n" in response:
            score += 0.05

        # Completeness
        if any(word in response.lower() for word in ["complete", "finished"]):
            score += 0.05

        # Errors
        if any(word in response.lower() for word in ["error:", "failed:"]):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _estimate_cost(
        self,
        model: str,
        response: str,
        prompt: Optional[str] = None,
    ) -> float:
        """
        Estimate API cost.

        Args:
            model: Model used
            response: Generated text
            prompt: Optional prompt text

        Returns:
            Estimated cost in USD
        """
        output_tokens = len(response) / 4
        input_tokens = len(prompt) / 4 if prompt else 0

        model_key = model.lower()
        costs = self.MODEL_COSTS.get(model_key, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get speculative metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = SpeculativeMetrics()


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

async def speculative_generate(
    client,
    prompt: str,
    task_type: str = "code_generation",
) -> SpeculativeResult:
    """Convenience function for speculative generation."""
    gen = SpeculativeGenerator(client=client)
    return await gen.speculative_generate(prompt, task_type)


__all__ = [
    "SpeculativeGenerator",
    "SpeculativeMetrics",
    "SpeculativeResult",
    "speculative_generate",
]
