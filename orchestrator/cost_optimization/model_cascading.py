"""
Model Cascading Module
=======================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements model cascading for 40-60% per-task cost reduction.

Strategy: Try cheap model first, escalate to premium only if quality insufficient.

Features:
- Configurable cascade chains per task type
- Quick quality evaluation
- Automatic cascade exit on success
- Cost tracking and savings metrics

Usage:
    from orchestrator.cost_optimization import ModelCascader

    cascader = ModelCascader(client=api_client)
    
    # Define cascade chain
    cascade = [
        ("deepseek-v3.2", 0.80),      # Try cheapest, accept if score ≥ 0.80
        ("claude-sonnet-4.6", 0.75),   # Mid-tier, accept if score ≥ 0.75
        ("claude-opus-4.6", 0.0),      # Premium, always accept
    ]
    
    result = await cascader.cascading_generate(task, cascade)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..log_config import get_logger

logger = get_logger(__name__)


@dataclass
class CascadeMetrics:
    """Metrics for model cascading."""
    total_attempts: int = 0
    cascade_exits_early: int = 0  # Exited at cheap/mid tier
    cascade_exits_premium: int = 0  # Required premium model
    avg_score: float = 0.0
    total_cost: float = 0.0
    estimated_savings: float = 0.0

    @property
    def early_exit_rate(self) -> float:
        """Calculate early exit rate (exits at cheap/mid tier)."""
        if self.total_attempts == 0:
            return 0.0
        return self.cascade_exits_early / self.total_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "total_attempts": self.total_attempts,
            "cascade_exits_early": self.cascade_exits_early,
            "cascade_exits_premium": self.cascade_exits_premium,
            "early_exit_rate": self.early_exit_rate,
            "avg_score": self.avg_score,
            "total_cost": self.total_cost,
            "estimated_savings": self.estimated_savings,
        }


@dataclass
class CascadeResult:
    """Result of cascading generation."""
    response: str
    model_used: str
    score: float
    cost: float
    attempts: int
    cascade_exit_tier: int
    all_scores: List[float] = field(default_factory=list)


class ModelCascader:
    """
    Model cascading for cost optimization.

    Usage:
        cascader = ModelCascader(client=api_client)
        result = await cascader.cascading_generate(task, cascade_chain)
    """

    # Default cascade chains per task type
    DEFAULT_CASCADE_CHAINS = {
        "code_generation": [
            ("deepseek-chat", 0.80),      # Try cheapest first
            ("claude-sonnet-4.6", 0.75),   # Mid-tier
            ("claude-opus-4.6", 0.0),      # Premium (always accept)
        ],
        "code_review": [
            ("deepseek-chat", 0.75),
            ("claude-sonnet-4.6", 0.70),
            ("claude-opus-4.6", 0.0),
        ],
        "decomposition": [
            ("deepseek-chat", 0.85),
            ("claude-sonnet-4.6", 0.80),
            ("claude-opus-4.6", 0.0),
        ],
        "evaluation": [
            ("deepseek-chat", 0.70),
            ("claude-sonnet-4.6", 0.0),
        ],
    }

    # Cost per 1M tokens (for savings estimation)
    MODEL_COSTS = {
        "deepseek-chat": {"input": 1.0, "output": 4.0},
        "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
        "claude-opus-4.6": {"input": 15.0, "output": 75.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
    }

    def __init__(self, client=None, evaluator_client=None):
        """
        Initialize model cascader.

        Args:
            client: API client for generation
            evaluator_client: Optional separate client for quick evaluation
        """
        self.client = client
        self.evaluator_client = evaluator_client or client
        self.metrics = CascadeMetrics()
        self._cascade_chains = dict(self.DEFAULT_CASCADE_CHAINS)
        self._score_cache: Dict[str, float] = {}

    def set_cascade_chain(
        self,
        task_type: str,
        chain: List[Tuple[str, float]],
    ) -> None:
        """
        Set custom cascade chain for task type.

        Args:
            task_type: Task type (e.g., "code_generation")
            chain: List of (model, min_score) tuples
        """
        self._cascade_chains[task_type] = chain
        logger.info(f"Cascade chain set for {task_type}: {chain}")

    def get_cascade_chain(self, task_type: str) -> List[Tuple[str, float]]:
        """
        Get cascade chain for task type.

        Args:
            task_type: Task type

        Returns:
            List of (model, min_score) tuples
        """
        return self._cascade_chains.get(
            task_type,
            self.DEFAULT_CASCADE_CHAINS.get("code_generation", []),
        )

    async def cascading_generate(
        self,
        task_prompt: str,
        task_type: str = "code_generation",
        max_tokens: int = 4000,
        **kwargs,
    ) -> CascadeResult:
        """
        Generate with model cascading.

        Args:
            task_prompt: Prompt text
            task_type: Task type for cascade chain selection
            max_tokens: Maximum output tokens
            **kwargs: Additional API parameters

        Returns:
            CascadeResult with response, model, score, cost
        """
        cascade_chain = self.get_cascade_chain(task_type)

        if not cascade_chain:
            logger.warning("No cascade chain found, using default")
            cascade_chain = self.DEFAULT_CASCADE_CHAINS["code_generation"]

        logger.info(
            f"Starting cascading generation for {task_type} "
            f"({len(cascade_chain)} models in chain)"
        )

        all_scores: List[float] = []
        total_cost: float = 0.0
        attempts: int = 0

        for tier_idx, (model, min_score) in enumerate(cascade_chain):
            attempts += 1
            self.metrics.total_attempts += 1

            logger.info(
                f"Cascade tier {tier_idx + 1}/{len(cascade_chain)}: "
                f"trying {model} (min_score={min_score})"
            )

            try:
                # Generate with current model
                start_time = time.time()
                response = await self._generate_with_model(
                    model=model,
                    prompt=task_prompt,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                gen_time = time.time() - start_time

                # Calculate cost
                cost = self._estimate_cost(model, response)
                total_cost += cost

                # Quick quality evaluation
                score = await self._quick_evaluate(
                    prompt=task_prompt,
                    response=response,
                    model=model,
                )
                all_scores.append(score)

                logger.info(
                    f"Model {model} scored {score:.3f} "
                    f"(min_required={min_score}, cost=${cost:.4f}, "
                    f"time={gen_time:.2f}s)"
                )

                # Check if score meets threshold
                if score >= min_score:
                    logger.info(
                        f"Cascade exit at tier {tier_idx + 1}: "
                        f"{model} scored {score:.3f} ≥ {min_score}"
                    )

                    self.metrics.cascade_exits_early += 1
                    self.metrics.avg_score = (
                        (self.metrics.avg_score * (self.metrics.total_attempts - 1) + score)
                        / self.metrics.total_attempts
                    )
                    self.metrics.total_cost += total_cost

                    # Estimate savings vs always using premium
                    premium_model = cascade_chain[-1][0]
                    premium_cost = self._estimate_premium_cost(
                        premium_model, task_prompt, max_tokens
                    )
                    self.metrics.estimated_savings += premium_cost - total_cost

                    return CascadeResult(
                        response=response,
                        model_used=model,
                        score=score,
                        cost=total_cost,
                        attempts=attempts,
                        cascade_exit_tier=tier_idx,
                        all_scores=all_scores,
                    )

                else:
                    logger.info(
                        f"Score {score:.3f} < {min_score}, "
                        f"continuing to next tier"
                    )

            except Exception as e:
                logger.warning(f"Tier {tier_idx + 1} ({model}) failed: {e}")
                # Continue to next tier
                continue

        # If we get here, use last result even if score is low
        logger.warning(
            f"Cascade completed without meeting threshold, "
            f"using last result (score={all_scores[-1] if all_scores else 0:.3f})"
        )

        self.metrics.cascade_exits_premium += 1
        self.metrics.avg_score = (
            (self.metrics.avg_score * (self.metrics.total_attempts - 1) + all_scores[-1])
            / self.metrics.total_attempts
        )
        self.metrics.total_cost += total_cost

        return CascadeResult(
            response=response,
            model_used=cascade_chain[-1][0],
            score=all_scores[-1] if all_scores else 0.0,
            cost=total_cost,
            attempts=attempts,
            cascade_exit_tier=len(cascade_chain) - 1,
            all_scores=all_scores,
        )

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

        # Call API
        response = await self.client.call(
            model=model,
            system_prompt=prompt,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract text from response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    async def _quick_evaluate(
        self,
        prompt: str,
        response: str,
        model: str,
    ) -> float:
        """
        Quick quality evaluation (0-1 score).

        Uses cheap heuristic evaluation or cached scores.

        Args:
            prompt: Original prompt
            response: Generated response
            model: Model that generated response

        Returns:
            Quality score (0-1)
        """
        # Check cache first
        cache_key = f"{prompt[:100]}|||{response[:100]}|||{model}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        try:
            # Quick heuristic evaluation
            score = await self._heuristic_score(prompt, response)

            # Cache score
            self._score_cache[cache_key] = score

            # Limit cache size
            if len(self._score_cache) > 1000:
                # Remove oldest 50%
                keys_to_remove = list(self._score_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._score_cache[key]

            return score

        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}")
            return 0.5  # Default score

    async def _heuristic_score(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Heuristic quality scoring.

        Fast, rule-based scoring without LLM evaluation.

        Args:
            prompt: Original prompt
            response: Generated response

        Returns:
            Quality score (0-1)
        """
        score = 0.5  # Base score

        # Length check (not too short, not too long)
        response_len = len(response)
        if 500 <= response_len <= 10000:
            score += 0.2
        elif response_len < 100:
            score -= 0.3  # Too short
        elif response_len > 50000:
            score -= 0.1  # Too long

        # Check for code blocks (for code tasks)
        if "```" in response:
            score += 0.1

        # Check for structure (newlines, paragraphs)
        if "\n\n" in response:
            score += 0.05

        # Check for completeness markers
        completeness_markers = ["complete", "finished", "implemented", "done"]
        if any(marker in response.lower() for marker in completeness_markers):
            score += 0.05

        # Penalize obvious errors
        error_markers = ["error:", "failed:", "cannot", "unable to"]
        if any(marker in response.lower() for marker in error_markers):
            score -= 0.1

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    def _estimate_cost(
        self,
        model: str,
        response: str,
        prompt: Optional[str] = None,
    ) -> float:
        """
        Estimate API cost for generation.

        Args:
            model: Model used
            response: Generated text
            prompt: Optional prompt text

        Returns:
            Estimated cost in USD
        """
        # Estimate tokens (4 chars ≈ 1 token)
        output_tokens = len(response) / 4
        input_tokens = len(prompt) / 4 if prompt else 0

        # Get model costs
        model_key = model.lower()
        costs = self.MODEL_COSTS.get(model_key, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost

    def _estimate_premium_cost(
        self,
        premium_model: str,
        prompt: str,
        max_tokens: int,
    ) -> float:
        """
        Estimate cost if premium model was used.

        Args:
            premium_model: Premium model name
            prompt: Prompt text
            max_tokens: Maximum output tokens

        Returns:
            Estimated premium cost in USD
        """
        input_tokens = len(prompt) / 4
        output_tokens = max_tokens

        model_key = premium_model.lower()
        costs = self.MODEL_COSTS.get(model_key, {"input": 15.0, "output": 75.0})

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cascading metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = CascadeMetrics()
        logger.info("Cascade metrics reset")


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

async def cascading_generate(
    client,
    prompt: str,
    task_type: str = "code_generation",
) -> CascadeResult:
    """
    Convenience function for cascading generation.

    Args:
        client: API client
        prompt: Prompt text
        task_type: Task type

    Returns:
        CascadeResult
    """
    cascader = ModelCascader(client=client)
    return await cascader.cascading_generate(prompt, task_type)


__all__ = [
    "ModelCascader",
    "CascadeMetrics",
    "CascadeResult",
    "cascading_generate",
]
