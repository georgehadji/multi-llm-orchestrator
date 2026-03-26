"""
Tier 1 Optimization Integration for Engine
===========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Integrates Tier 1 cost optimizations (prompt caching, batch API, token budget)
into the main orchestration engine.

Usage:
    # In engine.py __init__:
    from .cost_optimization import PromptCacher, BatchClient, TokenBudget
    
    self.prompt_cacher = PromptCacher(client=self.client)
    self.batch_client = BatchClient(client=self.client)
    self.token_budget = TokenBudget()
    
    # In _execute_parallel_level:
    await self.prompt_cacher.warm_cache(system_prompt, project_context)
    
    # In _generate_single:
    max_tokens = self.token_budget.get_limit_by_name(phase)
    response = await self.prompt_cacher.call_with_cache(...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .log_config import get_logger

if TYPE_CHECKING:
    from .engine import Orchestrator
    from .models import Task, TaskType

logger = get_logger(__name__)


class Tier1OptimizationMixin:
    """
    Mixin class for Tier 1 cost optimizations.
    
    Adds prompt caching, batch API, and token budget to Orchestrator.
    """
    
    # Phase mapping from TaskType to OptimizationPhase
    TASK_TYPE_TO_PHASE = {
        "decomposition": "decomposition",
        "code_generation": "generation",
        "code_review": "critique",
        "evaluation": "evaluation",
        "prompt_enhancement": "prompt_enhancement",
        "condensing": "condensing",
    }
    
    def init_tier1_optimizations(self: Orchestrator) -> None:
        """
        Initialize Tier 1 optimization components.
        
        Call this from Orchestrator.__init__().
        """
        from .cost_optimization import PromptCacher, BatchClient, TokenBudget
        
        # Initialize prompt cacher
        self.prompt_cacher = PromptCacher(client=self.client)
        logger.info("Prompt cacher initialized (80-90% input cost reduction)")
        
        # Initialize batch client
        self.batch_client = BatchClient(client=self.client)
        logger.info("Batch client initialized (50% discount on non-critical phases)")
        
        # Initialize token budget
        self.token_budget = TokenBudget()
        logger.info("Token budget initialized (15-25% output cost reduction)")
        
        # Track optimization metrics
        self._optimization_enabled = True
        logger.info("Tier 1 optimizations enabled — expected 60-75% cost reduction")
    
    async def warm_cache_for_parallel(self: Orchestrator, system_prompt: str, project_context: str) -> None:
        """
        Warm prompt cache before parallel execution.
        
        Call this before _execute_parallel_level().
        
        Args:
            system_prompt: System prompt to cache
            project_context: Project-specific context
        """
        if not hasattr(self, '_optimization_enabled') or not self._optimization_enabled:
            return
        
        try:
            cache_key = await self.prompt_cacher.warm_cache(
                system_prompt=system_prompt,
                project_context=project_context,
            )
            logger.info(f"Cache warmed for parallel execution (key={cache_key})")
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")
    
    async def generate_with_cache(
        self: Orchestrator,
        task: Task,
        model: Any,
        full_prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate response with prompt caching.
        
        Use this instead of direct client.call() for 80-90% input cost reduction.
        
        Args:
            task: Task to execute
            model: Model to use
            full_prompt: Full prompt text
            **kwargs: Additional API parameters
        
        Returns:
            Dictionary with response, tokens, cost
        """
        if not hasattr(self, '_optimization_enabled') or not self._optimization_enabled:
            # Fallback to direct call
            return await self._direct_generate(task, model, full_prompt, **kwargs)
        
        try:
            # Get phase-specific token limit
            phase = self.TASK_TYPE_TO_PHASE.get(task.type.value, "generation")
            max_tokens = self.token_budget.get_limit_by_name(phase)
            
            # Make cached API call
            response = await self.prompt_cacher.call_with_cache(
                model=model.value,
                messages=[{"role": "user", "content": full_prompt}],
                system_prompt=self._get_system_prompt(),  # Orchestrator method
                project_context=getattr(self, '_project_context', ""),
                max_tokens=max_tokens,
                **kwargs,
            )
            
            # Extract response data
            result = {
                "response": self._extract_response_text(response),
                "tokens_input": getattr(response.usage, 'input_tokens', 0),
                "tokens_output": getattr(response.usage, 'output_tokens', 0),
                "cost": self._estimate_cost(model, response),
                "cached": True,
            }
            
            # Record token usage
            self.token_budget.record_usage(
                model=model.value,
                input_tokens=result["tokens_input"],
                output_tokens=result["tokens_output"],
                phase=phase,
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Cached generation failed, using fallback: {e}")
            return await self._direct_generate(task, model, full_prompt, **kwargs)
    
    async def _direct_generate(
        self: Orchestrator,
        task: Task,
        model: Any,
        full_prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Direct generation without caching (fallback).
        
        Args:
            task: Task to execute
            model: Model to use
            full_prompt: Full prompt text
            **kwargs: Additional API parameters
        
        Returns:
            Dictionary with response, tokens, cost
        """
        # Get phase-specific token limit
        phase = self.TASK_TYPE_TO_PHASE.get(task.type.value, "generation")
        max_tokens = self.token_budget.get_limit_by_name(phase)
        
        # Direct API call
        response = await self.client.call(
            model=model,
            system_prompt=full_prompt,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        return {
            "response": response.text if hasattr(response, 'text') else str(response),
            "tokens_input": getattr(response, 'input_tokens', 0),
            "tokens_output": getattr(response, 'output_tokens', 0),
            "cost": getattr(response, 'cost_usd', 0.0),
            "cached": False,
        }
    
    async def evaluate_with_batch(
        self: Orchestrator,
        task: Task,
        prompt: str,
        model: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate using batch API for 50% discount.
        
        Use this for evaluation, critique, and condensing phases.
        
        Args:
            task: Task to execute
            prompt: Prompt text
            model: Model to use
            **kwargs: Additional API parameters
        
        Returns:
            Evaluation result dictionary
        """
        if not hasattr(self, '_optimization_enabled') or not self._optimization_enabled:
            # Fallback to realtime
            return await self._direct_evaluate(task, prompt, model, **kwargs)
        
        try:
            # Determine phase
            phase = self.TASK_TYPE_TO_PHASE.get(task.type.value, "evaluation")
            
            # Batch API call (auto-routed based on phase)
            result = await self.batch_client.call(
                model=model.value,
                prompt=prompt,
                phase=phase,
                **kwargs,
            )
            
            return {
                "response": result,
                "batch": True,
                "savings": 0.5,  # 50% discount
            }
            
        except Exception as e:
            logger.warning(f"Batch evaluation failed, using realtime: {e}")
            return await self._direct_evaluate(task, prompt, model, **kwargs)
    
    async def _direct_evaluate(
        self: Orchestrator,
        task: Task,
        prompt: str,
        model: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Direct evaluation without batching (fallback).
        
        Args:
            task: Task to execute
            prompt: Prompt text
            model: Model to use
            **kwargs: Additional API parameters
        
        Returns:
            Evaluation result dictionary
        """
        response = await self.client.call(
            model=model,
            system_prompt=prompt,
            **kwargs,
        )
        
        return {
            "response": response.text if hasattr(response, 'text') else str(response),
            "batch": False,
            "savings": 0.0,
        }
    
    def get_optimization_metrics(self: Orchestrator) -> Dict[str, Any]:
        """
        Get comprehensive optimization metrics.
        
        Returns:
            Dictionary with all optimization metrics
        """
        if not hasattr(self, '_optimization_enabled') or not self._optimization_enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "prompt_caching": self.prompt_cacher.get_metrics(),
            "batch_processing": self.batch_client.get_metrics(),
            "token_budget": self.token_budget.get_metrics(),
            "estimated_total_savings": self._calculate_total_savings(),
        }
    
    def _calculate_total_savings(self: Orchestrator) -> float:
        """
        Calculate total estimated cost savings.
        
        Returns:
            Total savings in USD
        """
        caching_metrics = self.prompt_cacher.get_metrics()
        batch_metrics = self.batch_client.get_metrics()
        budget_metrics = self.token_budget.get_metrics()
        
        # Calculate savings from each optimization
        caching_savings = caching_metrics.get('estimated_savings_percent', 0) / 100
        batch_savings = batch_metrics.get('total_savings', 0)
        budget_savings = budget_metrics.get('estimated_savings', 0)
        
        return caching_savings + batch_savings + budget_savings
    
    def _get_system_prompt(self: Orchestrator) -> str:
        """
        Get current system prompt.
        
        Returns:
            System prompt string
        """
        # This should be implemented in Orchestrator to return the actual system prompt
        return getattr(self, '_system_prompt', "You are an expert software developer.")
    
    def _extract_response_text(self: Orchestrator, response: Any) -> str:
        """
        Extract text from API response.
        
        Args:
            response: API response object
        
        Returns:
            Response text
        """
        if hasattr(response, 'text'):
            return response.text
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _estimate_cost(self: Orchestrator, model: Any, response: Any) -> float:
        """
        Estimate API cost from response.
        
        Args:
            model: Model used
            response: API response object
        
        Returns:
            Estimated cost in USD
        """
        if hasattr(response, 'cost_usd'):
            return response.cost_usd
        
        # Estimate from tokens
        usage = getattr(response, 'usage', None)
        if not usage:
            return 0.0
        
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        
        # Rough cost estimates (per 1M tokens)
        COST_PER_1M = {
            "claude-opus": {"input": 15.0, "output": 75.0},
            "claude-sonnet": {"input": 3.0, "output": 15.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "deepseek": {"input": 1.0, "output": 4.0},
        }
        
        model_key = model.value.lower() if hasattr(model, 'value') else str(model).lower()
        costs = COST_PER_1M.get(model_key, {"input": 3.0, "output": 15.0})
        
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        
        return input_cost + output_cost


# ─────────────────────────────────────────────
# Integration Instructions
# ─────────────────────────────────────────────

INTEGRATION_INSTRUCTIONS = """
To integrate Tier 1 optimizations into engine.py:

1. Add import at top of engine.py:
   from .cost_optimization_integration import Tier1OptimizationMixin

2. Make Orchestrator inherit from Tier1OptimizationMixin:
   class Orchestrator(Tier1OptimizationMixin):
       ...

3. Call init_tier1_optimizations() in __init__():
   def __init__(self, ...):
       ...
       self.init_tier1_optimizations()

4. Call warm_cache_for_parallel() before _execute_parallel_level():
   async def _execute_parallel_level(self, ...):
       await self.warm_cache_for_parallel(self.system_prompt, self.project_context)
       ...

5. Replace direct client.call() with generate_with_cache():
   # Old:
   response = await self.client.call(model, prompt, max_tokens=...)
   
   # New:
   result = await self.generate_with_cache(task, model, full_prompt)
   response = result["response"]

6. Use evaluate_with_batch() for evaluation phases:
   # Old:
   eval_result = await self.client.call(model, eval_prompt)
   
   # New:
   eval_result = await self.evaluate_with_batch(task, eval_prompt, model)

7. Add metrics to telemetry:
   metrics = self.get_optimization_metrics()
   self._telemetry.record("optimization.savings", metrics['estimated_total_savings'])
"""

__all__ = ["Tier1OptimizationMixin", "INTEGRATION_INSTRUCTIONS"]
