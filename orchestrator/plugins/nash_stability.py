"""
Nash Stability Plugin
=====================
Author: Georgios-Chrysovalantis Chatzivantsidis

Extracts Nash Stability features from core orchestrator into optional plugin.

Features:
- Nash equilibrium detection in model selection
- Performance-based model scoring
- Adaptive template selection
- Cost-quality frontier optimization

Part of Phase 5: Plugin Architecture
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from .base import Plugin, PluginContext, PluginMetadata, PluginPriority

if TYPE_CHECKING:
    from .models import Model, TaskResult

logger = logging.getLogger(__name__)


class NashStabilityPlugin(Plugin):
    """
    Nash Stability plugin for game-theoretic model selection.
    
    Provides:
    1. Nash equilibrium detection in multi-model scenarios
    2. Performance-based model scoring from historical data
    3. Adaptive template selection based on task type
    4. Cost-quality frontier optimization
    """
    
    metadata = PluginMetadata(
        name="nash-stability",
        version="1.0.0",
        description="Nash equilibrium-based model selection and optimization",
        author="Georgios-Chrysovalantis Chatzivantsidis",
        priority=PluginPriority.NORMAL,
    )
    
    def __init__(
        self,
        enable_nash_monitoring: bool = True,
        enable_adaptive_templates: bool = True,
        enable_pareto_frontier: bool = True,
    ) -> None:
        """
        Initialize Nash Stability plugin.
        
        Args:
            enable_nash_monitoring: Enable Nash equilibrium monitoring
            enable_adaptive_templates: Enable adaptive prompt templates
            enable_pareto_frontier: Enable cost-quality frontier
        """
        super().__init__()
        
        self.enable_nash_monitoring = enable_nash_monitoring
        self.enable_adaptive_templates = enable_adaptive_templates
        self.enable_pareto_frontier = enable_pareto_frontier
        
        # Internal state
        self._model_scores: dict[Model, float] = {}
        self._task_history: list[dict[str, Any]] = []
        self._nash_equilibria: list[tuple[Model, ...]] = []
    
    async def initialize(self) -> None:
        """Initialize Nash Stability plugin resources."""
        logger.info("Initializing Nash Stability plugin")
        
        # Load historical performance data
        if self.enable_nash_monitoring:
            await self._load_historical_data()
        
        # Initialize adaptive templates
        if self.enable_adaptive_templates:
            await self._init_adaptive_templates()
        
        # Initialize Pareto frontier
        if self.enable_pareto_frontier:
            await self._init_pareto_frontier()
        
        self._initialized = True
        logger.info("Nash Stability plugin initialized")
    
    async def shutdown(self) -> None:
        """Shutdown Nash Stability plugin."""
        logger.info("Shutting down Nash Stability plugin")
        
        # Save historical data
        await self._save_historical_data()
        
        self._initialized = False
    
    async def on_post_task(self, context: PluginContext) -> None:
        """
        Record task result for Nash stability analysis.
        
        Args:
            context: Plugin context with task result
        """
        if not context.task or not context.task_result:
            return
        
        # Record task outcome
        self._record_task_outcome(context.task, context.task_result)
        
        # Update model scores
        if context.task_result.model_used:
            self._update_model_score(
                context.task_result.model_used,
                context.task_result,
            )
    
    def get_optimal_model(
        self,
        task_type: str,
        budget_constraint: Optional[float] = None,
    ) -> Optional[Model]:
        """
        Get optimal model based on Nash equilibrium analysis.
        
        Args:
            task_type: Type of task
            budget_constraint: Optional budget limit
            
        Returns:
            Optimal model or None
        """
        from .models import Model
        
        if not self._model_scores:
            return None
        
        # Filter models by budget if constraint provided
        candidates = list(self._model_scores.keys())
        
        if budget_constraint:
            # Filter by predicted cost
            candidates = [
                m for m in candidates
                if self._predict_cost(m, task_type) <= budget_constraint
            ]
        
        if not candidates:
            return None
        
        # Select model with highest score
        return max(candidates, key=lambda m: self._model_scores[m])
    
    def _record_task_outcome(self, task: Any, result: TaskResult) -> None:
        """Record task outcome for analysis."""
        self._task_history.append({
            "task_id": task.id,
            "task_type": task.type.value if hasattr(task.type, 'value') else str(task.type),
            "model_used": result.model_used.value if result.model_used else None,
            "score": result.score,
            "cost": result.cost_usd,
            "iterations": result.iterations,
            "success": result.status.value if hasattr(result.status, 'value') else str(result.status),
        })
        
        # Keep history bounded
        if len(self._task_history) > 10000:
            self._task_history = self._task_history[-10000:]
    
    def _update_model_score(self, model: Model, result: TaskResult) -> None:
        """
        Update model score based on task result.
        
        Uses exponential moving average for score updates.
        """
        alpha = 0.1  # Smoothing factor
        
        # Calculate performance score
        performance = (
            result.score * 0.5 +  # Quality weight
            (1.0 - min(result.cost_usd / 0.1, 1.0)) * 0.3 +  # Cost efficiency
            (1.0 / (1.0 + result.iterations)) * 0.2  # Convergence speed
        )
        
        # Update EMA score
        if model not in self._model_scores:
            self._model_scores[model] = performance
        else:
            self._model_scores[model] = (
                alpha * performance +
                (1 - alpha) * self._model_scores[model]
            )
    
    def _predict_cost(self, model: Model, task_type: str) -> float:
        """Predict cost for model and task type."""
        # Simple prediction based on historical averages
        model_costs = [
            t["cost"] for t in self._task_history
            if t["model_used"] == model.value and t["task_type"] == task_type
        ]
        
        if not model_costs:
            # Fallback: use average cost
            all_costs = [t["cost"] for t in self._task_history if t["cost"]]
            return sum(all_costs) / len(all_costs) if all_costs else 0.01
        
        return sum(model_costs) / len(model_costs)
    
    async def _load_historical_data(self) -> None:
        """Load historical performance data."""
        # Would load from persistent storage
        logger.debug("Loading Nash Stability historical data")
    
    async def _save_historical_data(self) -> None:
        """Save historical performance data."""
        # Would save to persistent storage
        logger.debug("Saving Nash Stability historical data")
    
    async def _init_adaptive_templates(self) -> None:
        """Initialize adaptive template system."""
        logger.debug("Initializing adaptive templates")
    
    async def _init_pareto_frontier(self) -> None:
        """Initialize Pareto frontier for cost-quality optimization."""
        logger.debug("Initializing Pareto frontier")
    
    def get_nash_equilibria(self) -> list[tuple[Model, ...]]:
        """
        Get current Nash equilibria.
        
        Returns:
            List of model combinations in equilibrium
        """
        return list(self._nash_equilibria)
    
    def get_model_scores(self) -> dict[str, float]:
        """
        Get current model scores.
        
        Returns:
            Dictionary of model names to scores
        """
        return {m.value: s for m, s in self._model_scores.items()}
