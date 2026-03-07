"""
Nash Stability Auto-Tuning System
==================================

Self-optimization των hyperparameters με βάση το performance.
Προσαρμόζει dynamically exploration rate, EMA alpha, και άλλες παραμέτρους.

Features:
- Performance-based parameter adjustment
- Bayesian optimization (simplified)
- Multi-armed bandit for exploration/exploitation
- Drift detection triggers
- A/B testing framework for parameters

Usage:
    from orchestrator.nash_auto_tuning import AutoTuner
    
    tuner = AutoTuner()
    
    # Register tunable parameter
    tuner.register_parameter(
        name="exploration_rate",
        current_value=0.15,
        min_value=0.05,
        max_value=0.30,
        optimize_for="maximize",  # or "minimize"
    )
    
    # Auto-tune based on feedback
    await tuner.tune("exploration_rate", metric_value=0.85)
"""

from __future__ import annotations

import json
import random
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque, defaultdict

from .log_config import get_logger

logger = get_logger(__name__)


class OptimizationDirection(Enum):
    """Direction of optimization."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TuningStrategy(Enum):
    """Tuning strategy options."""
    EMA_TRACKING = "ema"           # Exponential moving average
    BAYESIAN = "bayesian"          # Simplified Bayesian
    BANDIT = "bandit"              # Multi-armed bandit
    GRID_SEARCH = "grid"           # Grid search
    ADAPTIVE = "adaptive"          # Adaptive based on variance


@dataclass
class ParameterConfig:
    """Configuration for a tunable parameter."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    direction: OptimizationDirection
    strategy: TuningStrategy
    
    # Tuning history
    history: List[Tuple[datetime, float, float]] = field(default_factory=list)
    # (timestamp, value, metric_result)
    
    # Configuration
    ema_alpha: float = 0.1
    min_samples_before_tune: int = 10
    max_change_per_step: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "direction": self.direction.value,
            "strategy": self.strategy.value,
            "history": [
                (t.isoformat(), v, m) for t, v, m in self.history
            ],
            "ema_alpha": self.ema_alpha,
        }


@dataclass
class TuningResult:
    """Result of a tuning operation."""
    parameter_name: str
    old_value: float
    new_value: float
    change: float
    reason: str
    confidence: float
    expected_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change": self.change,
            "reason": self.reason,
            "confidence": self.confidence,
            "expected_improvement": self.expected_improvement,
        }


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    metric_name: str
    window_size: int = 30
    threshold_std: float = 2.0  # Standard deviations
    min_samples: int = 20
    
    # State
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    recent_values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, value: float) -> Optional[Dict[str, Any]]:
        """Update drift detector and return alert if drift detected."""
        self.recent_values.append(value)
        
        # Not enough samples
        if len(self.recent_values) < self.min_samples:
            return None
        
        # Calculate baseline from first half
        if self.baseline_mean is None:
            half = len(self.recent_values) // 2
            baseline = list(self.recent_values)[:half]
            self.baseline_mean = statistics.mean(baseline)
            self.baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0
        
        # Check recent window
        if len(self.recent_values) >= self.window_size:
            recent = list(self.recent_values)[-self.window_size:]
            recent_mean = statistics.mean(recent)
            
            # Detect drift
            if self.baseline_std > 0:
                z_score = abs(recent_mean - self.baseline_mean) / self.baseline_std
                
                if z_score > self.threshold_std:
                    return {
                        "metric": self.metric_name,
                        "drift_detected": True,
                        "baseline_mean": self.baseline_mean,
                        "recent_mean": recent_mean,
                        "z_score": z_score,
                        "severity": "critical" if z_score > 3 else "warning",
                    }
        
        return None


class MultiArmedBandit:
    """
    Thompson Sampling for multi-armed bandit problem.
    Used for exploration/exploitation trade-off.
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Beta distribution parameters for each arm
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms
        self.pulls = [0] * n_arms
    
    def select_arm(self) -> int:
        """Select arm using Thompson Sampling."""
        samples = [
            random.betavariate(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]
        return max(range(self.n_arms), key=lambda i: samples[i])
    
    def update(self, arm: int, reward: float) -> None:
        """Update arm with observed reward."""
        self.pulls[arm] += 1
        # Reward should be in [0, 1]
        reward = max(0, min(1, reward))
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    
    def get_best_arm(self) -> int:
        """Get arm with highest expected reward."""
        expected = [
            self.alpha[i] / (self.alpha[i] + self.beta[i])
            for i in range(self.n_arms)
        ]
        return max(range(self.n_arms), key=lambda i: expected[i])


class AutoTuner:
    """
    Auto-tuning system for Nash stability hyperparameters.
    
    Monitors performance metrics and automatically adjusts:
    - Exploration rate (templates, frontier)
    - EMA alpha (convergence speed)
    - Privacy budget allocation
    - Cache TTL
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".nash_auto_tuning")
        self.storage_path.mkdir(exist_ok=True)
        
        # Registered parameters
        self._parameters: Dict[str, ParameterConfig] = {}
        
        # Drift detectors
        self._drift_detectors: Dict[str, DriftConfig] = {}
        
        # Bandits for A/B testing parameters
        self._bandits: Dict[str, MultiArmedBandit] = {}
        
        # Performance tracking
        self._metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Load saved state
        self._load_state()
        
        # Register default parameters
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default tunable parameters."""
        # Template exploration rate
        self.register_parameter(
            name="template_exploration_rate",
            current_value=0.15,
            min_value=0.05,
            max_value=0.30,
            direction=OptimizationDirection.MINIMIZE,  # Less exploration = more efficient
            strategy=TuningStrategy.ADAPTIVE,
        )
        
        # Knowledge graph similarity threshold
        self.register_parameter(
            name="kg_similarity_threshold",
            current_value=0.6,
            min_value=0.4,
            max_value=0.8,
            direction=OptimizationDirection.MAXIMIZE,  # Higher = better matches
            strategy=TuningStrategy.EMA_TRACKING,
        )
        
        # Pareto frontier confidence threshold
        self.register_parameter(
            name="frontier_min_confidence",
            current_value=0.3,
            min_value=0.1,
            max_value=0.5,
            direction=OptimizationDirection.MINIMIZE,  # Lower = more options
            strategy=TuningStrategy.BAYESIAN,
        )
        
        # EMA alpha for template scores
        self.register_parameter(
            name="template_ema_alpha",
            current_value=0.1,
            min_value=0.05,
            max_value=0.3,
            direction=OptimizationDirection.MAXIMIZE,  # Higher = faster adaptation
            strategy=TuningStrategy.ADAPTIVE,
        )
    
    def _load_state(self) -> None:
        """Load tuning state from disk."""
        state_file = self.storage_path / "tuning_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                
                for param_data in data.get("parameters", []):
                    param = ParameterConfig(
                        name=param_data["name"],
                        current_value=param_data["current_value"],
                        min_value=param_data["min_value"],
                        max_value=param_data["max_value"],
                        direction=OptimizationDirection(param_data["direction"]),
                        strategy=TuningStrategy(param_data["strategy"]),
                        ema_alpha=param_data.get("ema_alpha", 0.1),
                    )
                    self._parameters[param.name] = param
                
                logger.info(f"Loaded {len(self._parameters)} tuned parameters")
            except Exception as e:
                logger.error(f"Failed to load tuning state: {e}")
    
    def _save_state(self) -> None:
        """Save tuning state to disk."""
        try:
            data = {
                "parameters": [p.to_dict() for p in self._parameters.values()],
                "saved_at": datetime.utcnow().isoformat(),
            }
            state_file = self.storage_path / "tuning_state.json"
            state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save tuning state: {e}")
    
    def register_parameter(
        self,
        name: str,
        current_value: float,
        min_value: float,
        max_value: float,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        strategy: TuningStrategy = TuningStrategy.ADAPTIVE,
    ) -> None:
        """Register a parameter for auto-tuning."""
        self._parameters[name] = ParameterConfig(
            name=name,
            current_value=current_value,
            min_value=min_value,
            max_value=max_value,
            direction=direction,
            strategy=strategy,
        )
        logger.debug(f"Registered parameter: {name} = {current_value}")
    
    def get_parameter(self, name: str) -> Optional[float]:
        """Get current value of a parameter."""
        param = self._parameters.get(name)
        return param.current_value if param else None
    
    async def tune(
        self,
        parameter_name: str,
        metric_value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[TuningResult]:
        """
        Update parameter based on metric feedback.
        
        Args:
            parameter_name: Name of parameter to tune
            metric_value: Observed metric value
            context: Additional context for tuning
        
        Returns:
            Tuning result if parameter was changed
        """
        param = self._parameters.get(parameter_name)
        if not param:
            logger.warning(f"Unknown parameter: {parameter_name}")
            return None
        
        # Record history
        param.history.append((datetime.utcnow(), param.current_value, metric_value))
        
        # Check if we have enough samples
        if len(param.history) < param.min_samples_before_tune:
            return None
        
        # Apply tuning strategy
        old_value = param.current_value
        
        if param.strategy == TuningStrategy.EMA_TRACKING:
            new_value = self._tune_ema(param, metric_value)
        elif param.strategy == TuningStrategy.BAYESIAN:
            new_value = self._tune_bayesian(param, metric_value)
        elif param.strategy == TuningStrategy.BANDIT:
            new_value = self._tune_bandit(param_name, metric_value)
        elif param.strategy == TuningStrategy.ADAPTIVE:
            new_value = self._tune_adaptive(param, metric_value)
        else:
            new_value = old_value
        
        # Clamp to bounds
        new_value = max(param.min_value, min(param.max_value, new_value))
        
        # Check if change is significant
        change = abs(new_value - old_value)
        if change < 0.01:  # Minimum change threshold
            return None
        
        # Limit max change per step
        max_change = param.max_change_per_step * (param.max_value - param.min_value)
        if change > max_change:
            # Scale down the change
            direction = 1 if new_value > old_value else -1
            new_value = old_value + direction * max_change
        
        # Update parameter
        param.current_value = new_value
        
        # Save state
        self._save_state()
        
        # Create result
        result = TuningResult(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            change=new_value - old_value,
            reason=f"{param.strategy.value} tuning based on metric={metric_value:.3f}",
            confidence=min(1.0, len(param.history) / 50),
            expected_improvement=self._estimate_improvement(param, metric_value),
        )
        
        logger.info(
            f"Auto-tuned {parameter_name}: {old_value:.4f} → {new_value:.4f} "
            f"(change: {result.change:+.4f})"
        )
        
        return result
    
    def _tune_ema(self, param: ParameterConfig, metric_value: float) -> float:
        """Tune using EMA tracking."""
        # If metric is good, slowly move toward optimal
        recent_metrics = [m for _, _, m in param.history[-10:]]
        avg_metric = statistics.mean(recent_metrics)
        
        # Determine if we should increase or decrease
        is_good = avg_metric > 0.7  # Threshold for "good"
        
        if param.direction == OptimizationDirection.MAXIMIZE:
            if is_good:
                # Metric is good, can afford to be more aggressive
                return param.current_value * 1.05
            else:
                # Metric is poor, be more conservative
                return param.current_value * 0.95
        else:
            if is_good:
                return param.current_value * 0.95
            else:
                return param.current_value * 1.05
    
    def _tune_bayesian(self, param: ParameterConfig, metric_value: float) -> float:
        """Simplified Bayesian optimization."""
        # Try values around current, pick best based on history
        candidates = [
            param.current_value * 0.9,
            param.current_value,
            param.current_value * 1.1,
        ]
        
        best_value = param.current_value
        best_score = metric_value
        
        for candidate in candidates:
            candidate = max(param.min_value, min(param.max_value, candidate))
            
            # Find similar historical values
            similar = [
                (v, m) for _, v, m in param.history
                if abs(v - candidate) < 0.05
            ]
            
            if similar:
                avg_metric = statistics.mean(m for _, m in similar)
                if param.direction == OptimizationDirection.MAXIMIZE:
                    if avg_metric > best_score:
                        best_score = avg_metric
                        best_value = candidate
                else:
                    if avg_metric < best_score:
                        best_score = avg_metric
                        best_value = candidate
        
        return best_value
    
    def _tune_bandit(self, param_name: str, metric_value: float) -> float:
        """Tune using multi-armed bandit."""
        # Discretize parameter into arms
        param = self._parameters[param_name]
        n_arms = 10
        
        # Initialize bandit if needed
        if param_name not in self._bandits:
            self._bandits[param_name] = MultiArmedBandit(n_arms)
        
        bandit = self._bandits[param_name]
        
        # Map current value to arm
        current_arm = int((param.current_value - param.min_value) / 
                         (param.max_value - param.min_value) * (n_arms - 1))
        
        # Update bandit
        bandit.update(current_arm, metric_value)
        
        # Select new arm
        new_arm = bandit.select_arm()
        
        # Map arm back to value
        new_value = param.min_value + (new_arm / (n_arms - 1)) * (param.max_value - param.min_value)
        
        return new_value
    
    def _tune_adaptive(self, param: ParameterConfig, metric_value: float) -> float:
        """Adaptive tuning based on variance."""
        # Calculate recent variance
        recent_values = [m for _, _, m in param.history[-20:]]
        
        if len(recent_values) < 5:
            return param.current_value
        
        variance = statistics.variance(recent_values)
        
        # High variance = more exploration needed
        # Low variance = can exploit more
        if variance > 0.1:  # High variance
            adjustment = 1.1  # Increase parameter
        elif variance < 0.01:  # Low variance
            adjustment = 0.95  # Decrease parameter (converged)
        else:
            adjustment = 1.0
        
        # Direction depends on optimization goal
        if param.direction == OptimizationDirection.MINIMIZE:
            adjustment = 1 / adjustment
        
        return param.current_value * adjustment
    
    def _estimate_improvement(
        self,
        param: ParameterConfig,
        current_metric: float,
    ) -> float:
        """Estimate expected improvement from tuning."""
        # Based on historical performance at similar parameter values
        similar = [
            m for _, v, m in param.history
            if abs(v - param.current_value) < 0.05
        ]
        
        if similar:
            avg = statistics.mean(similar)
            return abs(avg - current_metric)
        
        return 0.05  # Default estimate
    
    def setup_drift_detection(
        self,
        metric_name: str,
        window_size: int = 30,
        threshold_std: float = 2.0,
    ) -> None:
        """Setup drift detection for a metric."""
        self._drift_detectors[metric_name] = DriftConfig(
            metric_name=metric_name,
            window_size=window_size,
            threshold_std=threshold_std,
        )
    
    def detect_drift(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Check for drift in a metric."""
        detector = self._drift_detectors.get(metric_name)
        if not detector:
            return None
        
        return detector.update(value)
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """Get comprehensive tuning report."""
        return {
            "parameters": {
                name: {
                    "current_value": param.current_value,
                    "range": [param.min_value, param.max_value],
                    "strategy": param.strategy.value,
                    "samples": len(param.history),
                    "last_tuned": param.history[-1][0].isoformat() if param.history else None,
                }
                for name, param in self._parameters.items()
            },
            "drift_detectors": {
                name: {
                    "samples": len(detector.recent_values),
                    "baseline_mean": detector.baseline_mean,
                    "baseline_std": detector.baseline_std,
                }
                for name, detector in self._drift_detectors.items()
            },
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate tuning recommendations."""
        recommendations = []
        
        for name, param in self._parameters.items():
            if len(param.history) < param.min_samples_before_tune:
                recommendations.append(
                    f"{name}: Need more samples ({len(param.history)}/{param.min_samples_before_tune})"
                )
            elif param.current_value == param.min_value:
                recommendations.append(
                    f"{name}: At minimum - consider expanding range"
                )
            elif param.current_value == param.max_value:
                recommendations.append(
                    f"{name}: At maximum - consider expanding range"
                )
        
        return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_auto_tuner: Optional[AutoTuner] = None


def get_auto_tuner() -> AutoTuner:
    """Get global auto-tuner."""
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = AutoTuner()
    return _auto_tuner


def reset_auto_tuner() -> None:
    """Reset global auto-tuner."""
    global _auto_tuner
    _auto_tuner = None
