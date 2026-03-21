"""
CostAnalytics — Cost analytics and forecasting
==============================================
Module for tracking, analyzing, and forecasting costs associated with LLM usage.

Pattern: Observer
Async: No — pure calculation and data processing
Layer: L6 Observability

Usage:
    from orchestrator.cost_analytics import CostAnalytics
    analytics = CostAnalytics()
    analytics.track_usage(model="gpt-4", input_tokens=1000, output_tokens=500)
    forecast = analytics.forecast_cost(project_duration_days=30)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .models import Model, COST_TABLE

logger = logging.getLogger("orchestrator.cost_analytics")


@dataclass
class UsageRecord:
    """Represents a single usage record."""
    
    model: Model
    input_tokens: int
    output_tokens: int
    timestamp: datetime
    cost: float
    project_id: Optional[str] = None
    task_id: Optional[str] = None


@dataclass
class CostBreakdown:
    """Represents a cost breakdown by various dimensions."""
    
    total_cost: float
    model_costs: Dict[Model, float]
    daily_costs: Dict[str, float]  # Date string -> cost
    project_costs: Dict[str, float]


@dataclass
class CostForecast:
    """Represents a cost forecast."""
    
    predicted_cost: float
    confidence_interval: Tuple[float, float]  # Lower and upper bounds
    trend: str  # "increasing", "decreasing", "stable"
    recommendations: List[str]


class CostAnalytics:
    """Tracks, analyzes, and forecasts costs associated with LLM usage."""

    def __init__(self):
        """Initialize the cost analytics tracker."""
        self.usage_records: List[UsageRecord] = []
        self.baseline_costs: Dict[Model, float] = {}  # Baseline cost per 1M tokens
    
    def track_usage(self, model: Model, input_tokens: int, output_tokens: int, 
                    project_id: Optional[str] = None, task_id: Optional[str] = None) -> float:
        """
        Track usage for a specific model and return the cost.
        
        Args:
            model: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            project_id: Optional project identifier
            task_id: Optional task identifier
            
        Returns:
            float: The cost of this usage
        """
        # Calculate cost based on the model's pricing
        cost_entry = COST_TABLE.get(model)
        if not cost_entry:
            logger.warning(f"Unknown model pricing for {model}, using default")
            # Use a default cost if model not found
            cost = (input_tokens * 0.01 + output_tokens * 0.03) / 1_000_000  # Default pricing
        else:
            input_cost_per_mil = cost_entry[0]
            output_cost_per_mil = cost_entry[1]
            cost = (input_tokens * input_cost_per_mil + output_tokens * output_cost_per_mil) / 1_000_000
        
        # Create usage record
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now(),
            cost=cost,
            project_id=project_id,
            task_id=task_id
        )
        
        # Add to records
        self.usage_records.append(record)
        
        logger.info(f"Tracked usage: {model.value}, {input_tokens} input + {output_tokens} output tokens, ${cost:.4f}")
        
        return cost
    
    def get_total_cost(self) -> float:
        """Get the total cost across all tracked usage."""
        return sum(record.cost for record in self.usage_records)
    
    def get_cost_breakdown(self) -> CostBreakdown:
        """Get a detailed cost breakdown."""
        total_cost = 0.0
        model_costs: Dict[Model, float] = {}
        daily_costs: Dict[str, float] = {}
        project_costs: Dict[str, float] = {}
        
        for record in self.usage_records:
            # Total cost
            total_cost += record.cost
            
            # Model costs
            if record.model not in model_costs:
                model_costs[record.model] = 0.0
            model_costs[record.model] += record.cost
            
            # Daily costs
            date_str = record.timestamp.strftime('%Y-%m-%d')
            if date_str not in daily_costs:
                daily_costs[date_str] = 0.0
            daily_costs[date_str] += record.cost
            
            # Project costs
            if record.project_id:
                if record.project_id not in project_costs:
                    project_costs[record.project_id] = 0.0
                project_costs[record.project_id] += record.cost
        
        return CostBreakdown(
            total_cost=total_cost,
            model_costs=model_costs,
            daily_costs=daily_costs,
            project_costs=project_costs
        )
    
    def get_model_cost_efficiency(self) -> Dict[Model, float]:
        """
        Calculate cost efficiency for each model (cost per token).
        
        Returns:
            Dict[Model, float]: Cost per million tokens for each model
        """
        model_efficiency: Dict[Model, float] = {}
        
        for record in self.usage_records:
            model = record.model
            total_tokens = record.input_tokens + record.output_tokens
            if total_tokens > 0:
                cost_per_token = record.cost / (total_tokens / 1_000_000)  # Cost per million tokens
                
                if model not in model_efficiency:
                    model_efficiency[model] = cost_per_token
                else:
                    # Average the cost efficiency
                    prev_efficiency = model_efficiency[model]
                    new_efficiency = (prev_efficiency + cost_per_token) / 2
                    model_efficiency[model] = new_efficiency
        
        return model_efficiency
    
    def forecast_cost(self, project_duration_days: int = 30) -> CostForecast:
        """
        Forecast costs for a given duration based on historical usage.
        
        Args:
            project_duration_days: Number of days to forecast
            
        Returns:
            CostForecast: The cost forecast with confidence intervals
        """
        if not self.usage_records:
            return CostForecast(
                predicted_cost=0.0,
                confidence_interval=(0.0, 0.0),
                trend="stable",
                recommendations=["No historical data available for forecasting"]
            )
        
        # Calculate average daily cost from history
        if len(self.usage_records) < 2:
            avg_daily_cost = self.usage_records[0].cost
        else:
            first_date = min(record.timestamp.date() for record in self.usage_records)
            last_date = max(record.timestamp.date() for record in self.usage_records)
            date_range = (last_date - first_date).days or 1  # At least 1 day
            
            total_cost = sum(record.cost for record in self.usage_records)
            avg_daily_cost = total_cost / date_range
        
        # Predict total cost for the duration
        predicted_cost = avg_daily_cost * project_duration_days
        
        # Calculate confidence interval based on historical variance
        daily_costs = self._get_daily_costs()
        if len(daily_costs) > 1:
            mean_daily = sum(daily_costs.values()) / len(daily_costs)
            variance = sum((cost - mean_daily) ** 2 for cost in daily_costs.values()) / len(daily_costs)
            std_dev = variance ** 0.5
            
            # Confidence interval: mean ± 1.96 * std_dev (95% confidence)
            margin = 1.96 * std_dev
            lower_bound = max(0, (mean_daily - margin) * project_duration_days)
            upper_bound = (mean_daily + margin) * project_duration_days
        else:
            lower_bound = predicted_cost * 0.8  # 20% lower
            upper_bound = predicted_cost * 1.2  # 20% higher
        
        # Determine trend
        trend = self._determine_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trend, avg_daily_cost)
        
        return CostForecast(
            predicted_cost=predicted_cost,
            confidence_interval=(lower_bound, upper_bound),
            trend=trend,
            recommendations=recommendations
        )
    
    def _get_daily_costs(self) -> Dict[str, float]:
        """Get costs grouped by date."""
        daily_costs: Dict[str, float] = {}
        
        for record in self.usage_records:
            date_str = record.timestamp.strftime('%Y-%m-%d')
            if date_str not in daily_costs:
                daily_costs[date_str] = 0.0
            daily_costs[date_str] += record.cost
        
        return daily_costs
    
    def _determine_trend(self) -> str:
        """Determine the cost trend based on recent history."""
        if len(self.usage_records) < 3:
            return "stable"
        
        # Get the last 7 days of data (or all if less than 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_records = [r for r in self.usage_records if r.timestamp >= week_ago]
        
        if len(recent_records) < 3:
            # Use all records if less than 3 in the last week
            recent_records = self.usage_records[-10:]  # Last 10 records
        
        # Calculate costs for early and late periods
        mid_point = len(recent_records) // 2
        early_period = recent_records[:mid_point]
        late_period = recent_records[mid_point:]
        
        if not early_period or not late_period:
            return "stable"
        
        early_avg = sum(r.cost for r in early_period) / len(early_period)
        late_avg = sum(r.cost for r in late_period) / len(late_period)
        
        # Define threshold for considering a change significant
        threshold = 0.1  # 10% difference
        
        if late_avg > early_avg * (1 + threshold):
            return "increasing"
        elif late_avg < early_avg * (1 - threshold):
            return "decreasing"
        else:
            return "stable"
    
    def _generate_recommendations(self, trend: str, avg_daily_cost: float) -> List[str]:
        """Generate cost optimization recommendations based on trend."""
        recommendations = []
        
        if trend == "increasing":
            recommendations.append(
                "Costs are increasing. Consider reviewing usage patterns and "
                "evaluating more cost-effective models for certain tasks."
            )
        elif avg_daily_cost > 10:  # If daily cost is high
            recommendations.append(
                "Daily costs are high. Consider implementing stricter budget "
                "controls or using more cost-efficient models for routine tasks."
            )
        
        # Identify the most expensive models
        model_costs = self.get_model_cost_efficiency()
        if model_costs:
            most_expensive = max(model_costs.items(), key=lambda x: x[1])
            recommendations.append(
                f"The most expensive model is {most_expensive[0].value} at ${most_expensive[1]:.4f} per million tokens. "
                f"Consider using alternatives for cost reduction."
            )
        
        # Suggest cost-saving measures
        recommendations.append(
            "Consider using cheaper models for preliminary tasks and more expensive "
            "models only for final refinement."
        )
        
        return recommendations
    
    def get_cost_per_project(self) -> Dict[str, float]:
        """Get total cost per project."""
        project_costs: Dict[str, float] = {}
        
        for record in self.usage_records:
            if record.project_id:
                if record.project_id not in project_costs:
                    project_costs[record.project_id] = 0.0
                project_costs[record.project_id] += record.cost
        
        return project_costs
    
    def get_top_cost_drivers(self, n: int = 5) -> List[Tuple[Model, float]]:
        """
        Get the top N models that contribute most to costs.
        
        Args:
            n: Number of top models to return
            
        Returns:
            List[Tuple[Model, float]]: Top models and their costs
        """
        model_costs = self.get_cost_breakdown().model_costs
        sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
        return sorted_models[:n]
    
    def reset_tracking(self):
        """Reset all usage tracking."""
        self.usage_records.clear()
        logger.info("Cost tracking reset")