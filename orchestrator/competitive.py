"""
Competitive — Competitive routing intelligence
=============================================
Module for competitive analysis and routing optimization based on market intelligence.

Pattern: Strategy
Async: Yes — for I/O-bound market data retrieval
Layer: L4 Supervisor

Usage:
    from orchestrator.competitive import CompetitiveIntelligence
    competitor = CompetitiveIntelligence()
    recommendation = await competitor.get_routing_recommendation(task_type="code_gen")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import Model, TaskType

logger = logging.getLogger("orchestrator.competitive")


@dataclass
class MarketDataPoint:
    """Represents a single market data point for a model."""

    model: Model
    price_per_mil_tokens: float  # Cost per million tokens
    latency_ms: float  # Average response latency
    availability: float  # Availability percentage (0.0-1.0)
    quality_score: float  # Quality score (0.0-1.0)
    timestamp: datetime


@dataclass
class CompetitiveRecommendation:
    """Represents a competitive routing recommendation."""

    recommended_model: Model
    cost_savings: float  # Potential cost savings compared to default
    performance_gains: float  # Potential performance gains
    risk_assessment: str  # Risk level ("low", "medium", "high")
    confidence: float  # Confidence in recommendation (0.0-1.0)
    alternatives: list[tuple[Model, float]]  # Alternative models with benefit scores


class CompetitiveIntelligence:
    """Provides competitive analysis and routing optimization based on market intelligence."""

    def __init__(self):
        """Initialize the competitive intelligence system."""
        self.market_data: list[MarketDataPoint] = []
        self.performance_history: dict[Model, list[float]] = {}  # Quality scores over time
        self.last_update = datetime.min

    async def update_market_data(self):
        """Update market data from various sources."""
        # Simulate fetching market data from various sources
        # In a real implementation, this would connect to APIs or databases
        logger.info("Updating market data...")

        # Simulate network delay
        await asyncio.sleep(0.1)

        # Generate simulated market data
        new_data_points = [
            MarketDataPoint(
                model=Model.OPENAI_GPT4,
                price_per_mil_tokens=30.0,
                latency_ms=800.0,
                availability=0.98,
                quality_score=0.92,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.DEEPSEEK_REASONER,
                price_per_mil_tokens=12.0,
                latency_ms=1200.0,
                availability=0.95,
                quality_score=0.89,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.DEEPSEEK_CHAT,
                price_per_mil_tokens=2.0,
                latency_ms=600.0,
                availability=0.97,
                quality_score=0.85,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.GOOGLE_GEMINI_PRO,
                price_per_mil_tokens=15.0,
                latency_ms=900.0,
                availability=0.96,
                quality_score=0.88,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.CLAUDE_3_OPUS,
                price_per_mil_tokens=75.0,
                latency_ms=1500.0,
                availability=0.94,
                quality_score=0.95,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.CLAUDE_3_5_SONNET,
                price_per_mil_tokens=15.0,
                latency_ms=1000.0,
                availability=0.96,
                quality_score=0.93,
                timestamp=datetime.now(),
            ),
            MarketDataPoint(
                model=Model.CLAUDE_3_HAIKU,
                price_per_mil_tokens=1.25,
                latency_ms=400.0,
                availability=0.97,
                quality_score=0.78,
                timestamp=datetime.now(),
            ),
        ]

        # Add new data points to market data
        self.market_data.extend(new_data_points)

        # Keep only recent data (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        self.market_data = [d for d in self.market_data if d.timestamp >= cutoff_date]

        self.last_update = datetime.now()
        logger.info(f"Market data updated with {len(new_data_points)} new data points")

    def get_model_rankings(self, task_type: TaskType) -> list[tuple[Model, float]]:
        """
        Get model rankings for a specific task type based on cost-performance ratio.

        Args:
            task_type: The type of task to rank models for

        Returns:
            List[Tuple[Model, float]]: Ranked models with scores
        """
        if not self.market_data:
            # If no market data, return default ranking
            return [
                (Model.DEEPSEEK_CHAT, 0.8),
                (Model.DEEPSEEK_REASONER, 0.7),
                (Model.OPENAI_GPT4, 0.6),
                (Model.GOOGLE_GEMINI_PRO, 0.5),
                (Model.CLAUDE_3_5_SONNET, 0.4),
            ]

        # Calculate composite scores for each model
        model_scores = {}

        for data_point in self.market_data:
            # Adjust quality score based on task type
            adjusted_quality = self._adjust_quality_for_task(data_point.quality_score, task_type)

            # Calculate cost-performance ratio (higher is better)
            # Normalize price to 0-1 scale (lower price is better)
            max_price = max(md.price_per_mil_tokens for md in self.market_data)
            normalized_price = 1 - (data_point.price_per_mil_tokens / max_price)

            # Calculate composite score
            # Weight: 40% quality, 30% cost efficiency, 20% availability, 10% latency
            composite_score = (
                adjusted_quality * 0.4
                + normalized_price * 0.3
                + data_point.availability * 0.2
                + (1 - min(data_point.latency_ms / 2000.0, 1.0)) * 0.1  # Lower latency is better
            )

            # Accumulate scores for models that appear multiple times
            if data_point.model in model_scores:
                # Average the scores
                prev_score = model_scores[data_point.model][0]
                count = model_scores[data_point.model][1] + 1
                avg_score = ((prev_score * (count - 1)) + composite_score) / count
                model_scores[data_point.model] = (avg_score, count)
            else:
                model_scores[data_point.model] = (composite_score, 1)

        # Convert to list and sort by score
        ranked_models = [(model, score[0]) for model, score in model_scores.items()]
        ranked_models.sort(key=lambda x: x[1], reverse=True)

        return ranked_models

    def _adjust_quality_for_task(self, base_quality: float, task_type: TaskType) -> float:
        """Adjust quality score based on task type."""
        # Different models excel at different tasks
        task_multipliers = {
            TaskType.CODE_GEN: {
                Model.DEEPSEEK_CHAT: 1.1,  # Good at coding
                Model.DEEPSEEK_REASONER: 1.05,
                Model.CLAUDE_3_5_SONNET: 1.15,  # Excellent at coding
                Model.OPENAI_GPT4: 1.0,
                Model.GOOGLE_GEMINI_PRO: 0.95,
            },
            TaskType.REASONING: {
                Model.DEEPSEEK_REASONER: 1.15,  # Optimized for reasoning
                Model.CLAUDE_3_OPUS: 1.2,  # Excellent reasoning
                Model.CLAUDE_3_5_SONNET: 1.1,
                Model.OPENAI_GPT4: 1.0,
                Model.GOOGLE_GEMINI_PRO: 1.05,
            },
            TaskType.TEXT_GEN: {
                Model.CLAUDE_3_OPUS: 1.1,  # Great for text generation
                Model.CLAUDE_3_5_SONNET: 1.05,
                Model.OPENAI_GPT4: 1.0,
                Model.DEEPSEEK_CHAT: 0.95,
                Model.GOOGLE_GEMINI_PRO: 1.0,
            },
            TaskType.OTHER: {
                Model.DEEPSEEK_CHAT: 1.05,  # Versatile
                Model.CLAUDE_3_5_SONNET: 1.0,
                Model.OPENAI_GPT4: 1.0,
                Model.GOOGLE_GEMINI_PRO: 1.0,
                Model.DEEPSEEK_REASONER: 0.95,
            },
        }

        multiplier = task_multipliers.get(task_type, {}).get(Model.DEEPSEEK_CHAT, 1.0)
        return min(base_quality * multiplier, 1.0)  # Cap at 1.0

    async def get_routing_recommendation(self, task_type: TaskType) -> CompetitiveRecommendation:
        """
        Get a competitive routing recommendation for a specific task type.

        Args:
            task_type: The type of task to get a recommendation for

        Returns:
            CompetitiveRecommendation: The routing recommendation
        """
        # Update market data if it's been more than an hour
        if datetime.now() - self.last_update > timedelta(hours=1):
            await self.update_market_data()

        # Get model rankings for this task type
        rankings = self.get_model_rankings(task_type)

        if not rankings:
            # Fallback to default model
            return CompetitiveRecommendation(
                recommended_model=Model.DEEPSEEK_CHAT,
                cost_savings=0.0,
                performance_gains=0.0,
                risk_assessment="low",
                confidence=0.5,
                alternatives=[],
            )

        # Get the top recommended model
        recommended_model, recommended_score = rankings[0]

        # Calculate potential benefits compared to default model
        default_model = Model.DEEPSEEK_CHAT
        default_score = next((score for model, score in rankings if model == default_model), 0.5)

        cost_savings = self._calculate_cost_savings(recommended_model, default_model)
        performance_gains = max(0, recommended_score - default_score)

        # Assess risk level
        risk_level = self._assess_risk_level(recommended_model)

        # Prepare alternatives
        alternatives = [(model, score) for model, score in rankings[1:4]]  # Top 3 alternatives

        return CompetitiveRecommendation(
            recommended_model=recommended_model,
            cost_savings=cost_savings,
            performance_gains=performance_gains,
            risk_assessment=risk_level,
            confidence=min(recommended_score, 1.0),
            alternatives=alternatives,
        )

    def _calculate_cost_savings(self, recommended_model: Model, default_model: Model) -> float:
        """Calculate potential cost savings of using recommended model vs default."""
        recommended_price = next(
            (md.price_per_mil_tokens for md in self.market_data if md.model == recommended_model),
            float("inf"),
        )
        default_price = next(
            (md.price_per_mil_tokens for md in self.market_data if md.model == default_model), 0
        )

        # Savings = default_price - recommended_price (positive if recommended is cheaper)
        return default_price - recommended_price

    def _assess_risk_level(self, model: Model) -> str:
        """Assess the risk level of using a particular model."""
        # Get the latest data point for this model
        model_data = [md for md in self.market_data if md.model == model]
        if not model_data:
            return "high"  # No data means high risk

        latest_data = max(model_data, key=lambda x: x.timestamp)

        # Risk factors: low availability, high latency, low quality
        if latest_data.availability < 0.95 or latest_data.quality_score < 0.8:
            return "high"
        elif latest_data.availability < 0.98 or latest_data.quality_score < 0.85:
            return "medium"
        else:
            return "low"

    async def get_cost_optimization_opportunities(self) -> list[dict[str, any]]:
        """
        Identify cost optimization opportunities based on market data.

        Returns:
            List[Dict]: Opportunities for cost optimization
        """
        if not self.market_data:
            await self.update_market_data()

        opportunities = []

        # Find models that offer better price-to-performance ratios
        for data_point in self.market_data:
            # Compare against the most expensive model in the same quality bracket
            same_quality_models = [
                md
                for md in self.market_data
                if abs(md.quality_score - data_point.quality_score) < 0.05
            ]

            if same_quality_models:
                # Find the cheapest model with similar quality
                cheapest_in_bracket = min(same_quality_models, key=lambda x: x.price_per_mil_tokens)

                if cheapest_in_bracket.model != data_point.model:
                    savings_per_mil = (
                        data_point.price_per_mil_tokens - cheapest_in_bracket.price_per_mil_tokens
                    )

                    if savings_per_mil > 0:  # Only if there are actual savings
                        opportunities.append(
                            {
                                "current_model": data_point.model,
                                "recommended_model": cheapest_in_bracket.model,
                                "savings_per_mil_tokens": savings_per_mil,
                                "quality_difference": abs(
                                    data_point.quality_score - cheapest_in_bracket.quality_score
                                ),
                                "opportunity_description": (
                                    f"Switch from {data_point.model.value} to {cheapest_in_bracket.model.value} "
                                    f"for ${savings_per_mil:.2f} savings per million tokens "
                                    f"with similar quality ({cheapest_in_bracket.quality_score:.2f})"
                                ),
                            }
                        )

        # Sort by savings potential
        opportunities.sort(key=lambda x: x["savings_per_mil_tokens"], reverse=True)

        return opportunities

    def get_performance_trends(self, model: Model) -> list[tuple[datetime, float]]:
        """
        Get performance trends for a specific model.

        Args:
            model: The model to get trends for

        Returns:
            List[Tuple[datetime, float]]: Timestamp and quality score pairs
        """
        model_data = [md for md in self.market_data if md.model == model]

        # Sort by timestamp
        model_data.sort(key=lambda x: x.timestamp)

        # Return list of (timestamp, quality_score)
        return [(md.timestamp, md.quality_score) for md in model_data]

    async def trigger_market_analysis(self, force_refresh: bool = False):
        """
        Trigger a market analysis, optionally forcing a refresh of data.

        Args:
            force_refresh: Whether to force a refresh of market data
        """
        if force_refresh or datetime.now() - self.last_update > timedelta(hours=1):
            await self.update_market_data()

        logger.info("Market analysis completed")
