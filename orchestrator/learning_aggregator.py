"""
LearningAggregator — Persistent cross-run learning
================================================
Module for aggregating model performance across ALL runs (not session-local) to enable
persistent learning and continuous improvement.

Pattern: Observer
Async: Yes — for I/O-bound storage operations
Layer: L6 Observability

Usage:
    from orchestrator.learning_aggregator import LearningAggregator
    aggregator = LearningAggregator(storage_path="./learning_data/")
    await aggregator.record_task_result(task_type="code_gen", model="gpt-4", score=0.85)
    recommendations = await aggregator.get_routing_recommendations(task_type="code_gen")
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Model, TaskType

logger = logging.getLogger("orchestrator.learning_aggregator")


@dataclass
class TaskPerformanceRecord:
    """Represents a single task performance record."""
    
    task_id: str
    task_type: TaskType
    model: Model
    score: float  # Performance score (0.0-1.0)
    timestamp: datetime
    cost: float  # Cost of the operation
    tokens_used: int  # Number of tokens used
    execution_time: float  # Execution time in seconds
    feedback: Optional[str] = None  # Optional feedback on the result


@dataclass
class ModelPerformanceStats:
    """Aggregated performance statistics for a model on a specific task type."""
    
    model: Model
    task_type: TaskType
    avg_score: float
    avg_cost: float
    avg_tokens: int
    avg_time: float
    total_runs: int
    success_rate: float
    last_updated: datetime


@dataclass
class RoutingRecommendation:
    """Recommendation for model routing based on historical performance."""
    
    task_type: TaskType
    recommended_model: Model
    expected_score: float
    cost_efficiency: float  # Higher is better
    confidence: float  # 0.0-1.0
    alternatives: List[Tuple[Model, float]]  # Model and expected score


class LearningAggregator:
    """Aggregates model performance across ALL runs for persistent learning."""

    def __init__(self, storage_path: str = "./learning_data/", 
                 retention_days: int = 90, min_samples_for_recommendation: int = 5):
        """
        Initialize the learning aggregator.
        
        Args:
            storage_path: Path to store learning data
            retention_days: Number of days to retain performance records
            min_samples_for_recommendation: Minimum samples needed for recommendations
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.min_samples_for_recommendation = min_samples_for_recommendation
        
        # In-memory cache for recent performance data
        self.performance_cache: Dict[Tuple[TaskType, Model], List[TaskPerformanceRecord]] = {}
        self.aggregated_stats: Dict[Tuple[TaskType, Model], ModelPerformanceStats] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def record_task_result(self, task_id: str, task_type: TaskType, model: Model, 
                                score: float, cost: float, tokens_used: int, 
                                execution_time: float, feedback: Optional[str] = None):
        """
        Record the result of a task for learning purposes.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task performed
            model: Model used for the task
            score: Performance score (0.0-1.0)
            cost: Cost of the operation
            tokens_used: Number of tokens used
            execution_time: Execution time in seconds
            feedback: Optional feedback on the result
        """
        async with self._lock:
            record = TaskPerformanceRecord(
                task_id=task_id,
                task_type=task_type,
                model=model,
                score=score,
                timestamp=datetime.now(),
                cost=cost,
                tokens_used=tokens_used,
                execution_time=execution_time,
                feedback=feedback
            )
            
            # Add to cache
            cache_key = (task_type, model)
            if cache_key not in self.performance_cache:
                self.performance_cache[cache_key] = []
            self.performance_cache[cache_key].append(record)
            
            # Persist to storage
            await self._persist_record(record)
            
            # Update aggregated stats
            await self._update_aggregated_stats(cache_key)
            
            logger.info(f"Recorded task result: {task_type.value} with {model.value}, score: {score:.2f}")
    
    async def _persist_record(self, record: TaskPerformanceRecord):
        """Persist a record to storage."""
        # Create a filename based on date
        date_str = record.timestamp.strftime("%Y-%m-%d")
        filename = f"performance_{date_str}.jsonl"
        filepath = self.storage_path / filename
        
        record_dict = {
            "task_id": record.task_id,
            "task_type": record.task_type.value,
            "model": record.model.value,
            "score": record.score,
            "timestamp": record.timestamp.isoformat(),
            "cost": record.cost,
            "tokens_used": record.tokens_used,
            "execution_time": record.execution_time,
            "feedback": record.feedback
        }
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist record: {e}")
    
    async def _update_aggregated_stats(self, cache_key: Tuple[TaskType, Model]):
        """Update aggregated statistics for a task type and model."""
        task_type, model = cache_key
        records = self.performance_cache[cache_key]
        
        if not records:
            return
        
        # Calculate statistics
        scores = [r.score for r in records]
        costs = [r.cost for r in records]
        tokens = [r.tokens_used for r in records]
        times = [r.execution_time for r in records]
        
        avg_score = sum(scores) / len(scores)
        avg_cost = sum(costs) / len(costs)
        avg_tokens = int(sum(tokens) / len(tokens))
        avg_time = sum(times) / len(times)
        total_runs = len(records)
        success_rate = len([s for s in scores if s >= 0.7]) / len(scores)  # Success threshold
        
        stats = ModelPerformanceStats(
            model=model,
            task_type=task_type,
            avg_score=avg_score,
            avg_cost=avg_cost,
            avg_tokens=avg_tokens,
            avg_time=avg_time,
            total_runs=total_runs,
            success_rate=success_rate,
            last_updated=datetime.now()
        )
        
        self.aggregated_stats[cache_key] = stats
    
    async def get_routing_recommendations(self, task_type: TaskType) -> Optional[RoutingRecommendation]:
        """
        Get routing recommendations for a task type based on historical performance.
        
        Args:
            task_type: Type of task to get recommendations for
            
        Returns:
            RoutingRecommendation or None if insufficient data
        """
        async with self._lock:
            # Find all models that have been used for this task type
            relevant_stats = [
                stats for (t, m), stats in self.aggregated_stats.items()
                if t == task_type and stats.total_runs >= self.min_samples_for_recommendation
            ]
            
            if len(relevant_stats) < 2:
                logger.info(f"Insufficient data for routing recommendations for {task_type.value}")
                return None
            
            # Sort by a combination of score and cost efficiency
            # Higher score and lower cost are better
            def score_key(stats: ModelPerformanceStats) -> float:
                # Normalize score (higher is better) and cost efficiency (lower cost is better)
                normalized_score = stats.avg_score
                # Invert cost for efficiency calculation (lower cost = higher efficiency)
                cost_efficiency = 1.0 / (stats.avg_cost + 0.0001)  # Add small value to avoid division by zero
                return normalized_score * cost_efficiency
            
            relevant_stats.sort(key=score_key, reverse=True)
            
            # Get the best model
            best_stats = relevant_stats[0]
            
            # Calculate confidence based on number of runs and recency
            confidence = min(best_stats.total_runs / 20.0, 1.0)  # Max confidence at 20 runs
            
            # Prepare alternatives
            alternatives = [
                (stats.model, stats.avg_score) 
                for stats in relevant_stats[1:4]  # Top 3 alternatives
            ]
            
            # Calculate cost efficiency (lower cost per unit score is better)
            cost_efficiency = best_stats.avg_score / (best_stats.avg_cost + 0.0001)
            
            recommendation = RoutingRecommendation(
                task_type=task_type,
                recommended_model=best_stats.model,
                expected_score=best_stats.avg_score,
                cost_efficiency=cost_efficiency,
                confidence=confidence,
                alternatives=alternatives
            )
            
            return recommendation
    
    async def load_historical_data(self):
        """Load historical performance data from storage."""
        async with self._lock:
            # Find all performance files
            for filepath in self.storage_path.glob("performance_*.jsonl"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    record_data = json.loads(line)
                                    
                                    # Convert string values back to enums
                                    task_type = TaskType(record_data["task_type"])
                                    model = Model(record_data["model"])
                                    
                                    record = TaskPerformanceRecord(
                                        task_id=record_data["task_id"],
                                        task_type=task_type,
                                        model=model,
                                        score=record_data["score"],
                                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                                        cost=record_data["cost"],
                                        tokens_used=record_data["tokens_used"],
                                        execution_time=record_data["execution_time"],
                                        feedback=record_data.get("feedback")
                                    )
                                    
                                    # Add to cache
                                    cache_key = (task_type, model)
                                    if cache_key not in self.performance_cache:
                                        self.performance_cache[cache_key] = []
                                    self.performance_cache[cache_key].append(record)
                                    
                                except (json.JSONDecodeError, KeyError, ValueError) as e:
                                    logger.warning(f"Skipping invalid record in {filepath}: {e}")
                                    continue
                except Exception as e:
                    logger.error(f"Failed to load historical data from {filepath}: {e}")
            
            # Update aggregated stats after loading
            for cache_key in self.performance_cache:
                await self._update_aggregated_stats(cache_key)
            
            logger.info(f"Loaded historical data: {len(self.performance_cache)} task-model combinations")
    
    async def get_model_comparison(self, task_type: TaskType) -> List[ModelPerformanceStats]:
        """
        Get a comparison of all models for a specific task type.
        
        Args:
            task_type: Type of task to compare models for
            
        Returns:
            List of ModelPerformanceStats sorted by performance
        """
        async with self._lock:
            relevant_stats = [
                stats for (t, m), stats in self.aggregated_stats.items()
                if t == task_type and stats.total_runs >= self.min_samples_for_recommendation
            ]
            
            # Sort by a combination of score and cost efficiency
            def comparison_key(stats: ModelPerformanceStats) -> float:
                # Higher score and lower cost are better
                normalized_score = stats.avg_score
                cost_efficiency = 1.0 / (stats.avg_cost + 0.0001)
                return normalized_score * cost_efficiency
            
            relevant_stats.sort(key=comparison_key, reverse=True)
            return relevant_stats
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from the learning data.
        
        Returns:
            Dict with learning insights
        """
        async with self._lock:
            total_records = sum(len(records) for records in self.performance_cache.values())
            total_task_types = len(set(task_type for (task_type, model) in self.performance_cache.keys()))
            total_models = len(set(model for (task_type, model) in self.performance_cache.keys()))
            
            # Find the best performing model for each task type
            best_models = {}
            for task_type in TaskType:
                recommendation = await self.get_routing_recommendations(task_type)
                if recommendation:
                    best_models[task_type.value] = recommendation.recommended_model.value
            
            # Calculate overall trends
            all_scores = [record.score for records in self.performance_cache.values() for record in records]
            avg_overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            # Calculate improvement over time (simplified)
            if len(all_scores) > 10:  # Need enough data points
                recent_scores = all_scores[-10:]  # Last 10 scores
                older_scores = all_scores[:10]   # First 10 scores
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                trending_positive = recent_avg > older_avg
            else:
                trending_positive = None
            
            return {
                "total_records_analyzed": total_records,
                "task_types_covered": total_task_types,
                "models_evaluated": total_models,
                "best_models_by_task": best_models,
                "average_overall_score": avg_overall_score,
                "trending_improvement": trending_positive,
                "last_updated": datetime.now().isoformat()
            }
    
    async def cleanup_old_records(self):
        """Clean up old performance records based on retention policy."""
        async with self._lock:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            cleaned_count = 0
            for (task_type, model), records in self.performance_cache.items():
                # Filter out old records
                filtered_records = [r for r in records if r.timestamp >= cutoff_date]
                self.performance_cache[(task_type, model)] = filtered_records
                cleaned_count += len(records) - len(filtered_records)
            
            # Also clean up old files from storage
            for filepath in self.storage_path.glob("performance_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = filepath.name.split("_")[1].split(".")[0]  # Gets YYYY-MM-DD from performance_YYYY-MM-DD.jsonl
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        filepath.unlink()
                        logger.info(f"Deleted old performance file: {filepath}")
                except (ValueError, IndexError):
                    # If we can't parse the date, skip the file
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} old performance records")
    
    async def export_learning_data(self, export_path: str) -> bool:
        """
        Export learning data to a file.
        
        Args:
            export_path: Path to export the data to
            
        Returns:
            bool: True if export was successful
        """
        try:
            export_data = {
                "aggregated_stats": {
                    f"{task_type.value}_{model.value}": {
                        "model": model.value,
                        "task_type": task_type.value,
                        "avg_score": stats.avg_score,
                        "avg_cost": stats.avg_cost,
                        "avg_tokens": stats.avg_tokens,
                        "avg_time": stats.avg_time,
                        "total_runs": stats.total_runs,
                        "success_rate": stats.success_rate,
                        "last_updated": stats.last_updated.isoformat()
                    }
                    for (task_type, model), stats in self.aggregated_stats.items()
                },
                "learning_insights": await self.get_learning_insights(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported learning data to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")
            return False
    
    async def import_learning_data(self, import_path: str) -> bool:
        """
        Import learning data from a file.
        
        Args:
            import_path: Path to import the data from
            
        Returns:
            bool: True if import was successful
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import aggregated stats
            for key, stats_data in import_data.get("aggregated_stats", {}).items():
                task_type = TaskType(stats_data["task_type"])
                model = Model(stats_data["model"])
                
                stats = ModelPerformanceStats(
                    model=model,
                    task_type=task_type,
                    avg_score=stats_data["avg_score"],
                    avg_cost=stats_data["avg_cost"],
                    avg_tokens=stats_data["avg_tokens"],
                    avg_time=stats_data["avg_time"],
                    total_runs=stats_data["total_runs"],
                    success_rate=stats_data["success_rate"],
                    last_updated=datetime.fromisoformat(stats_data["last_updated"])
                )
                
                self.aggregated_stats[(task_type, model)] = stats
            
            logger.info(f"Imported learning data from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import learning data: {e}")
            return False


# Global learning aggregator instance for convenience
_global_learning_aggregator: Optional[LearningAggregator] = None


async def get_global_learning_aggregator(storage_path: str = "./learning_data/") -> LearningAggregator:
    """
    Get the global learning aggregator instance, creating it if needed.
    
    Args:
        storage_path: Path to store learning data
        
    Returns:
        LearningAggregator instance
    """
    global _global_learning_aggregator
    if _global_learning_aggregator is None:
        _global_learning_aggregator = LearningAggregator(storage_path=storage_path)
        await _global_learning_aggregator.load_historical_data()
    return _global_learning_aggregator


async def record_task_result_global(task_id: str, task_type: TaskType, model: Model, 
                                  score: float, cost: float, tokens_used: int, 
                                  execution_time: float, feedback: Optional[str] = None):
    """
    Record a task result using the global learning aggregator.
    
    Args:
        task_id: Unique identifier for the task
        task_type: Type of task performed
        model: Model used for the task
        score: Performance score (0.0-1.0)
        cost: Cost of the operation
        tokens_used: Number of tokens used
        execution_time: Execution time in seconds
        feedback: Optional feedback on the result
    """
    aggregator = await get_global_learning_aggregator()
    await aggregator.record_task_result(
        task_id, task_type, model, score, cost, tokens_used, execution_time, feedback
    )


async def get_routing_recommendations_global(task_type: TaskType) -> Optional[RoutingRecommendation]:
    """
    Get routing recommendations using the global learning aggregator.
    
    Args:
        task_type: Type of task to get recommendations for
        
    Returns:
        RoutingRecommendation or None if insufficient data
    """
    aggregator = await get_global_learning_aggregator()
    return await aggregator.get_routing_recommendations(task_type)