"""
Model Leaderboard with Benchmark Suite
======================================

Automated benchmarking that periodically runs standardized tasks across all
providers and publishes a leaderboard (cost, quality, latency) that feeds
into routing decisions.

Features:
- Standardized benchmark tasks for each TaskType
- Periodic automated runs
- Historical tracking of model performance
- Routing weight updates based on results
- Export to dashboard/API

Usage:
    from orchestrator.leaderboard import ModelLeaderboard, BenchmarkSuite
    
    suite = BenchmarkSuite()
    lb = ModelLeaderboard(suite)
    results = await lb.run_benchmarks()
    await lb.update_routing_weights()
"""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import hashlib

from .log_config import get_logger
from .models import Model, TaskType, COST_TABLE, ROUTING_TABLE
from .api_clients import UnifiedClient
from .policy import ModelProfile
from .feedback_loop import FeedbackLoop, ProductionOutcome, OutcomeStatus

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Data Models
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkDifficulty(Enum):
    """Difficulty level of benchmark tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    name: str
    task_type: TaskType
    difficulty: BenchmarkDifficulty
    prompt: str
    expected_patterns: List[str]  # Patterns expected in good output
    validation_code: Optional[str] = None  # Optional code to validate output
    timeout_seconds: float = 60.0
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(self.name.encode()).hexdigest()[:12]


@dataclass
class BenchmarkResult:
    """Result of running a benchmark task on a model."""
    task_id: str
    model: Model
    
    # Timing
    latency_ms: float
    time_to_first_token_ms: float
    total_duration_ms: float
    
    # Quality
    quality_score: float  # 0.0 - 1.0
    passed_validation: bool
    pattern_match_score: float
    
    # Cost
    input_tokens: int
    output_tokens: int
    cost_usd: float
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    raw_output: str = field(default="", repr=False)
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate throughput."""
        if self.total_duration_ms <= 0:
            return 0.0
        return (self.input_tokens + self.output_tokens) / (self.total_duration_ms / 1000)
    
    @property
    def efficiency_score(self) -> float:
        """
        Calculate efficiency score (quality per dollar).
        
        Higher is better.
        """
        if self.cost_usd <= 0:
            return self.quality_score * 100  # Free models get high score
        return (self.quality_score * 100) / self.cost_usd


@dataclass
class ModelBenchmarkSummary:
    """Summary of a model's performance across all benchmarks."""
    model: Model
    
    # Overall scores
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    
    # Success rates
    tasks_completed: int = 0
    tasks_failed: int = 0
    validation_pass_rate: float = 0.0
    
    # Efficiency
    avg_efficiency_score: float = 0.0
    
    # Per-task-type scores
    by_task_type: Dict[TaskType, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    benchmark_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def composite_score(self) -> float:
        """
        Calculate composite ranking score.
        
        Weights: Quality 40%, Efficiency 30%, Speed 20%, Reliability 10%
        """
        quality = self.avg_quality * 0.4
        
        # Efficiency: normalize to 0-1 range (assume max efficiency ~1000)
        efficiency = min(self.avg_efficiency_score / 1000, 1.0) * 0.3
        
        # Speed: normalize (assume good latency < 5000ms)
        speed = max(0, 1.0 - (self.avg_latency_ms / 5000)) * 0.2
        
        # Reliability
        total = self.tasks_completed + self.tasks_failed
        reliability = (self.tasks_completed / total * 0.1) if total > 0 else 0.0
        
        return quality + efficiency + speed + reliability


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""
    rank: int
    model: Model
    provider: str
    
    # Scores
    composite_score: float
    quality_score: float
    efficiency_score: float
    speed_score: float
    reliability_score: float
    
    # Costs
    avg_cost_per_1k_tokens: float
    
    # Best for
    recommended_for: List[TaskType] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Standard Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Standardized benchmark tasks for model evaluation.
    """
    
    def __init__(self):
        self.tasks: List[BenchmarkTask] = []
        self._initialize_tasks()
    
    def _initialize_tasks(self) -> None:
        """Initialize the standard benchmark tasks."""
        
        # CODE_GENERATION tasks
        self.tasks.extend([
            BenchmarkTask(
                id="codegen-001",
                name="FastAPI Endpoint",
                task_type=TaskType.CODE_GEN,
                difficulty=BenchmarkDifficulty.EASY,
                prompt="""Create a FastAPI endpoint `/users/{user_id}` that:
1. Accepts a user_id path parameter
2. Returns a JSON response with user details
3. Includes proper error handling for invalid user_id
4. Has type hints and docstrings

Return only the Python code.""",
                expected_patterns=[
                    "@app.get", "async def", "user_id", 
                    "JSONResponse", "HTTPException", "Optional[", "Union[",
                ],
            ),
            BenchmarkTask(
                id="codegen-002",
                name="Async Data Pipeline",
                task_type=TaskType.CODE_GEN,
                difficulty=BenchmarkDifficulty.MEDIUM,
                prompt="""Create an async data processing pipeline that:
1. Fetches data from multiple URLs concurrently
2. Processes results with error handling
3. Implements rate limiting
4. Returns aggregated results

Use asyncio and aiohttp.""",
                expected_patterns=[
                    "async def", "await", "asyncio.gather",
                    "aiohttp", "Semaphore", "try:", "except",
                ],
            ),
            BenchmarkTask(
                id="codegen-003",
                name="Database Transaction Handler",
                task_type=TaskType.CODE_GEN,
                difficulty=BenchmarkDifficulty.HARD,
                prompt="""Create a robust database transaction handler with:
1. Connection pooling
2. Automatic retries with exponential backoff
3. Deadlock detection and handling
4. Proper transaction isolation
5. Context manager interface

Use SQLAlchemy.""",
                expected_patterns=[
                    "async def", "@contextlib.asynccontextmanager",
                    "sessionmaker", "create_async_engine",
                    "retry", "backoff", "isolation_level",
                ],
            ),
        ])
        
        # CODE_REVIEW tasks
        self.tasks.extend([
            BenchmarkTask(
                id="review-001",
                name="Review Python Function",
                task_type=TaskType.CODE_REVIEW,
                difficulty=BenchmarkDifficulty.EASY,
                prompt="""Review this Python function and identify issues:

```python
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
```

List specific improvements with line references.""",
                expected_patterns=[
                    "enumerate", "list comprehension", "generator",
                    "type hints", "docstring", "inefficient",
                ],
            ),
        ])
        
        # REASONING tasks
        self.tasks.extend([
            BenchmarkTask(
                id="reason-001",
                name="Architecture Decision",
                task_type=TaskType.REASONING,
                difficulty=BenchmarkDifficulty.MEDIUM,
                prompt="""Given these requirements:
- 10,000 concurrent users
- Real-time chat with message history
- File sharing up to 100MB
- Mobile and web clients

Compare WebSocket vs Server-Sent Events vs Long Polling.
Recommend one with clear trade-offs.""",
                expected_patterns=[
                    "WebSocket", "Server-Sent Events", "Long Polling",
                    "scalability", "latency", "trade-off", "recommend",
                ],
            ),
        ])
        
        # EVALUATE tasks
        self.tasks.extend([
            BenchmarkTask(
                id="eval-001",
                name="Code Quality Assessment",
                task_type=TaskType.EVALUATE,
                difficulty=BenchmarkDifficulty.MEDIUM,
                prompt="""Evaluate this code snippet for:
1. Security vulnerabilities
2. Performance issues
3. Maintainability
4. Python best practices

```python
import pickle

def load_user_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
```

Score each category 0-10 and provide specific fixes.""",
                expected_patterns=[
                    "pickle", "insecure", "deserialization", "arbitrary code",
                    "JSON", "schema", "validation", "score",
                ],
            ),
        ])
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[BenchmarkTask]:
        """Get all tasks of a specific type."""
        return [t for t in self.tasks if t.task_type == task_type]
    
    def get_tasks_by_difficulty(self, difficulty: BenchmarkDifficulty) -> List[BenchmarkTask]:
        """Get all tasks of a specific difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]


# ═══════════════════════════════════════════════════════════════════════════════
# Model Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════

class ModelLeaderboard:
    """
    Automated model benchmarking and ranking system.
    """
    
    def __init__(
        self,
        suite: Optional[BenchmarkSuite] = None,
        storage_path: Optional[Path] = None,
        api_client_factory: Optional[Callable[[Model], UnifiedClient]] = None,
    ):
        self.suite = suite or BenchmarkSuite()
        self.storage_path = storage_path or Path(".leaderboard")
        self.storage_path.mkdir(exist_ok=True)
        self._api_client_factory = api_client_factory
        
        # Storage for results
        self._results: List[BenchmarkResult] = []
        self._summaries: Dict[Model, ModelBenchmarkSummary] = {}
        
        self._load_results()
    
    def _load_results(self) -> None:
        """Load historical results from disk."""
        results_file = self.storage_path / "results.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                for result_data in data.get("results", []):
                    result = BenchmarkResult(
                        task_id=result_data["task_id"],
                        model=Model(result_data["model"]),
                        latency_ms=result_data["latency_ms"],
                        time_to_first_token_ms=result_data["time_to_first_token_ms"],
                        total_duration_ms=result_data["total_duration_ms"],
                        quality_score=result_data["quality_score"],
                        passed_validation=result_data["passed_validation"],
                        pattern_match_score=result_data["pattern_match_score"],
                        input_tokens=result_data["input_tokens"],
                        output_tokens=result_data["output_tokens"],
                        cost_usd=result_data["cost_usd"],
                        timestamp=datetime.fromisoformat(result_data["timestamp"]),
                        error=result_data.get("error"),
                    )
                    self._results.append(result)
                
                self._recompute_summaries()
            except Exception as e:
                logger.error(f"Failed to load benchmark results: {e}")
    
    def _save_results(self) -> None:
        """Save results to disk."""
        results_file = self.storage_path / "results.json"
        try:
            data = {
                "last_updated": datetime.utcnow().isoformat(),
                "results": [
                    {
                        **asdict(r),
                        "model": r.model.value,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in self._results
                ],
            }
            results_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def _get_client(self, model: Model) -> UnifiedClient:
        """Get API client for a model."""
        if self._api_client_factory:
            return self._api_client_factory(model)
        # Default client creation
        from .api_clients import create_client_for_model
        return create_client_for_model(model)
    
    async def run_benchmarks(
        self,
        models: Optional[List[Model]] = None,
        tasks: Optional[List[BenchmarkTask]] = None,
        max_concurrent: int = 3,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for specified models and tasks.
        
        Args:
            models: Models to benchmark (default: all)
            tasks: Tasks to run (default: all in suite)
            max_concurrent: Max concurrent API calls
        
        Returns:
            List of benchmark results
        """
        models = models or list(Model)
        tasks = tasks or self.suite.tasks
        
        logger.info(f"Running {len(tasks)} benchmarks on {len(models)} models")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[BenchmarkResult] = []
        
        async def run_single(model: Model, task: BenchmarkTask) -> BenchmarkResult:
            async with semaphore:
                return await self._run_single_benchmark(model, task)
        
        # Create all tasks
        coroutines = [
            run_single(model, task)
            for model in models
            for task in tasks
        ]
        
        # Run with progress tracking
        completed = 0
        total = len(coroutines)
        
        for coro in asyncio.as_completed(coroutines):
            result = await coro
            results.append(result)
            completed += 1
            
            if completed % 5 == 0:
                logger.info(f"Benchmark progress: {completed}/{total}")
        
        # Store and save
        self._results.extend(results)
        self._recompute_summaries()
        self._save_results()
        
        logger.info(f"Completed {len(results)} benchmarks")
        return results
    
    async def _run_single_benchmark(
        self,
        model: Model,
        task: BenchmarkTask,
    ) -> BenchmarkResult:
        """Run a single benchmark task."""
        import time
        
        client = self._get_client(model)
        start_time = time.monotonic()
        
        try:
            # Make the API call
            response = await client.call(
                prompt=task.prompt,
                model=model.value,
                max_tokens=2000,
                temperature=0.2,
            )
            
            end_time = time.monotonic()
            duration_ms = (end_time - start_time) * 1000
            
            # Calculate quality
            output = response.get("content", "")
            quality, pattern_score = self._score_output(output, task)
            
            # Calculate cost
            input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response.get("usage", {}).get("completion_tokens", 0)
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            return BenchmarkResult(
                task_id=task.id,
                model=model,
                latency_ms=duration_ms,
                time_to_first_token_ms=duration_ms * 0.3,  # Estimate
                total_duration_ms=duration_ms,
                quality_score=quality,
                passed_validation=quality >= 0.7,
                pattern_match_score=pattern_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                raw_output=output,
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model.value} on {task.id}: {e}")
            return BenchmarkResult(
                task_id=task.id,
                model=model,
                latency_ms=0,
                time_to_first_token_ms=0,
                total_duration_ms=0,
                quality_score=0,
                passed_validation=False,
                pattern_match_score=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                error=str(e),
            )
    
    def _score_output(self, output: str, task: BenchmarkTask) -> Tuple[float, float]:
        """Score the output quality."""
        output_lower = output.lower()
        
        # Check for expected patterns
        patterns_found = sum(
            1 for pattern in task.expected_patterns
            if pattern.lower() in output_lower
        )
        pattern_score = patterns_found / len(task.expected_patterns) if task.expected_patterns else 0.5
        
        # Overall quality (simplified)
        quality = pattern_score
        if len(output) > 100:
            quality += 0.1
        if len(output) > 500:
            quality += 0.1
        
        return min(1.0, quality), pattern_score
    
    def _calculate_cost(
        self,
        model: Model,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage."""
        costs = COST_TABLE.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return round(input_cost + output_cost, 6)
    
    def _recompute_summaries(self) -> None:
        """Recompute model summaries from results."""
        from collections import defaultdict
        
        by_model: Dict[Model, List[BenchmarkResult]] = defaultdict(list)
        for result in self._results:
            by_model[result.model].append(result)
        
        for model, results in by_model.items():
            if not results:
                continue
            
            completed = [r for r in results if r.error is None]
            failed = [r for r in results if r.error is not None]
            
            summary = ModelBenchmarkSummary(
                model=model,
                avg_quality=statistics.mean([r.quality_score for r in completed]) if completed else 0,
                avg_latency_ms=statistics.mean([r.latency_ms for r in completed]) if completed else 0,
                avg_cost_usd=statistics.mean([r.cost_usd for r in completed]) if completed else 0,
                total_cost_usd=sum([r.cost_usd for r in results]),
                tasks_completed=len(completed),
                tasks_failed=len(failed),
                validation_pass_rate=len([r for r in completed if r.passed_validation]) / len(completed) if completed else 0,
                avg_efficiency_score=statistics.mean([r.efficiency_score for r in completed]) if completed else 0,
                benchmark_count=len(results),
            )
            
            # Compute per-task-type scores
            by_type: Dict[TaskType, List[float]] = defaultdict(list)
            for r in results:
                task = next((t for t in self.suite.tasks if t.id == r.task_id), None)
                if task:
                    by_type[task.task_type].append(r.quality_score)
            
            summary.by_task_type = {
                tt: {"avg_quality": statistics.mean(scores)}
                for tt, scores in by_type.items()
            }
            
            self._summaries[model] = summary
    
    def get_leaderboard(self, task_type: Optional[TaskType] = None) -> List[LeaderboardEntry]:
        """Generate leaderboard entries."""
        entries = []
        
        for model, summary in self._summaries.items():
            # Skip models with insufficient data
            if summary.benchmark_count < 3:
                continue
            
            from .models import get_provider
            
            entry = LeaderboardEntry(
                rank=0,  # Will be assigned after sorting
                model=model,
                provider=get_provider(model),
                composite_score=summary.composite_score,
                quality_score=summary.avg_quality,
                efficiency_score=summary.avg_efficiency_score / 1000,  # Normalize
                speed_score=max(0, 1.0 - (summary.avg_latency_ms / 5000)),
                reliability_score=summary.validation_pass_rate,
                avg_cost_per_1k_tokens=summary.avg_cost_usd * 1000,
            )
            
            # Determine best task types for this model
            entry.recommended_for = self._get_recommended_tasks(model, summary)
            
            entries.append(entry)
        
        # Sort by composite score
        entries.sort(key=lambda e: e.composite_score, reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1
        
        return entries
    
    def _get_recommended_tasks(
        self,
        model: Model,
        summary: ModelBenchmarkSummary,
    ) -> List[TaskType]:
        """Determine which task types this model is best for."""
        recommended = []
        
        for task_type, scores in summary.by_task_type.items():
            avg_quality = scores.get("avg_quality", 0)
            if avg_quality >= 0.7:
                recommended.append(task_type)
        
        return recommended
    
    def get_best_model_for_task(self, task_type: TaskType) -> Optional[Model]:
        """Get the best model for a specific task type based on benchmarks."""
        best_model = None
        best_score = -1
        
        for model, summary in self._summaries.items():
            type_scores = summary.by_task_type.get(task_type, {})
            score = type_scores.get("avg_quality", 0)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model
    
    async def update_routing_weights(self) -> Dict[str, Any]:
        """
        Update routing weights based on benchmark results.
        
        Returns the updated routing configuration.
        """
        from .models import ROUTING_TABLE
        
        updates = {}
        
        for task_type, models in ROUTING_TABLE.items():
            # Reorder based on benchmark scores
            scored_models = [
                (model, self._summaries.get(model, ModelBenchmarkSummary(model)).composite_score)
                for model in models
            ]
            scored_models.sort(key=lambda x: x[1], reverse=True)
            
            updates[task_type.value] = [m[0].value for m in scored_models]
        
        # Save updated routing
        routing_file = self.storage_path / "routing_weights.json"
        routing_file.write_text(json.dumps({
            "updated_at": datetime.utcnow().isoformat(),
            "routing": updates,
        }, indent=2))
        
        logger.info("Updated routing weights based on benchmarks")
        return updates
    
    def export_to_dashboard_format(self) -> Dict[str, Any]:
        """Export leaderboard in dashboard-friendly format."""
        leaderboard = self.get_leaderboard()
        
        return {
            "last_updated": datetime.utcnow().isoformat(),
            "total_benchmarks": len(self._results),
            "leaderboard": [
                {
                    "rank": e.rank,
                    "model": e.model.value,
                    "provider": e.provider,
                    "composite_score": round(e.composite_score, 3),
                    "quality": round(e.quality_score, 3),
                    "efficiency": round(e.efficiency_score, 3),
                    "speed": round(e.speed_score, 3),
                    "reliability": round(e.reliability_score, 3),
                    "cost_per_1k": round(e.avg_cost_per_1k_tokens, 4),
                    "recommended_for": [t.value for t in e.recommended_for],
                }
                for e in leaderboard
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_leaderboard: Optional[ModelLeaderboard] = None


def get_leaderboard() -> ModelLeaderboard:
    """Get global leaderboard instance."""
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = ModelLeaderboard()
    return _leaderboard


def reset_leaderboard() -> None:
    """Reset global leaderboard (for testing)."""
    global _leaderboard
    _leaderboard = None
