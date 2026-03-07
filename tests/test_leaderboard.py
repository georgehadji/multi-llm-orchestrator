"""
Tests for Model Leaderboard
"""

import pytest
from datetime import datetime

from orchestrator.leaderboard import (
    ModelLeaderboard,
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkResult,
    BenchmarkDifficulty,
    ModelBenchmarkSummary,
    get_leaderboard,
    reset_leaderboard,
)
from orchestrator.models import Model, TaskType


class TestBenchmarkTask:
    def test_task_creation(self):
        task = BenchmarkTask(
            name="test-task",
            task_type=TaskType.CODE_GEN,
            difficulty=BenchmarkDifficulty.EASY,
            prompt="Generate a function",
            expected_patterns=["def", "return"],
        )
        
        assert task.name == "test-task"
        assert task.task_type == TaskType.CODE_GEN
        assert task.difficulty == BenchmarkDifficulty.EASY
        assert task.id is not None  # Auto-generated
    
    def test_task_custom_id(self):
        task = BenchmarkTask(
            id="custom-123",
            name="test-task",
            task_type=TaskType.CODE_GEN,
            difficulty=BenchmarkDifficulty.EASY,
            prompt="Generate a function",
            expected_patterns=["def"],
        )
        
        assert task.id == "custom-123"


class TestBenchmarkSuite:
    def test_suite_initialization(self):
        suite = BenchmarkSuite()
        
        # Should have tasks
        assert len(suite.tasks) > 0
        
        # Should have CODE_GEN tasks
        codegen_tasks = suite.get_tasks_by_type(TaskType.CODE_GEN)
        assert len(codegen_tasks) > 0
    
    def test_get_by_difficulty(self):
        suite = BenchmarkSuite()
        
        easy_tasks = suite.get_tasks_by_difficulty(BenchmarkDifficulty.EASY)
        hard_tasks = suite.get_tasks_by_difficulty(BenchmarkDifficulty.HARD)
        
        # All easy tasks should actually be easy
        assert all(t.difficulty == BenchmarkDifficulty.EASY for t in easy_tasks)


class TestBenchmarkResult:
    def test_efficiency_score_free(self):
        result = BenchmarkResult(
            task_id="test",
            model=Model.GLM_4_FLASH,  # Free model
            latency_ms=1000,
            time_to_first_token_ms=300,
            total_duration_ms=1000,
            quality_score=0.8,
            passed_validation=True,
            pattern_match_score=0.9,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,  # Free
        )
        
        # Free model with quality 0.8 should have high efficiency
        assert result.efficiency_score > 0
    
    def test_efficiency_score_paid(self):
        result = BenchmarkResult(
            task_id="test",
            model=Model.GPT_4O,
            latency_ms=1000,
            time_to_first_token_ms=300,
            total_duration_ms=1000,
            quality_score=0.9,
            passed_validation=True,
            pattern_match_score=0.9,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.01,
        )
        
        # Should calculate efficiency
        assert result.efficiency_score == (0.9 * 100) / 0.01


class TestModelBenchmarkSummary:
    def test_composite_score_calculation(self):
        summary = ModelBenchmarkSummary(
            model=Model.GPT_4O,
            avg_quality=0.9,
            avg_latency_ms=500,
            avg_efficiency_score=500,
            validation_pass_rate=0.95,
        )
        
        score = summary.composite_score
        
        # Should be between 0 and 1
        assert 0.0 <= score <= 1.0
        
        # High quality should give good score
        assert score > 0.5
    
    def test_update_with_result(self):
        summary = ModelBenchmarkSummary(model=Model.GPT_4O)
        
        # Initially no deployments
        assert summary.total_deployments == 0
        
        # Would need actual update logic tested via leaderboard


class TestModelLeaderboard:
    def setup_method(self):
        reset_leaderboard()
        self.lb = ModelLeaderboard()
    
    def test_leaderboard_initialization(self):
        assert self.lb is not None
        assert len(self.lb.suite.tasks) > 0
    
    def test_get_leaderboard_empty(self):
        # With no results, should return empty
        leaderboard = self.lb.get_leaderboard()
        
        # Should filter out models with insufficient data
        assert len(leaderboard) == 0
    
    def test_export_to_dashboard(self):
        export = self.lb.export_to_dashboard_format()
        
        assert "last_updated" in export
        assert "total_benchmarks" in export
        assert "leaderboard" in export
    
    def test_get_best_model_no_data(self):
        # With no data, should return None
        best = self.lb.get_best_model_for_task(TaskType.CODE_GEN)
        assert best is None


class TestGlobalLeaderboard:
    def setup_method(self):
        reset_leaderboard()
    
    def test_get_leaderboard_singleton(self):
        lb1 = get_leaderboard()
        lb2 = get_leaderboard()
        
        assert lb1 is lb2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
