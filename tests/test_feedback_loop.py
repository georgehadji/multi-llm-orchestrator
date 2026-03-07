"""
Tests for Production Feedback Loop
"""

import pytest
from datetime import datetime

from orchestrator.feedback_loop import (
    FeedbackLoop,
    ProductionOutcome,
    RuntimeError,
    PerformanceMetrics,
    UserFeedback,
    CodebaseFingerprint,
    OutcomeStatus,
    get_feedback_loop,
    reset_feedback_loop,
)
from orchestrator.models import Model, TaskType


class TestProductionOutcome:
    def test_calculate_success_score_success(self):
        outcome = ProductionOutcome(
            project_id="test",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        )
        
        score = outcome.calculate_success_score()
        assert score == 1.0
    
    def test_calculate_success_score_with_errors(self):
        outcome = ProductionOutcome(
            project_id="test",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
            runtime_errors=[
                RuntimeError(error_type="TypeError", message="Test error", count=2),
            ],
        )
        
        score = outcome.calculate_success_score()
        assert score < 1.0
        assert score > 0.0
    
    def test_calculate_success_score_with_user_feedback(self):
        outcome = ProductionOutcome(
            project_id="test",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
            user_feedback=UserFeedback(rating=3),  # 3/5
        )
        
        score = outcome.calculate_success_score()
        assert score == 0.6  # 1.0 * 0.6


class TestCodebaseFingerprint:
    def test_similarity_identical(self):
        fp1 = CodebaseFingerprint(
            languages=["python", "javascript"],
            framework="fastapi",
            patterns=["mvc", "repository"],
        )
        fp2 = CodebaseFingerprint(
            languages=["python", "javascript"],
            framework="fastapi",
            patterns=["mvc", "repository"],
        )
        
        similarity = fp1.similarity(fp2)
        assert similarity == 1.0
    
    def test_similarity_different(self):
        fp1 = CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
        )
        fp2 = CodebaseFingerprint(
            languages=["rust"],
            framework="actix",
        )
        
        similarity = fp1.similarity(fp2)
        assert similarity == 0.0
    
    def test_similarity_partial(self):
        fp1 = CodebaseFingerprint(
            languages=["python", "javascript"],
            framework="fastapi",
        )
        fp2 = CodebaseFingerprint(
            languages=["python", "typescript"],
            framework="fastapi",
        )
        
        similarity = fp1.similarity(fp2)
        assert 0.0 < similarity < 1.0


class TestFeedbackLoop:
    def setup_method(self):
        reset_feedback_loop()
        self.loop = FeedbackLoop()
    
    @pytest.mark.asyncio
    async def test_record_outcome(self):
        outcome = ProductionOutcome(
            project_id="test-project",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        )
        
        result = await self.loop.record_outcome(outcome)
        
        assert result["success"] is True
        assert result["model"] == "gpt-4o"
        assert result["success_score"] == 1.0
        assert result["updated_record"]["total_deployments"] == 1
    
    @pytest.mark.asyncio
    async def test_record_multiple_outcomes(self):
        # Record success
        await self.loop.record_outcome(ProductionOutcome(
            project_id="test",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        ))
        
        # Record failure
        await self.loop.record_outcome(ProductionOutcome(
            project_id="test",
            deployment_id="dep-2",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="def456",
            status=OutcomeStatus.FAILURE,
        ))
        
        # Check record
        score = self.loop.get_model_score(Model.GPT_4O, TaskType.CODE_GEN)
        assert 0.0 < score < 1.0  # Should be between 0 and 1
    
    def test_get_model_score_unknown(self):
        # Unknown model should return neutral score
        score = self.loop.get_model_score(Model.GPT_4O, TaskType.CODE_GEN)
        assert score == 0.5  # Default for unknown


class TestModelPerformanceRecord:
    def test_update_record(self):
        from orchestrator.feedback_loop import ModelPerformanceRecord
        
        record = ModelPerformanceRecord(
            model=Model.GPT_4O,
            task_type=TaskType.CODE_GEN,
        )
        
        outcome = ProductionOutcome(
            project_id="test",
            deployment_id="dep-1",
            task_type=TaskType.CODE_GEN,
            model_used=Model.GPT_4O,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        )
        
        record.update(outcome)
        
        assert record.total_deployments == 1
        assert record.success_count == 1
        assert record.avg_success_score > 0.5
    
    def test_success_rate_calculation(self):
        from orchestrator.feedback_loop import ModelPerformanceRecord
        
        record = ModelPerformanceRecord(
            model=Model.GPT_4O,
            task_type=TaskType.CODE_GEN,
        )
        
        # 2 successes, 1 failure
        for i, status in enumerate([
            OutcomeStatus.SUCCESS,
            OutcomeStatus.SUCCESS,
            OutcomeStatus.FAILURE,
        ]):
            record.update(ProductionOutcome(
                project_id="test",
                deployment_id=f"dep-{i}",
                task_type=TaskType.CODE_GEN,
                model_used=Model.GPT_4O,
                generated_code_hash=f"hash{i}",
                status=status,
            ))
        
        assert record.success_rate == 2.0 / 3.0


class TestGlobalFeedbackLoop:
    def setup_method(self):
        reset_feedback_loop()
    
    def test_get_feedback_loop(self):
        loop = get_feedback_loop()
        assert loop is not None
        
        loop2 = get_feedback_loop()
        assert loop is loop2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
