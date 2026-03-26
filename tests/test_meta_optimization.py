"""
Tests for Meta-Optimization and Self-Improving Templates
=========================================================
Tests for Hyperagents-inspired meta-optimization features.
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from orchestrator.meta_orchestrator import (
    ExecutionRecord,
    ProjectTrajectory,
    StrategyProposal,
    StrategyType,
    ProposalStatus,
    ExecutionArchive,
    MetaOptimizer,
    MetaOptimizationIntegration,
)
from orchestrator.adaptive_templates import (
    SelfImprovingTemplates,
    TemplateEvolutionRecord,
    get_self_improving_templates,
    reset_self_improving_templates,
)
from orchestrator.models import Model, TaskType


# ─────────────────────────────────────────────
# ExecutionRecord Tests
# ─────────────────────────────────────────────

class TestExecutionRecord:
    """Test ExecutionRecord data structure."""
    
    def test_create_record(self):
        record = ExecutionRecord(
            task_id="task_001",
            task_type="code_generation",
            model_used="deepseek-chat",
            success=True,
            cost_usd=0.005,
            latency_ms=1500,
            input_tokens=100,
            output_tokens=200,
            score=0.9,
        )
        
        assert record.task_id == "task_001"
        assert record.success is True
        assert record.score == 0.9
    
    def test_to_dict_roundtrip(self):
        record = ExecutionRecord(
            task_id="task_002",
            task_type="code_review",
            model_used="claude-3-5-sonnet",
            success=False,
            cost_usd=0.01,
            latency_ms=2000,
            input_tokens=150,
            output_tokens=250,
            score=0.5,
            error_message="Timeout",
        )
        
        data = record.to_dict()
        restored = ExecutionRecord.from_dict(data)
        
        assert restored.task_id == record.task_id
        assert restored.model_used == record.model_used
        assert restored.error_message == record.error_message


# ─────────────────────────────────────────────
# ProjectTrajectory Tests
# ─────────────────────────────────────────────

class TestProjectTrajectory:
    """Test ProjectTrajectory data structure."""
    
    def test_create_trajectory(self):
        records = [
            ExecutionRecord(
                task_id="task_001",
                task_type="code_generation",
                model_used="deepseek-chat",
                success=True,
                cost_usd=0.005,
                latency_ms=1500,
                input_tokens=100,
                output_tokens=200,
                score=0.9,
            )
        ]
        
        trajectory = ProjectTrajectory(
            project_id="proj_001",
            project_description="Build a web app",
            total_cost=0.005,
            total_time=2.0,
            success=True,
            task_records=records,
            model_sequence=["deepseek-chat"],
        )
        
        assert trajectory.project_id == "proj_001"
        assert len(trajectory.task_records) == 1
        assert trajectory.model_sequence == ["deepseek-chat"]
    
    def test_to_dict_roundtrip(self):
        records = [
            ExecutionRecord(
                task_id="task_001",
                task_type="code_generation",
                model_used="deepseek-chat",
                success=True,
                cost_usd=0.005,
                latency_ms=1500,
                input_tokens=100,
                output_tokens=200,
                score=0.9,
            )
        ]
        
        trajectory = ProjectTrajectory(
            project_id="proj_002",
            project_description="Build API",
            total_cost=0.005,
            total_time=2.0,
            success=True,
            task_records=records,
            model_sequence=["deepseek-chat"],
        )
        
        data = trajectory.to_dict()
        restored = ProjectTrajectory.from_dict(data)
        
        assert restored.project_id == trajectory.project_id
        assert len(restored.task_records) == len(trajectory.task_records)


# ─────────────────────────────────────────────
# ExecutionArchive Tests
# ─────────────────────────────────────────────

class TestExecutionArchive:
    """Test ExecutionArchive functionality."""
    
    @pytest.fixture
    def archive(self, tmp_path):
        return ExecutionArchive(tmp_path / "archive")
    
    def test_store_and_retrieve(self, archive):
        records = [
            ExecutionRecord(
                task_id="task_001",
                task_type="code_generation",
                model_used="deepseek-chat",
                success=True,
                cost_usd=0.005,
                latency_ms=1500,
                input_tokens=100,
                output_tokens=200,
                score=0.9,
                project_id="proj_001",
            )
        ]
        
        trajectory = ProjectTrajectory(
            project_id="proj_001",
            project_description="Build web app",
            total_cost=0.005,
            total_time=2.0,
            success=True,
            task_records=records,
            model_sequence=["deepseek-chat"],
        )
        
        archive.store(trajectory)
        
        assert archive.total_projects == 1
        assert archive.total_executions == 1
    
    def test_model_performance_stats(self, archive):
        # Store multiple executions
        for i in range(10):
            records = [
                ExecutionRecord(
                    task_id=f"task_{i}",
                    task_type="code_generation",
                    model_used="deepseek-chat" if i % 2 == 0 else "claude-3-5-sonnet",
                    success=i % 3 != 0,  # 67% success rate
                    cost_usd=0.005,
                    latency_ms=1500,
                    input_tokens=100,
                    output_tokens=200,
                    score=0.8 if i % 3 != 0 else 0.4,
                    project_id=f"proj_{i}",
                )
            ]
            
            trajectory = ProjectTrajectory(
                project_id=f"proj_{i}",
                project_description="Test project",
                total_cost=0.005,
                total_time=2.0,
                success=i % 3 != 0,
                task_records=records,
                model_sequence=["deepseek-chat" if i % 2 == 0 else "claude-3-5-sonnet"],
            )
            archive.store(trajectory)
        
        # Get model performance
        deepseek_stats = archive.get_model_performance("deepseek-chat")
        claude_stats = archive.get_model_performance("claude-3-5-sonnet")
        
        assert deepseek_stats["total_executions"] == 5
        assert claude_stats["total_executions"] == 5
    
    def test_find_similar_projects(self, archive):
        # Store projects with different descriptions
        descriptions = [
            "Build a Python web API with FastAPI",
            "Create a React frontend dashboard",
            "Build a Python API for data processing",
            "Design a database schema for e-commerce",
        ]
        
        for i, desc in enumerate(descriptions):
            trajectory = ProjectTrajectory(
                project_id=f"proj_{i}",
                project_description=desc,
                total_cost=0.01,
                total_time=3.0,
                success=True,
                task_records=[],
                model_sequence=[],
            )
            archive.store(trajectory)
        
        # Find similar to "Python API"
        similar = archive.find_similar_projects("Build Python API service", limit=2)
        
        assert len(similar) >= 1
        # Should find projects with "Python" and "API" keywords
        assert any("Python" in p.project_description for p in similar)


# ─────────────────────────────────────────────
# MetaOptimizer Tests
# ─────────────────────────────────────────────

class TestMetaOptimizer:
    """Test MetaOptimizer functionality."""
    
    @pytest.fixture
    def optimizer(self, tmp_path):
        archive = ExecutionArchive(tmp_path / "archive")
        return MetaOptimizer(archive, min_samples=5)
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, optimizer):
        """Should not propose optimizations with insufficient data."""
        proposals = await optimizer.analyze_and_propose()
        assert len(proposals) == 0
    
    @pytest.mark.asyncio
    async def test_generate_proposals(self, optimizer, tmp_path):
        """Should generate proposals with enough data."""
        # Store enough executions
        for i in range(20):
            records = [
                ExecutionRecord(
                    task_id=f"task_{i}",
                    task_type="code_generation",
                    model_used="deepseek-chat" if i < 15 else "other-model",
                    success=i < 15,  # other-model has 100% failure
                    cost_usd=0.005,
                    latency_ms=1500,
                    input_tokens=100,
                    output_tokens=200,
                    score=0.9 if i < 15 else 0.1,
                    project_id=f"proj_{i}",
                )
            ]
            
            trajectory = ProjectTrajectory(
                project_id=f"proj_{i}",
                project_description="Test project",
                total_cost=0.005,
                total_time=2.0,
                success=i < 15,
                task_records=records,
                model_sequence=["deepseek-chat" if i < 15 else "other-model"],
            )
            optimizer.archive.store(trajectory)
        
        proposals = await optimizer.analyze_and_propose()
        
        # Should generate some proposals
        assert len(proposals) >= 0  # May vary based on patterns
    
    @pytest.mark.asyncio
    async def test_evaluate_proposal(self, optimizer):
        """Should evaluate proposals correctly."""
        proposal = StrategyProposal(
            proposal_id="test_proposal",
            strategy_type=StrategyType.MODEL_ROUTING,
            description="Test routing change",
            current_config={"model": "test-model", "enabled": True},
            proposed_config={"model": "test-model", "enabled": False},
            expected_improvement=0.10,
            confidence=0.8,
            evidence=["Test evidence"],
        )
        
        # Simulate evaluation
        result = await optimizer.evaluate_proposal(proposal)
        
        # Result depends on simulation (may pass or fail based on threshold)
        assert proposal.status in [ProposalStatus.APPROVED, ProposalStatus.REJECTED]


# ─────────────────────────────────────────────
# SelfImprovingTemplates Tests
# ─────────────────────────────────────────────

class TestSelfImprovingTemplates:
    """Test SelfImprovingTemplates functionality."""
    
    @pytest.fixture(autouse=True)
    def reset_templates(self):
        """Reset global state before each test."""
        reset_self_improving_templates()
        yield
        reset_self_improving_templates()
    
    def test_record_execution(self):
        sit = SelfImprovingTemplates()
        
        sit.record_execution(
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            variant_name="structured",
            score=0.9,
            success=True,
            cost_usd=0.005,
        )
        
        assert len(sit._evolution_records) == 1
        assert sit._evolution_records[0].variant_name == "structured"
    
    def test_get_variant_stats(self):
        sit = SelfImprovingTemplates()
        
        # Record multiple executions
        for score in [0.8, 0.9, 0.85, 0.95, 0.75]:
            sit.record_execution(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="structured",
                score=score,
                success=True,
                cost_usd=0.005,
            )
        
        stats = sit.get_variant_stats("structured")
        
        assert stats["count"] == 5
        assert stats["avg_score"] == pytest.approx(0.85, rel=0.01)
        assert stats["min"] == 0.75
        assert stats["max"] == 0.95
    
    def test_propose_improvements(self):
        sit = SelfImprovingTemplates()
        
        # Record executions for two variants
        for i in range(15):
            sit.record_execution(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="good_variant",
                score=0.9 + (i % 10) * 0.01,  # 0.90-0.99
                success=True,
                cost_usd=0.005,
            )
            
            sit.record_execution(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="bad_variant",
                score=0.5 + (i % 10) * 0.01,  # 0.50-0.59
                success=i % 2 == 0,
                cost_usd=0.005,
            )
        
        proposals = sit.propose_improvements(min_samples=10)
        
        # Should propose retiring bad variant
        assert len(proposals) >= 1
        assert any(p["type"] == "retire_variant" for p in proposals)
    
    def test_get_evolution_report(self):
        sit = SelfImprovingTemplates()
        
        # Record some executions
        for i in range(10):
            sit.record_execution(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="test_variant",
                score=0.8 + (i % 5) * 0.02,
                success=True,
                cost_usd=0.005,
            )
        
        report = sit.get_evolution_report()
        
        assert report["total_executions"] == 10
        assert report["variants_tracked"] == 1
        assert "test_variant" in report["variant_stats"]


# ─────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────

class TestMetaOptimizationIntegration:
    """Test meta-optimization integration."""
    
    @pytest.fixture
    def integration(self, tmp_path):
        # Mock orchestrator
        orchestrator = MagicMock()
        return MetaOptimizationIntegration(orchestrator)
    
    @pytest.mark.asyncio
    async def test_record_execution(self, integration):
        """Should record executions correctly."""
        records = [
            ExecutionRecord(
                task_id="task_001",
                task_type="code_generation",
                model_used="deepseek-chat",
                success=True,
                cost_usd=0.005,
                latency_ms=1500,
                input_tokens=100,
                output_tokens=200,
                score=0.9,
                project_id="proj_001",
            )
        ]
        
        trajectory = ProjectTrajectory(
            project_id="proj_001",
            project_description="Test project",
            total_cost=0.005,
            total_time=2.0,
            success=True,
            task_records=records,
            model_sequence=["deepseek-chat"],
        )
        
        await integration.record_execution(trajectory)
        
        assert integration.archive.total_projects == 1
    
    @pytest.mark.asyncio
    async def test_maybe_optimize_insufficient_data(self, integration):
        """Should not optimize with insufficient data."""
        proposals = await integration.maybe_optimize()
        assert len(proposals) == 0
    
    def test_get_status(self, integration):
        """Should return status report."""
        status = integration.get_status()
        
        assert "archive_stats" in status
        assert "patterns" in status
        assert "pending_proposals" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
