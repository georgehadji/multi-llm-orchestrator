"""
Integration tests for Nash-Stable Orchestrator.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from orchestrator.nash_stable_orchestrator import (
    NashStableOrchestrator,
    NashStabilityMetrics,
)
from orchestrator.models import Model, TaskType
from orchestrator.feedback_loop import CodebaseFingerprint
from orchestrator.budget import Budget
from orchestrator.pareto_frontier import Objective


class TestNashStableOrchestrator:
    """Integration test suite for Nash-stable orchestrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create Nash-stable orchestrator."""
        return NashStableOrchestrator(
            budget=Budget(max_usd=5.0),
            org_id="test-org",
            privacy_budget=1.0,
        )
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.knowledge_graph is not None
        assert orchestrator.adaptive_templates is not None
        assert orchestrator.pareto_frontier is not None
        assert orchestrator.federated is not None
    
    @pytest.mark.asyncio
    async def test_get_model_recommendation(self, orchestrator):
        """Test getting model recommendations."""
        fingerprint = CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
            patterns=["repository"],
        )
        
        recommendations = await orchestrator.get_model_recommendation(
            task_type=TaskType.CODE_GEN,
            fingerprint=fingerprint,
            budget=5.0,
        )
        
        assert "recommendations" in recommendations
        assert "all_sources" in recommendations
        assert isinstance(recommendations["recommendations"], list)
    
    @pytest.mark.asyncio
    async def test_run_task_with_adaptive_template(self, orchestrator):
        """Test running task with adaptive template."""
        from orchestrator.models import Task
        
        task = Task(
            task_id="test-task",
            task_type=TaskType.CODE_GEN,
            description="Write a hello world function",
        )
        
        result = await orchestrator.run_task_with_adaptive_template(
            task=task,
            model=Model.DEEPSEEK_CHAT,
            context={"language": "python", "criteria": "Simple and clean"},
        )
        
        assert "template_name" in result
        assert "rendered_prompt" in result
        assert "selection_strategy" in result
    
    def test_update_metrics(self, orchestrator):
        """Test metrics updating."""
        # Initial state
        assert orchestrator.metrics.nash_stability_score == 0.0
        
        # Update
        orchestrator._update_metrics()
        
        # Should have some values
        assert orchestrator.metrics.knowledge_graph_nodes >= 0
        assert orchestrator.metrics.templates_tested >= 0
    
    def test_get_nash_stability_report(self, orchestrator):
        """Test getting stability report."""
        report = orchestrator.get_nash_stability_report()
        
        assert "nash_stability_score" in report
        assert "interpretation" in report
        assert "accumulated_assets" in report
        assert "competitive_moat" in report
        assert "switching_cost_analysis" in report
    
    @pytest.mark.asyncio
    async def test_compare_with_competitor(self, orchestrator):
        """Test competitor comparison."""
        comparison = await orchestrator.compare_with_competitor("competitor-x")
        
        assert "comparison" in comparison
        assert "our_platform" in comparison["comparison"]
        assert "competitor" in comparison["comparison"]
        assert "advantage_analysis" in comparison
    
    def test_interpret_stability_score(self, orchestrator):
        """Test stability score interpretation."""
        assert "Early stage" in orchestrator._interpret_stability_score(0.1)
        assert "Growing" in orchestrator._interpret_stability_score(0.3)
        assert "Moderate" in orchestrator._interpret_stability_score(0.5)
        assert "Strong" in orchestrator._interpret_stability_score(0.7)
        assert "Dominant" in orchestrator._interpret_stability_score(0.9)


class TestNashStabilityMetrics:
    """Test NashStabilityMetrics dataclass."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = NashStabilityMetrics()
        
        assert metrics.knowledge_graph_nodes == 0
        assert metrics.nash_stability_score == 0.0
    
    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = NashStabilityMetrics(
            knowledge_graph_nodes=100,
            patterns_learned=50,
            nash_stability_score=0.75,
        )
        
        data = metrics.to_dict()
        assert data["knowledge_graph"]["nodes"] == 100
        assert data["knowledge_graph"]["patterns_learned"] == 50
        assert data["nash_stability_score"] == 0.75


class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with all features."""
        # Create orchestrator
        orch = NashStableOrchestrator(
            budget=Budget(max_usd=2.0),
            org_id="integration-test",
        )
        
        # 1. Get recommendations before any usage
        initial_recs = await orch.get_model_recommendation(
            task_type=TaskType.CODE_GEN,
        )
        
        # 2. Use adaptive template
        from orchestrator.models import Task
        task = Task(
            task_id="test",
            task_type=TaskType.CODE_GEN,
            description="Create a function",
        )
        
        template_result = await orch.run_task_with_adaptive_template(
            task=task,
            model=Model.DEEPSEEK_CHAT,
            context={"language": "python"},
        )
        
        # 3. Report results to improve templates
        await orch.adaptive_templates.report_result(
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            variant_name=template_result["template_name"],
            score=0.9,
            success=True,
        )
        
        # 4. Get Pareto frontier
        frontier = await orch.pareto_frontier.get_pareto_frontier(
            task_type=TaskType.CODE_GEN,
            objectives=[Objective.QUALITY, Objective.COST],
        )
        
        # 5. Get stability report
        report = orch.get_nash_stability_report()
        
        # Verify everything worked
        assert template_result is not None
        assert frontier is not None
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
