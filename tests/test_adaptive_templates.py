"""
Tests for Adaptive Prompt Template System.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import os
import stat
import logging

logger = logging.getLogger(__name__)


def _handle_remove_readonly(func, path, exc_info):
    """Handle permission errors when removing temp dirs."""
    exc_type, exc_value, _ = exc_info
    if not issubclass(exc_type, PermissionError):
        raise exc_value

    # Attempt to make the path writable before retrying
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        logger.debug("Failed to chmod %s before cleanup", path)

    try:
        func(path)
    except Exception as inner_exc:
        logger.warning("Cleanup still failed for %s: %s", path, inner_exc)

from orchestrator.adaptive_templates import (
    AdaptiveTemplateSystem,
    TemplateVariant,
    TemplateStyle,
    TemplatePerformance,
    ContextProfile,
)
from orchestrator.models import Model, TaskType


class TestAdaptiveTemplateSystem:
    """Test suite for adaptive template system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, onerror=_handle_remove_readonly)

    @pytest.fixture
    def ats(self, temp_dir):
        """Create adaptive template system."""
        return AdaptiveTemplateSystem(storage_path=temp_dir)
    
    def test_initialization(self, ats):
        """Test system initialization."""
        assert ats is not None
        # Should have default templates
        variants = ats._get_variants_for_task(TaskType.CODE_GEN)
        assert len(variants) > 0
    
    def test_register_variant(self, ats):
        """Test registering a template variant."""
        variant = TemplateVariant(
            name="test_variant",
            template="Test: {task}",
            style=TemplateStyle.CONCISE,
        )
        
        ats.register_variant(TaskType.CODE_GEN, variant)
        
        variants = ats._get_variants_for_task(TaskType.CODE_GEN)
        assert any(v.name == "test_variant" for v in variants)
    
    def test_template_render(self):
        """Test template rendering."""
        variant = TemplateVariant(
            name="test",
            template="Write {language} code for: {task}",
        )
        
        rendered = variant.render(language="python", task="hello world")
        assert "python" in rendered
        assert "hello world" in rendered
    
    def test_context_profile_similarity(self):
        """Test context profile similarity."""
        cp1 = ContextProfile(
            language="python",
            framework="fastapi",
            complexity="medium",
        )
        cp2 = ContextProfile(
            language="python",
            framework="fastapi",
            complexity="medium",
        )
        cp3 = ContextProfile(
            language="rust",
            framework="actix",
            complexity="high",
        )
        
        sim_same = cp1.similarity(cp2)
        sim_diff = cp1.similarity(cp3)
        
        assert sim_same > sim_diff
        assert sim_same == 1.0  # Identical
    
    @pytest.mark.asyncio
    async def test_select_template(self, ats):
        """Test template selection."""
        variant, metadata = await ats.select_template(
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            context={"language": "python"},
        )
        
        assert variant is not None
        assert "strategy" in metadata
        assert variant.template is not None
    
    @pytest.mark.asyncio
    async def test_report_result(self, ats):
        """Test reporting template results."""
        await ats.report_result(
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            variant_name="concise",
            score=0.85,
            success=True,
        )
        
        # Check performance was recorded
        perf = ats._performance.get((TaskType.CODE_GEN, Model.DEEPSEEK_CHAT, "concise"))
        assert perf is not None
        assert perf.total_uses == 1
    
    @pytest.mark.asyncio
    async def test_template_convergence(self, ats):
        """Test that templates converge to best performer."""
        # Report multiple results favoring one variant
        for _ in range(20):
            await ats.report_result(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="structured",
                score=0.9,
                success=True,
            )
            await ats.report_result(
                task_type=TaskType.CODE_GEN,
                model=Model.DEEPSEEK_CHAT,
                variant_name="concise",
                score=0.6,
                success=True,
            )
        
        # Get stats
        stats = ats.get_template_stats(
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
        )
        
        assert stats["total_performance_records"] > 0
        
        # Check top performer is structured
        if stats["top_performers"]:
            top = stats["top_performers"][0]
            assert top["variant"] == "structured"
            assert top["ema_score"] > 0.8
    
    def test_get_template_stats(self, ats):
        """Test getting template statistics."""
        stats = ats.get_template_stats()
        
        assert "total_variants" in stats
        assert "by_task_type" in stats
        assert "top_performers" in stats


class TestTemplatePerformance:
    """Test TemplatePerformance class."""
    
    def test_update_score(self):
        """Test score updating with EMA."""
        perf = TemplatePerformance(
            variant_name="test",
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
        )
        
        # Update with high scores
        for _ in range(10):
            perf.update_score(0.9, success=True)
        
        assert perf.ema_score > 0.5  # Should have increased
        assert perf.total_uses == 10
        assert perf.confidence > 0.1
    
    def test_success_rate(self):
        """Test success rate calculation."""
        perf = TemplatePerformance(
            variant_name="test",
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            total_uses=10,
            total_successes=8,
            total_failures=2,
        )
        
        assert perf.success_rate == 0.8
    
    def test_score_variance(self):
        """Test variance calculation."""
        perf = TemplatePerformance(
            variant_name="test",
            task_type=TaskType.CODE_GEN,
            model=Model.DEEPSEEK_CHAT,
            scores=[0.8, 0.85, 0.9, 0.82, 0.88],
        )
        
        var = perf.score_variance
        assert var > 0  # Should have some variance
        assert var < 1


class TestTemplateVariant:
    """Test TemplateVariant class."""
    
    def test_get_hash(self):
        """Test template hashing."""
        variant = TemplateVariant(
            name="test",
            template="Hello {name}",
        )
        
        hash1 = variant.get_hash()
        hash2 = variant.get_hash()
        
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 12
    
    def test_render_missing_variable(self):
        """Test rendering with missing variables."""
        variant = TemplateVariant(
            name="test",
            template="Hello {name}, your task is {task}",
        )
        
        # Should handle missing variables gracefully
        rendered = variant.render(name="World")
        assert "Hello World" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
