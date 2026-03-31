"""
Test Enhanced Dashboard Components
===================================
Quick tests to verify the enhanced dashboard works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all dashboard components can be imported."""
    print("Testing imports...")

    try:
        from orchestrator.dashboard_enhanced import (
            EnhancedDashboardServer,
            EnhancedDataProvider,
            DashboardIntegration,
            ArchitectureInfo,
            ProjectInfo,
            ActiveTaskInfo,
            ModelStatusInfo,
            run_enhanced_dashboard,
        )

        print("✅ All dashboard components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_architecture_info():
    """Test ArchitectureInfo dataclass."""
    print("\nTesting ArchitectureInfo...")

    from orchestrator.dashboard_enhanced import ArchitectureInfo

    arch = ArchitectureInfo(
        style="microservices",
        paradigm="object_oriented",
        api_style="rest",
        database_type="relational",
        primary_language="python",
        frameworks=["fastapi", "pydantic"],
        libraries=["uvicorn", "httpx"],
        databases=["postgresql"],
        tools=["pytest", "black"],
        constraints=["Max complexity 10"],
        patterns=["CQRS", "Event Sourcing"],
        rationale="Scalability requirements",
    )

    assert arch.style == "microservices"
    assert len(arch.frameworks) == 2
    assert "CQRS" in arch.patterns

    print(f"✅ ArchitectureInfo created: {arch.style} / {arch.primary_language}")
    return True


def test_project_info():
    """Test ProjectInfo dataclass."""
    print("\nTesting ProjectInfo...")

    from orchestrator.dashboard_enhanced import ProjectInfo

    project = ProjectInfo(
        project_id="test-123",
        description="Test project",
        success_criteria="All tests pass",
        status="running",
        total_tasks=10,
        completed_tasks=5,
        progress_percent=50.0,
        budget_used=2.5,
        budget_total=5.0,
    )

    assert project.project_id == "test-123"
    assert project.progress_percent == 50.0

    print(f"✅ ProjectInfo created: {project.completed_tasks}/{project.total_tasks} tasks")
    return True


def test_active_task_info():
    """Test ActiveTaskInfo dataclass."""
    print("\nTesting ActiveTaskInfo...")

    from orchestrator.dashboard_enhanced import ActiveTaskInfo

    task = ActiveTaskInfo(
        task_id="task_001",
        task_type="code_generation",
        prompt="Write a function",
        status="running",
        iteration=2,
        max_iterations=3,
        score=0.85,
        model_used="gpt-4o",
    )

    assert task.task_id == "task_001"
    assert task.score == 0.85

    print(f"✅ ActiveTaskInfo created: {task.task_id} (score: {task.score})")
    return True


def test_model_status_info():
    """Test ModelStatusInfo dataclass."""
    print("\nTesting ModelStatusInfo...")

    from orchestrator.dashboard_enhanced import ModelStatusInfo

    model = ModelStatusInfo(
        name="gpt-4o",
        provider="openai",
        available=True,
        health_status="healthy",
        success_rate=0.98,
        avg_latency=120,
        call_count=150,
    )

    assert model.name == "gpt-4o"
    assert model.available is True

    print(f"✅ ModelStatusInfo created: {model.name} ({model.provider})")
    return True


def test_data_provider():
    """Test EnhancedDataProvider initialization."""
    print("\nTesting EnhancedDataProvider...")

    from orchestrator.dashboard_enhanced import EnhancedDataProvider

    provider = EnhancedDataProvider()

    assert provider._cache_ttl == 3
    assert provider._current_project_id is None

    print("✅ EnhancedDataProvider initialized")
    return True


def test_dashboard_integration():
    """Test DashboardIntegration with mock data."""
    print("\nTesting DashboardIntegration...")

    from orchestrator.dashboard_enhanced import (
        DashboardIntegration,
        EnhancedDataProvider,
    )
    from orchestrator.models import Model

    provider = EnhancedDataProvider()
    integration = DashboardIntegration(provider)

    # Test model notifications
    integration.on_model_success(Model.GPT_4O)
    integration.on_model_failure(Model.DEEPSEEK_CHAT)

    # Verify failure was recorded
    assert provider._consecutive_failures[Model.DEEPSEEK_CHAT] == 1

    print("✅ DashboardIntegration working")
    return True


async def test_async_methods():
    """Test async methods of data provider."""
    print("\nTesting async methods...")

    from orchestrator.dashboard_enhanced import EnhancedDataProvider

    provider = EnhancedDataProvider()

    # Test metrics
    metrics = await provider.get_metrics()
    assert "total_calls" in metrics
    assert "timestamp" in metrics

    print(f"✅ Async methods working: {metrics['total_calls']} calls")
    return True


def test_html_generation():
    """Test that HTML is generated correctly."""
    print("\nTesting HTML generation...")

    from orchestrator.dashboard_enhanced import EnhancedDashboardServer

    server = EnhancedDashboardServer(host="127.0.0.1", port=9999)
    html = server._get_html()

    assert len(html) > 10000
    assert "MISSION CONTROL" in html
    assert "Architecture" in html
    assert "Project" in html
    assert "Models Status" in html

    print(f"✅ HTML generated: {len(html)} bytes")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Enhanced Dashboard Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_architecture_info,
        test_project_info,
        test_active_task_info,
        test_model_status_info,
        test_data_provider,
        test_dashboard_integration,
        test_html_generation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    # Run async tests
    async_tests = [test_async_methods]
    for test in async_tests:
        try:
            if asyncio.run(test()):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests passed! Dashboard is ready to use.")
        print("\nTo start the dashboard:")
        print(
            '  python -c "from orchestrator.dashboard_enhanced import run_enhanced_dashboard; run_enhanced_dashboard()"'
        )

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
