"""
Test Ant Design Dashboard
=========================
Tests for the Ant Design dashboard components.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that dashboard components can be imported."""
    print("Testing imports...")
    
    try:
        from orchestrator.dashboard_antd import (
            AntDesignDashboardServer,
            run_ant_design_dashboard,
            DashboardState,
        )
        print("✅ Ant Design dashboard components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_dataclass():
    """Test DashboardState dataclass."""
    print("\nTesting DashboardState...")
    
    from orchestrator.dashboard_antd import DashboardState
    
    state = DashboardState(
        project={"name": "test", "status": "running"},
        architecture={"style": "microservices"},
        active_task={"task_id": "task_001"},
        models=[{"name": "gpt-4o"}],
        metrics={"calls": 100},
    )
    
    assert state.project["name"] == "test"
    assert state.architecture["style"] == "microservices"
    assert len(state.models) == 1
    
    print("✅ DashboardState created successfully")
    return True


def test_server_init():
    """Test server initialization."""
    print("\nTesting server initialization...")
    
    from orchestrator.dashboard_antd import AntDesignDashboardServer
    
    server = AntDesignDashboardServer(host="127.0.0.1", port=9999)
    
    assert server.host == "127.0.0.1"
    assert server.port == 9999
    assert server.app is not None
    
    print("✅ Server initialized successfully")
    return True


def test_html_generation():
    """Test HTML generation."""
    print("\nTesting HTML generation...")
    
    from orchestrator.dashboard_antd import AntDesignDashboardServer
    
    server = AntDesignDashboardServer()
    html = server._get_html()
    
    # Check for key components
    assert len(html) > 10000, "HTML too short"
    assert "Ant Design" in html, "Missing Ant Design reference"
    assert "antd@5." in html, "Missing Ant Design CSS"
    assert "react@18" in html, "Missing React"
    assert "Mission Control" in html, "Missing title"
    assert "ProjectCard" in html or "Project Overview" in html, "Missing project card"
    assert "Architecture" in html, "Missing architecture section"
    assert "Model Status" in html, "Missing model table"
    
    print(f"✅ HTML generated: {len(html)} bytes")
    return True


def test_mock_data():
    """Test mock data generation."""
    print("\nTesting mock data...")
    
    from orchestrator.dashboard_antd import AntDesignDashboardServer
    
    server = AntDesignDashboardServer()
    data = server._get_mock_data()
    
    # Check structure
    assert "project" in data
    assert "architecture" in data
    assert "active_task" in data
    assert "models" in data
    assert "metrics" in data
    
    # Check content
    assert data["project"]["status"] in ["idle", "running", "completed", "failed"]
    assert len(data["models"]) > 0
    assert all("name" in m for m in data["models"])
    assert all("provider" in m for m in data["models"])
    
    print(f"✅ Mock data generated: {len(data['models'])} models")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Ant Design Dashboard Tests")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_dataclass,
        test_server_init,
        test_html_generation,
        test_mock_data,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ All tests passed! Ant Design Dashboard is ready.")
        print("\nTo start the dashboard:")
        print("  python -c \"from orchestrator.dashboard_antd import run_ant_design_dashboard; run_ant_design_dashboard()\"")
        print("\nOr use the script:")
        print("  python scripts/run_dashboard.py")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
