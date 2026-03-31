"""
Test Live Dashboard
===================
Tests for the gamified live dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that dashboard components can be imported."""
    print("Testing imports...")

    try:
        from orchestrator.dashboard_live import (
            LiveDashboardServer,
            DashboardLiveIntegration,
            LiveTask,
            TestExecution,
            Achievement,
            DashboardState,
            run_live_dashboard,
        )

        print("✅ All live dashboard components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_dataclasses():
    """Test dataclass creation."""
    print("\nTesting dataclasses...")

    from orchestrator.dashboard_live import LiveTask, Achievement, DashboardState

    # Test LiveTask
    task = LiveTask(
        task_id="task_001",
        task_type="code_generation",
        prompt="Write a function",
        status="running",
        iteration=2,
        score=0.85,
    )
    assert task.task_id == "task_001"
    assert task.status == "running"
    print(f"  ✅ LiveTask: {task.task_id} (status={task.status})")

    # Test Achievement
    achievement = Achievement(
        id="first_task",
        title="Task Master",
        description="Complete first task",
        icon="🎯",
    )
    assert achievement.id == "first_task"
    assert achievement.icon == "🎯"
    print(f"  ✅ Achievement: {achievement.title} {achievement.icon}")

    # Test DashboardState
    state = DashboardState(
        project_id="test-123",
        project_status="running",
        level=5,
        xp=250,
    )
    assert state.project_id == "test-123"
    assert state.level == 5
    print(f"  ✅ DashboardState: Level {state.level}, XP {state.xp}")

    return True


def test_achievements():
    """Test achievement system."""
    print("\nTesting achievement system...")

    from orchestrator.dashboard_live import LiveDashboardServer

    server = LiveDashboardServer()

    # Check achievements initialized
    assert len(server.achievements_db) > 0
    assert "first_task" in server.achievements_db
    assert "speed_demon" in server.achievements_db

    # Check achievement properties
    achievement = server.achievements_db["first_task"]
    assert achievement.title == "Task Master"
    assert achievement.icon == "🎯"

    print(f"  ✅ {len(server.achievements_db)} achievements initialized")
    for key, ach in server.achievements_db.items():
        print(f"     - {ach.icon} {ach.title}")

    return True


def test_server_init():
    """Test server initialization."""
    print("\nTesting server initialization...")

    from orchestrator.dashboard_live import LiveDashboardServer

    server = LiveDashboardServer(host="127.0.0.1", port=9999)

    assert server.host == "127.0.0.1"
    assert server.port == 9999
    assert server.state is not None
    assert len(server.connections) == 0

    print("✅ Server initialized successfully")
    return True


def test_xp_system():
    """Test XP and leveling system."""
    print("\nTesting XP system...")

    from orchestrator.dashboard_live import LiveDashboardServer

    server = LiveDashboardServer()

    # Initial state
    assert server.state.level == 1
    assert server.state.xp == 0

    # Add XP (below threshold)
    server._add_xp(50)
    assert server.state.xp == 50
    assert server.state.level == 1  # Still level 1

    # Add more XP to level up
    server._add_xp(60)  # Total 110, exceeds 100
    assert server.state.level == 2
    assert server.state.xp == 10  # 110 - 100

    print(f"✅ XP system: Level {server.state.level}, XP {server.state.xp}")
    return True


def test_html_generation():
    """Test HTML generation."""
    print("\nTesting HTML generation...")

    from orchestrator.dashboard_live import LiveDashboardServer

    server = LiveDashboardServer()
    html = server._get_html()

    # Check for key components
    assert len(html) > 20000, "HTML too short"
    assert "WebSocket" in html or "ws://" in html, "Missing WebSocket"
    assert "confetti" in html.lower(), "Missing confetti"
    assert "gamification" in html.lower() or "xp" in html.lower(), "Missing gamification"
    assert "achievement" in html.lower(), "Missing achievements"
    assert "Mission Control LIVE" in html, "Missing title"
    assert "Level" in html, "Missing level display"

    print(f"✅ HTML generated: {len(html)} bytes")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Live Dashboard Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_dataclasses,
        test_achievements,
        test_server_init,
        test_xp_system,
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
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests passed! Live Dashboard is ready.")
        print("\n🎮 To start the gamified dashboard:")
        print(
            '  python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"'
        )
        print("\n🎯 Features:")
        print("  • WebSocket real-time updates")
        print("  • XP, levels, and achievements")
        print("  • Confetti celebrations")
        print("  • Sound effects")
        print("  • Toast notifications")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
