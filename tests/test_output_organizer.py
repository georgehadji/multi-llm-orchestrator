"""
Test Output Organizer
=====================
Quick tests to verify the output organizer works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all organizer components can be imported."""
    print("Testing imports...")

    try:
        from orchestrator.output_organizer import (
            OutputOrganizer,
            organize_project_output,
            OrganizationReport,
            TestResult,
            suppress_cache_messages,
            CacheMessageSuppressor,
        )

        print("✅ All organizer components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_dataclasses():
    """Test dataclass creation."""
    print("\nTesting dataclasses...")

    from orchestrator.output_organizer import TestResult, OrganizationReport

    # Test TestResult
    result = TestResult(
        test_file="test_example.py",
        passed=True,
        duration_ms=125.5,
        output="test output",
        coverage_percent=85.0,
    )
    assert result.test_file == "test_example.py"
    assert result.passed is True
    assert result.coverage_percent == 85.0
    print(f"  ✅ TestResult: {result.test_file} (passed={result.passed})")

    # Test OrganizationReport
    report = OrganizationReport()
    report.tasks_moved = ["task_001.py", "task_002.py"]
    report.tests_run = [result]

    assert len(report.tasks_moved) == 2
    assert len(report.tests_run) == 1
    print(
        f"  ✅ OrganizationReport: {len(report.tasks_moved)} tasks, {len(report.tests_run)} tests"
    )

    return True


def test_report_serialization():
    """Test report serialization."""
    print("\nTesting report serialization...")

    from orchestrator.output_organizer import OrganizationReport, TestResult

    report = OrganizationReport()
    report.tasks_moved = ["task_001.py"]
    report.tests_created = ["test_main.py"]
    report.tests_run = [
        TestResult(
            test_file="test_main.py",
            passed=True,
            duration_ms=100.0,
            output="OK",
            coverage_percent=80.0,
        )
    ]

    data = report.to_dict()

    assert "tasks_moved" in data
    assert "tests_run" in data
    assert "summary" in data
    assert data["summary"]["total_tests"] == 1
    assert data["summary"]["passed_tests"] == 1

    print(f"  ✅ Report serialized: {data['summary']}")
    return True


def test_cache_suppressor():
    """Test cache message suppressor."""
    print("\nTesting cache suppressor...")

    import logging
    from orchestrator.output_organizer import CacheMessageSuppressor, suppress_cache_messages

    # Test context manager
    logger = logging.getLogger("orchestrator.cache")
    original_level = logger.level

    with CacheMessageSuppressor():
        # During suppression, level should be WARNING
        assert logger.level == logging.WARNING

    # After context, level restored
    assert logger.level == original_level
    print("  ✅ CacheMessageSuppressor works")

    # Test global suppression (should not raise)
    try:
        suppress_cache_messages()
        print("  ✅ suppress_cache_messages() called successfully")
    except Exception as e:
        print(f"  ⚠️ suppress_cache_messages() raised: {e}")

    return True


def test_organizer_init():
    """Test OutputOrganizer initialization."""
    print("\nTesting OutputOrganizer initialization...")

    from orchestrator.output_organizer import OutputOrganizer

    organizer = OutputOrganizer(
        output_dir=Path("./test_output"),
        auto_generate_tests=True,
        run_tests=True,
        min_coverage=80.0,
    )

    assert organizer.output_dir == Path("./test_output")
    assert organizer.auto_generate_tests is True
    assert organizer.run_tests is True
    assert organizer.min_coverage == 80.0

    print(f"  ✅ OutputOrganizer initialized: auto_generate={organizer.auto_generate_tests}")
    return True


def test_is_test_file():
    """Test test file detection."""
    print("\nTesting test file detection...")

    from orchestrator.output_organizer import OutputOrganizer

    organizer = OutputOrganizer(Path("."))

    # Test files
    assert organizer._is_test_file(Path("test_main.py")) is True
    assert organizer._is_test_file(Path("main_test.py")) is True
    assert organizer._is_test_file(Path("tests/unit/test_main.py")) is True

    # Non-test files
    assert organizer._is_test_file(Path("main.py")) is False
    assert organizer._is_test_file(Path("utils.py")) is False

    print("  ✅ Test file detection works")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Output Organizer Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_dataclasses,
        test_report_serialization,
        test_cache_suppressor,
        test_organizer_init,
        test_is_test_file,
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
        print("\n✅ All tests passed! Output Organizer is ready to use.")
        print("\nTo use in your project:")
        print("  from orchestrator.output_organizer import OutputOrganizer")
        print('  organizer = OutputOrganizer(Path("./output"))')
        print("  await organizer.organize_project()")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
