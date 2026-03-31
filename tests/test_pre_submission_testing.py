"""
Tests for Pre-Submission Automated Testing
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_pre_submission_testing.py -v
"""

import pytest
import tempfile
from pathlib import Path

from orchestrator.pre_submission_testing import (
    PreSubmissionTester,
    CheckType,
    CheckResult,
    ReviewResult,
    run_pre_submission_review,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def tester():
    """Create pre-submission tester."""
    return PreSubmissionTester()


@pytest.fixture
def temp_app_project():
    """Create temporary iOS app project structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app_path = Path(tmpdir)

        # Create basic iOS project structure
        (app_path / "MyApp.xcodeproj").mkdir()
        (app_path / "MyApp" / "AppDelegate.swift").parent.mkdir(parents=True)

        # Create AppDelegate
        app_delegate = app_path / "MyApp" / "AppDelegate.swift"
        app_delegate.write_text("""
import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        return true
    }
}
""")

        # Create ContentView
        content_view = app_path / "MyApp" / "ContentView.swift"
        content_view.write_text("""
import SwiftUI
import CoreData

struct ContentView: View {
    @Environment(\.managedObjectContext) private var viewContext
    
    var body: some View {
        TabView {
            Text("Home")
                .tabItem { Label("Home", systemImage: "house") }
            Text("Settings")
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .accessibilityLabel("Main navigation")
    }
}

class OfflineManager {
    static let shared = OfflineManager()
    func saveData() {}
}
""")

        # Create Info.plist
        info_plist = app_path / "MyApp" / "Info.plist"
        import plistlib

        # Create valid plist file using plistlib
        plist_data = {
            "CFBundleName": "MyApp",
            "CFBundleVersion": "1",
            "CFBundleShortVersionString": "1.0",
            "UISupportedInterfaceOrientations": ["UIInterfaceOrientationPortrait"],
            "NSCameraUsageDescription": "We need camera access",
        }
        with open(info_plist, "wb") as f:
            plistlib.dump(plist_data, f)

        yield app_path


@pytest.fixture
def problematic_app_project():
    """Create problematic iOS app project for testing failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app_path = Path(tmpdir)

        # Create basic structure
        (app_path / "MyApp.xcodeproj").mkdir()
        (app_path / "MyApp").mkdir()

        # Create problematic Swift file with placeholders and dynamic code
        bad_file = app_path / "MyApp" / "BadCode.swift"
        bad_file.write_text("""
import UIKit

class BadViewController: UIViewController {
    // TODO: Implement this
    // FIXME: This is broken
    // Lorem ipsum placeholder text
    
    func loadCode() {
        eval("print('hello')")
        let cls = NSClassFromString("UIView")
    }
    
    func fetchData() {
        // Hardcoded IPv4
        let url = "http://192.168.1.1/api"
    }
}
""")

        yield app_path


# ─────────────────────────────────────────────
# Test CheckResult
# ─────────────────────────────────────────────


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test check result creation."""
        result = CheckResult(
            check_type=CheckType.BUILD_VERIFICATION,
            passed=True,
            message="Build verified",
        )

        assert result.check_type == CheckType.BUILD_VERIFICATION
        assert result.passed is True
        assert result.message == "Build verified"

    def test_check_result_to_dict(self):
        """Test check result serialization."""
        result = CheckResult(
            check_type=CheckType.PRIVACY_COMPLIANCE,
            passed=False,
            message="Privacy issues found",
            details="Missing keys",
            severity="critical",
            guideline="5.1.1",
        )

        result_dict = result.to_dict()

        assert result_dict["check_type"] == "privacy_compliance"
        assert result_dict["passed"] is False
        assert result_dict["severity"] == "critical"


# ─────────────────────────────────────────────
# Test ReviewResult
# ─────────────────────────────────────────────


class TestReviewResult:
    """Test ReviewResult dataclass."""

    def test_review_result_creation(self):
        """Test review result creation."""
        checks = [
            CheckResult(
                check_type=CheckType.BUILD_VERIFICATION,
                passed=True,
                message="OK",
            )
        ]

        result = ReviewResult(
            passed=True,
            checks=checks,
            estimated_approval_probability=95.0,
        )

        assert result.passed is True
        assert len(result.checks) == 1
        assert result.estimated_approval_probability == 95.0

    def test_review_result_to_dict(self):
        """Test review result serialization."""
        checks = [
            CheckResult(
                check_type=CheckType.COMPLETENESS,
                passed=True,
                message="OK",
            )
        ]

        result = ReviewResult(
            passed=True,
            checks=checks,
            estimated_approval_probability=90.0,
        )

        result_dict = result.to_dict()

        assert result_dict["passed"] is True
        assert len(result_dict["checks"]) == 1
        assert "estimated_approval_probability" in result_dict


# ─────────────────────────────────────────────
# Test PreSubmissionTester
# ─────────────────────────────────────────────


class TestPreSubmissionTester:
    """Test PreSubmissionTester class."""

    def test_tester_initialization(self, tester):
        """Test tester initializes correctly."""
        assert tester is not None
        assert tester.MAX_LAUNCH_TIME == 3.0
        assert len(tester.PLACEHOLDER_PATTERNS) > 0

    @pytest.mark.asyncio
    async def test_verify_build_success(self, tester, temp_app_project):
        """Test build verification with valid project."""
        result = await tester._verify_build(temp_app_project)

        assert result.check_type == CheckType.BUILD_VERIFICATION
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_verify_build_no_project(self, tester):
        """Test build verification with no project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tester._verify_build(Path(tmpdir))

            assert result.check_type == CheckType.BUILD_VERIFICATION
            assert result.passed is False
            assert result.severity == "critical"

    @pytest.mark.asyncio
    async def test_launch_time_success(self, tester, temp_app_project):
        """Test launch time with valid project."""
        result = await tester._test_launch_time(temp_app_project)

        assert result.check_type == CheckType.LAUNCH_TIME
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_launch_time_no_app_delegate(self, tester):
        """Test launch time without AppDelegate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app_path = Path(tmpdir)
            (app_path / "MyApp.xcodeproj").mkdir()

            result = await tester._test_launch_time(app_path)

            assert result.check_type == CheckType.LAUNCH_TIME
            assert result.passed is False

    @pytest.mark.asyncio
    async def test_crash_test_success(self, tester, temp_app_project):
        """Test crash detection with clean code."""
        result = await tester._crash_test(temp_app_project)

        assert result.check_type == CheckType.CRASH_DETECTION
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_completeness_check_success(self, tester, temp_app_project):
        """Test completeness check with clean code."""
        result = await tester._completeness_check(temp_app_project)

        assert result.check_type == CheckType.COMPLETENESS
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_completeness_check_failures(self, tester, problematic_app_project):
        """Test completeness check with placeholders."""
        result = await tester._completeness_check(problematic_app_project)

        assert result.check_type == CheckType.COMPLETENESS
        assert result.passed is False
        assert "TODO" in result.details or "FIXME" in result.details

    @pytest.mark.asyncio
    async def test_privacy_audit_success(self, tester, temp_app_project):
        """Test privacy audit with valid Info.plist."""
        result = await tester._privacy_audit(temp_app_project)

        assert result.check_type == CheckType.PRIVACY_COMPLIANCE
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_hig_compliance_success(self, tester, temp_app_project):
        """Test HIG compliance with valid code."""
        result = await tester._hig_compliance_check(temp_app_project)

        assert result.check_type == CheckType.HIG_COMPLIANCE
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_offline_test_success(self, tester, temp_app_project):
        """Test offline support detection."""
        result = await tester._offline_test(temp_app_project)

        assert result.check_type == CheckType.NETWORK_INDEPENDENCE
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_ipv6_test_success(self, tester, temp_app_project):
        """Test IPv6 compatibility."""
        result = await tester._ipv6_test(temp_app_project)

        assert result.check_type == CheckType.IPV6_COMPATIBILITY
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_execution_scan_success(self, tester, temp_app_project):
        """Test code execution scan with clean code."""
        result = await tester._code_execution_scan(temp_app_project)

        assert result.check_type == CheckType.CODE_EXECUTION_SCAN
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_execution_scan_failures(self, tester, problematic_app_project):
        """Test code execution scan with dynamic code."""
        result = await tester._code_execution_scan(problematic_app_project)

        assert result.check_type == CheckType.CODE_EXECUTION_SCAN
        assert result.passed is False
        assert result.severity == "critical"
        assert "eval" in result.details.lower() or "NSClassFromString" in result.details

    @pytest.mark.asyncio
    async def test_metadata_validation_success(self, tester, temp_app_project):
        """Test metadata validation with valid Info.plist."""
        result = await tester._metadata_validation(temp_app_project)

        assert result.check_type == CheckType.METADATA_VALIDATION
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_full_review_success(self, tester, temp_app_project):
        """Test full review with valid project."""
        result = await tester.run_full_review(temp_app_project)

        assert isinstance(result, ReviewResult)
        assert result.passed is True
        assert len(result.checks) == 10
        assert result.estimated_approval_probability >= 80.0

    @pytest.mark.asyncio
    async def test_full_review_failures(self, tester, problematic_app_project):
        """Test full review with problematic project."""
        result = await tester.run_full_review(problematic_app_project)

        assert isinstance(result, ReviewResult)
        assert result.passed is False
        assert len(result.critical_issues) > 0
        assert result.estimated_approval_probability < 80.0

    def test_calculate_approval_probability(self, tester):
        """Test approval probability calculation."""
        # All passed
        results_all_pass = [
            CheckResult(check_type=CheckType.BUILD_VERIFICATION, passed=True, message="OK"),
            CheckResult(check_type=CheckType.COMPLETENESS, passed=True, message="OK"),
        ]
        prob = tester._calculate_approval_probability(results_all_pass)
        assert prob == 100.0

        # Mixed results
        results_mixed = [
            CheckResult(
                check_type=CheckType.BUILD_VERIFICATION,
                passed=True,
                message="OK",
                severity="critical",
            ),
            CheckResult(
                check_type=CheckType.COMPLETENESS, passed=False, message="Fail", severity="critical"
            ),
        ]
        prob = tester._calculate_approval_probability(results_mixed)
        assert prob == 50.0


# ─────────────────────────────────────────────
# Test Convenience Function
# ─────────────────────────────────────────────


class TestConvenienceFunction:
    """Test convenience function."""

    @pytest.mark.asyncio
    async def test_run_pre_submission_review(self, temp_app_project):
        """Test convenience function."""
        result = await run_pre_submission_review(temp_app_project)

        assert isinstance(result, ReviewResult)
        assert len(result.checks) == 10


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────


class TestIntegration:
    """Test integration with other modules."""

    @pytest.mark.asyncio
    async def test_integration_with_app_store_validator(self, temp_app_project):
        """Test integration with App Store validator."""
        from orchestrator.app_store_validator import AppStoreValidator

        tester = PreSubmissionTester()
        result = await tester.run_full_review(temp_app_project)

        # Result can be used with validator
        validator = AppStoreValidator()

        # Both should agree on critical issues
        assert result is not None
        assert validator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
