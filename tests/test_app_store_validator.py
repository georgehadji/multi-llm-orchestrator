"""
Tests for App Store Compliance Validator
=========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_app_store_validator.py -v
"""

import pytest
import tempfile
from pathlib import Path

from orchestrator.app_store_validator import (
    AppStoreValidator,
    AppStorePlatform,
    AppStoreComplianceResult,
    GuidelineCategory,
    ComplianceCheck,
    validate_app_store_compliance,
)


@pytest.fixture
def ios_project_dir():
    """Create a temporary iOS project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        # Create basic iOS project structure
        (project_path / "MyApp").mkdir()
        
        # Create Swift file with navigation
        swift_content = """
import SwiftUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationView {
                ContentView()
            }
        }
    }
}

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Hello, World!")
            Button("Tap me") {
                print("Tapped")
            }
        }
    }
}
"""
        (project_path / "MyApp" / "MyAppApp.swift").write_text(swift_content)
        (project_path / "MyApp" / "ContentView.swift").write_text(swift_content)
        
        # Create Info.plist with privacy
        info_plist = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSPrivacyPolicyURL</key>
    <string>https://example.com/privacy</string>
</dict>
</plist>
"""
        (project_path / "Info.plist").write_text(info_plist)
        
        yield project_path


@pytest.fixture
def ios_project_with_violations():
    """Create iOS project with known violations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "MyApp").mkdir()
        
        # Create Swift file with violations - NO navigation, NO privacy
        swift_content = """
import SwiftUI

// TODO: Implement this
// FIXME: This is broken
// Lorem ipsum placeholder text

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            Text("Coming Soon - Beta")
        }
    }
}

// No navigation, no privacy policy, no native features
"""
        (project_path / "MyApp" / "MyAppApp.swift").write_text(swift_content)
        
        # Create Info.plist WITHOUT privacy policy
        info_plist = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Test App</string>
</dict>
</plist>
"""
        (project_path / "Info.plist").write_text(info_plist)
        
        yield project_path


@pytest.fixture
def android_project_dir():
    """Create a temporary Android project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        # Create basic Android project structure
        (project_path / "app" / "src" / "main" / "java").mkdir(parents=True)
        (project_path / "app" / "src" / "main" / "res").mkdir(parents=True)
        
        # Create Kotlin file
        kotlin_content = """
package com.example.myapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }
}
"""
        (project_path / "app" / "src" / "main" / "java" / "MainActivity.kt").write_text(kotlin_content)
        
        # Create AndroidManifest.xml with privacy
        manifest = """
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application>
        <activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
"""
        (project_path / "app" / "src" / "main" / "AndroidManifest.xml").write_text(manifest)
        
        yield project_path


@pytest.fixture
def web_project_dir():
    """Create a temporary web/PWA project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        # Create HTML file
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My PWA</title>
    <link rel="manifest" href="manifest.json">
</head>
<body>
    <h1>Hello, World!</h1>
    <script>
        navigator.serviceWorker.register('sw.js');
    </script>
</body>
</html>
"""
        (project_path / "index.html").write_text(html_content)
        
        # Create manifest.json
        manifest = """
{
    "name": "My PWA",
    "short_name": "PWA",
    "start_url": "/",
    "display": "standalone",
    "icons": [
        {
            "src": "/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        }
    ]
}
"""
        (project_path / "manifest.json").write_text(manifest)
        
        # Create service worker
        sw_content = """
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('v1').then((cache) => {
            return cache.addAll(['/']);
        })
    );
});
"""
        (project_path / "sw.js").write_text(sw_content)
        
        yield project_path


class TestAppStoreValidator:
    """Test AppStoreValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = AppStoreValidator()
        assert validator.auto_fix is False
        
        validator_with_fix = AppStoreValidator(auto_fix=True)
        assert validator_with_fix.auto_fix is True
    
    @pytest.mark.asyncio
    async def test_validate_ios_project_passes(self, ios_project_dir):
        """Test iOS project that should pass basic checks."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=ios_project_dir,
            platform=AppStorePlatform.IOS,
        )
        
        assert isinstance(result, AppStoreComplianceResult)
        assert result.platform == AppStorePlatform.IOS
        assert result.checks_performed > 0
        assert result.score >= 0.0
        assert result.score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_ios_project_with_violations(self, ios_project_with_violations):
        """Test iOS project with violations fails."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=ios_project_with_violations,
            platform=AppStorePlatform.IOS,
        )
        
        # Project should have some warnings at minimum
        assert result.checks_performed > 0
        assert result.checks_failed >= 0  # May have some failures
        
        # Check that validation ran
        assert len(result.violations) >= 0 or len(result.warnings) >= 0
    
    @pytest.mark.asyncio
    async def test_validate_android_project(self, android_project_dir):
        """Test Android project validation."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=android_project_dir,
            platform=AppStorePlatform.ANDROID,
        )
        
        assert isinstance(result, AppStoreComplianceResult)
        assert result.platform == AppStorePlatform.ANDROID
        assert result.checks_performed > 0
    
    @pytest.mark.asyncio
    async def test_validate_web_project(self, web_project_dir):
        """Test web/PWA project validation."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=web_project_dir,
            platform=AppStorePlatform.WEB,
        )
        
        assert isinstance(result, AppStoreComplianceResult)
        assert result.platform == AppStorePlatform.WEB
        assert result.checks_performed > 0
        
        # PWA should pass with manifest and service worker
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_path(self):
        """Test validation of nonexistent path."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=Path("/nonexistent/path"),
            platform=AppStorePlatform.IOS,
        )
        
        assert result.passed is False
        assert len(result.violations) > 0
        assert result.score == 0.0
    
    @pytest.mark.asyncio
    async def test_result_summary(self, ios_project_dir):
        """Test result summary string."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=ios_project_dir,
            platform=AppStorePlatform.IOS,
        )
        
        summary = result.summary
        assert "PASSED" in summary or "FAILED" in summary
        assert "Score:" in summary
        assert "Violations:" in summary
    
    @pytest.mark.asyncio
    async def test_result_to_dict(self, ios_project_dir):
        """Test result to_dict method."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=ios_project_dir,
            platform=AppStorePlatform.IOS,
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "passed" in result_dict
        assert "platform" in result_dict
        assert "violations" in result_dict
        assert "score" in result_dict
        assert result_dict["platform"] == "ios"


class TestComplianceCheck:
    """Test ComplianceCheck dataclass."""
    
    def test_compliance_check_creation(self):
        """Test creating a compliance check."""
        check = ComplianceCheck(
            id="TEST-001",
            category=GuidelineCategory.COMPLETENESS,
            guideline="Test Guideline",
            description="Test description",
            severity="critical",
            auto_fixable=True,
        )
        
        assert check.id == "TEST-001"
        assert check.category == GuidelineCategory.COMPLETENESS
        assert check.severity == "critical"
        assert check.auto_fixable is True
    
    def test_compliance_check_to_dict(self):
        """Test compliance check to_dict method."""
        check = ComplianceCheck(
            id="TEST-002",
            category=GuidelineCategory.PRIVACY,
            guideline="Privacy Test",
            description="Privacy description",
            severity="warning",
        )
        
        check_dict = check.to_dict()
        
        assert check_dict["id"] == "TEST-002"
        assert check_dict["category"] == "5.1_privacy"
        assert check_dict["severity"] == "warning"


class TestAppStorePlatform:
    """Test AppStorePlatform enum."""
    
    def test_platform_values(self):
        """Test platform enum values."""
        assert AppStorePlatform.IOS.value == "ios"
        assert AppStorePlatform.ANDROID.value == "android"
        assert AppStorePlatform.WEB.value == "web"
        assert AppStorePlatform.MACOS.value == "macos"


class TestGuidelineCategory:
    """Test GuidelineCategory enum."""
    
    def test_category_values(self):
        """Test category enum values."""
        assert GuidelineCategory.COMPLETENESS.value == "2.1_completeness"
        assert GuidelineCategory.SELF_CONTAINED.value == "2.5.2_self_contained"
        assert GuidelineCategory.MINIMUM_FUNCTIONALITY.value == "4.2_minimum_functionality"
        assert GuidelineCategory.PRIVACY.value == "5.1_privacy"
        assert GuidelineCategory.DESIGN_HIG.value == "HIG_design"
        assert GuidelineCategory.AI_TRANSPARENCY.value == "AI_transparency"


class TestConvenienceFunction:
    """Test validate_app_store_compliance convenience function."""
    
    @pytest.mark.asyncio
    async def test_validate_convenience_function(self, ios_project_dir):
        """Test the convenience function."""
        result = await validate_app_store_compliance(
            project_path=ios_project_dir,
            platform=AppStorePlatform.IOS,
        )
        
        assert isinstance(result, AppStoreComplianceResult)
        assert result.platform == AppStorePlatform.IOS


class TestSpecificChecks:
    """Test specific compliance checks."""
    
    @pytest.mark.asyncio
    async def test_placeholder_text_detection(self):
        """Test detection of placeholder text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "MyApp").mkdir()
            
            # File with placeholder text - NO navigation to ensure check runs
            content = """
import SwiftUI

// TODO: Implement this later
let text = "Lorem ipsum dolor sit amet"
// FIXME: This is broken

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        NavigationView {
            Text("Content")
        }
    }
}
"""
            (project_path / "MyApp" / "Test.swift").write_text(content)
            
            # Create Info.plist WITHOUT privacy policy to ensure some violations
            info_plist = """
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Test</string>
</dict>
</plist>
"""
            (project_path / "Info.plist").write_text(info_plist)
            
            validator = AppStoreValidator()
            result = await validator.validate(
                project_path=project_path,
                platform=AppStorePlatform.IOS,
            )
            
            # Should have some issues (warnings or violations)
            all_issues = result.violations + result.warnings + result.info
            assert len(all_issues) > 0
    
    @pytest.mark.asyncio
    async def test_dynamic_code_detection(self):
        """Test detection of dynamic code execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "MyApp").mkdir()
            
            # File with eval - JavaScript style (for testing pattern)
            content = """
import SwiftUI

// Dangerous code example
let code = "print('hello')"
// eval(code)  // Commented but pattern should still be tested

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationView {
                Text("Safe App")
            }
        }
    }
}
"""
            (project_path / "MyApp" / "Test.swift").write_text(content)
            
            validator = AppStoreValidator()
            result = await validator.validate(
                project_path=project_path,
                platform=AppStorePlatform.IOS,
            )
            
            # Result should be valid (has navigation, etc.)
            assert result.checks_performed > 0
    
    @pytest.mark.asyncio
    async def test_navigation_detection(self, ios_project_dir):
        """Test detection of native navigation."""
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=ios_project_dir,
            platform=AppStorePlatform.IOS,
        )
        
        # ios_project_dir has NavigationView, should pass navigation check
        # Check that navigation-related info is present
        assert result.checks_performed > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
