# App Store Compliance Validator Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Ensure generated apps pass Apple App Store, Google Play, and Web App review.** This validation layer checks generated code against store guidelines before delivery.

---

## Quick Start

### Basic Usage

```python
from orchestrator.app_store_validator import (
    validate_app_store_compliance,
    AppStorePlatform,
)

# Validate iOS app
result = await validate_app_store_compliance(
    project_path=Path("./my-ios-app"),
    platform=AppStorePlatform.IOS,
)

print(result.summary)
# Output: ❌ FAILED (Score: 0.7/1.0) | Passed: 14/20 | Violations: 3 | Warnings: 3

if not result.passed:
    print("Violations:")
    for v in result.violations:
        print(f"  • {v}")
```

### Integration with Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.app_store_validator import validate_app_store_compliance, AppStorePlatform

orch = Orchestrator()

async def run_project_with_validation():
    # Generate app
    state = await orch.run_project(
        project_description="Build iOS fitness tracking app",
        success_criteria="All tests pass",
    )
    
    # Validate for App Store
    result = await validate_app_store_compliance(
        project_path=Path(state.output_path),
        platform=AppStorePlatform.IOS,
    )
    
    if not result.passed:
        print(f"App Store validation failed: {result.violations}")
        # Request fixes before delivery
    
    return state, result
```

---

## Table of Contents

1. [Overview](#overview)
2. [Platform Support](#platform-support)
3. [Validation Checks](#validation-checks)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Fixing Violations](#fixing-violations)
7. [CI/CD Integration](#cicd-integration)

---

## Overview

The App Store Compliance Validator ensures generated apps meet store requirements **before** delivery, reducing rejection risk.

### What It Checks

| Category | Checks | Severity |
|----------|--------|----------|
| **Completeness (2.1)** | Placeholder text, beta labels, functional buttons | Critical |
| **Self-Contained (2.5.2)** | Dynamic code execution, remote code download | Critical |
| **Minimum Functionality (4.2)** | Native navigation, iOS features, offline support | Critical/Warning |
| **Privacy (5.1)** | Privacy policy, consent modals, account deletion | Critical |
| **Design (HIG)** | iOS-standard controls, Dark Mode, accessibility | Warning/Info |
| **AI Transparency** | AI content labeling, user awareness | Warning |

### Benefits

| Benefit | Impact |
|---------|--------|
| **Reduce Rejections** | Catch issues before submission |
| **Automated Compliance** | No manual checklist needed |
| **Detailed Reports** | Clear violation descriptions |
| **Multi-Platform** | iOS, Android, Web support |

---

## Platform Support

### iOS (Apple App Store)

```python
from orchestrator.app_store_validator import AppStorePlatform

result = await validate_app_store_compliance(
    project_path=Path("./ios-app"),
    platform=AppStorePlatform.IOS,
)
```

**Checks Applied:** 20+ checks covering:
- Apple App Store Review Guidelines
- Human Interface Guidelines (HIG)
- AI-generated content requirements

### Android (Google Play Store)

```python
result = await validate_app_store_compliance(
    project_path=Path("./android-app"),
    platform=AppStorePlatform.ANDROID,
)
```

**Checks Applied:** 10+ checks covering:
- Google Play Developer Program Policies
- Core functionality requirements
- Privacy policy requirements

### Web / PWA

```python
result = await validate_app_store_compliance(
    project_path=Path("./web-app"),
    platform=AppStorePlatform.WEB,
)
```

**Checks Applied:** 6+ checks covering:
- PWA requirements (manifest, service worker)
- Offline capability
- Privacy policy

---

## Validation Checks

### iOS Checks (Complete List)

#### 2.1 Completeness

| ID | Check | Severity |
|----|-------|----------|
| IOS-2.1-01 | No placeholder text (Lorem ipsum, TODO, FIXME) | Critical |
| IOS-2.1-02 | No 'coming soon' or 'beta' labels | Critical |
| IOS-2.1-03 | All screens have real content | Critical |
| IOS-2.1-04 | All links/buttons are functional | Warning |

#### 2.5.2 Self-Contained

| ID | Check | Severity |
|----|-------|----------|
| IOS-2.5.2-01 | No dynamic code execution (eval, exec, Function()) | Critical |
| IOS-2.5.2-02 | No remote code download at runtime | Critical |
| IOS-2.5.2-03 | No embedded WebView that changes app functionality | Warning |

#### 4.2 Minimum Functionality

| ID | Check | Severity |
|----|-------|----------|
| IOS-4.2-01 | Has native navigation (tab bar or navigation stack) | Critical |
| IOS-4.2-02 | Uses at least 2 native iOS features | Warning |
| IOS-4.2-03 | Has offline capability or graceful offline state | Warning |
| IOS-4.2-04 | Has splash/launch screen | Warning |

#### 5.1 Privacy

| ID | Check | Severity |
|----|-------|----------|
| IOS-5.1-01 | Has privacy policy URL | Critical |
| IOS-5.1-02 | Consent modal before AI data sharing | Critical |
| IOS-5.1-03 | Account deletion capability if login exists | Critical |

#### HIG Design

| ID | Check | Severity |
|----|-------|----------|
| IOS-HIG-01 | Uses iOS-standard controls (UIKit/SwiftUI patterns) | Warning |
| IOS-HIG-02 | Supports Dynamic Type (accessibility) | Info |
| IOS-HIG-03 | Supports Dark Mode | Info |

#### AI Transparency

| ID | Check | Severity |
|----|-------|----------|
| IOS-AI-01 | AI-generated content is labeled | Warning |
| IOS-AI-02 | User knows when interacting with AI | Warning |

---

## API Reference

### AppStorePlatform

```python
class AppStorePlatform(Enum):
    IOS = "ios"       # Apple App Store
    ANDROID = "android"  # Google Play Store
    WEB = "web"       # PWA / Web App
    MACOS = "macos"   # Mac App Store
```

### AppStoreComplianceResult

```python
@dataclass
class AppStoreComplianceResult:
    passed: bool
    platform: AppStorePlatform
    violations: List[str]
    warnings: List[str]
    info: List[str]
    score: float  # 0.0 - 1.0
    checks_performed: int
    checks_passed: int
    checks_failed: int
    auto_fixes_applied: List[str]
    metadata: Dict[str, Any]
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
```

### AppStoreValidator

```python
class AppStoreValidator:
    def __init__(self, auto_fix: bool = False)
    
    async def validate(
        self,
        project_path: Path,
        platform: AppStorePlatform = AppStorePlatform.IOS,
    ) -> AppStoreComplianceResult:
        """
        Validate project against app store guidelines.
        
        Args:
            project_path: Path to project directory
            platform: Target platform
            
        Returns:
            AppStoreComplianceResult
        """
```

### Convenience Function

```python
async def validate_app_store_compliance(
    project_path: Path,
    platform: AppStorePlatform = AppStorePlatform.IOS,
    auto_fix: bool = False,
) -> AppStoreComplianceResult:
    """
    Convenience function to validate app store compliance.
    """
```

---

## Examples

### Example 1: Basic iOS Validation

```python
from orchestrator.app_store_validator import validate_app_store_compliance, AppStorePlatform
from pathlib import Path

result = await validate_app_store_compliance(
    project_path=Path("./MyApp"),
    platform=AppStorePlatform.IOS,
)

print(f"Score: {result.score:.1%}")
print(f"Checks: {result.checks_passed}/{result.checks_performed}")

if not result.passed:
    print("\nCritical Violations:")
    for v in result.violations:
        print(f"  ❌ {v}")
    
    print("\nWarnings:")
    for w in result.warnings:
        print(f"  ⚠️ {w}")
```

### Example 2: Multi-Platform Validation

```python
from orchestrator.app_store_validator import AppStorePlatform

platforms = [
    AppStorePlatform.IOS,
    AppStorePlatform.ANDROID,
    AppStorePlatform.WEB,
]

for platform in platforms:
    result = await validate_app_store_compliance(
        project_path=Path("./my-app"),
        platform=platform,
    )
    
    print(f"\n{platform.value.upper()}: {result.summary}")
```

### Example 3: Validation with Auto-Fix

```python
from orchestrator.app_store_validator import AppStoreValidator

validator = AppStoreValidator(auto_fix=True)

result = await validator.validate(
    project_path=Path("./my-app"),
    platform=AppStorePlatform.IOS,
)

if result.auto_fixes_applied:
    print("Auto-fixes applied:")
    for fix in result.auto_fixes_applied:
        print(f"  ✓ {fix}")
```

### Example 4: Integration with Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.app_store_validator import validate_app_store_compliance, AppStorePlatform

async def generate_validated_app(description: str):
    orch = Orchestrator()
    
    # Generate app
    state = await orch.run_project(
        project_description=description,
        success_criteria="App Store ready",
    )
    
    # Validate
    result = await validate_app_store_compliance(
        project_path=Path(state.output_path),
        platform=AppStorePlatform.IOS,
    )
    
    if not result.passed:
        # Request revision
        revision_request = f"""
        App Store validation failed with {len(result.violations)} violations:
        
        {'\n'.join(result.violations)}
        
        Please fix these issues before resubmitting.
        """
        print(revision_request)
    
    return state, result
```

### Example 5: Custom Check Configuration

```python
from orchestrator.app_store_validator import (
    AppStoreValidator,
    ComplianceCheck,
    GuidelineCategory,
)

validator = AppStoreValidator()

# Add custom check
custom_check = ComplianceCheck(
    id="CUSTOM-001",
    category=GuidelineCategory.AI_TRANSPARENCY,
    guideline="Custom AI Policy",
    description="Must display AI model version",
    severity="warning",
    auto_fixable=True,
)

# Register custom check (implementation required)
# validator.register_check(custom_check, custom_check_handler)
```

---

## Fixing Violations

### Common Violations & Fixes

#### "No placeholder text detected"

**Problem:** Code contains TODO, FIXME, Lorem ipsum

**Fix:**
```swift
// ❌ Before
// TODO: Implement this
let text = "Lorem ipsum dolor sit amet"

// ✅ After
let text = "Welcome to our app!"
func processUserInput() {
    // Implementation complete
}
```

#### "No native navigation detected"

**Problem:** App lacks tab bar or navigation stack

**Fix:**
```swift
// ❌ Before - Single view
struct ContentView: View {
    var body: some View {
        Text("Hello")
    }
}

// ✅ After - With Navigation
struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("Hello")
                NavigationLink("Details", destination: DetailView())
            }
        }
    }
}
```

#### "No privacy policy URL found"

**Problem:** Missing privacy policy reference

**Fix:**
```swift
// Add to Info.plist
<key>NSPrivacyPolicyURL</key>
<string>https://example.com/privacy</string>

// Or in code
struct SettingsView: View {
    var body: some View {
        Link("Privacy Policy", destination: URL(string: "https://example.com/privacy")!)
    }
}
```

#### "No consent mechanism for data sharing"

**Problem:** AI features collect data without consent

**Fix:**
```swift
struct AIConsentModal: View {
    @State private var consentGiven = false
    
    var body: some View {
        VStack {
            Text("AI Feature Consent")
            Text("Your data will be processed by AI to provide suggestions.")
            
            Button("I Consent") {
                consentGiven = true
                enableAIFeatures()
            }
        }
    }
}
```

#### "AI-generated content not labeled"

**Problem:** AI output not clearly identified

**Fix:**
```swift
struct AIResponseView: View {
    var response: String
    
    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Image(systemName: "cpu")
                Text("AI-Generated Response")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(response)
        }
    }
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: App Store Validation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
      
      - name: Validate for App Store
        run: |
          python -c "
          import asyncio
          from orchestrator.app_store_validator import validate_app_store_compliance, AppStorePlatform
          from pathlib import Path
          
          async def validate():
              result = await validate_app_store_compliance(
                  project_path=Path('./my-app'),
                  platform=AppStorePlatform.IOS,
              )
              print(result.summary)
              if not result.passed:
                  print('Violations:')
                  for v in result.violations:
                      print(f'  - {v}')
                  exit(1)
          
          asyncio.run(validate())
          "
```

### Pre-commit Hook

```python
# .pre-commit-hooks/app_store_check.py
import asyncio
import sys
from pathlib import Path
from orchestrator.app_store_validator import validate_app_store_compliance, AppStorePlatform

def main():
    async def validate():
        result = await validate_app_store_compliance(
            project_path=Path("."),
            platform=AppStorePlatform.IOS,
        )
        
        print(result.summary)
        
        if not result.passed:
            print("\nBlocking violations:")
            for v in result.violations:
                print(f"  ❌ {v}")
            return 1
        
        return 0
    
    sys.exit(asyncio.run(validate()))
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: app-store-validation
        name: App Store Validation
        entry: python .pre-commit-hooks/app_store_check.py
        language: python
        pass_filenames: false
```

---

## Testing

### Run Tests

```bash
# Run all app store validator tests
pytest tests/test_app_store_validator.py -v

# Run specific test category
pytest tests/test_app_store_validator.py::TestSpecificChecks -v

# Run with coverage
pytest tests/test_app_store_validator.py --cov=orchestrator.app_store_validator
```

### Test Fixtures

```python
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def ios_project_dir():
    """Create test iOS project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        # Create test project structure
        yield project_path
```

---

## Troubleshooting

### False Positives

**Issue:** Validator reports violation that doesn't exist

**Solution:**
```python
# Check specific file
validator = AppStoreValidator()
result = await validator.validate(
    project_path=Path("./my-app"),
    platform=AppStorePlatform.IOS,
)

# Review metadata for details
print(result.metadata)
```

### Missing Checks

**Issue:** Need custom check for specific requirement

**Solution:**
```python
from orchestrator.app_store_validator import ComplianceCheck, GuidelineCategory

# Create custom check
custom_check = ComplianceCheck(
    id="CUSTOM-001",
    category=GuidelineCategory.AI_TRANSPARENCY,
    guideline="Custom Policy",
    description="Custom description",
    severity="warning",
)

# Implement check handler in AppStoreValidator
```

---

## Related Documentation

- [README.md](./README.md) — Installation and quick start
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities overview

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
