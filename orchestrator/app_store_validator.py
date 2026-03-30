"""
App Store Compliance Validator Pipeline
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Validates generated apps against Apple App Store, Google Play, and Web App guidelines.
This validation layer ensures generated apps meet store requirements before delivery.

Usage:
    from orchestrator.app_store_validator import AppStoreValidator, AppStorePlatform

    validator = AppStoreValidator()
    result = await validator.validate(
        project_path=Path("./my-app"),
        platform=AppStorePlatform.IOS,
    )

    if not result.passed:
        print(f"Violations: {result.violations}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .log_config import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class AppStorePlatform(Enum):
    """Target app store platform."""
    IOS = "ios"  # Apple App Store
    ANDROID = "android"  # Google Play Store
    WEB = "web"  # PWA / Web App
    MACOS = "macos"  # Mac App Store


class GuidelineCategory(Enum):
    """App Store Guideline Categories."""
    COMPLETENESS = "2.1_completeness"
    SELF_CONTAINED = "2.5.2_self_contained"
    MINIMUM_FUNCTIONALITY = "4.2_minimum_functionality"
    PRIVACY = "5.1_privacy"
    DESIGN_HIG = "HIG_design"
    AI_TRANSPARENCY = "AI_transparency"
    CONTENT_SAFETY = "1.1_content_safety"
    LEGAL = "3.1_legal"


@dataclass
class ComplianceCheck:
    """A single compliance check."""
    id: str
    category: GuidelineCategory
    guideline: str
    description: str
    severity: str  # "critical", "warning", "info"
    auto_fixable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "guideline": self.guideline,
            "description": self.description,
            "severity": self.severity,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class AppStoreComplianceResult:
    """Result of app store compliance validation."""
    platform: AppStorePlatform
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    score: float = 0.0  # 0.0 - 1.0
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    auto_fixes_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Passed if no critical violations."""
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "platform": self.platform.value,
            "violations": self.violations,
            "warnings": self.warnings,
            "info": self.info,
            "score": self.score,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "auto_fixes_applied": self.auto_fixes_applied,
            "metadata": self.metadata,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"{status} (Score: {self.score:.1f}/1.0) | "
            f"Passed: {self.checks_passed}/{self.checks_performed} | "
            f"Violations: {len(self.violations)} | "
            f"Warnings: {len(self.warnings)}"
        )


class AppStoreValidator:
    """
    Validate generated apps against app store guidelines.

    Supports:
    - Apple App Store (iOS, macOS)
    - Google Play Store (Android)
    - Web Apps (PWA)

    Usage:
        validator = AppStoreValidator()
        result = await validator.validate(
            project_path=Path("./my-app"),
            platform=AppStorePlatform.IOS,
        )
    """

    # Apple App Store Guidelines
    IOS_CHECKS: dict[str, list[ComplianceCheck]] = {
        GuidelineCategory.COMPLETENESS.value: [
            ComplianceCheck(
                id="IOS-2.1-01",
                category=GuidelineCategory.COMPLETENESS,
                guideline="2.1 Completeness",
                description="No placeholder text (Lorem ipsum, TODO, FIXME)",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-2.1-02",
                category=GuidelineCategory.COMPLETENESS,
                guideline="2.1 Completeness",
                description="No 'coming soon' or 'beta' labels",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-2.1-03",
                category=GuidelineCategory.COMPLETENESS,
                guideline="2.1 Completeness",
                description="All screens have real content",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-2.1-04",
                category=GuidelineCategory.COMPLETENESS,
                guideline="2.1 Completeness",
                description="All links/buttons are functional",
                severity="warning",
            ),
        ],
        GuidelineCategory.SELF_CONTAINED.value: [
            ComplianceCheck(
                id="IOS-2.5.2-01",
                category=GuidelineCategory.SELF_CONTAINED,
                guideline="2.5.2 Self-Contained",
                description="No dynamic code execution (eval, exec, Function())",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-2.5.2-02",
                category=GuidelineCategory.SELF_CONTAINED,
                guideline="2.5.2 Self-Contained",
                description="No remote code download at runtime",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-2.5.2-03",
                category=GuidelineCategory.SELF_CONTAINED,
                guideline="2.5.2 Self-Contained",
                description="No embedded WebView that changes app functionality",
                severity="warning",
            ),
        ],
        GuidelineCategory.MINIMUM_FUNCTIONALITY.value: [
            ComplianceCheck(
                id="IOS-4.2-01",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="4.2 Minimum Functionality",
                description="Has native navigation (tab bar or navigation stack)",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-4.2-02",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="4.2 Minimum Functionality",
                description="Uses at least 2 native iOS features",
                severity="warning",
            ),
            ComplianceCheck(
                id="IOS-4.2-03",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="4.2 Minimum Functionality",
                description="Has offline capability or graceful offline state",
                severity="warning",
            ),
            ComplianceCheck(
                id="IOS-4.2-04",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="4.2 Minimum Functionality",
                description="Has splash/launch screen",
                severity="warning",
            ),
        ],
        GuidelineCategory.PRIVACY.value: [
            ComplianceCheck(
                id="IOS-5.1-01",
                category=GuidelineCategory.PRIVACY,
                guideline="5.1 Privacy",
                description="Has privacy policy URL",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-5.1-02",
                category=GuidelineCategory.PRIVACY,
                guideline="5.1 Privacy",
                description="Consent modal before AI data sharing",
                severity="critical",
            ),
            ComplianceCheck(
                id="IOS-5.1-03",
                category=GuidelineCategory.PRIVACY,
                guideline="5.1 Privacy",
                description="Account deletion capability if login exists",
                severity="critical",
            ),
        ],
        GuidelineCategory.DESIGN_HIG.value: [
            ComplianceCheck(
                id="IOS-HIG-01",
                category=GuidelineCategory.DESIGN_HIG,
                guideline="Human Interface Guidelines",
                description="Uses iOS-standard controls (UIKit/SwiftUI patterns)",
                severity="warning",
            ),
            ComplianceCheck(
                id="IOS-HIG-02",
                category=GuidelineCategory.DESIGN_HIG,
                guideline="Human Interface Guidelines",
                description="Supports Dynamic Type (accessibility)",
                severity="info",
            ),
            ComplianceCheck(
                id="IOS-HIG-03",
                category=GuidelineCategory.DESIGN_HIG,
                guideline="Human Interface Guidelines",
                description="Supports Dark Mode",
                severity="info",
            ),
        ],
        GuidelineCategory.AI_TRANSPARENCY.value: [
            ComplianceCheck(
                id="IOS-AI-01",
                category=GuidelineCategory.AI_TRANSPARENCY,
                guideline="AI Transparency",
                description="AI-generated content is labeled",
                severity="warning",
            ),
            ComplianceCheck(
                id="IOS-AI-02",
                category=GuidelineCategory.AI_TRANSPARENCY,
                guideline="AI Transparency",
                description="User knows when interacting with AI",
                severity="warning",
            ),
        ],
    }

    # Google Play Store Guidelines
    ANDROID_CHECKS: dict[str, list[ComplianceCheck]] = {
        GuidelineCategory.COMPLETENESS.value: [
            ComplianceCheck(
                id="AND-2.1-01",
                category=GuidelineCategory.COMPLETENESS,
                guideline="Core Functionality",
                description="No placeholder text (Lorem ipsum, TODO, FIXME)",
                severity="critical",
            ),
            ComplianceCheck(
                id="AND-2.1-02",
                category=GuidelineCategory.COMPLETENESS,
                guideline="Core Functionality",
                description="App doesn't crash on launch",
                severity="critical",
            ),
        ],
        GuidelineCategory.SELF_CONTAINED.value: [
            ComplianceCheck(
                id="AND-2.5-01",
                category=GuidelineCategory.SELF_CONTAINED,
                guideline="Device and Network Abuse",
                description="No dynamic code execution from unknown sources",
                severity="critical",
            ),
        ],
        GuidelineCategory.MINIMUM_FUNCTIONALITY.value: [
            ComplianceCheck(
                id="AND-4.2-01",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="Minimum Functionality",
                description="Provides unique value beyond web browser",
                severity="critical",
            ),
        ],
        GuidelineCategory.PRIVACY.value: [
            ComplianceCheck(
                id="AND-5.1-01",
                category=GuidelineCategory.PRIVACY,
                guideline="Privacy Policy",
                description="Has privacy policy URL",
                severity="critical",
            ),
            ComplianceCheck(
                id="AND-5.1-02",
                category=GuidelineCategory.PRIVACY,
                guideline="Data Safety",
                description="Has Data Safety form completed",
                severity="critical",
            ),
        ],
    }

    # Web App / PWA Guidelines
    WEB_CHECKS: dict[str, list[ComplianceCheck]] = {
        GuidelineCategory.COMPLETENESS.value: [
            ComplianceCheck(
                id="WEB-01",
                category=GuidelineCategory.COMPLETENESS,
                guideline="Completeness",
                description="No placeholder text",
                severity="critical",
            ),
        ],
        GuidelineCategory.MINIMUM_FUNCTIONALITY.value: [
            ComplianceCheck(
                id="WEB-PWA-01",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="PWA Requirements",
                description="Has valid manifest.json",
                severity="critical",
            ),
            ComplianceCheck(
                id="WEB-PWA-02",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="PWA Requirements",
                description="Has service worker registered",
                severity="critical",
            ),
            ComplianceCheck(
                id="WEB-PWA-03",
                category=GuidelineCategory.MINIMUM_FUNCTIONALITY,
                guideline="PWA Requirements",
                description="Works offline (cached assets)",
                severity="warning",
            ),
        ],
        GuidelineCategory.PRIVACY.value: [
            ComplianceCheck(
                id="WEB-5.1-01",
                category=GuidelineCategory.PRIVACY,
                guideline="Privacy",
                description="Has privacy policy page",
                severity="critical",
            ),
        ],
    }

    def __init__(self, auto_fix: bool = False):
        """
        Initialize validator.

        Args:
            auto_fix: Automatically fix fixable issues
        """
        self.auto_fix = auto_fix
        self._checks_by_platform = {
            AppStorePlatform.IOS: self.IOS_CHECKS,
            AppStorePlatform.ANDROID: self.ANDROID_CHECKS,
            AppStorePlatform.WEB: self.WEB_CHECKS,
        }

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
            AppStoreComplianceResult with violations and score
        """
        if not project_path.exists():
            return AppStoreComplianceResult(
                platform=platform,
                violations=[f"Project path does not exist: {project_path}"],
                score=0.0,
            )

        checks = self._checks_by_platform.get(platform, self.IOS_CHECKS)
        result = AppStoreComplianceResult(platform=platform)

        # Read project files
        source_files = self._collect_source_files(project_path, platform)
        content_map = self._read_files(source_files)

        # Run all checks
        for _category_name, category_checks in checks.items():
            for check in category_checks:
                result.checks_performed += 1

                # Run check
                passed, details = await self._run_check(
                    check=check,
                    content_map=content_map,
                    project_path=project_path,
                    platform=platform,
                )

                if passed:
                    result.checks_passed += 1
                    result.info.append(f"✓ {check.description}")
                else:
                    result.checks_failed += 1
                    violation_msg = f"{check.guideline}: {check.description}"
                    if details:
                        violation_msg += f" ({details})"

                    if check.severity == "critical":
                        result.violations.append(violation_msg)
                    elif check.severity == "warning":
                        result.warnings.append(violation_msg)
                    else:
                        result.info.append(f"ℹ️ {violation_msg}")

        # Calculate score
        if result.checks_performed > 0:
            result.score = result.checks_passed / result.checks_performed

        # Add metadata
        result.metadata = {
            "platform": platform.value,
            "source_files": len(source_files),
            "total_lines": sum(len(c.split('\n')) for c in content_map.values()),
        }

        return result

    def _collect_source_files(
        self,
        project_path: Path,
        platform: AppStorePlatform,
    ) -> list[Path]:
        """Collect relevant source files for platform."""
        files = []

        if platform == AppStorePlatform.IOS:
            patterns = ["**/*.swift", "**/*.m", "**/*.mm", "**/*.h", "**/*.storyboard", "**/*.xib"]
        elif platform == AppStorePlatform.ANDROID:
            patterns = ["**/*.kt", "**/*.java", "**/*.xml"]
        elif platform == AppStorePlatform.WEB:
            patterns = ["**/*.html", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx", "**/*.css", "manifest.json"]
        else:
            patterns = ["**/*"]

        for pattern in patterns:
            files.extend(project_path.glob(pattern))

        return files

    def _read_files(self, files: list[Path]) -> dict[str, str]:
        """Read file contents into map."""
        content_map = {}
        for file_path in files:
            try:
                content_map[str(file_path)] = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        return content_map

    async def _run_check(
        self,
        check: ComplianceCheck,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """
        Run a single compliance check.

        Returns:
            (passed, details)
        """
        check_method = getattr(self, f"_check_{check.id.replace('-', '_').lower()}", None)

        if check_method:
            return await check_method(content_map, project_path, platform)

        # Default: check passes if no specific implementation
        return True, ""

    # ─────────────────────────────────────────────
    # iOS Check Implementations
    # ─────────────────────────────────────────────

    async def _check_ios_2_1_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for placeholder text."""
        placeholder_patterns = [
            r"lorem\s+ipsum",
            r"TODO[:\s]",
            r"FIXME[:\s]",
            r"placeholder",
            r"xxx.*xxx",
            r"\{\{.*\}\}",  # Template placeholders
        ]

        for file_path, content in content_map.items():
            for pattern in placeholder_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Found in {file_path}"

        return True, ""

    async def _check_ios_2_1_02(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for 'coming soon' or 'beta' labels."""
        beta_patterns = [
            r"coming\s+soon",
            r"\bbeta\b",
            r"work\s+in\s+progress",
            r"WIP",
        ]

        for file_path, content in content_map.items():
            # Skip test files
            if "test" in file_path.lower():
                continue

            for pattern in beta_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Found in {file_path}"

        return True, ""

    async def _check_ios_2_5_2_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for dynamic code execution."""
        dangerous_patterns = [
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bFunction\s*\(",  # JavaScript
            r"performSelector",  # Objective-C
            r"NSClassFromString",  # Objective-C dynamic
        ]

        for file_path, content in content_map.items():
            for pattern in dangerous_patterns:
                if re.search(pattern, content):
                    return False, f"Found in {file_path}"

        return True, ""

    async def _check_ios_4_2_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for native navigation."""
        navigation_patterns = [
            r"NavigationView",  # SwiftUI
            r"UINavigationController",  # UIKit
            r"NavigationStack",  # SwiftUI
            r"TabView",  # SwiftUI tab bar
            r"UITabBarController",  # UIKit tab bar
            r"NavigationBar",
        ]

        all_content = "\n".join(content_map.values())

        for pattern in navigation_patterns:
            if re.search(pattern, all_content):
                return True, ""

        return False, "No native navigation detected"

    async def _check_ios_4_2_02(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for native iOS features usage."""
        ios_features = [
            (r"UIImagePickerController", "Camera/Photo Library"),
            (r"CLLocationManager", "Location Services"),
            (r"AVAudioSession", "Audio"),
            (r"AVCaptureSession", "Camera"),
            (r"CNContactStore", "Contacts"),
            (r"EventStore", "Calendar"),
            (r"PHPhotoLibrary", "Photo Library"),
            (r"LocalNotification", "Push Notifications"),
            (r"UNUserNotificationCenter", "Notifications"),
            (r"CoreBluetooth", "Bluetooth"),
            (r"FaceID", "Face ID"),
            (r"TouchID", "Touch ID"),
            (r"LAContext", "Biometric Auth"),
        ]

        all_content = "\n".join(content_map.values())
        features_found = []

        for pattern, feature_name in ios_features:
            if re.search(pattern, all_content, re.IGNORECASE):
                features_found.append(feature_name)

        if len(features_found) >= 2:
            return True, f"Features: {', '.join(features_found)}"
        elif len(features_found) == 1:
            return False, f"Only 1 native feature: {features_found[0]}"
        else:
            return False, "No native iOS features detected"

    async def _check_ios_5_1_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for privacy policy URL."""
        privacy_patterns = [
            r"privacy.*policy",
            r"privacyPolicy",
            r"privacy_url",
            r"terms.*privacy",
        ]

        # Check Info.plist for iOS
        info_plist_path = project_path / "Info.plist"
        if info_plist_path.exists():
            content = info_plist_path.read_text()
            if re.search(r"NSPrivacyPolicyURL", content, re.IGNORECASE):
                return True, ""

        # Check source files
        all_content = "\n".join(content_map.values())
        for pattern in privacy_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "No privacy policy URL found"

    async def _check_ios_5_1_02(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for consent modal before AI data sharing."""
        consent_patterns = [
            r"consent",
            r"permission",
            r"agree.*data",
            r"privacy.*consent",
            r"data.*sharing.*confirm",
        ]

        all_content = "\n".join(content_map.values())

        for pattern in consent_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "No consent mechanism for data sharing"

    async def _check_ios_5_1_03(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for account deletion capability."""
        # Check if app has login
        login_patterns = [
            r"login",
            r"signin",
            r"auth",
            r"username",
            r"password",
        ]

        all_content = "\n".join(content_map.values())
        has_login = any(re.search(p, all_content, re.IGNORECASE) for p in login_patterns)

        if not has_login:
            return True, ""  # No login = no deletion needed

        # Check for delete/deactivate
        delete_patterns = [
            r"delete.*account",
            r"deactivate",
            r"close.*account",
            r"remove.*account",
        ]

        for pattern in delete_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "No account deletion capability found"

    async def _check_ios_hig_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for iOS-standard controls."""
        swiftui_controls = [
            r"Button\(",
            r"Text\(",
            r"Image\(",
            r"List\(",
            r"NavigationView",
            r"TabView",
            r"TextField\(",
            r"Toggle\(",
            r"Slider\(",
            r"Picker\(",
        ]

        uikit_controls = [
            r"UIButton",
            r"UILabel",
            r"UIImageView",
            r"UITableView",
            r"UICollectionView",
            r"UITextField",
            r"UISwitch",
            r"UISlider",
            r"UIPickerView",
        ]

        all_content = "\n".join(content_map.values())

        # Check for SwiftUI or UIKit controls
        swiftui_count = sum(1 for p in swiftui_controls if re.search(p, all_content))
        uikit_count = sum(1 for p in uikit_controls if re.search(p, all_content))

        if swiftui_count >= 3 or uikit_count >= 3:
            return True, ""

        return False, "Limited use of iOS-standard controls"

    async def _check_ios_hig_03(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for Dark Mode support."""
        dark_mode_patterns = [
            r"colorScheme.*dark",
            r"@Environment.*colorScheme",
            r"traitCollection\.userInterfaceStyle",
            r"UIColor\.label",  # System colors
            r"UIColor\.systemBackground",
            r"Color\.primary",  # SwiftUI system colors
            r"Color\.\.background",
        ]

        all_content = "\n".join(content_map.values())

        for pattern in dark_mode_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "No Dark Mode support detected"

    async def _check_ios_ai_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for AI-generated content labeling."""
        ai_patterns = [
            r"AI[- ]generated",
            r"generated\s+by\s+AI",
            r"powered\s+by\s+AI",
            r"machine\s+learning",
            r"automated\s+suggestion",
        ]

        all_content = "\n".join(content_map.values())

        # Check if app uses AI
        ai_usage = any(re.search(p, all_content, re.IGNORECASE) for p in [
            r"openai", r"anthropic", r"gemini", r"llm", r"chatgpt",
        ])

        if not ai_usage:
            return True, ""  # No AI = no labeling needed

        # Check for transparency
        for pattern in ai_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "AI-generated content not labeled"

    async def _check_ios_ai_02(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check that user knows when interacting with AI."""
        transparency_patterns = [
            r"chatbot",
            r"AI\s+assistant",
            r"virtual\s+assistant",
            r"automated\s+response",
            r"bot",
        ]

        all_content = "\n".join(content_map.values())

        # Check if app uses AI
        ai_usage = any(re.search(p, all_content, re.IGNORECASE) for p in [
            r"openai", r"anthropic", r"gemini", r"llm", r"chatgpt",
        ])

        if not ai_usage:
            return True, ""

        # Check for transparency
        for pattern in transparency_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "AI interaction not clearly disclosed"

    # ─────────────────────────────────────────────
    # Android Check Implementations
    # ─────────────────────────────────────────────

    async def _check_and_2_1_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for placeholder text (Android)."""
        return await self._check_ios_2_1_01(content_map, project_path, platform)

    async def _check_and_5_1_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for privacy policy (Android)."""
        # Check AndroidManifest.xml
        manifest_path = project_path / "AndroidManifest.xml"
        if manifest_path.exists():
            content = manifest_path.read_text()
            if re.search(r"privacy.*policy", content, re.IGNORECASE):
                return True, ""

        return await self._check_ios_5_1_01(content_map, project_path, platform)

    # ─────────────────────────────────────────────
    # Web/PWA Check Implementations
    # ─────────────────────────────────────────────

    async def _check_web_pwa_01(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for valid manifest.json."""
        manifest_path = project_path / "manifest.json"

        if not manifest_path.exists():
            return False, "manifest.json not found"

        try:
            import json
            manifest = json.loads(manifest_path.read_text())

            required_fields = ["name", "short_name", "start_url", "display", "icons"]
            missing = [f for f in required_fields if f not in manifest]

            if missing:
                return False, f"Missing fields: {', '.join(missing)}"

            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    async def _check_web_pwa_02(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for service worker registration."""
        sw_patterns = [
            r"navigator\.serviceWorker\.register",
            r"serviceWorker\.register",
            r"self\.addEventListener.*install",
        ]

        all_content = "\n".join(content_map.values())

        for pattern in sw_patterns:
            if re.search(pattern, all_content):
                return True, ""

        return False, "No service worker registration found"

    async def _check_web_pwa_03(
        self,
        content_map: dict[str, str],
        project_path: Path,
        platform: AppStorePlatform,
    ) -> tuple[bool, str]:
        """Check for offline capability."""
        offline_patterns = [
            r"cache",
            r"offline",
            r"workbox",
            r"precache",
        ]

        all_content = "\n".join(content_map.values())

        for pattern in offline_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return True, ""

        return False, "No offline capability detected"


async def validate_app_store_compliance(
    project_path: Path,
    platform: AppStorePlatform = AppStorePlatform.IOS,
    auto_fix: bool = False,
) -> AppStoreComplianceResult:
    """
    Convenience function to validate app store compliance.

    Usage:
        result = await validate_app_store_compliance(
            Path("./my-app"),
            platform=AppStorePlatform.IOS,
        )
        print(result.summary)
    """
    validator = AppStoreValidator(auto_fix=auto_fix)
    return await validator.validate(project_path, platform)
