"""
iOS/HIG-Aware Code Generation Prompts
======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Apple Human Interface Guidelines (HIG) injection for iOS code generation.
Ensures generated iOS code follows App Store requirements and HIG standards.

Usage:
    from orchestrator.ios_hig_prompts import IOS_GENERATION_CONTEXT, get_ios_prompt

    prompt = get_ios_prompt(project_description, include_hig=True)
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# iOS Generation Context (App Store Requirements)
# ─────────────────────────────────────────────

IOS_GENERATION_CONTEXT = """
CRITICAL: This code will be submitted to the Apple App Store.
Follow these requirements STRICTLY:

1. DESIGN (Human Interface Guidelines):
   - Use SwiftUI or UIKit standard components
   - Tab bars for main navigation (UITabBarController / TabView)
   - Navigation stacks (NavigationStack) for drill-down
   - Standard system icons (SF Symbols: sfSymbolName)
   - Support Dark Mode via @Environment(\\.colorScheme)
   - Support Dynamic Type for accessibility (@ScaledMetric)
   - Follow iOS safe areas and layout margins
   - Use standard iOS gestures (swipe to go back, pull to refresh)

2. COMPLETENESS (App Store Guideline 2.1):
   - NO placeholder content ("Lorem ipsum", "TODO", "Coming soon")
   - Every screen must have real, meaningful content
   - Every button must have a working action
   - All UI elements must be functional
   - No broken links or dead ends

3. PRIVACY (App Store Guideline 5.1):
   - Include Info.plist privacy descriptions for ALL permissions:
     * NSPhotoLibraryUsageDescription
     * NSCameraUsageDescription
     * NSLocationWhenInUseUsageDescription
     * NSUserTrackingUsageDescription
   - Show consent modal before sharing data with AI services
   - Include in-app privacy policy link (in Settings or About)
   - Implement account deletion if login exists (Settings → Delete Account)
   - Use App Privacy labels accurately

4. PERFORMANCE (App Store Guideline 2.5):
   - App must launch in <3 seconds
   - No blocking main thread operations (use async/await)
   - Use lazy loading for images (AsyncImage)
   - Handle offline state gracefully (show offline UI)
   - Memory efficient (no memory leaks)
   - Battery efficient (background tasks only when necessary)

5. SELF-CONTAINED (App Store Guideline 2.5.2):
   - NO eval(), exec(), or dynamic code execution
   - NO downloading code at runtime
   - All functionality must be in the app bundle
   - No embedded WebView that changes app functionality
   - No JIT compilation

6. APP STORE ASSETS:
   - Include LaunchScreen.storyboard or iOS Launch Image
   - Include App Icon in Assets.xcassets/AppIcon
   - Support all required screen sizes (iPhone, iPad if universal)
   - Include proper Info.plist configuration:
     * CFBundleName
     * CFBundleVersion
     * CFBundleShortVersionString
     * UISupportedInterfaceOrientations

7. ACCESSIBILITY (App Store Guideline 5.1.3):
   - All images must have accessibility labels
   - All buttons must have accessibility hints
   - Support VoiceOver (accessibilityElement)
   - Support Dynamic Type (font sizes)
   - Minimum touch target size: 44x44 points

8. NAVIGATION:
   - Clear navigation hierarchy
   - Back buttons must work
   - Deep links must resolve correctly
   - Tab bar items (3-5 main sections)

9. ERROR HANDLING:
   - User-friendly error messages
   - Retry mechanisms for network failures
   - Graceful degradation for unsupported features
   - No crash on invalid input

10. TESTING:
    - Include unit tests for critical logic
    - Include UI tests for main user flows
    - All tests must pass before submission
    - Test on real devices (not just simulator)

Example SwiftUI Structure:
```swift
import SwiftUI

@main
struct AppNameApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag(0)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(1)
        }
    }
}
```

VIOLATION = REJECTION. Follow these guidelines EXACTLY.
"""

# ─────────────────────────────────────────────
# HIG-Specific Prompt Templates
# ─────────────────────────────────────────────

HIG_NAVIGATION_TEMPLATE = """
CRITICAL: Implement proper iOS navigation following HIG.

Requirements:
- Use TabView for main navigation (3-5 tabs maximum)
- Use NavigationStack for drill-down navigation
- Each tab should have a clear purpose
- Use SF Symbols for tab icons
- Support swipe-to-go-back gesture

Example:
```swift
TabView {
    HomeView()
        .tabItem { Label("Home", systemImage: "house") }
    SearchView()
        .tabItem { Label("Search", systemImage: "magnifyingglass") }
    ProfileView()
        .tabItem { Label("Profile", systemImage: "person") }
}
```
"""

HIG_DARK_MODE_TEMPLATE = """
CRITICAL: Support Dark Mode following HIG.

Requirements:
- Use @Environment(\\.colorScheme) to detect mode
- Use semantic colors (Color.primary, Color.background)
- Test in both Light and Dark modes
- Ensure sufficient contrast (WCAG AA minimum)
- Custom colors must have light/dark variants

Example:
```swift
struct ContentView: View {
    @Environment(\\.colorScheme) var colorScheme

    var body: some View {
        Text("Hello")
            .foregroundColor(colorScheme == .dark ? .white : .black)
            .padding()
            .background(Color(.systemBackground))
    }
}
```
"""

HIG_ACCESSIBILITY_TEMPLATE = """
CRITICAL: Support accessibility following HIG.

Requirements:
- All images must have .accessibilityLabel()
- All buttons must have .accessibilityHint()
- Support Dynamic Type with @ScaledMetric
- Minimum touch target: 44x44 points
- Test with VoiceOver enabled

Example:
```swift
struct ContentView: View {
    @ScaledMetric(relativeTo: .body) var fontSize: CGFloat = 16

    var body: some View {
        Button(action: onSave) {
            Image(systemName: "square.and.arrow.down")
                .font(.system(size: fontSize))
        }
        .accessibilityLabel("Save")
        .accessibilityHint("Saves the current document")
        .frame(minWidth: 44, minHeight: 44)
    }
}
```
"""

HIG_PRIVACY_TEMPLATE = """
CRITICAL: Implement privacy features following App Store Guidelines.

Requirements:
1. Info.plist privacy descriptions:
   - Add NSPhotoLibraryUsageDescription
   - Add NSCameraUsageDescription
   - Add NSLocationWhenInUseUsageDescription (if needed)
   - Add NSUserTrackingUsageDescription (if tracking)

2. Consent Modal:
   - Show before accessing photos/camera/location
   - Explain WHY you need the permission
   - Provide "Not Now" option

3. Privacy Policy:
   - Include link to privacy policy
   - Place in Settings or About screen
   - Explain what data is collected and why

4. Account Deletion (if login exists):
   - Settings → Delete Account
   - Confirm deletion
   - Delete all user data
   - Confirm deletion completed

Example Consent Modal:
```swift
struct PhotoPermissionModal: View {
    @State private var showPermissionAlert = false

    var body: some View {
        Button("Select Photo") {
            showPermissionAlert = true
        }
        .alert("Photo Access Required", isPresented: $showPermissionAlert) {
            Button("Open Settings", action: openSettings)
            Button("Not Now", role: .cancel) { }
        } message: {
            Text("We need photo access to let you upload profile pictures. This is optional.")
        }
    }
}
```
"""

HIG_LAUNCH_SCREEN_TEMPLATE = """
CRITICAL: Create proper launch screen following HIG.

Requirements:
- Use LaunchScreen.storyboard (preferred) OR
- Use iOS Launch Image asset
- Match app's initial screen layout
- No text (text changes by localization)
- Use app icon and brand colors
- Launch time must be <3 seconds

LaunchScreen.storyboard structure:
- Root view controller with safe area constraints
- App icon in center
- Brand color background
- No custom code in launch screen
"""

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def get_ios_prompt(
    project_description: str,
    include_hig: bool = True,
    include_navigation: bool = True,
    include_dark_mode: bool = True,
    include_accessibility: bool = True,
    include_privacy: bool = True,
    include_launch_screen: bool = True,
) -> str:
    """
    Generate iOS-specific prompt with HIG guidelines.

    Args:
        project_description: Project to generate
        include_hig: Include main HIG context (default: True)
        include_navigation: Include navigation template (default: True)
        include_dark_mode: Include dark mode template (default: True)
        include_accessibility: Include accessibility template (default: True)
        include_privacy: Include privacy template (default: True)
        include_launch_screen: Include launch screen template (default: True)

    Returns:
        Complete iOS generation prompt
    """
    prompt_parts = [
        "# iOS App Generation Request",
        "",
        f"Project: {project_description}",
        "",
        "## Platform: iOS (SwiftUI)",
        "",
    ]

    if include_hig:
        prompt_parts.append("## Apple HIG & App Store Requirements")
        prompt_parts.append(IOS_GENERATION_CONTEXT)
        prompt_parts.append("")

    if include_navigation:
        prompt_parts.append("## Navigation Requirements")
        prompt_parts.append(HIG_NAVIGATION_TEMPLATE)
        prompt_parts.append("")

    if include_dark_mode:
        prompt_parts.append("## Dark Mode Requirements")
        prompt_parts.append(HIG_DARK_MODE_TEMPLATE)
        prompt_parts.append("")

    if include_accessibility:
        prompt_parts.append("## Accessibility Requirements")
        prompt_parts.append(HIG_ACCESSIBILITY_TEMPLATE)
        prompt_parts.append("")

    if include_privacy:
        prompt_parts.append("## Privacy Requirements")
        prompt_parts.append(HIG_PRIVACY_TEMPLATE)
        prompt_parts.append("")

    if include_launch_screen:
        prompt_parts.append("## Launch Screen Requirements")
        prompt_parts.append(HIG_LAUNCH_SCREEN_TEMPLATE)
        prompt_parts.append("")

    prompt_parts.append("## Task")
    prompt_parts.append("Generate a complete, production-ready iOS app that follows ALL guidelines above.")
    prompt_parts.append("The app must be ready for App Store submission with no modifications.")

    return "\n".join(prompt_parts)


def get_hig_checklist() -> dict[str, list[str]]:
    """
    Get HIG compliance checklist.

    Returns:
        Dictionary with checklist categories and items
    """
    return {
        "Design": [
            "Uses SwiftUI or UIKit standard components",
            "Tab bars for main navigation",
            "Navigation stacks for drill-down",
            "SF Symbols for icons",
            "Dark Mode support",
            "Dynamic Type support",
        ],
        "Completeness": [
            "No placeholder content",
            "All screens have real content",
            "All buttons have working actions",
            "No TODO/FIXME comments",
        ],
        "Privacy": [
            "Info.plist privacy descriptions",
            "Consent modals for permissions",
            "Privacy policy link",
            "Account deletion (if login)",
        ],
        "Performance": [
            "Launch time <3 seconds",
            "No blocking main thread",
            "Lazy loading for images",
            "Offline state handling",
        ],
        "Self-Contained": [
            "No eval()/exec()",
            "No runtime code download",
            "All functionality in bundle",
        ],
        "Accessibility": [
            "Image accessibility labels",
            "Button accessibility hints",
            "VoiceOver support",
            "Dynamic Type support",
            "44x44 minimum touch targets",
        ],
    }


def validate_hig_compliance(code: str) -> tuple[bool, list[str]]:
    """
    Validate code for HIG compliance.

    Args:
        code: Swift code to validate

    Returns:
        Tuple of (is_compliant, list of violations)
    """
    violations = []

    # Check for placeholder content
    placeholders = ["lorem ipsum", "todo", "fixme", "coming soon", "placeholder"]
    code_lower = code.lower()
    for placeholder in placeholders:
        if placeholder in code_lower:
            violations.append(f"Found placeholder content: '{placeholder}'")

    # Check for dynamic code execution
    dangerous_patterns = ["eval(", "exec(", "NSClassFromString", "performSelector"]
    for pattern in dangerous_patterns:
        if pattern in code:
            violations.append(f"Found dynamic code execution: '{pattern}'")

    # Check for TabView (navigation requirement)
    if "TabView" not in code and "UITabBarController" not in code:
        violations.append("Missing TabView/UITabBarController for main navigation")

    # Check for Dark Mode support
    if "@Environment(\\.colorScheme)" not in code and "userInterfaceStyle" not in code:
        violations.append("Missing Dark Mode support (@Environment(.colorScheme))")

    # Check for accessibility
    if ".accessibilityLabel(" not in code and "accessibilityLabel" not in code:
        violations.append("Missing accessibility labels")

    is_compliant = len(violations) == 0

    return is_compliant, violations


# ─────────────────────────────────────────────
# Integration with Multi-Platform Generator
# ─────────────────────────────────────────────

def inject_hig_context(
    project_description: str,
    target_platform: str,
) -> str:
    """
    Inject HIG context if target is iOS.

    Args:
        project_description: Original project description
        target_platform: Target platform (ios, android, web, etc.)

    Returns:
        Enhanced project description with HIG context if iOS
    """
    if target_platform.lower() in ["ios", "swiftui", "iphone", "ipad"]:
        return get_ios_prompt(project_description)

    return project_description
