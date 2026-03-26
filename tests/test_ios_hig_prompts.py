"""
Tests for iOS HIG-Aware Prompts
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_ios_hig_prompts.py -v
"""

import pytest

from orchestrator.ios_hig_prompts import (
    IOS_GENERATION_CONTEXT,
    get_ios_prompt,
    get_hig_checklist,
    validate_hig_compliance,
    inject_hig_context,
)


# ─────────────────────────────────────────────
# Test iOS Generation Context
# ─────────────────────────────────────────────

class TestIOSGenerationContext:
    """Test iOS generation context."""
    
    def test_context_exists(self):
        """Test context string exists."""
        assert IOS_GENERATION_CONTEXT is not None
        assert len(IOS_GENERATION_CONTEXT) > 100
    
    def test_context_has_design_requirements(self):
        """Test context includes design requirements."""
        assert "DESIGN" in IOS_GENERATION_CONTEXT
        assert "SwiftUI" in IOS_GENERATION_CONTEXT or "UIKit" in IOS_GENERATION_CONTEXT
        assert "TabView" in IOS_GENERATION_CONTEXT or "UITabBarController" in IOS_GENERATION_CONTEXT
        assert "Dark Mode" in IOS_GENERATION_CONTEXT
    
    def test_context_has_completeness_requirements(self):
        """Test context includes completeness requirements."""
        assert "COMPLETENESS" in IOS_GENERATION_CONTEXT
        assert "placeholder" in IOS_GENERATION_CONTEXT.lower()
        assert "Lorem ipsum" in IOS_GENERATION_CONTEXT or "TODO" in IOS_GENERATION_CONTEXT
    
    def test_context_has_privacy_requirements(self):
        """Test context includes privacy requirements."""
        assert "PRIVACY" in IOS_GENERATION_CONTEXT
        assert "Info.plist" in IOS_GENERATION_CONTEXT
        assert "privacy" in IOS_GENERATION_CONTEXT.lower()
    
    def test_context_has_performance_requirements(self):
        """Test context includes performance requirements."""
        assert "PERFORMANCE" in IOS_GENERATION_CONTEXT
        assert "launch" in IOS_GENERATION_CONTEXT.lower()
        assert "<3 seconds" in IOS_GENERATION_CONTEXT or "3 seconds" in IOS_GENERATION_CONTEXT
    
    def test_context_has_self_contained_requirements(self):
        """Test context includes self-contained requirements."""
        assert "SELF-CONTAINED" in IOS_GENERATION_CONTEXT
        assert "eval" in IOS_GENERATION_CONTEXT.lower()
        assert "dynamic code" in IOS_GENERATION_CONTEXT.lower()


# ─────────────────────────────────────────────
# Test get_ios_prompt
# ─────────────────────────────────────────────

class TestGetIOSPrompt:
    """Test get_ios_prompt function."""
    
    def test_basic_prompt(self):
        """Test basic iOS prompt generation."""
        prompt = get_ios_prompt("Build a todo app", include_hig=True)
        
        assert "Build a todo app" in prompt
        assert "iOS" in prompt
        assert "SwiftUI" in prompt
    
    def test_prompt_with_all_sections(self):
        """Test prompt with all HIG sections."""
        prompt = get_ios_prompt(
            "Build a fitness tracker",
            include_hig=True,
            include_navigation=True,
            include_dark_mode=True,
            include_accessibility=True,
            include_privacy=True,
            include_launch_screen=True,
        )
        
        assert "Apple HIG" in prompt
        assert "Navigation Requirements" in prompt
        assert "Dark Mode Requirements" in prompt
        assert "Accessibility Requirements" in prompt
        assert "Privacy Requirements" in prompt
        assert "Launch Screen Requirements" in prompt
    
    def test_prompt_minimal(self):
        """Test prompt with minimal sections."""
        prompt = get_ios_prompt(
            "Build a calculator",
            include_hig=False,
            include_navigation=False,
            include_dark_mode=False,
            include_accessibility=False,
            include_privacy=False,
            include_launch_screen=False,
        )
        
        assert "Build a calculator" in prompt
        assert "iOS" in prompt
        # Should not have detailed sections
        assert "Apple HIG" not in prompt


# ─────────────────────────────────────────────
# Test HIG Checklist
# ─────────────────────────────────────────────

class TestHIGChecklist:
    """Test HIG checklist."""
    
    def test_checklist_exists(self):
        """Test checklist exists."""
        checklist = get_hig_checklist()
        
        assert checklist is not None
        assert isinstance(checklist, dict)
    
    def test_checklist_categories(self):
        """Test checklist has all categories."""
        checklist = get_hig_checklist()
        
        assert "Design" in checklist
        assert "Completeness" in checklist
        assert "Privacy" in checklist
        assert "Performance" in checklist
        assert "Self-Contained" in checklist
        assert "Accessibility" in checklist
    
    def test_checklist_items(self):
        """Test checklist items are non-empty."""
        checklist = get_hig_checklist()
        
        for category, items in checklist.items():
            assert len(items) > 0
            assert all(isinstance(item, str) for item in items)


# ─────────────────────────────────────────────
# Test validate_hig_compliance
# ─────────────────────────────────────────────

class TestValidateHIGCompliance:
    """Test HIG compliance validation."""
    
    def test_compliant_code(self):
        """Test compliant code passes validation."""
        compliant_code = """
import SwiftUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView()
                    .tabItem { Label("Home", systemImage: "house") }
            }
        }
    }
}

struct ContentView: View {
    @Environment(\\.colorScheme) var colorScheme
    
    var body: some View {
        Button(action: {}) {
            Text("Save")
                .accessibilityLabel("Save")
                .accessibilityHint("Saves the current item")
                .frame(minWidth: 44, minHeight: 44)
        }
    }
}
"""
        is_compliant, violations = validate_hig_compliance(compliant_code)
        
        # Should have no major violations
        assert len(violations) == 0 or all("placeholder" not in v.lower() for v in violations)
    
    def test_non_compliant_code_placeholders(self):
        """Test code with placeholders fails validation."""
        non_compliant = """
struct ContentView: View {
    var body: some View {
        Text("TODO: Implement this")
    }
}
"""
        is_compliant, violations = validate_hig_compliance(non_compliant)
        
        assert is_compliant is False
        assert any("placeholder" in v.lower() for v in violations)
    
    def test_non_compliant_code_dynamic_execution(self):
        """Test code with dynamic execution fails validation."""
        non_compliant = """
class DynamicLoader {
    func loadCode() {
        eval("print('hello')")
    }
}
"""
        is_compliant, violations = validate_hig_compliance(non_compliant)
        
        assert is_compliant is False
        assert any("dynamic code" in v.lower() or "eval" in v.lower() for v in violations)
    
    def test_non_compliant_code_no_navigation(self):
        """Test code without navigation fails validation."""
        non_compliant = """
struct ContentView: View {
    var body: some View {
        Text("Hello World")
    }
}
"""
        is_compliant, violations = validate_hig_compliance(non_compliant)
        
        assert any("navigation" in v.lower() or "tab" in v.lower() for v in violations)
    
    def test_non_compliant_code_no_dark_mode(self):
        """Test code without dark mode fails validation."""
        non_compliant = """
struct ContentView: View {
    var body: some View {
        Text("Hello")
    }
}
"""
        is_compliant, violations = validate_hig_compliance(non_compliant)
        
        assert any("dark mode" in v.lower() or "color" in v.lower() for v in violations)


# ─────────────────────────────────────────────
# Test inject_hig_context
# ─────────────────────────────────────────────

class TestInjectHIGContext:
    """Test HIG context injection."""
    
    def test_inject_for_ios(self):
        """Test HIG context is injected for iOS."""
        project = "Build a fitness app"
        result = inject_hig_context(project, "ios")
        
        assert "Build a fitness app" in result
        assert "iOS" in result
        assert "HIG" in result or "App Store" in result
    
    def test_inject_for_swiftui(self):
        """Test HIG context is injected for SwiftUI."""
        project = "Build a todo app"
        result = inject_hig_context(project, "swiftui")
        
        assert "Build a todo app" in result
        assert "iOS" in result
    
    def test_no_inject_for_android(self):
        """Test HIG context is NOT injected for Android."""
        project = "Build a fitness app"
        result = inject_hig_context(project, "android")
        
        assert result == project  # Should be unchanged
    
    def test_no_inject_for_web(self):
        """Test HIG context is NOT injected for Web."""
        project = "Build a fitness app"
        result = inject_hig_context(project, "web")
        
        assert result == project  # Should be unchanged


# ─────────────────────────────────────────────
# Test Integration with Multi-Platform Generator
# ─────────────────────────────────────────────

class TestIntegration:
    """Test integration with multi-platform generator."""
    
    def test_ios_prompt_in_generator(self):
        """Test iOS prompt can be used with generator."""
        from orchestrator.multi_platform_generator import OutputTarget
        
        # Verify iOS target exists
        assert OutputTarget.SWIFTUI_IOS is not None
        
        # Verify HIG prompts can be imported
        from orchestrator.ios_hig_prompts import get_ios_prompt
        prompt = get_ios_prompt("Test app")
        
        assert prompt is not None
        assert len(prompt) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
