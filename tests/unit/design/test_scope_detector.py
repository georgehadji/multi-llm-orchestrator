"""Unit tests for scope_detector."""

import pytest

from orchestrator.design.scope_detector import detect_scope
from orchestrator.models import DesignScope


@pytest.mark.unit
class TestScopeDetector:
    def test_component_button_short_prompt(self):
        assert detect_scope("Build a primary button component", "Button.tsx") == DesignScope.COMPONENT

    def test_component_input_explicit(self):
        assert detect_scope("Just the email input field", "Input.tsx") == DesignScope.COMPONENT

    def test_component_modal(self):
        assert detect_scope("Create a modal dialog", "Modal.jsx") == DesignScope.COMPONENT

    def test_page_deep_path(self):
        """Files in deep paths (e.g. app/about/page.tsx) are pages."""
        assert detect_scope("Build a landing page with hero, features, and CTA", "app/about/page.tsx") == DesignScope.PAGE

    def test_page_marketing_site(self):
        assert detect_scope("Create a marketing website with multiple sections", "src/pages/index.html") == DesignScope.PAGE

    def test_component_short_with_explicit_phrase(self):
        assert detect_scope("Only the card", "Card.tsx") == DesignScope.COMPONENT

    def test_page_no_component_signals_deep_path(self):
        """Dashboard page in deep path is a page."""
        assert detect_scope("Build a full dashboard with sidebar and charts", "app/dashboard/page.tsx") == DesignScope.PAGE

    def test_component_in_components_folder(self):
        assert detect_scope("Badge component", "components/Badge.tsx") == DesignScope.COMPONENT

    def test_component_bare_file(self):
        """Bare .tsx files (1 path part) are treated as components."""
        assert detect_scope("About page", "About.tsx") == DesignScope.COMPONENT

    def test_page_long_prompt_no_component_words(self):
        """Long prompt without component words and .html path → page."""
        prompt = "Build a landing page that displays user profile information with photograph, name, biography, and social links in a responsive grid layout with navigation and footer sections"
        assert detect_scope(prompt, "app/profile/page.html") == DesignScope.PAGE
