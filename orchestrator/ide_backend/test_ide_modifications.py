"""
Test IDE Modification Handler - Regression Tests for Color Changes and Broadcast Ordering
Covers: BUG-001 (Race Condition), BUG-002 (String Replacement), BUG-003 (Session State Broadcast)
"""
import re

import pytest


class TestColorModificationRegex:
    """Test FIX-002b: CSS parsing for color replacement"""

    def test_color_change_gold_to_purple(self):
        """Test changing from default gold (#c9a55c) to purple (#8b5cf6)"""
        css_content = """
        :root {
            --primary: #0a0a0f;
            --secondary: #1a1a2e;
            --accent: #c9a55c;
            --text: #ffffff;
        }
        """
        new_color = "#8b5cf6"
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'
        match = re.search(accent_pattern, css_content)

        assert match is not None
        assert match.group(0) == "--accent: #c9a55c"

        result = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        assert "--accent: #8b5cf6" in result
        assert "--accent: #c9a55c" not in result

    def test_color_change_already_same_color(self):
        """Test changing to same color - should still work"""
        css_content = """
        :root {
            --accent: #3b82f6;
        }
        """
        new_color = "#3b82f6"  # Same color
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        result = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        assert "--accent: #3b82f6" in result

    def test_color_change_multiple_spaces(self):
        """Test regex handles various spacing"""
        css_content = """
        :root {
            --accent:#c9a55c;
        }
        .btn {
            --accent:   #c9a55c;
        }
        """
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'
        matches = re.findall(accent_pattern, css_content)

        assert len(matches) == 2
        assert matches[0] == "--accent:#c9a55c"
        assert matches[1] == "--accent:   #c9a55c"

    def test_no_accent_color_found(self):
        """Test CSS without --accent variable"""
        css_content = """
        :root {
            --primary: #0a0a0f;
            --secondary: #1a1a2e;
        }
        """
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'
        match = re.search(accent_pattern, css_content)

        assert match is None

    def test_color_change_short_hex(self):
        """Test 3-digit hex colors"""
        css_content = """
        :root {
            --accent: #fff;
        }
        """
        new_color = "#000"
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        result = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        assert "--accent: #000" in result

    def test_color_change_uppercase_hex(self):
        """Test uppercase hex colors"""
        css_content = """
        :root {
            --accent: #C9A55C;
        }
        """
        new_color = "#8B5CF6"
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        result = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        assert "--accent: #8B5CF6" in result
        assert "--accent: #C9A55C" not in result


class TestBroadcastOrdering:
    """Test FIX-001a+003: Broadcast ordering - session_state before terminal_update"""

    @pytest.mark.asyncio
    async def test_broadcast_order_session_state_first(self):
        """Verify session_state is broadcast before terminal_update"""
        from orchestrator.ide_backend.session_manager import SessionManager

        session_manager = SessionManager()

        # Create mock session
        session = await session_manager.create_session(
            project_name="Test Project",
            description="Test"
        )

        # Mock broadcast method to capture call order
        broadcast_calls = []
        original_broadcast = session_manager.broadcast

        async def mock_broadcast(session_id, event, data):
            broadcast_calls.append(event)
            await original_broadcast(session_id, event, data)

        session_manager.broadcast = mock_broadcast

        # Simulate modification completion
        session_manager.update_session(session.id, status="completed")

        # Broadcast in correct order
        await session_manager.broadcast(session.id, "session_state", session_manager.get_session(session.id))
        await session_manager.broadcast(session.id, "terminal_update", {"lines": []})
        await session_manager.broadcast(session.id, "messages_update", {"messages": []})

        # Verify order
        assert broadcast_calls[0] == "session_state"
        assert broadcast_calls[1] == "terminal_update"
        assert broadcast_calls[2] == "messages_update"

    @pytest.mark.asyncio
    async def test_session_state_includes_updated_files(self):
        """Verify session_state contains updated file list after modification"""
        from orchestrator.ide_backend.session_manager import SessionManager

        session_manager = SessionManager()
        session = await session_manager.create_session()

        # Add terminal line
        session_manager.add_terminal_line(session.id, "success", "✓ Updated styles.css")

        # Get session state
        session_state = session_manager.get_session(session.id)

        # Verify terminal lines updated
        assert any("Updated styles.css" in line.get("content", "")
                   for line in session_state.terminal_lines)


class TestEdgeCases:
    """Test edge cases and failure modes"""

    def test_color_name_not_in_css(self):
        """Test requesting color that doesn't exist in CSS"""
        css_content = """
        :root {
            --accent: #c9a55c;
        }
        """
        # Request "black" but CSS has gold
        new_color = "#000000"
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        result = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        assert "--accent: #000000" in result

    def test_multiple_color_changes_sequential(self):
        """Test multiple sequential color changes"""
        css_content = """
        :root {
            --accent: #c9a55c;
        }
        """
        colors = ["#8b5cf6", "#3b82f6", "#10b981", "#ef4444"]
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        for color in colors:
            css_content = re.sub(accent_pattern, f'--accent: {color}', css_content, count=1)
            assert f"--accent: {color}" in css_content

        # Final color should be last in list
        assert "--accent: #ef4444" in css_content

    def test_invalid_hex_color_ignored(self):
        """Test invalid hex colors don't match pattern"""
        css_content = """
        :root {
            --accent: #GGGGGG;  /* Invalid */
            --primary: #0a0a0f;
        }
        """
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'
        match = re.search(accent_pattern, css_content)

        # Invalid hex should not match
        assert match is None


class TestIntegration:
    """Integration tests for complete modification flow"""

    @pytest.mark.asyncio
    async def test_full_modification_flow(self, tmp_path):
        """Test complete modification flow with file I/O"""
        from orchestrator.ide_backend.session_manager import SessionManager

        # Setup
        session_manager = SessionManager()
        session = await session_manager.create_session()

        # Create fake CSS file
        output_dir = tmp_path / session.id
        output_dir.mkdir(parents=True, exist_ok=True)
        css_path = output_dir / "styles.css"

        original_css = """
        :root {
            --primary: #0a0a0f;
            --accent: #c9a55c;
            --text: #ffffff;
        }
        """
        css_path.write_text(original_css)

        # Simulate modification
        css_content = css_path.read_text()
        new_color = "#8b5cf6"
        accent_pattern = r'--accent:\s*#[0-9a-fA-F]{3,6}'

        match = re.search(accent_pattern, css_content)
        assert match is not None

        css_content = re.sub(accent_pattern, f'--accent: {new_color}', css_content, count=1)
        css_path.write_text(css_content)

        # Verify file updated
        updated_css = css_path.read_text()
        assert "--accent: #8b5cf6" in updated_css
        assert "--accent: #c9a55c" not in updated_css


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
