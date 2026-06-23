"""
Unit tests for WebsiteGenerator._sanitize_output().
Covers all 4 bug-class fixers: fences, casts, template literals, truncation.
"""

import pytest


@pytest.fixture(scope="module")
def sanitize():
    from orchestrator.generators.website_generator import WebsiteGenerator

    return WebsiteGenerator._sanitize_output


# ═══════════════════════════════════════════════════════════════════
# 1. Markdown code fence stripping
# ═══════════════════════════════════════════════════════════════════


class TestFenceStripping:
    def test_strips_tsx_fence(self, sanitize):
        out, w = sanitize("```tsx\nconst x = 1;\n```", "test")
        assert out == "const x = 1;"
        assert any("fences" in x for x in w)

    def test_strips_fence_no_lang(self, sanitize):
        out, w = sanitize("```\nhello\n```", "test")
        assert out == "hello"
        assert any("fences" in x for x in w)

    def test_strips_only_outer_fence(self, sanitize):
        raw = "```tsx\nconst code = '```';\n```"
        out, w = sanitize(raw, "test")
        assert "const code" in out
        assert any("fences" in x for x in w)

    def test_passes_through_clean(self, sanitize):
        raw = (
            "export default function Hero() {\n"
            "  return (\n"
            "    <section className='hero'>\n"
            "      <h1>Welcome to My Portfolio</h1>\n"
            "      <p>This is a test component with enough length to pass the minimum threshold.</p>\n"
            "    </section>\n"
            "  );\n"
            "}\n"
        )
        out, w = sanitize(raw, "test")
        assert out == raw.rstrip()
        assert not w


# ═══════════════════════════════════════════════════════════════════
# 2. as React.CSSProperties cast removal
# ═══════════════════════════════════════════════════════════════════


class TestCastRemoval:
    def test_removes_single_cast(self, sanitize):
        raw = "style={{color: red} as React.CSSProperties}"
        out, w = sanitize(raw, "test")
        assert "as React.CSSProperties" not in out
        assert out.endswith("}}")
        assert any("cast" in x for x in w)

    def test_removes_multiple_casts(self, sanitize):
        raw = (
            "style={{a:1} as React.CSSProperties}\n"
            "style={{b:2} as React.CSSProperties}"
        )
        out, w = sanitize(raw, "test")
        assert "as React.CSSProperties" not in out
        assert out.count("}}") == 2
        assert any("cast" in x for x in w)

    def test_no_cast_passthrough(self, sanitize):
        raw = "style={{color: 'red'}}"
        out, w = sanitize(raw, "test")
        assert "style={{color: 'red'}}" in out


# ═══════════════════════════════════════════════════════════════════
# 3. Template literal conversion
# ═══════════════════════════════════════════════════════════════════


class TestTemplateLiteralConversion:
    def test_converts_simple_variable(self, sanitize):
        raw = "`linear-gradient(${ink} 1px)`"
        out, w = sanitize(raw, "test")
        assert "${ink}" not in out
        assert any("converted" in x for x in w)

    def test_converts_dotted_variable(self, sanitize):
        raw = "`background: ${tokens.paper}`"
        out, w = sanitize(raw, "test")
        assert "${tokens.paper}" not in out
        assert "tokens.paper" in out

    def test_converts_multiple_placeholders(self, sanitize):
        raw = "`color: ${primary}; size: ${secondary}px`"
        out, w = sanitize(raw, "test")
        assert "${primary}" not in out
        assert "${secondary}" not in out
        assert "primary" in out
        assert "secondary" in out
        assert any("converted" in x for x in w)

    def test_preserves_complex_expressions(self, sanitize):
        """Expressions that aren't simple variable names should be left as-is."""
        raw = "`items: ${items.length + 1} total`"
        out, w = sanitize(raw, "test")
        # Complex expression: not converted
        assert "${items.length + 1}" in out or "items.length + 1" in out

    def test_no_placeholder_no_conversion(self, sanitize):
        raw = "`background: transparent`"
        out, w = sanitize(raw, "test")
        assert "`background: transparent`" in out or "background: transparent" in out
        assert not any("converted" in x for x in w)


# ═══════════════════════════════════════════════════════════════════
# 4. Truncation detection
# ═══════════════════════════════════════════════════════════════════


class TestTruncationDetection:
    def test_detects_unbalanced_braces(self, sanitize):
        out, w = sanitize("function f() { return <div>", "test")
        assert any("truncated" in x or "unbalanced" in x for x in w)
        assert any("short" in x for x in w)

    def test_detects_trailing_open_tag(self, sanitize):
        out, w = sanitize("function f() { return <div", "test")
        assert any("truncated" in x for x in w)

    def test_detects_trailing_exit(self, sanitize):
        out, w = sanitize("\n  exit", "test")
        assert any("truncated" in x for x in w)

    def test_full_component_no_truncation(self, sanitize):
        raw = (
            "export default function Hero() {\n"
            "  return (\n"
            "    <section className='hero'>\n"
            "      <h1>Hello</h1>\n"
            "      <p>Test component with sufficient length</p>\n"
            "      <p>More content to exceed 200 char threshold easily achieved</p>\n"
            "    </section>\n"
            "  );\n"
            "}\n"
        )
        out, w = sanitize(raw, "test")
        assert not any("truncated" in x for x in w)
        assert not any("unbalanced" in x for x in w)
        assert not any("short" in x for x in w)


# ═══════════════════════════════════════════════════════════════════
# 5. Brace-in-string false positives (Fix 3 target)
# ═══════════════════════════════════════════════════════════════════


class TestBraceInStrings:
    def test_braces_in_double_quotes(self, sanitize):
        """Braces inside double-quoted strings should NOT trigger unbalanced warning."""
        raw = (
            'const url = "https://api.com/{id}";\n'
            'export default function F() {\n'
            '  return <div />;\n'
            '}\n'
        )
        out, w = sanitize(raw, "test")
        # Fix #3: should NOT have "unbalanced" warning
        assert not any("unbalanced" in x for x in w)

    def test_braces_in_single_quotes(self, sanitize):
        """Braces inside single-quoted strings should NOT trigger unbalanced warning."""
        raw = (
            "const msg = '{ count: 5 }';\n"
            "export default function F() {\n"
            "  return <div />;\n"
            "}\n"
        )
        out, w = sanitize(raw, "test")
        assert not any("unbalanced" in x for x in w)

    def test_braces_in_template_literal(self, sanitize):
        """Braces inside backtick template literals should NOT trigger unbalanced warning."""
        raw = (
            'const cls = `item-${index}`;\n'
            'export default function F() {\n'
            '  return <div className={cls} />;\n'
            '}\n'
        )
        out, w = sanitize(raw, "test")
        assert not any("unbalanced" in x for x in w)

    def test_braces_in_line_comments(self, sanitize):
        """Braces inside // comments should NOT trigger unbalanced warning."""
        raw = (
            "// { this is a comment with braces }\n"
            "export default function F() {\n"
            "  return <div />;\n"
            "}\n"
        )
        out, w = sanitize(raw, "test")
        assert not any("unbalanced" in x for x in w)

    def test_braces_in_block_comments(self, sanitize):
        """Braces inside /* */ comments should NOT trigger unbalanced warning."""
        raw = (
            "/* { count: 5, items: [1, 2, 3] } */\n"
            "export default function F() {\n"
            "  return <div />;\n"
            "}\n"
        )
        out, w = sanitize(raw, "test")
        assert not any("unbalanced" in x for x in w)
