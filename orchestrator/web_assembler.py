"""
Web Project Assembler
=====================
Author: Reasonix

Assembles raw task outputs (task_XXX_code_generation.html, .js, .css) into a
coherent, browser-ready web project structure:

    app/
    ├── index.html       ← best HTML task, with OG/CSP/security tags injected
    ├── style.css        ← extracted from all inline <style> blocks + .css tasks
    ├── script.js        ← extracted from all inline <script> blocks + .js tasks
    ├── favicon.svg      ← generated placeholder
    └── site.webmanifest ← generated manifest

Runs AFTER task output collection, as a counterpart to the Python
``ProjectAssembler``. Only active for non-Python projects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .models import ProjectState, TaskType

logger = logging.getLogger("orchestrator.web_assembler")

# ── Default OG / security tags injected if missing ──────────────────────────

_DEFAULT_OG_TAGS = """\
    <!-- Open Graph -->
    <meta property="og:title" content="{title}">
    <meta property="og:description" content="{description}">
    <meta property="og:image" content="og-image.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:url" content="">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="{title}">
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{title}">
    <meta name="twitter:description" content="{description}">
    <meta name="twitter:image" content="og-image.png">
    <!-- Security -->
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';">
    <meta http-equiv="X-Content-Type-Options" content="nosniff">
    <meta http-equiv="X-Frame-Options" content="DENY">
    <meta name="referrer" content="strict-origin-when-cross-origin">"""

_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="48" fill="#1a1a2e" stroke="#e94560" stroke-width="2"/>
  <text x="50" y="68" text-anchor="middle" font-size="50" fill="#e94560">✦</text>
</svg>"""

_SITE_WEBMANIFEST = """{
  "name": "{title}",
  "short_name": "{title}",
  "start_url": ".",
  "display": "standalone",
  "theme_color": "#1a1a2e",
  "background_color": "#0f0f23",
  "icons": [
    { "src": "favicon.svg", "sizes": "any", "type": "image/svg+xml" }
  ]
}"""


class WebProjectAssembler:
    """Assembles raw task outputs into a coherent web project."""

    def assemble(self, output_dir: Path, state: ProjectState) -> list[str]:
        """
        Entry point. Scans task outputs, extracts HTML/CSS/JS, writes app/.

        Returns list of created file paths relative to output_dir.
        """
        app_dir = output_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)

        # Collect all task outputs by extension
        html_parts: list[str] = []
        css_parts: list[str] = []
        js_parts: list[str] = []
        best_html: str = ""
        best_html_len: int = 0

        for task_id in state.execution_order or state.results:
            result = state.results.get(task_id)
            task = state.tasks.get(task_id)
            if not result or not result.output:
                continue

            output = result.output
            ext = _guess_ext(task, output)

            if ext == ".html":
                html_parts.append(output)
                # Track the best (largest) HTML file for index.html
                stripped = _strip_fences(output)
                if len(stripped) > best_html_len:
                    best_html = stripped
                    best_html_len = len(stripped)
            elif ext == ".css":
                css_parts.append(_extract_css_from_output(output))
            elif ext in (".js", ".ts"):
                js_parts.append(_extract_js_from_output(output))

        # Also extract inline <style> and <script> from HTML outputs
        for html in html_parts:
            css_parts.append(_extract_inline_css(html))
            js_parts.append(_extract_inline_js(html))

        created: list[str] = []

        # ── Write index.html ─────────────────────────────────────────────
        if best_html:
            title = state.project_description[:60]
            description = state.success_criteria[:150] if state.success_criteria else title
            index_html = _inject_meta_tags(
                _strip_inline_css_js(best_html), title=title, description=description
            )
            (app_dir / "index.html").write_text(index_html, encoding="utf-8")
            created.append("index.html")
            logger.info("Web Assembler: wrote index.html (%d chars)", len(index_html))
        else:
            logger.warning("Web Assembler: no HTML content found — creating minimal index.html")
            _write_minimal_html(app_dir, state)
            created.append("index.html")

        # ── Write style.css ──────────────────────────────────────────────
        if css_parts:
            combined_css = "\n\n/* --- next section --- */\n\n".join(css_parts)
            (app_dir / "style.css").write_text(combined_css, encoding="utf-8")
            created.append("style.css")
            logger.info("Web Assembler: wrote style.css (%d chars)", len(combined_css))

        # ── Write script.js ──────────────────────────────────────────────
        if js_parts:
            combined_js = "\n\n// --- next section ---\n\n".join(js_parts)
            (app_dir / "script.js").write_text(combined_js, encoding="utf-8")
            created.append("script.js")
            logger.info("Web Assembler: wrote script.js (%d chars)", len(combined_js))

        # ── Write favicon + manifest ─────────────────────────────────────
        (app_dir / "favicon.svg").write_text(_FAVICON_SVG, encoding="utf-8")
        created.append("favicon.svg")
        manifest = _SITE_WEBMANIFEST.format(title=state.project_description[:40])
        (app_dir / "site.webmanifest").write_text(manifest, encoding="utf-8")
        created.append("site.webmanifest")

        return created


# ── Internal helpers ──────────────────────────────────────────────────────────


def _guess_ext(task: Any, output: str) -> str:
    """Guess file extension from task type + output content."""
    from .output_writer import _ext_for as _writer_ext_for

    try:
        target_lang = getattr(task, "target_language", "") or ""
        return _writer_ext_for(
            task.type if task else TaskType.CODE_GEN, output, target_language=target_lang
        )
    except Exception:
        # Fallback: check content for HTML tags
        if re.search(r"<!DOCTYPE\s+html|<html|<head|<body", output, re.IGNORECASE):
            return ".html"
        # Check for JS patterns
        if re.search(r"^(import |const |let |function |'use strict')", output, re.MULTILINE):
            return ".js"
        return ".html"  # default for web projects


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_css_from_output(output: str) -> str:
    """Extract CSS from a .css task output (strip fences, return content)."""
    return _strip_fences(output)


def _extract_js_from_output(output: str) -> str:
    """Extract JS from a .js task output (strip fences, return content)."""
    return _strip_fences(output)


def _extract_inline_css(html: str) -> str:
    """Extract content from <style> blocks in HTML."""
    matches = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    return "\n\n".join(m.strip() for m in matches if m.strip())


def _extract_inline_js(html: str) -> str:
    """Extract content from non-CDN <script> blocks in HTML."""
    # Match <script> blocks that are NOT loading external URLs
    matches = re.findall(
        r"<script(?![^>]*\bsrc\s*=\s*['\"])[^>]*>(.*?)</script>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    return "\n\n".join(m.strip() for m in matches if m.strip())


def _strip_inline_css_js(html: str) -> str:
    """Remove inline <style> and non-CDN <script> blocks from HTML."""
    # Remove <style> blocks
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove non-CDN <script> blocks
    html = re.sub(
        r"<script(?![^>]*\bsrc\s*=\s*['\"])[^>]*>.*?</script>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return html


def _inject_meta_tags(html: str, title: str = "", description: str = "") -> str:
    """Inject OG, Twitter Card, CSP, and security meta tags into <head> if missing."""
    tags = _DEFAULT_OG_TAGS.format(title=title or "Project", description=description or "")

    # Only inject if <head> exists and OG tags are missing
    if "<head" not in html.lower():
        return html
    if "og:title" in html.lower():
        logger.debug("OG tags already present — skipping injection")
        return html

    # Insert after <head> or after <meta charset> if present
    charset_match = re.search(r"<meta\s+charset[^>]*>", html, re.IGNORECASE)
    if charset_match:
        insert_pos = charset_match.end()
        html = html[:insert_pos] + "\n" + tags + "\n" + html[insert_pos:]
    else:
        head_match = re.search(r"<head[^>]*>", html, re.IGNORECASE)
        if head_match:
            insert_pos = head_match.end()
            html = html[:insert_pos] + "\n" + tags + "\n" + html[insert_pos:]

    return html


def _write_minimal_html(app_dir: Path, state: ProjectState) -> None:
    """Write a minimal index.html when no HTML content is found."""
    title = state.project_description[:60]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {_DEFAULT_OG_TAGS.format(title=title, description=state.success_criteria[:150])}
    <link rel="stylesheet" href="style.css">
    <link rel="icon" type="image/svg+xml" href="favicon.svg">
    <link rel="manifest" href="site.webmanifest">
</head>
<body>
    <h1>{title}</h1>
    <p>Generated by AI Orchestrator — see <code>script.js</code> for logic.</p>
    <script src="script.js" defer></script>
</body>
</html>"""
    (app_dir / "index.html").write_text(html, encoding="utf-8")
