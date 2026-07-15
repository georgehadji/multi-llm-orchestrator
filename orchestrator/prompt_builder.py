"""
orchestrator/prompt_builder.py
──────────────────────────────
Single source of truth for all LLM prompt templates used by the orchestrator.

Rules:
  - No I/O, no asyncio, no engine imports.
  - All public methods are static — no instance state.
  - Returns plain strings or (user_prompt, system_prompt) tuples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.models import AttemptRecord


class DecompositionPrompt:
    """Builds the project decomposition prompt sent to the decomposition model."""

    @staticmethod
    def build(
        project: str,
        criteria: str,
        app_context_block: str,
        valid_types: list[str],
    ) -> str:
        return (
            f"You are a project decomposition engine. Break this project into\n"
            f"atomic, executable tasks.\n"
            f"\n"
            f"PROJECT: {project}\n"
            f"\n"
            f"SUCCESS CRITERIA: {criteria}\n"
            f"{app_context_block}"
            f"Return ONLY a JSON array. Each element must have:\n"
            f'- "id": string (e.g., "task_001")\n'
            f'- "type": one of {valid_types}\n'
            f'- "prompt": detailed instruction for the task executor. For code_generation tasks, MUST include:\n'
            f"  - Code must be THOROUGHLY COMMENTED\n"
            f"  - EVERY file MUST start with: /** Author: Georgios-Chrysovalantis Chatzivantsidis */\n"
            f'- "dependencies": list of task id strings this depends on (empty if none)\n'
            f'- "hard_validators": list of validator names \u2014 ONLY use these for code tasks:\n'
            f'  - "python_syntax": only for code_generation tasks that produce Python code\n'
            f'  - "json_schema": only for tasks that must return valid JSON\n'
            f'  - "pytest": only for code_generation tasks with runnable tests\n'
            f'  - "ruff": only for code_generation tasks requiring lint checks\n'
            f'  - "latex": only for tasks producing LaTeX documents\n'
            f'  - "length": for tasks requiring minimum/maximum output length\n'
            f"  - Use [] (empty list) for non-code tasks (reasoning, writing, analysis, evaluation)\n"
            f"\n"
            f"RULES:\n"
            f"- Tasks must be atomic (one clear deliverable each)\n"
            f"- Dependencies must form a DAG (no cycles)\n"
            f"- Include code_review tasks after code_generation tasks\n"
            f"- Include at least one evaluation task at the end\n"
            f"- 5-15 tasks total for a medium project\n"
            f"- Do NOT add hard_validators to reasoning, writing, analysis, or evaluation tasks\n"
            f"\n"
            f"Return ONLY the JSON array, no markdown fences, no explanation."
        )


class SystemPrompt:
    """Builds the system prompt for task execution, varying by quality mode."""

    @staticmethod
    def build(task_type: str = "", mode: str = "production", target_language: str = "") -> str:
        prompt = SystemPrompt._production(task_type) if mode == "production" else SystemPrompt._standard()
        if target_language:
            prompt = SystemPrompt._inject_language_guidance(prompt, target_language)
        return prompt

    @staticmethod
    def karpathy_guidelines() -> str:
        """Karpathy behavioral principles — injected into every agent system prompt."""
        return (
            "\n\n## Behavioral Guidelines\n\n"
            "### 1. Think Before Coding\n"
            "- State assumptions explicitly. If uncertain, ASK before implementing.\n"
            "- If multiple interpretations exist, present ALL of them.\n"
            "- If a simpler approach exists, say so. Push back when warranted.\n"
            "- If something is unclear, STOP. Name what is confusing. Ask.\n\n"
            "### 2. Simplicity First\n"
            "- Minimum code that solves the problem. Nothing speculative.\n"
            "- No features beyond what was asked. No abstractions for single-use code.\n"
            '- No "flexibility" or "configurability" that was not requested.\n'
            "- If 200 lines could be 50, rewrite it.\n"
            '- Ask: "Would a senior engineer say this is overcomplicated?"\n\n'
            "### 3. Surgical Changes\n"
            "- Touch only what you must. Clean up only your own mess.\n"
            '- Do not "improve" adjacent code, comments, or formatting.\n'
            "- Do not refactor things that are not broken.\n"
            "- Match existing style, even if you would do it differently.\n"
            "- The test: Every changed line traces directly to the user request.\n\n"
            "### 4. Goal-Driven Execution\n"
            "- Define success criteria. Loop until verified.\n"
            '- "Add validation" -> "Write tests for invalid inputs, then make them pass"\n'
            '- "Fix the bug" -> "Write a test that reproduces it, then make it pass"\n'
            "- For multi-step tasks, state a brief plan with verification per step."
        )

    @staticmethod
    def _standard() -> str:
        return (
            "You are an expert software engineer executing a task. "
            "Produce high-quality, complete output. "
            "Follow best practices and ensure all code is valid and runnable."
            + SystemPrompt.karpathy_guidelines()
        )

    @staticmethod
    def _production(task_type: str = "") -> str:
        base = (
            "You are a senior software engineer delivering production-grade output. "
            "Requirements:\n"
            "1. Full type annotations on every function and class.\n"
            "2. Comprehensive error handling and input validation.\n"
            "3. Unit tests for every public function (pytest style).\n"
            "4. Docstrings on every module, class, and public function.\n"
            "5. Logging via the standard library logger (not print).\n"
            "6. No TODOs, no placeholder implementations.\n"
            "7. Follow SOLID principles and keep cyclomatic complexity <= 10.\n"
            "8. Include a brief inline comment for any non-obvious logic.\n"
        )
        if task_type in ("code_gen", "code_generation"):
            base += (
                "9. Return ONLY raw code -- no markdown fences, no prose outside code.\n"
                "10. Code must pass mypy --strict.\n"
            )
        return base + SystemPrompt.karpathy_guidelines()

    @staticmethod
    def _inject_language_guidance(prompt: str, target_language: str) -> str:
        """Replace Python-specific requirements with target-language best practices.

        Appends language-specific guidance covering:
        - Industry best practices (semantic HTML, BEM CSS, modern JS)
        - Security (CSP, XSS prevention, secure headers)
        - Open Graph / Twitter Card metadata
        - Accessibility (WCAG 2.1 AA)
        - Separate file structure (HTML + CSS + JS)
        """
        lang = target_language.lower().strip()

        # ── Strip Python-specific requirements (shared across all web languages) ──
        _python_replacements = {
            "Full type annotations on every function and class.": "",
            "Unit tests for every public function (pytest style).": "",
            "Docstrings on every module, class, and public function.": "",
            "Code must pass mypy --strict.": "",
            "Logging via the standard library logger (not print).": "",
            "Follow SOLID principles and keep cyclomatic complexity <= 10.": "",
            "Include a brief inline comment for any non-obvious logic.": "",
        }
        for old, new in _python_replacements.items():
            prompt = prompt.replace(old, new)

        # ── Build language-specific guidance block ──────────────────────
        guidance = SystemPrompt._build_web_guidance(lang)
        if guidance:
            # Strip trailing whitespace lines from Python requirements that became empty
            prompt = prompt.rstrip() + "\n\n" + guidance
        return prompt

    @staticmethod
    def _build_web_guidance(lang: str) -> str:
        """Build comprehensive web-development guidance for a target language."""
        if lang not in ("html", "css", "scss", "javascript", "js", "typescript", "ts"):
            return ""

        lines = ["## Web Development Requirements (Production-Grade)", ""]

        # ── HTML-specific ───────────────────────────────────────────────
        if lang == "html":
            lines += [
                "### HTML5 Best Practices",
                "1. Use semantic elements: <header>, <nav>, <main>, <section>, <article>, <aside>, <footer>.",
                "2. Every page MUST have: <!DOCTYPE html>, lang attribute, charset utf-8, viewport meta tag.",
                "3. Use aria-* attributes for accessibility (WCAG 2.1 AA minimum).",
                "4. All images MUST have alt text. Decorative images use alt=\"\".",
                "5. Form inputs MUST have associated <label> elements.",
                "6. Use heading hierarchy correctly: single <h1> per page, no skipped levels.",
                "7. External links: rel=\"noopener noreferrer\", internal links: plain <a href>.",
                "8. Lazy-load offscreen images and iframes: loading=\"lazy\".",
                "9. Use <picture> + srcset for responsive images.",
                "10. No inline styles. No inline event handlers (onclick=\"...\"). Keep HTML structural.",
                "",
                "### CSS (linked as separate style.css file)",
                "1. MUST be a SEPARATE .css file linked via <link rel=\"stylesheet\" href=\"style.css\">.",
                "2. Use CSS custom properties for colors, spacing, fonts, breakpoints.",
                "3. Mobile-first responsive: start with base styles, add @media (min-width: ...).",
                "4. Use logical properties: margin-inline, padding-block (RTL-compatible).",
                "5. BEM naming: .block__element--modifier for component classes.",
                "6. System font stack: font-family: system-ui, -apple-system, sans-serif.",
                "7. Smooth transitions: transition: 200ms ease on interactive elements.",
                "8. Focus styles: :focus-visible with visible outline. Never outline: none without replacement.",
                "9. Print stylesheet: @media print to hide nav/footer.",
                "10. Dark mode: @media (prefers-color-scheme: dark) with reduced brightness.",
                "",
                "### JavaScript (linked as separate script.js file)",
                "1. MUST be a SEPARATE .js file linked via <script src=\"script.js\" defer></script>.",
                "2. Use 'use strict' at the top of every .js file.",
                "3. Use const/let, never var. Prefer const by default.",
                "4. Use addEventListener, never inline onclick attributes.",
                "5. Wrap DOM-dependent code in DOMContentLoaded event.",
                "6. Debounce scroll/resize handlers (300ms).",
                "7. Use event delegation on parent containers, not per-element listeners.",
                "8. Prefer fetch() with async/await over XMLHttpRequest.",
                "9. Sanitize user input before inserting into DOM (textContent, not innerHTML).",
                "10. No eval(), no document.write(), no inline <script> tags.",
                "",
                "### Security (MANDATORY)",
                "1. Content Security Policy: <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';\">",
                "2. Prevent MIME sniffing: <meta http-equiv=\"X-Content-Type-Options\" content=\"nosniff\">",
                "3. Prevent clickjacking: <meta http-equiv=\"X-Frame-Options\" content=\"DENY\"> (or use frame-ancestors in CSP)",
                "4. Referrer policy: <meta name=\"referrer\" content=\"strict-origin-when-cross-origin\">",
                "5. All forms use HTTPS action URLs (or // for protocol-relative).",
                "6. Sanitize all user input before display — use textContent, not innerHTML.",
                "7. No secrets, API keys, or tokens in HTML/JS/CSS source.",
                "8. Subresource Integrity (SRI) for any CDN-loaded scripts/styles.",
                "",
                "### Open Graph + Twitter Card",
                "1. <meta property=\"og:title\" content=\"...\"> — page title (max 60 chars)",
                "2. <meta property=\"og:description\" content=\"...\"> — compelling description (max 160 chars)",
                "3. <meta property=\"og:image\" content=\"https://...\"> — 1200x630px JPG/PNG, full URL",
                "4. <meta property=\"og:image:width\" content=\"1200\"> <meta property=\"og:image:height\" content=\"630\">",
                "5. <meta property=\"og:url\" content=\"https://...\"> — canonical URL",
                "6. <meta property=\"og:type\" content=\"website\"> — or 'article' for blog posts",
                "7. <meta property=\"og:site_name\" content=\"...\">",
                "8. <meta name=\"twitter:card\" content=\"summary_large_image\">",
                "9. <meta name=\"twitter:title\" content=\"...\"> <meta name=\"twitter:description\" content=\"...\"> <meta name=\"twitter:image\" content=\"...\">",
                "10. OG image MUST be a real image URL or a generated placeholder. Use a data URI placeholder only if no image generation is available.",
                "",
                "### Favicon + App Icons",
                "1. <link rel=\"icon\" type=\"image/svg+xml\" href=\"/favicon.svg\"> (modern)",
                "2. <link rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"/favicon-32x32.png\">",
                "3. <link rel=\"apple-touch-icon\" sizes=\"180x180\" href=\"/apple-touch-icon.png\">",
                "4. <link rel=\"manifest\" href=\"/site.webmanifest\">",
                "5. <meta name=\"theme-color\" content=\"#...\"> for browser chrome color",
                "",
                "### OUTPUT FORMAT — Named File Blocks (MANDATORY)",
                "You MUST wrap each separate file in a named code block using this exact format:",
                "",
                "**index.html**",
                "```html",
                "<!DOCTYPE html>",
                "<html lang=\"en\">",
                "...complete HTML file with OG meta tags, CSP, favicon links...",
                "</html>",
                "```",
                "",
                "**style.css**",
                "```css",
                ":root { --color-primary: #...; }",
                "...complete CSS file...",
                "```",
                "",
                "**script.js**",
                "```javascript",
                "'use strict';",
                "...complete JS file...",
                "```",
                "",
                "CRITICAL RULES:",
                "1. Every file MUST be in its own **filename.ext** header + fenced code block.",
                "2. Do NOT put CSS inside <style> tags — use a separate **style.css** block.",
                "3. Do NOT put JS inside <script> tags — use a separate **script.js** block.",
                "4. The index.html file uses <link> and <script src> to reference external files.",
                "5. Each named block must contain the COMPLETE file content, not snippets.",
            ]
        # ── CSS-specific ─────────────────────────────────────────────────
        elif lang in ("css", "scss"):
            lines += [
                "### CSS Best Practices",
                "1. CSS custom properties at :root for design tokens (--color-primary, --spacing-md, etc.).",
                "2. Mobile-first: base styles for 375px, then @media (min-width: 768px), 1024px, 1440px.",
                "3. BEM naming: .block__element--modifier. No ID selectors for styling.",
                "4. Use logical properties: margin-inline, padding-block, inset-inline for RTL support.",
                "5. System font stack with fallbacks.",
                "6. Box-sizing: border-box globally via *, *::before, *::after.",
                "7. Smooth scrolling: scroll-behavior: smooth on html.",
                "8. Reduced motion: @media (prefers-reduced-motion: reduce) disables animations.",
                "9. Dark mode: @media (prefers-color-scheme: dark) inverts surface colors.",
                "10. Print styles: @media print hides non-content elements.",
                "11. Focus states: :focus-visible with outline-offset, never outline:none alone.",
                "12. Use modern layout: Grid for page structure, Flexbox for components.",
                "13. gap property instead of margin hacks for spacing.",
                "14. aspect-ratio for media containers instead of padding-top hacks.",
                "15. container-type / container queries for component-level responsiveness.",
            ]
        # ── JS/TS-specific ───────────────────────────────────────────────
        elif lang in ("javascript", "js", "typescript", "ts"):
            lines += [
                "### JavaScript Best Practices",
                "1. 'use strict' at file top. const by default, let when mutation needed. Never var.",
                "2. Use ES modules: export/import. Each file exports ONE primary concern.",
                "3. Async/await over raw Promises. Always try/catch async operations.",
                "4. Event delegation on parent containers. No per-element listeners for lists.",
                "5. Debounce scroll/resize/input handlers (300ms). Throttle animation handlers (16ms).",
                "6. Use AbortController for fetch() timeouts and cancellations.",
                "7. Prefer template literals over string concatenation.",
                "8. Optional chaining (?.) and nullish coalescing (??) over && checks.",
                "9. Avoid any type in TypeScript. Use unknown + type guards if type is uncertain.",
                "10. DOM access cached: const el = document.querySelector('#id') at module top.",
                "11. Use requestAnimationFrame for visual updates, setTimeout for deferred logic.",
                "12. No inline event handlers (onclick=\"\"). Use addEventListener.",
                "13. Error boundary pattern: window.addEventListener('error', handler).",
                "14. Use IntersectionObserver for scroll-triggered animations, not scroll events.",
                "15. LocalStorage/IndexedDB wrapped in try/catch (private browsing may throw).",
                "",
                "### Security",
                "1. All user input sanitized: use textContent (not innerHTML), DOMPurify if HTML needed.",
                "2. CSRF tokens on state-changing requests if backend is involved.",
                "3. No eval(), no new Function(), no innerHTML with user data.",
                "4. No secrets in client code. API keys live on backend only.",
                "5. Validate and sanitize URL parameters before use.",
                "6. Use rel=\"noopener noreferrer\" on target=\"_blank\" links.",
            ]
        return "\n".join(lines)


class DeltaPrompt:
    """
    Builds the enriched retry prompt after a failed validation attempt.

    Security: XML-style tags delimit the feedback block so adversarial LLM
    output cannot escape it with injected sentinels or tag sequences.
    All user-controlled fields are sanitised before embedding.
    """

    @staticmethod
    def build(original_prompt: str, record: "AttemptRecord") -> str:
        def _sanitize(text: str) -> str:
            """Strip XML delimiters and the plain sentinel from user-supplied data."""
            text = text.replace("<ORCHESTRATOR_FEEDBACK>", "")
            text = text.replace("</ORCHESTRATOR_FEEDBACK>", "")
            text = text.replace("PREVIOUS ATTEMPT FAILED:", "[PREVIOUS ATTEMPT]:")
            return text

        safe_reason = _sanitize(record.failure_reason)
        safe_validators = [_sanitize(v) for v in record.validators_failed]
        validators_str = ", ".join(safe_validators) if safe_validators else "none"

        snippet_section = ""
        if record.output_snippet:
            safe_snippet = _sanitize(record.output_snippet)
            snippet_section = f"\n- Output snippet: {safe_snippet}"

        additional_guidance = ""
        if "F821" in record.failure_reason or "Undefined name" in record.failure_reason:
            additional_guidance = (
                "\n\nIMPORT ERROR DETECTED: You used a name without importing it first.\n"
                "FIX: Add the required import statement at the TOP of your code.\n"
                "Example: 'from nba_api.stats.endpoints import playerdashboardbyyearoveryear'\n"
                "Example: 'from requests import RequestException'\n"
                "Check ALL function/class names used and ensure they are imported."
            )
        elif "F401" in record.failure_reason or "imported but unused" in record.failure_reason:
            additional_guidance = (
                "\n\nUNUSED IMPORT DETECTED: Remove imports you do not use.\n"
                "FIX: Either remove the unused import OR use the imported name in your code."
            )
        elif "E402" in record.failure_reason or "import not at top" in record.failure_reason:
            additional_guidance = (
                "\n\nIMPORT POSITION ERROR: Move all imports to the TOP of the file.\n"
                "FIX: Place all import statements before any code (functions, classes, etc.)."
            )
        elif (
            "unterminated triple-quoted string" in record.failure_reason
            or "Syntax error" in record.failure_reason
        ):
            additional_guidance = (
                "\n\nSYNTAX ERROR DETECTED: Unclosed string literal or code structure issue.\n"
                "FIX:\n"
                '1. Check ALL triple-quoted strings (""") are properly CLOSED\n'
                "2. Ensure all parentheses (), brackets [], and braces {} are matched\n"
                "3. Verify all string literals have matching opening and closing quotes\n"
                "4. Check that if/else/for/while blocks have proper indentation\n"
                "5. Run your code through a Python syntax checker BEFORE submitting\n"
                '\nCRITICAL: Every opening """ must have a closing """ on a later line!'
            )
        elif "invalid-syntax" in record.failure_reason:
            additional_guidance = (
                "\n\nSYNTAX ERROR DETECTED: Invalid Python code structure.\n"
                "FIX:\n"
                "1. Check for missing colons after if/for/while/def/class statements\n"
                "2. Ensure proper indentation (use spaces, not tabs)\n"
                "3. Verify all parentheses, brackets, and braces are properly matched\n"
                "4. Check that string literals are properly quoted and closed"
            )

        return (
            f"{original_prompt}\n\n"
            f"<ORCHESTRATOR_FEEDBACK>\n"
            f"PREVIOUS ATTEMPT FAILED:\n"
            f"- Attempt: {record.attempt_num}\n"
            f"- Model: {record.model_used}\n"
            f"- Reason: {safe_reason}\n"
            f"- Validators failed: {validators_str}"
            f"{snippet_section}"
            f"{additional_guidance}\n\n"
            f"Please correct specifically: {safe_reason}\n"
            f"</ORCHESTRATOR_FEEDBACK>"
        )


class CritiquePrompt:
    """Builds critique/review prompts for cross-model review and quality scoring."""

    @staticmethod
    def build(task_prompt: str, output: str) -> tuple[str, str]:
        """
        Cross-model review prompt (engine.py usage).

        Returns:
            (user_prompt, system_prompt) tuple.
        """
        user_prompt = (
            f"Review this output for correctness, completeness, and quality. "
            f"Be specific about flaws and suggest concrete improvements.\n\n"
            f"ORIGINAL TASK: {task_prompt}\n\n"
            f"OUTPUT TO REVIEW:\n{output}"
        )
        system_prompt = (
            "You are a critical reviewer. Find flaws, be specific. "
            "Apply the Simplicity First principle -- flag over-complication."
        )
        return user_prompt, system_prompt

    @staticmethod
    def build_score(
        task_prompt: str,
        output: str,
        task_type_value: str = "",
    ) -> str:
        """
        Scoring critique prompt (engine_core/critique_cycle.py usage).
        Returns a user prompt string; caller supplies the system prompt.
        """
        if task_type_value == "code_review":
            return (
                f"Review the following code for quality, correctness, and best practices.\n"
                f'Provide a score from 0.0 to 1.0 in JSON format: {{"score": 0.85, "reasoning": "..."}}\n\n'
                f"Original requirement:\n{task_prompt}\n\n"
                f"Code to review:\n```\n{output}\n```"
            )
        return (
            f"Review the following output for quality and correctness.\n"
            f'Provide a score from 0.0 to 1.0 in JSON format: {{"score": 0.85, "reasoning": "..."}}\n\n'
            f"Original prompt:\n{task_prompt}\n\n"
            f"Generated output:\n```\n{output}\n```"
        )


class RevisionPrompt:
    """Builds the revision prompt when a critique requires rework."""

    @staticmethod
    def build(
        task_prompt: str,
        critique_text: str,
        task_type_value: str = "",
    ) -> tuple[str, str]:
        """
        Returns:
            (user_prompt, system_prompt) tuple.
        """
        user_prompt = (
            f"{task_prompt}\n\n"
            f"[Revision required] {critique_text}\n"
            f"Please revise your previous response to address the above."
        )
        if task_type_value:
            system_prompt = (
                f"You are an expert executing a {task_type_value} task. "
                f"Produce high-quality, complete output."
            )
        else:
            system_prompt = "You are an expert. Produce high-quality, complete output."
        return user_prompt, system_prompt

    @staticmethod
    def with_critique_context(
        task_prompt: str,
        critique_report: "CritiqueReport",
        task_type_value: str = "",
    ) -> tuple[str, str]:
        """
        Build revision prompt from a typed CritiqueReport.

        Unlike build() which receives raw text, this method receives a
        structured CritiqueReport with severity levels, categories, and
        suggestions.

        Returns:
            (user_prompt, system_prompt) tuple.
        """
        user_prompt = (
            f"{task_prompt}\n\n"
            f"[Revision required] Score: {critique_report.score:.1f}/10\n"
            f"{critique_report.to_prompt_context()}\n\n"
            f"Focus on fixing BLOCKER and MAJOR items first."
        )
        if task_type_value:
            system_prompt = (
                f"You are an expert executing a {task_type_value} task. "
                f"Address each critique item by severity. "
                f"BLOCKER items must be fixed. MAJOR items should be fixed or explained."
            )
        else:
            system_prompt = (
                "You are an expert. Address each critique item by severity. "
                "BLOCKER items must be fixed. MAJOR items should be fixed or explained."
            )
        return user_prompt, system_prompt
