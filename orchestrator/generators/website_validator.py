"""
Website Quality Validator for DSDG
====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Validates generated websites against quality standards:
- Accessibility (WCAG 2.1 AA)
- Performance (Lighthouse)
- Design System Compliance
- Responsive Design
- SEO Basics
- Content Quality
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

# FIXED: from .design_system import DesignSystem, QualityCheck, QualityReport
from ..design_system import DesignSystem, QualityCheck, QualityReport

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class WebsiteQualityValidator:
    """
    Validate generated website against quality standards.

    Performs comprehensive checks across multiple dimensions:
    1. Accessibility (WCAG 2.1 AA)
    2. Performance (Lighthouse-like metrics)
    3. Design System Compliance
    4. Responsive Design
    5. SEO Basics
    6. Content Quality
    """

    def __init__(self):
        self._checks_performed: list[QualityCheck] = []

    async def validate(
        self,
        output_dir: Path,
        design_system: DesignSystem | None = None,
    ) -> QualityReport:
        """
        Run all quality checks on generated website.

        Parameters
        ----------
        output_dir : Directory containing generated website
        design_system : Optional design system for compliance checking

        Returns
        -------
        QualityReport with all check results
        """
        self._checks_performed = []

        # 1. Accessibility check (WCAG 2.1 AA)
        accessibility_check = await self._check_accessibility(output_dir)
        self._checks_performed.append(accessibility_check)

        # 2. Performance check (simulated Lighthouse)
        performance_check = await self._check_performance(output_dir)
        self._checks_performed.append(performance_check)

        # 3. Design System Compliance
        if design_system:
            compliance_check = await self._check_design_tokens(output_dir, design_system)
            self._checks_performed.append(compliance_check)

        # 4. Responsive Design
        responsive_check = await self._check_responsive(output_dir)
        self._checks_performed.append(responsive_check)

        # 5. SEO Basics
        seo_check = await self._check_seo(output_dir)
        self._checks_performed.append(seo_check)

        # 6. Content Quality
        content_check = await self._check_content_quality(output_dir)
        self._checks_performed.append(content_check)

        # 7. Security: Rate Limiting
        rate_limit_check = await self._check_rate_limiting(output_dir)
        self._checks_performed.append(rate_limit_check)

        # 8. Security: Email Verification Flow
        auth_check = await self._check_auth_flow(output_dir)
        self._checks_performed.append(auth_check)

        # 9. Security: No API Keys on Frontend
        secret_check = await self._check_secret_exposure(output_dir)
        self._checks_performed.append(secret_check)

        # Build report
        report = QualityReport(
            checks=self._checks_performed,
            lighthouse_score=performance_check.score if performance_check.passed else 0,
            wcag_level="AA" if accessibility_check.passed else "A",
            responsive_breakpoints_tested=4,
            seo_score=seo_check.score if seo_check.passed else 0,
        )

        logger.info(
            f"WebsiteQualityValidator: score={report.score:.2f}, "
            f"passed={report.passed}, lighthouse={report.lighthouse_score}"
        )

        return report

    # ── Shared HTML/asset helpers ────────────────────────────────────────────

    @staticmethod
    def _html_files(output_dir: Path) -> list[Path]:
        return [p for p in output_dir.glob("**/*.html") if "node_modules" not in p.parts]

    @staticmethod
    def _component_files(output_dir: Path) -> list[Path]:
        return [
            p
            for ext in ("*.tsx", "*.jsx")
            for p in output_dir.glob(f"**/{ext}")
            if "node_modules" not in p.parts
        ]

    @classmethod
    def _markup(cls, output_dir: Path) -> str:
        """All rendered markup and components concatenated.

        Checks read THIS, not just components. Globbing only tsx/jsx meant a
        static HTML site was never read at all: zero files scanned, zero issues
        found, a perfect score.
        """
        parts = []
        for path in cls._html_files(output_dir) + cls._component_files(output_dir):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
        return "\n".join(parts)

    @staticmethod
    def _css(output_dir: Path) -> str:
        parts = []
        for path in output_dir.glob("**/*.css"):
            if "node_modules" in path.parts:
                continue
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
        return "\n".join(parts)

    @staticmethod
    def _grade(passed_criteria: int, total_criteria: int) -> float:
        """Fraction of criteria met, rounded to 2dp. Graded, never binary."""
        if total_criteria <= 0:
            return 0.0
        return round(passed_criteria / total_criteria, 2)

    async def _check_accessibility(self, output_dir: Path) -> QualityCheck:
        """Check accessibility against deterministic WCAG 2.1 proxies.

        Graded: the score is the fraction of criteria met. Previously this
        globbed only tsx/jsx, so a static HTML site scored a constant 1.00 —
        it never read the page it was judging.
        """
        markup = self._markup(output_dir)
        css = self._css(output_dir)
        if not markup.strip():
            return QualityCheck(
                name="Accessibility (WCAG 2.1 AA)",
                passed=True,
                score=1.0,
                details="No markup to assess — not applicable.",
                applicable=False,
            )

        failures: list[str] = []

        def criterion(ok: bool, complaint: str) -> bool:
            if not ok:
                failures.append(complaint)
            return ok

        html_only = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore") for p in self._html_files(output_dir)
        )

        # 1. Document language.
        criterion(
            not html_only.strip() or re.search(r"<html[^>]*\slang=", html_only, re.I) is not None,
            "<html> has no lang attribute",
        )
        # 2. Every image carries an alt attribute.
        imgs = re.findall(r"<img\b[^>]*>", markup, re.I)
        criterion(
            all(re.search(r"\salt\s*=", tag, re.I) for tag in imgs),
            f"{sum(1 for t in imgs if not re.search(r'.salt.s*=', t, re.I))} <img> without alt",
        )
        # 3. Exactly one h1.
        h1s = len(re.findall(r"<h1\b", markup, re.I))
        criterion(h1s == 1, f"expected exactly one <h1>, found {h1s}")
        # 4. Landmarks rather than a soup of divs.
        criterion(
            any(f"<{tag}" in markup.lower() for tag in ("main", "nav", "header", "footer")),
            "no landmark elements (main/nav/header/footer)",
        )
        # 5. Form inputs are labelled.
        inputs = re.findall(r"<input\b[^>]*>", markup, re.I)
        labelled = markup.lower().count("<label")
        criterion(
            not inputs or labelled >= len([i for i in inputs if 'type="hidden"' not in i.lower()]),
            f"{len(inputs)} input(s) but only {labelled} <label>(s)",
        )
        # 6. Buttons have an accessible name.
        empty_buttons = len(
            re.findall(r"<button\b(?![^>]*aria-label)[^>]*>\s*</button>", markup, re.I)
        )
        criterion(empty_buttons == 0, f"{empty_buttons} button(s) with no accessible name")
        # 7. Visible focus indication.
        criterion(
            ":focus" in css or "focus:" in markup or ":focus-visible" in css,
            "no focus styles for keyboard navigation",
        )
        # 8. Zoom is not disabled.
        criterion(
            not re.search(r"user-scalable\s*=\s*no|maximum-scale\s*=\s*1", markup, re.I),
            "viewport disables pinch zoom",
        )

        total = 8
        score = self._grade(total - len(failures), total)
        return QualityCheck(
            name="Accessibility (WCAG 2.1 AA)",
            passed=not failures,
            score=score,
            details=(
                f"{total - len(failures)}/{total} criteria met"
                + (f" — {'; '.join(failures[:4])}" if failures else "")
            ),
            recommendations=failures,
        )

    async def _check_performance(self, output_dir: Path) -> QualityCheck:
        """Check page weight and request shape against a budget.

        Graded against real weight. Previously only a >500KB JS bundle could
        lower the score and images were never measured, so every site scored a
        constant 0.96.
        """
        markup = self._markup(output_dir)
        if not markup.strip():
            return QualityCheck(
                name="Performance (Lighthouse)",
                passed=True,
                score=1.0,
                details="Nothing to weigh — not applicable.",
                applicable=False,
            )

        def total_bytes(patterns: tuple[str, ...]) -> int:
            size = 0
            for pattern in patterns:
                for path in output_dir.glob(f"**/{pattern}"):
                    if "node_modules" in path.parts:
                        continue
                    try:
                        size += path.stat().st_size
                    except OSError:
                        continue
            return size

        js = total_bytes(("*.js", "*.tsx", "*.jsx"))
        css_bytes = total_bytes(("*.css",))
        images = total_bytes(("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif", "*.avif", "*.svg"))

        failures: list[str] = []

        def criterion(ok: bool, complaint: str) -> None:
            if not ok:
                failures.append(complaint)

        # Budgets chosen to pass a well-built brochure site and fail a careless one.
        criterion(js <= 500 * 1024, f"JS {js / 1024:.0f}KB over the 500KB budget")
        criterion(css_bytes <= 150 * 1024, f"CSS {css_bytes / 1024:.0f}KB over the 150KB budget")
        criterion(
            images <= 2 * 1024 * 1024, f"images {images / 1024 / 1024:.1f}MB over the 2MB budget"
        )

        # Render-blocking third-party scripts in <head> without defer/async.
        head = re.search(r"<head\b.*?</head>", markup, re.I | re.S)
        blocking = 0
        if head:
            for tag in re.findall(r"<script\b[^>]*src=[^>]*>", head.group(0), re.I):
                if not re.search(r"\b(defer|async|type=[\"']module[\"'])", tag, re.I):
                    blocking += 1
        criterion(blocking <= 1, f"{blocking} render-blocking <script> in <head>")

        # Below-the-fold images should be lazy where there are several.
        img_count = len(re.findall(r"<img\b", markup, re.I))
        criterion(
            img_count <= 1 or 'loading="lazy"' in markup or "loading={'lazy'}" in markup,
            f'{img_count} images and none marked loading="lazy"',
        )

        total = 5
        score = self._grade(total - len(failures), total)
        return QualityCheck(
            name="Performance (Lighthouse)",
            passed=not failures,
            score=score,
            details=(
                f"JS {js / 1024:.1f}KB, CSS {css_bytes / 1024:.1f}KB, "
                f"images {images / 1024:.0f}KB, {blocking} blocking script(s)"
                + (f" — {'; '.join(failures[:3])}" if failures else "")
            ),
            recommendations=failures,
        )

    async def _check_design_tokens(
        self,
        output_dir: Path,
        design_system: DesignSystem,
    ) -> QualityCheck:
        """
        Check design system compliance.

        Checks:
        - Only design system colors are used
        - Typography matches design system
        - Spacing uses design tokens
        """
        issues = []

        # Get allowed colors from design system
        allowed_colors = {
            design_system.colors.primary.lower(),
            design_system.colors.accent.lower(),
            design_system.colors.surface.lower(),
            design_system.colors.surface_alt.lower(),
            design_system.colors.text_primary.lower(),
            design_system.colors.text_secondary.lower(),
            design_system.colors.border.lower(),
            design_system.colors.success.lower(),
            design_system.colors.error.lower(),
            # Common Tailwind utilities that are okay
            "#000000",
            "#ffffff",
            "#000",
            "#fff",
            "transparent",
            "currentColor",
        }

        # Find all TSX/JSX files
        component_files = list(output_dir.glob("**/*.tsx")) + list(output_dir.glob("**/*.jsx"))

        arbitrary_colors_found = []

        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8")

            # Find arbitrary hex colors (not in design system)
            hex_colors = re.findall(r"#[0-9a-fA-F]{3,6}", content)
            for color in hex_colors:
                if color.lower() not in allowed_colors:
                    arbitrary_colors_found.append(f"{file_path.name}: {color}")

        if arbitrary_colors_found:
            # Only flag if more than 3 arbitrary colors (allow some flexibility)
            if len(arbitrary_colors_found) > 3:
                issues.append(
                    f"Found {len(arbitrary_colors_found)} arbitrary colors not in design system"
                )

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.6, 1.0 - (len(issues) * 0.2))

        return QualityCheck(
            name="Design System Compliance",
            passed=passed,
            score=score,
            details=f"{'Compliant' if passed else 'Non-compliant colors found'}",
            recommendations=[] if passed else ["Use only design system color tokens"],
        )

    async def _check_responsive(self, output_dir: Path) -> QualityCheck:
        """Score responsive readiness on DISTINCT breakpoints actually declared.

        Previously: `breakpoints/4 if has_responsive_classes else 0.5`, where
        has_responsive_classes was set only by Tailwind prefixes in tsx/jsx. A
        static HTML site with eight media queries scored the same 0.50 as one
        with none — the check was blind to its primary output format.
        """
        markup = self._markup(output_dir)
        css = self._css(output_dir)
        if not markup.strip() and not css.strip():
            return QualityCheck(
                name="Responsive Design",
                passed=True,
                score=1.0,
                details="Nothing to assess — not applicable.",
                applicable=False,
            )

        # Distinct breakpoints, so ten copies of the same one do not read as ten.
        # Units matter: em/rem breakpoints are common in mobile-first sheets and
        # a px-only regex scored them as zero.
        widths = {
            (value, unit.lower())
            for value, unit in re.findall(
                r"@media[^{]*?(?:min|max)-width:\s*([\d.]+)(px|em|rem)", css, re.I
            )
        }
        # Tailwind responsive prefixes count as breakpoints too.
        tailwind = {p for p in ("sm:", "md:", "lg:", "xl:", "2xl:") if p in markup}
        breakpoints = len(widths) + len(tailwind)

        # Graded and monotonic in breakpoint count.
        ladder = {0: 0.0, 1: 0.4, 2: 0.6, 3: 0.8}
        base = ladder.get(breakpoints, 1.0)

        penalties: list[str] = []

        # Mobile-first: base styles are the phone layout and `min-width` queries
        # add capability upward. A `max-width`-dominant sheet is desktop-first —
        # responsive, but built the wrong way round, so a phone parses the wide
        # layout and then overrides it. A handful of max-width rules is normal
        # in a mobile-first sheet, so this triggers only when they DOMINATE.
        min_q = len(re.findall(r"@media[^{]*?min-width", css, re.I))
        max_q = len(re.findall(r"@media[^{]*?max-width", css, re.I))
        if max_q and max_q > min_q:
            penalties.append(
                f"desktop-first CSS: {max_q} max-width vs {min_q} min-width queries — "
                f"write mobile-first (base styles for phones, min-width to scale up)"
            )
            base -= 0.3

        if not re.search(r'<meta[^>]+name=["\']viewport["\']', markup, re.I):
            penalties.append("no viewport meta tag")
            base -= 0.2
        fixed_container = re.search(r"\.(container|wrapper)\s*\{[^}]*width:\s*\d{3,}px", css, re.I)
        if fixed_container:
            penalties.append("container pinned to a fixed pixel width")
            base -= 0.2

        score = round(max(0.0, min(1.0, base)), 2)
        return QualityCheck(
            name="Responsive Design",
            passed=breakpoints >= 3 and not penalties,
            score=score,
            details=(
                f"Breakpoints declared: {breakpoints} "
                f"({len(widths)} CSS, {len(tailwind)} utility)"
                + (f" — {'; '.join(penalties)}" if penalties else "")
            ),
            recommendations=penalties,
        )

    async def _check_seo(self, output_dir: Path) -> QualityCheck:
        """
        Check SEO basics.

        Checks:
        - Meta tags present
        - Title tag exists
        - Heading hierarchy (h1 → h2 → h3)
        - Semantic HTML
        """
        issues = []
        recommendations = []

        # Find main page file
        page_files = (
            list(output_dir.glob("page.tsx"))
            + list(output_dir.glob("page.jsx"))
            + list(output_dir.glob("App.tsx"))
            + list(output_dir.glob("App.jsx"))
            + list(output_dir.glob("index.html"))
        )

        if not page_files:
            # No main page found, check any HTML/TSX file
            page_files = list(output_dir.glob("**/*.tsx"))[:1]

        for file_path in page_files:
            content = file_path.read_text(encoding="utf-8")

            # Check for title
            if "<title" not in content and "metadata" not in content.lower():
                recommendations.append(f"{file_path.name}: Add page title/metadata")

            # Check for meta description
            if "description" not in content.lower() and "<meta" not in content:
                recommendations.append(f"{file_path.name}: Add meta description")

            # Check heading hierarchy
            has_h1 = "<h1" in content or "<H1" in content
            has_h2 = "<h2" in content or "<H2" in content

            if not has_h1:
                issues.append(f"{file_path.name}: Missing <h1> heading")

            if has_h1 and not has_h2:
                recommendations.append("Consider adding <h2> subheadings for structure")

        passed = len(issues) == 0
        score = 0.95 if passed and not recommendations else max(0.7, 1.0 - (len(issues) * 0.15))

        return QualityCheck(
            name="SEO Basics",
            passed=passed,
            score=score,
            details=f"{'Good SEO structure' if passed else 'SEO issues found'}",
            recommendations=recommendations,
        )

    async def _check_content_quality(self, output_dir: Path) -> QualityCheck:
        """
        Check content quality.

        Checks:
        - No Lorem ipsum placeholder text
        - No TODO/FIXME comments in final code
        - No generic placeholder content
        """
        issues = []

        # Kept deliberately tight: a false positive blocks a good build, so only
        # unambiguous placeholders belong here.
        placeholder_patterns = [
            r"lorem\s+ipsum",
            r"TODO[:\s]",
            r"FIXME[:\s]",
            r"placeholder\s+content",
            r"your\s+content\s+here",
            r"insert\s+.*\s+here",
            r"untitled\s+project",
            r"your\s+company\s+name",
            r"company\s+name\s+here",
            r"@example\.com",
        ]
        # A title that is empty or generic ships straight into search results and
        # browser tabs — highest-signal placeholder there is.
        generic_title_patterns = [
            r"^document$",
            r"^home$",
            r"^index$",
            r"^new\s+page$",
            r"^my\s+site$",
        ]

        # Scan rendered markup, not just components. Globbing only tsx/jsx meant a
        # static HTML site was never read at all: zero files scanned, zero issues
        # found, perfect score. HTML is the primary deliverable for framework=html.
        content_files = [
            p
            for ext in ("*.tsx", "*.jsx", "*.html", "*.ts", "*.js")
            for p in output_dir.glob(f"**/{ext}")
            if "node_modules" not in p.parts
        ]

        for file_path in content_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for pattern in placeholder_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues.append(
                        f"{file_path.name}: Found placeholder content ({len(matches)} matches)"
                    )

        for html_file in [p for p in output_dir.glob("**/*.html") if "node_modules" not in p.parts]:
            try:
                markup = html_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            title_match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            if not title:
                issues.append(f"{html_file.name}: missing or empty <title>")
            elif any(re.search(pat, title, re.IGNORECASE) for pat in generic_title_patterns):
                issues.append(f"{html_file.name}: generic placeholder <title> ({title!r})")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.5, 1.0 - (len(issues) * 0.15))

        details = "No placeholder content found" if passed else "Placeholder content detected"

        return QualityCheck(
            name="Content Quality",
            passed=passed,
            score=score,
            details=details,
            recommendations=[] if passed else ["Replace placeholder content with real copy"],
        )

    # ── Security Checks ──────────────────────────────────────────────────────

    @staticmethod
    def _has_server_endpoints(output_dir: Path) -> bool:
        """True when the artifact CONTAINS server-side code that could be abused.

        Deliberately narrower than "the page submits data somewhere". A static
        export whose form posts to /api/contact carries no backend to rate-limit
        — the endpoint lives in another deployable — so judging it here is a
        false positive that penalises every brochure site with a contact form.
        Applicability requires request-handling code to actually be present.
        """
        import re as _re

        for api_dir in ("pages/api", "app/api", "api", "functions", "netlify/functions"):
            candidate = output_dir / api_dir
            if candidate.is_dir() and any(candidate.rglob("*")):
                return True

        if any(output_dir.rglob("*.py")):
            return True

        # Server-side request handlers, not client-side calls to someone else's.
        handler_markers = [
            r"app\.(post|put|patch)\s*\(",
            r"router\.(post|put|patch)\s*\(",
            r"export\s+(async\s+)?function\s+(POST|PUT|PATCH)\b",
            r"createServer\s*\(",
            r"fastify\.(post|put|patch)\s*\(",
        ]
        scan = [
            p
            for ext in ("*.js", "*.jsx", "*.ts", "*.tsx", "*.mjs")
            for p in output_dir.rglob(ext)
            if "node_modules" not in p.parts
        ]
        for fpath in scan[:100]:
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(_re.search(pat, content, _re.IGNORECASE) for pat in handler_markers):
                return True
        return False

    async def _check_rate_limiting(self, output_dir: Path) -> QualityCheck:
        """Verify contact/auth endpoints include IP-based rate limiting."""
        import re

        # Applicability guard. Without it this check fails EVERY static site
        # forever, which flattens the aggregate score into a constant and
        # destroys its ability to discriminate good output from bad. Mirrors the
        # "not applicable" behaviour _check_auth_flow already uses.
        if not self._has_server_endpoints(output_dir):
            return QualityCheck(
                name="Rate Limiting",
                passed=True,
                score=1.0,
                details="No server-side endpoints or form submissions found — not applicable.",
                applicable=False,
            )

        rate_limit_patterns = [
            r"rate.limit",
            r"RateLimit",
            r"ratelimit",
            r"too.many.requests",
            r"status.*429|429.*Too Many",
            r"X-RateLimit",
            r"express-rate-limit",
            r"upstash/ratelimit",
            r"@upstash/ratelimit",
            r"maxRequests",
            r"max_requests",
            r"Ratelimit\(",
        ]
        files_to_check = (
            list(output_dir.rglob("*.ts"))
            + list(output_dir.rglob("*.tsx"))
            + list(output_dir.rglob("*.js"))
            + list(output_dir.rglob("*.jsx"))
            + list(output_dir.rglob("*.py"))
        )
        # Scan contact/auth files first
        priority = [
            f
            for f in files_to_check
            if any(k in f.name.lower() for k in ("contact", "auth", "register", "signup", "login"))
        ]
        rest = [f for f in files_to_check if f not in priority]
        found = False
        for fpath in (priority + rest)[:50]:
            try:
                c = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat in rate_limit_patterns:
                    if re.search(pat, c, re.IGNORECASE):
                        found = True
                        break
            except Exception:
                continue
            if found:
                break

        passed = found
        return QualityCheck(
            name="Rate Limiting",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=(
                "Rate-limit patterns detected"
                if passed
                else (
                    "No rate limiting found. Contact forms and registration "
                    "endpoints must include IP-based rate limiting."
                )
            ),
            recommendations=(
                []
                if passed
                else [
                    "Add express-rate-limit or upstash/ratelimit middleware to API routes",
                    "Implement in-memory rate-limit map if no Redis available",
                    "Return HTTP 429 with Retry-After header when limit exceeded",
                ]
            ),
        )

    async def _check_auth_flow(self, output_dir: Path) -> QualityCheck:
        """Verify email verification flow exists in auth pages."""
        import re

        auth_patterns = [
            r"verifyEmail",
            r"verify.email",
            r"verify_email",
            r"emailVerification",
            r"email_verification",
            r"verificationToken",
            r"verification.token",
            r"checkEmail",
            r"check.email",
            r"sendVerification",
        ]
        auth_files = (
            list(output_dir.rglob("*auth*"))
            + list(output_dir.rglob("*signup*"))
            + list(output_dir.rglob("*register*"))
            + list(output_dir.rglob("*login*"))
            + list(output_dir.rglob("*verify*"))
        )
        if not auth_files:
            return QualityCheck(
                name="Email Verification",
                passed=True,
                score=1.0,
                details="No auth pages found — not applicable.",
                applicable=False,
                recommendations=[],
            )
        found = False
        for fpath in auth_files[:20]:
            try:
                c = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat in auth_patterns:
                    if re.search(pat, c, re.IGNORECASE):
                        found = True
                        break
            except Exception:
                continue
            if found:
                break

        passed = found
        return QualityCheck(
            name="Email Verification",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=(
                "Email verification flow detected"
                if passed
                else ("Auth pages found but no email verification flow detected.")
            ),
            recommendations=(
                []
                if passed
                else [
                    "Send verification email with a unique token after registration",
                    "Add /api/auth/verify-email route to mark user as verified",
                    "Prevent login until email is verified",
                ]
            ),
        )

    async def _check_secret_exposure(self, output_dir: Path) -> QualityCheck:
        """Scan for API keys or secrets leaked in frontend code."""
        import re

        secret_patterns = [
            # OpenAI / Anthropic / Google AI keys
            r"sk-(?:proj-)?[a-zA-Z0-9]{20,}",
            r"AIza[0-9A-Za-z\-_]{35}",
            # GitHub tokens (all variants)
            r"gh[opsu]_[a-zA-Z0-9]{36,}",
            r"github_pat_[a-zA-Z0-9_]{40,}",
            # HuggingFace
            r"hf_[a-zA-Z0-9]{34}",
            # Stripe live keys
            r"(?:sk|rk)_live_[a-zA-Z0-9]{24,}",
            # AWS access keys
            r"AKIA[0-9A-Z]{16}",
            # Supabase / Firebase config
            r"supabase\.(?:url|key|anon)",
            r"firebase\.(?:apiKey|authDomain|projectId)",
            # Database URLs with credentials
            r"postgres(?:ql)?://[^:]+:[^@]+@",
            r"mongodb(?:\+srv)?://[^:]+:[^@]+@",
            # Generic secret patterns
            r'[A-Z_]+_(?:SECRET|TOKEN|KEY|PASSWORD)\s*[:=]\s*["\x60\'(]',
            r"process\.env\.NEXT_PUBLIC_(?!.*URL\b)[A-Z_]+",
            # Sentry DSNs
            r"https://[a-f0-9]+@o\d+\.ingest\.sentry\.io/\d+",
        ]
        # .js/.ts/.mjs were missing here, so on a static site — where the code
        # lives in script.js — this security check scanned no JavaScript at all
        # and reported a clean bill of health for every artifact it was given.
        frontend_files = [
            f
            for ext in ("*.tsx", "*.jsx", "*.ts", "*.js", "*.mjs", "*.html")
            for f in output_dir.rglob(ext)
            if "node_modules" not in f.parts
        ]
        # Exclude API route files and server-only dirs (cross-platform safe):
        # a key in server-side code is not a frontend leak.
        frontend_files = [
            f for f in frontend_files if "api" not in f.parts and "server" not in f.parts
        ]
        leaks = []
        for fpath in frontend_files[:50]:
            try:
                c = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat in secret_patterns:
                    for m in re.finditer(pat, c, re.IGNORECASE):
                        match_text = m.group(0)
                        masked = match_text[:12] + "***" if len(match_text) > 12 else match_text
                        leaks.append(f"{fpath.relative_to(output_dir)}: {masked}")
            except Exception:
                continue

        passed = len(leaks) == 0
        return QualityCheck(
            name="Secret Exposure",
            passed=passed,
            score=0.0 if leaks else 1.0,
            details=(
                "No secrets found in frontend code"
                if passed
                else (
                    f"Potential secret exposure in {len(leaks)} location(s). "
                    "API keys must never appear in client-side code."
                )
            ),
            recommendations=leaks[:5] if not passed else [],
        )


async def validate_website(
    output_dir: Path,
    design_system: DesignSystem | None = None,
) -> QualityReport:
    """
    Convenience function to validate a website.

    Usage:
        report = await validate_website(
            output_dir=Path("./my-website"),
            design_system=design_system,
        )
    """
    validator = WebsiteQualityValidator()
    return await validator.validate(output_dir, design_system)
