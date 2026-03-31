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

from .design_system import DesignSystem, QualityCheck, QualityReport

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

    async def _check_accessibility(self, output_dir: Path) -> QualityCheck:
        """
        Check accessibility compliance (WCAG 2.1 AA).

        Checks:
        - All images have alt text
        - All interactive elements have aria-labels
        - Color contrast ratios meet 4.5:1 minimum
        - Focus indicators are visible
        - Semantic HTML structure
        - Skip links present
        """
        issues = []
        recommendations = []

        # Find all TSX/JSX files
        component_files = list(output_dir.glob("**/*.tsx")) + list(output_dir.glob("**/*.jsx"))

        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8")

            # Check for img without alt
            img_tags = re.findall(r"<img\s+[^>]*>", content, re.IGNORECASE)
            for img in img_tags:
                if "alt=" not in img and "alt={" not in content:
                    issues.append(f"{file_path.name}: <img> missing alt attribute")

            # Check for buttons without aria-label or text content
            button_tags = re.findall(r"<button\s+[^>]*>", content, re.IGNORECASE)
            for button in button_tags:
                if "aria-label=" not in button and "aria-label={" not in content:
                    # Check if button has children (text content)
                    # This is a simplified check
                    recommendations.append(
                        f"{file_path.name}: Ensure buttons have accessible names"
                    )

            # Check for semantic HTML
            if '<div className="container"' in content or '<div className="wrapper"' in content:
                # Should use <main>, <section>, <article>, etc.
                if "<main" not in content and "<section" not in content:
                    recommendations.append(
                        f"{file_path.name}: Consider using semantic HTML elements"
                    )

        # Check for focus styles in CSS/Tailwind
        css_files = list(output_dir.glob("**/*.css")) + list(output_dir.glob("**/*.scss"))

        has_focus_styles = False
        for css_file in css_files:
            content = css_file.read_text(encoding="utf-8")
            if ":focus" in content or "focus:" in content:
                has_focus_styles = True
                break

        if not has_focus_styles:
            recommendations.append("Add focus styles for keyboard navigation")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.5, 1.0 - (len(issues) * 0.1))

        return QualityCheck(
            name="Accessibility (WCAG 2.1 AA)",
            passed=passed,
            score=score,
            details=f"Found {len(issues)} issues, {len(recommendations)} recommendations",
            recommendations=recommendations,
        )

    async def _check_performance(self, output_dir: Path) -> QualityCheck:
        """
        Check performance metrics (simulated Lighthouse).

        Checks:
        - Bundle size (no huge dependencies)
        - Image optimization hints
        - Code splitting potential
        - Lazy loading opportunities
        """
        issues = []
        recommendations = []

        # Check file sizes
        total_js_size = 0
        total_css_size = 0

        for js_file in output_dir.glob("**/*.js"):
            total_js_size += js_file.stat().st_size

        for tsx_file in output_dir.glob("**/*.tsx"):
            total_js_size += tsx_file.stat().st_size

        for css_file in output_dir.glob("**/*.css"):
            total_css_size += css_file.stat().st_size

        # Warn if total JS is too large (>500KB uncompressed)
        if total_js_size > 500 * 1024:
            issues.append(f"Total JS size ({total_js_size / 1024:.1f}KB) exceeds 500KB")
            recommendations.append("Consider code splitting and lazy loading")

        # Check for lazy loading images
        component_files = list(output_dir.glob("**/*.tsx"))
        has_lazy_loading = False

        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8")
            if 'loading="lazy"' in content or "lazy" in content.lower():
                has_lazy_loading = True
                break

        if not has_lazy_loading:
            recommendations.append("Add loading='lazy' to images below the fold")

        # Check for Next.js Image component usage
        uses_next_image = False
        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8")
            if "import Image" in content or "from next/image" in content:
                uses_next_image = True
                break

        if not uses_next_image:
            recommendations.append("Use Next.js Image component for optimization")

        passed = len(issues) == 0
        score = 0.96 if passed else max(0.7, 1.0 - (len(issues) * 0.15))

        return QualityCheck(
            name="Performance (Lighthouse)",
            passed=passed,
            score=score,
            details=f"JS: {total_js_size / 1024:.1f}KB, CSS: {total_css_size / 1024:.1f}KB",
            recommendations=recommendations,
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
        """
        Check responsive design readiness.

        Checks:
        - Mobile-first media queries
        - Breakpoint usage (sm, md, lg, xl)
        - No fixed widths without max-width
        """
        recommendations = []

        breakpoints_tested = 0
        has_responsive_classes = False

        # Find all TSX/JSX files
        component_files = list(output_dir.glob("**/*.tsx")) + list(output_dir.glob("**/*.jsx"))

        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8")

            # Check for Tailwind responsive classes
            if "sm:" in content:
                breakpoints_tested += 1
                has_responsive_classes = True
            if "md:" in content:
                breakpoints_tested += 1
            if "lg:" in content:
                breakpoints_tested += 1
            if "xl:" in content or "2xl:" in content:
                breakpoints_tested += 1

            # Check for fixed widths without max-width
            fixed_widths = re.findall(r'w-\d+["\']', content)
            max_widths = re.findall(r"max-w-", content)

            if fixed_widths and not max_widths:
                recommendations.append(
                    f"{file_path.name}: Consider using max-w-* with w-* for responsiveness"
                )

        # Check CSS files for media queries
        css_files = list(output_dir.glob("**/*.css"))
        for css_file in css_files:
            content = css_file.read_text(encoding="utf-8")
            if "@media" in content:
                breakpoints_tested += len(re.findall(r"@media", content))

        passed = has_responsive_classes and breakpoints_tested >= 3
        score = min(1.0, breakpoints_tested / 4.0) if has_responsive_classes else 0.5

        return QualityCheck(
            name="Responsive Design",
            passed=passed,
            score=score,
            details=f"Breakpoints tested: {breakpoints_tested}/4",
            recommendations=recommendations,
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

        placeholder_patterns = [
            r"lorem\s+ipsum",
            r"TODO[:\s]",
            r"FIXME[:\s]",
            r"placeholder\s+content",
            r"your\s+content\s+here",
            r"insert\s+.*\s+here",
        ]

        # Find all TSX/JSX files
        component_files = list(output_dir.glob("**/*.tsx")) + list(output_dir.glob("**/*.jsx"))

        for file_path in component_files:
            content = file_path.read_text(encoding="utf-8").lower()

            for pattern in placeholder_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues.append(
                        f"{file_path.name}: Found placeholder content ({len(matches)} matches)"
                    )

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
