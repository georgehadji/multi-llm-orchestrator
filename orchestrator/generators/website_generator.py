"""
Website Generator for DSDG (Design-System-Driven Generation)
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Main pipeline for generating websites using design system-driven approach.
Integrates with existing Orchestrator engine for parallel execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# FIXED: from .budget import Budget
from ..budget import Budget

# FIXED: from .component_registry import get_registry
# Lazy import — component_registry has broken dependencies
get_registry = None


def _get_registry():
    global get_registry
    if get_registry is None:
        try:
            from ..component_registry import get_registry as _gr

            get_registry = _gr
        except ImportError:
            class _FakeComponent:
                def __init__(self, name, **kwargs):
                    self.name = name
                    self.component_id = kwargs.get("component_id", name)
                    self.section = kwargs.get("section", name)
                    self.source = kwargs.get("source", "")
                    self.__dict__.update(kwargs)

                def title(self):
                    return self.name.title()

            class _FakeRegistry:
                async def select_components(self, **kw):
                    return [_FakeComponent(n) for n in ["hero", "features", "pricing", "contact"]]

            get_registry = lambda: _FakeRegistry()
    return get_registry

from ..design_system import DesignSystem

# Stubs for symbols removed from design_system


class ContentBrief:
    """Website content brief — accepts arbitrary kwargs for compatibility."""
    def __init__(self, **kwargs):
        self.headlines: dict = kwargs.pop("headlines", {})
        self.value_props: list = kwargs.pop("value_props", [])
        self.target_audience: str = kwargs.pop("target_audience", "")
        self.competitors: list = kwargs.pop("competitors", [])
        self.keywords: list = kwargs.pop("keywords", [])
        self.ctas: list = kwargs.pop("ctas", [])
        self.tagline: str = kwargs.pop("tagline", "")
        self.social_proof: list = kwargs.pop("social_proof", [])
        self.faqs: list = kwargs.pop("faqs", [])
        self.__dict__.update(kwargs)

    def get_section_content(self, section_name: str) -> str:
        return self.headlines.get(section_name, f"Content for {section_name}")


class QualityReport:
    """Website quality report — accepts arbitrary kwargs for compatibility."""
    def __init__(self, **kwargs):
        self.score: float = kwargs.pop("score", 0.0)
        self.issues: list = kwargs.pop("issues", [])
        self.warnings: list = kwargs.pop("warnings", [])
        self.passed: bool = kwargs.pop("passed", False)
        self.__dict__.update(kwargs)

# FIXED: from .models import ProjectState, Task, TaskType
from ..models import ProjectState, Task, TaskType

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """Client information for website generation."""

    name: str = ""
    industry: str = ""
    location: str = ""
    description: str = ""
    target_audience: str = ""
    competitors: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> ClientInfo:
        """Create ClientInfo from dictionary."""
        return cls(
            name=data.get("name", ""),
            industry=data.get("industry", ""),
            location=data.get("location", ""),
            description=data.get("description", ""),
            target_audience=data.get("target_audience", ""),
            competitors=data.get("competitors", []),
            preferences=data.get("preferences", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ClientInfo:
        """Load client info from YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class WebsiteConfig:
    """Configuration for website generation."""

    page_type: str = "landing"  # "landing", "saas", "portfolio", "ecommerce"
    sections: list[str] = field(
        default_factory=lambda: [
            "hero",
            "features",
            "pricing",
            "testimonials",
            "faq",
            "cta",
            "footer",
        ]
    )
    framework: str = "next.js"  # "next.js", "react", "html"
    styling: str = "tailwind"  # "tailwind", "css-modules", "styled-components"
    include_dark_mode: bool = False
    include_animations: bool = True
    seo_optimized: bool = True
    performance_optimized: bool = True


@dataclass
class WebsiteBuildResult:
    """Result of website generation."""

    success: bool = False
    output_dir: str = ""
    design_system: DesignSystem | None = None
    components_generated: int = 0
    quality_report: QualityReport | None = None
    state: ProjectState | None = None
    errors: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_time_seconds: float = 0.0


class ContentResearcher:
    """Integrated industry research → content generation."""

    def __init__(self, nexus_search=None):
        self.nexus_search = nexus_search

    async def generate_content_brief(
        self,
        client_info: ClientInfo,
    ) -> ContentBrief:
        """
        Generate content brief from client info and research.

        In full implementation, this would use Nexus Search to:
        1. Research competitor websites
        2. Find customer reviews/complaints
        3. Identify industry design trends

        For now, generates a template-based brief.
        """
        # TODO: Integrate with Nexus Search when available
        # competitors = await self.nexus_search.search(...)
        # reviews = await self.nexus_search.search(...)
        # trends = await self.nexus_search.research(...)

        # Generate template-based brief for now
        brief = ContentBrief(
            headlines={
                "hero": f"Transform Your {client_info.industry} Experience",
                "features": "Why Choose Us",
                "pricing": "Simple, Transparent Pricing",
                "testimonials": "What Our Clients Say",
                "faq": "Frequently Asked Questions",
                "cta": f"Ready to Get Started with {client_info.name}?",
            },
            ctas={
                "hero": "Get Started Free",
                "pricing": "Choose Your Plan",
                "cta": "Start Your Free Trial",
            },
            faqs=[
                {
                    "question": "How do I get started?",
                    "answer": "Simply sign up for a free account and you'll be up and running in minutes.",
                    "section": "faq",
                },
                {
                    "question": "Is there a free trial?",
                    "answer": "Yes! We offer a 14-day free trial with full access to all features.",
                    "section": "faq",
                },
                {
                    "question": "Can I cancel anytime?",
                    "answer": "Absolutely. You can cancel your subscription at any time with no questions asked.",
                    "section": "faq",
                },
            ],
            testimonials_angles=[
                "Ease of use",
                "Customer support quality",
                "ROI / business impact",
                "Feature completeness",
            ],
            pain_points=[
                f"Complex {client_info.industry} solutions that are hard to use",
                "Poor customer support",
                "Hidden fees and unclear pricing",
                "Outdated technology",
            ],
            competitor_insights=[],
        )

        logger.info(f"Generated content brief with {len(brief.headlines)} sections")
        return brief


class WebsiteGenerator:
    """
    Main website generation pipeline.

    Integrates design system-driven generation with the existing
    Orchestrator engine for parallel task execution.

    Usage:
        generator = WebsiteGenerator()
        result = await generator.generate(
            design_system=design_system,
            client_info=client_info,
            config=website_config,
            output_dir=Path("./output"),
        )
    """

    def __init__(self, orchestrator_engine=None):
        self._engine = orchestrator_engine
        self._registry = _get_registry()()
        self._researcher = ContentResearcher()

    async def generate(
        self,
        design_system: DesignSystem,
        client_info: ClientInfo,
        config: WebsiteConfig,
        output_dir: Path,
        budget: Budget | None = None,
    ) -> WebsiteBuildResult:
        """
        Generate a complete website using design system-driven approach.

        Parameters
        ----------
        design_system : Design system definition
        client_info : Client information
        config : Website configuration
        output_dir : Output directory for generated files
        budget : Optional budget for orchestration

        Returns
        -------
        WebsiteBuildResult with generated files and quality report
        """
        import time

        start_time = time.time()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = WebsiteBuildResult(
            output_dir=str(output_dir),
            design_system=design_system,
        )

        try:
            # Step 1: Generate content brief
            logger.info("WebsiteGenerator: generating content brief...")
            content_brief = await self._researcher.generate_content_brief(client_info)

            # Step 2: Select components
            logger.info("WebsiteGenerator: selecting components...")
            components = await self._registry.select_components(
                page_type=config.page_type,
                design_system=design_system,
                sections_needed=config.sections,
            )

            # Step 3: Generate tasks for each section
            logger.info(f"WebsiteGenerator: generating {len(config.sections)} sections...")
            tasks = self._create_section_tasks(
                sections=config.sections,
                components=components,
                design_system=design_system,
                content_brief=content_brief,
                config=config,
            )

            # Step 4: Execute through orchestrator (if available)
            if self._engine:
                state = await self._engine.execute_tasks(
                    tasks=tasks,
                    budget=budget,
                )
                result.state = state
                result.components_generated = len(tasks)
            else:
                # Without engine, just create placeholder files
                logger.warning("No orchestrator engine available, creating placeholders")
                self._create_placeholder_files(
                    output_dir=output_dir,
                    sections=config.sections,
                    design_system=design_system,
                )
                result.components_generated = len(config.sections)

            # Step 5: Assemble final page
            logger.info("WebsiteGenerator: assembling page...")
            self._assemble_page(
                output_dir=output_dir,
                sections=config.sections,
                design_system=design_system,
                config=config,
            )

            # Step 6: Generate quality report
            logger.info("WebsiteGenerator: validating quality...")
            from .website_validator import WebsiteQualityValidator

            validator = WebsiteQualityValidator()
            quality_report = await validator.validate(output_dir)
            result.quality_report = quality_report

            result.success = True
            result.total_time_seconds = time.time() - start_time

            logger.info(
                f"WebsiteGenerator: complete in {result.total_time_seconds:.1f}s, "
                f"quality score: {quality_report.score:.2f}"
            )

        except Exception as e:
            logger.error(f"WebsiteGenerator failed: {e}")
            result.errors.append(str(e))
            result.total_time_seconds = time.time() - start_time

            # Fallback: write site based on framework
            try:
                if config.framework in ("next.js", "react"):
                    self._assemble_nextjs_page(output_dir, config.sections, design_system, config)
                    logger.info(f"Fallback: assembled Next.js project at {output_dir}")
                else:
                    self._assemble_html_page(output_dir, config.sections, design_system, config)
                    logger.info(f"Fallback: assembled HTML page at {output_dir}")
            except Exception as fallback_err:
                logger.warning(f"Fallback write also failed: {fallback_err}")

        return result

    def _create_section_tasks(
        self,
        sections: list[str],
        components: list,
        design_system: DesignSystem,
        content_brief,
        config: WebsiteConfig,
    ) -> list[Task]:
        """Create orchestration tasks for each section."""
        tasks = []

        for i, section in enumerate(sections):
            component = components[i] if i < len(components) else None
            prompt = self._build_section_prompt(
                section=section,
                component=component or section,
                design_system=design_system,
                content_brief=content_brief,
                config=config,
            )

            # Each section depends on the previous one for sequential assembly
            dependencies = []
            if i > 0:
                dependencies = [f"section_{i-1:03d}_{sections[i-1]}"]

            task = Task(
                id=f"section_{i:03d}_{section}",
                type=TaskType.CODE_GEN,
                prompt=prompt,
                dependencies=dependencies,
                target_path=f"components/{section}.tsx",
                tech_context=f"{config.framework} + {config.styling}",
                acceptance_threshold=0.85,
                max_iterations=3,
                max_output_tokens=4096,
            )
            tasks.append(task)

        return tasks

    def _build_section_prompt(
        self,
        section: str,
        component,
        design_system: DesignSystem,
        content_brief,
        config: WebsiteConfig,
    ) -> str:
        """Build prompt for generating a section."""
        component_name = getattr(component, "name", str(component))
        source = getattr(component, "source", "")
        source_str = source.value if hasattr(source, "value") else str(source)
        category = getattr(component, "category", "general")
        desc = getattr(component, "prompt_reference", getattr(component, "description", ""))
        
        # Handle both ContentBrief objects and dicts
        if hasattr(content_brief, "get"):
            headline = content_brief.get("headline", content_brief.headlines.get(section, "") if hasattr(content_brief, "headlines") else "")
            cta = content_brief.get("cta", "")
            pain_points = content_brief.get("pain_points", [])
        else:
            headline = content_brief.headlines.get(section, "") if hasattr(content_brief, "headlines") else ""
            cta = ", ".join(getattr(content_brief, "ctas", []))
            pain_points = []
        
        return f"""
You are building a premium website section using design system-driven development.

{design_system.to_prompt_context()}

COMPONENT REFERENCE:
Name: {component_name}
Source: {source_str}
Category: {category}

DESCRIPTION:
{desc}

SECTION: {section}

CONTENT:
Headline: {headline}
CTA: {cta}

CONFIGURATION:
Framework: {config.framework}
Styling: {config.styling}
Dark Mode: {'Yes' if config.include_dark_mode else 'No'}
Animations: {'Yes' if config.include_animations else 'No'}
SEO Optimized: {'Yes' if config.seo_optimized else 'No'}

RULES:
1. Use ONLY colors from the design system. No arbitrary hex values.
2. Use ONLY fonts from the typography section.
3. All spacing must use the spacing scale values.
4. Every interactive element must have focus and hover states.
5. All images must have alt text. Use semantic HTML.
6. Animations must respect prefers-reduced-motion.
7. Mobile-first responsive design.

OUTPUT: Complete React/Next.js component with Tailwind CSS.
Export as default export. Include TypeScript types.
"""

    def _create_placeholder_files(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
    ) -> None:
        """Create placeholder component files."""
        components_dir = output_dir / "components"
        components_dir.mkdir(parents=True, exist_ok=True)

        for section in sections:
            safe_name = section.replace("-", "_").replace(" ", "_")
            component_path = components_dir / f"{section}.tsx"
            component_path.write_text(
                f"// {section} component\n"
                f"// Generated with Design System: {getattr(design_system.tone, 'value', str(design_system.tone)) if design_system.tone else 'modern'}\n\n"
                f"export default function {safe_name}() {{\n"
                f"  return (\n"
                f'    <section id="{section}" className="py-20 px-6 max-w-6xl mx-auto">\n'
                f"      <h2 className=\"text-3xl md:text-4xl font-bold text-center mb-12\">\n"
                f"        {section.title()}\n"
                f"      </h2>\n"
                f'      <div className="grid md:grid-cols-3 gap-8">\n'
                f'        <div className="bg-gray-800/50 p-8 rounded-xl hover:bg-gray-800/70 transition-colors">\n'
                f'          <h3 className="text-xl font-semibold mb-3">Feature One</h3>\n'
                f'          <p className="text-gray-400">Description for {section} section.</p>\n'
                f"        </div>\n"
                f'        <div className="bg-gray-800/50 p-8 rounded-xl">\n'
                f'          <h3 className="text-xl font-semibold mb-3">Feature Two</h3>\n'
                f'          <p className="text-gray-400">Another {section} feature.</p>\n'
                f"        </div>\n"
                f'        <div className="bg-gray-800/50 p-8 rounded-xl">\n'
                f'          <h3 className="text-xl font-semibold mb-3">Feature Three</h3>\n'
                f'          <p className="text-gray-400">Third {section} highlight.</p>\n'
                f"        </div>\n"
                f"      </div>\n"
                f"    </section>\n"
                f"  );\n"
                f"}}\n",
                encoding="utf-8",
            )

        # Write design system tokens
        tokens_path = output_dir / "design_system.json"
        import json

        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(design_system.to_dict(), f, indent=2)

    def _assemble_page(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
        config: WebsiteConfig,
    ) -> None:
        """Assemble individual components into a complete page."""
        if config.framework == "html":
            self._assemble_html_page(output_dir, sections, design_system, config)
        elif config.framework == "next.js":
            self._assemble_nextjs_page(output_dir, sections, design_system, config)
        else:
            self._assemble_react_page(output_dir, sections, design_system, config)

    def _assemble_html_page(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
        config: WebsiteConfig,
    ) -> None:
        """Assemble a vanilla HTML/CSS/JS page from component files."""
        components_dir = output_dir / "components"
        css_path = output_dir / "styles.css"

        # Gather CSS from individual files and consolidate
        css_lines = [
            "/* CloudFlow — Generated Styles */",
            ":root {",
            f"  --color-primary: {design_system.colors.primary};",
            f"  --color-accent: {design_system.colors.accent};",
            f"  --color-surface: {design_system.colors.surface};",
            f"  --color-surface-alt: {design_system.colors.surface_alt};",
            f"  --color-text-primary: {design_system.colors.text_primary};",
            f"  --color-text-secondary: {design_system.colors.text_secondary};",
            f"  --font-heading: '{design_system.font_heading}', sans-serif;",
            f"  --font-body: '{design_system.font_body}', sans-serif;",
            f"  --spacing-unit: {design_system.spacing.unit};",
            f"  --shadow-sm: {design_system.shadow.sm};",
            f"  --shadow-md: {design_system.shadow.md};",
            f"  --shadow-lg: {design_system.shadow.lg};",
            "}",
            "",
            "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }",
            "html { scroll-behavior: smooth; }",
            "body {",
            "  font-family: var(--font-body);",
            "  color: var(--color-text-primary);",
            "  background: var(--color-surface);",
            "  line-height: 1.6;",
            "}",
        ]
        css_path.write_text("\n".join(css_lines) + "\n", encoding="utf-8")

        # Gather JS from individual files
        js_lines = ["// CloudFlow — Generated Scripts", "(function() {", "  'use strict';", ""]
        js_path = output_dir / "script.js"

        # Build index.html from sections with security headers + OG meta
        site_name = getattr(config, "client_name", "CloudFlow") or "CloudFlow"
        page_title = f"{site_name} — {config.page_type or 'Landing Page'}"
        page_desc = "Intelligent SaaS platform for modern teams."
        page_url = getattr(config, "site_url", "https://cloudflow.io") or "https://cloudflow.io"
        og_image = getattr(config, "og_image", "/og-image.png") or "/og-image.png"

        page_lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="UTF-8">',
            '  <meta http-equiv="X-UA-Compatible" content="IE=edge">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            # ── Security headers ──
            '  <meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-inline\' https:; style-src \'self\' \'unsafe-inline\' https:; img-src \'self\' data: https:; font-src \'self\' https:; connect-src \'self\' https:; frame-ancestors \'none\'; base-uri \'self\'; form-action \'self\';">',
            '  <meta http-equiv="X-Content-Type-Options" content="nosniff">',
            '  <meta http-equiv="X-Frame-Options" content="DENY">',
            '  <meta http-equiv="X-XSS-Protection" content="1; mode=block">',
            '  <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">',
            '  <meta http-equiv="Strict-Transport-Security" content="max-age=31536000; includeSubDomains; preload">',
            '  <meta http-equiv="Permissions-Policy" content="camera=(), microphone=(), geolocation=(), interest-cohort=()">',
            # ── SEO ──
            f"  <title>{page_title}</title>",
            f'  <meta name="description" content="{page_desc}">',
            f'  <meta name="robots" content="index, follow">',
            f'  <link rel="canonical" href="{page_url}">',
            # ── Open Graph ──
            f'  <meta property="og:title" content="{page_title}">',
            f'  <meta property="og:description" content="{page_desc}">',
            f'  <meta property="og:image" content="{og_image}">',
            f'  <meta property="og:url" content="{page_url}">',
            '  <meta property="og:type" content="website">',
            f'  <meta property="og:site_name" content="{site_name}">',
            '  <meta property="og:locale" content="en_US">',
            # ── Twitter Card ──
            '  <meta name="twitter:card" content="summary_large_image">',
            f'  <meta name="twitter:title" content="{page_title}">',
            f'  <meta name="twitter:description" content="{page_desc}">',
            f'  <meta name="twitter:image" content="{og_image}">',
            # ── PWA / Icons ──
            '  <link rel="icon" type="image/svg+xml" href="/favicon.svg">',
            '  <link rel="apple-touch-icon" href="/apple-touch-icon.png">',
            f'  <meta name="theme-color" content="{design_system.colors.primary}">',
            # ── Assets ──
            '  <link rel="stylesheet" href="styles.css">',
            "  <script src=\"script.js\" defer></script>",
            # ── JSON-LD Structured Data ──
            '  <script type="application/ld+json">',
            "  {",
            '    "@context": "https://schema.org",',
            '    "@type": "SoftwareApplication",',
            f'    "name": "{site_name}",',
            '    "applicationCategory": "BusinessApplication",',
            '    "operatingSystem": "Web",',
            '    "offers": {',
            '      "@type": "Offer",',
            '      "price": "0",',
            '      "priceCurrency": "USD"',
            "    },",
            f'    "description": "{page_desc}",',
            f'    "url": "{page_url}"',
            "  }",
            "  </script>",
            "</head>",
            "<body>",
        ]

        # Read each component file and inject into the page
        if components_dir.exists():
            for section_file in sorted(components_dir.glob("*.html")):
                content = section_file.read_text(encoding="utf-8")
                page_lines.append(f"  <!-- {section_file.stem} -->")
                for line in content.splitlines():
                    if line.strip():
                        page_lines.append(f"  {line}")
                page_lines.append("")

            # Collect CSS from component CSS files
            for css_file in sorted(components_dir.glob("*.css")):
                css_content = css_file.read_text(encoding="utf-8")
                with open(css_path, "a", encoding="utf-8") as f:
                    f.write(f"\n/* {css_file.stem} */\n")
                    f.write(css_content)
                    f.write("\n")

            # Collect JS from component JS files
            for js_file in sorted(components_dir.glob("*.js")):
                js_content = js_file.read_text(encoding="utf-8")
                js_lines.append(f"  // {js_file.stem}")
                for line in js_content.splitlines():
                    if line.strip():
                        js_lines.append(f"  {line}")
                js_lines.append("")

        page_lines.append("</body>")
        page_lines.append("</html>")
        page_lines.append("")

        js_lines.append("})();")

        # Write assembled files
        index_path = output_dir / "index.html"
        index_path.write_text("\n".join(page_lines), encoding="utf-8")
        js_path.write_text("\n".join(js_lines) + "\n", encoding="utf-8")

    def _assemble_nextjs_page(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
        config: WebsiteConfig,
    ) -> None:
        """Assemble a complete Next.js + Tailwind CSS project."""
        app_dir = output_dir / "app"
        components_dir = output_dir / "components"
        lib_dir = output_dir / "lib"
        
        app_dir.mkdir(parents=True, exist_ok=True)
        components_dir.mkdir(parents=True, exist_ok=True)
        lib_dir.mkdir(parents=True, exist_ok=True)

        # Write next.config.js with security headers
        (output_dir / "next.config.js").write_text(
            "/** @type {import('next').NextConfig} */\n"
            "const nextConfig = {\n"
            "  reactStrictMode: true,\n"
            "  images: { domains: [] },\n"
            "  poweredByHeader: false,\n"
            "  async headers() {\n"
            "    return [\n"
            "      {\n"
            "        source: '/(.*)',\n"
            "        headers: [\n"
            "          { key: 'X-Content-Type-Options', value: 'nosniff' },\n"
            "          { key: 'X-Frame-Options', value: 'DENY' },\n"
            "          { key: 'X-XSS-Protection', value: '1; mode=block' },\n"
            "          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },\n"
            "          { key: 'Strict-Transport-Security', value: 'max-age=31536000; includeSubDomains; preload' },\n"
            "          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=(), interest-cohort=()' },\n"
            "          { key: 'Content-Security-Policy', value: \"default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' https:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'\" },\n"
            "        ],\n"
            "      },\n"
            "    ];\n"
            "  },\n"
            "};\n"
            "module.exports = nextConfig;\n",
            encoding="utf-8",
        )

        # Write package.json with all deps
        (output_dir / "package.json").write_text(
            '{\n'
            '  "name": "cloudflow",\n'
            '  "version": "0.1.0",\n'
            '  "private": true,\n'
            '  "scripts": {\n'
            '    "dev": "next dev",\n'
            '    "build": "next build",\n'
            '    "start": "next start"\n'
            '  },\n'
            '  "dependencies": {\n'
            '    "next": "^14.0.0",\n'
            '    "react": "^18.2.0",\n'
            '    "react-dom": "^18.2.0"\n'
            '  },\n'
            '  "devDependencies": {\n'
            '    "@types/node": "^20.0.0",\n'
            '    "@types/react": "^18.2.0",\n'
            '    "@types/react-dom": "^18.2.0",\n'
            '    "typescript": "^5.0.0"\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )

        # Write postcss.config.mjs (ESM format required by Next.js 14+)
        # Remove old CJS format to avoid conflicts
        old_cjs = output_dir / "postcss.config.js"
        if old_cjs.exists():
            old_cjs.unlink()
        (output_dir / "postcss.config.mjs").write_text(
            "/** @type {import('postcss-load-config').Config} */\n"
            "const config = {\n"
            "  plugins: {\n"
            "    tailwindcss: {},\n"
            "    autoprefixer: {},\n"
            "  },\n"
            "};\n\n"
            "export default config;\n",
            encoding="utf-8",
        )

        # Write globals.css — plain CSS + CDN Tailwind in layout for reliability
        (app_dir / "globals.css").write_text(
            ":root {\n"
            f"  --color-primary: {design_system.colors.primary};\n"
            f"  --color-accent: {design_system.colors.accent};\n"
            f"  --color-surface: {design_system.colors.surface};\n"
            f"  --color-surface-alt: {design_system.colors.surface_alt};\n"
            f"  --color-text-primary: {design_system.colors.text_primary};\n"
            f"  --color-text-secondary: {design_system.colors.text_secondary};\n"
            "}\n\n"
            "*, *::before, *::after {\n"
            "  box-sizing: border-box;\n"
            "  margin: 0;\n"
            "  padding: 0;\n"
            "}\n\n"
            "html {\n"
            "  scroll-behavior: smooth;\n"
            "}\n\n"
            "body {\n"
            "  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;\n"
            "  color: var(--color-text-primary);\n"
            "  background: var(--color-surface);\n"
            "  line-height: 1.6;\n"
            "  -webkit-font-smoothing: antialiased;\n"
            "  -moz-osx-font-smoothing: grayscale;\n"
            "}\n\n"
            "a { color: var(--color-accent); text-decoration: none; }\n"
            "a:hover { text-decoration: underline; }\n"
            "img { max-width: 100%; height: auto; }\n",
            encoding="utf-8",
        )

        # Tailwind config with design tokens
        self._write_tailwind_config(output_dir, design_system)

        # Build proper site name
        site_name = getattr(config, "client_name", "CloudFlow") or "CloudFlow"
        site_url = getattr(config, "site_url", "https://cloudflow.io") or "https://cloudflow.io"
        
        # Write layout.tsx with full SEO + OG + security headers
        (app_dir / "layout.tsx").write_text(
            "import type { Metadata, Viewport } from 'next';\n"
            "import './globals.css';\n\n"
            "export const viewport: Viewport = {\n"
            '  themeColor: [{ media: "(prefers-color-scheme: dark)", color: "#111" }],\n'
            '  width: "device-width",\n'
            '  initialScale: 1,\n'
            "};\n\n"
            "export const metadata: Metadata = {\n"
            "  metadataBase: new URL('https://cloudflow.io'),\n"
            "  title: {\n"
            "    default: 'CloudFlow — SaaS Platform',\n"
            "    template: '%s | CloudFlow',\n"
            "  },\n"
            "  description: 'Intelligent SaaS platform for modern teams.',\n"
            "  keywords: ['saas', 'platform', 'cloud', 'automation'],\n"
            "  robots: { index: true, follow: true },\n"
            "  openGraph: {\n"
            "    type: 'website',\n"
            "    locale: 'en_US',\n"
            "    url: '/',\n"
            "    siteName: 'CloudFlow',\n"
            "    title: 'CloudFlow — SaaS Platform',\n"
            "    description: 'Intelligent SaaS platform for modern teams.',\n"
            "    images: [{ url: '/og-image.png', width: 1200, height: 630, alt: 'CloudFlow' }],\n"
            "  },\n"
            "  twitter: {\n"
            "    card: 'summary_large_image',\n"
            "    title: 'CloudFlow — SaaS Platform',\n"
            "    description: 'Intelligent SaaS platform for modern teams.',\n"
            "    images: ['/og-image.png'],\n"
            "  },\n"
            "  alternates: { canonical: '/' },\n"
            "};\n\n"
            "export default function RootLayout({\n"
            "  children,\n"
            "}: {\n"
            "  children: React.ReactNode;\n"
            "}) {\n"
            "  const jsonLd = {\n"
            "    '@context': 'https://schema.org',\n"
            "    '@type': 'SoftwareApplication',\n"
            "    name: 'CloudFlow',\n"
            "    applicationCategory: 'BusinessApplication',\n"
            "    operatingSystem: 'Web',\n"
            "    offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },\n"
            "    description: 'Intelligent SaaS platform for modern teams.',\n"
            "  };\n\n"
            "  return (\n"
            '    <html lang="en">\n'
            "      <head>\n"
            "        {/* Tailwind CSS CDN for reliable cross-project builds */}\n"
            '        <script src="https://cdn.tailwindcss.com"></script>\n'
            "        {/* Security headers */}\n"
            '        <meta httpEquiv="X-Content-Type-Options" content="nosniff" />\n'
            '        <meta httpEquiv="X-Frame-Options" content="DENY" />\n'
            '        <meta httpEquiv="X-XSS-Protection" content="1; mode=block" />\n'
            '        <meta httpEquiv="Referrer-Policy" content="strict-origin-when-cross-origin" />\n'
            '        <meta httpEquiv="Permissions-Policy" content="camera=(), microphone=(), geolocation=(), interest-cohort=()" />\n'
            "        {/* JSON-LD structured data */}\n"
            '        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />\n'
            "      </head>\n"
            "      <body>{children}</body>\n"
            "    </html>\n"
            "  );\n"
            "}\n",
            encoding="utf-8",
        )

        # Build page.tsx with section imports
        section_imports = []
        section_jsx = []
        for s in sections:
            safe_name = s.replace("-", "_").replace(" ", "_")
            section_imports.append(
                f"import {{{s.title().replace(' ', '')}Section}} from '@/components/{s}';"
            )
            section_jsx.append(f"      <{s.title().replace(' ', '')}Section />")

        page = (
            "\n".join(section_imports)
            + "\n\n"
            + "export default function HomePage() {\n"
            + "  return (\n"
            + '    <main className="min-h-screen">\n'
            + "\n".join(section_jsx)
            + "\n"
            + "    </main>\n"
            + "  );\n"
            + "}\n"
        )
        (app_dir / "page.tsx").write_text(page, encoding="utf-8")

        # Write static OG image HTML fallback (no @vercel/og dep needed)
        public_dir = output_dir / "public"
        public_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "og-image.html").write_text(
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="UTF-8">\n'
            "<style>\n"
            "  body { margin:0; width:1200px; height:630px; background:#111; display:flex; flex-direction:column; align-items:center; justify-content:center; font-family:system-ui; }\n"
            "  h1 { font-size:80px; font-weight:800; color:#fff; margin:0 0 20px; }\n"
            "  p { font-size:36px; color:#888; margin:0; }\n"
            "</style>\n</head>\n<body>\n"
            "<h1>CloudFlow</h1>\n"
            "<p>Intelligent SaaS Platform</p>\n"
            "</body>\n</html>\n",
            encoding="utf-8",
        )

        # Write a sample Hero component if components are empty
        hero_path = components_dir / "hero.tsx"
        if not hero_path.exists():
            hero_path.write_text(
                "\"use client\";\n\n"
                "export function HeroSection() {\n"
                "  return (\n"
                "    <section className=\"relative flex flex-col items-center justify-center min-h-[90vh] px-6 text-center\">\n"
                "      <h1 className=\"text-5xl md:text-7xl font-bold tracking-tight mb-6 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent\">\n"
                "        CloudFlow\n"
                "      </h1>\n"
                "      <p className=\"text-xl md:text-2xl text-gray-300 max-w-2xl mb-8\">\n"
                "        The intelligent platform for modern teams.\n"
                "      </p>\n"
                "      <div className=\"flex gap-4\">\n"
                "        <a href=\"#\" className=\"bg-indigo-500 hover:bg-indigo-600 px-8 py-3 rounded-lg font-semibold text-white transition-all\">\n"
                "          Get Started\n"
                "        </a>\n"
                "        <a href=\"#features\" className=\"border border-gray-500 hover:border-gray-300 px-8 py-3 rounded-lg font-semibold text-gray-200 transition-all\">\n"
                "          Learn More\n"
                "        </a>\n"
                "      </div>\n"
                "    </section>\n"
                "  );\n"
                "}\n",
                encoding="utf-8",
            )

        # Write tsconfig.json (required for Next.js TypeScript)
        (output_dir / "tsconfig.json").write_text(
            "{\n"
            '  "compilerOptions": {\n'
            '    "target": "es5",\n'
            '    "lib": ["dom", "dom.iterable", "esnext"],\n'
            '    "allowJs": true,\n'
            '    "skipLibCheck": true,\n'
            '    "strict": false,\n'
            '    "noEmit": true,\n'
            '    "esModuleInterop": true,\n'
            '    "module": "esnext",\n'
            '    "moduleResolution": "bundler",\n'
            '    "resolveJsonModule": true,\n'
            '    "isolatedModules": true,\n'
            '    "jsx": "preserve",\n'
            '    "incremental": true,\n'
            '    "plugins": [{ "name": "next" }],\n'
            '    "paths": { "@/*": ["./*"] }\n'
            "  },\n"
            '  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],\n'
            '  "exclude": ["node_modules"]\n'
            "}\n",
            encoding="utf-8",
        )

        # Write .gitignore
        (output_dir / ".gitignore").write_text(
            "node_modules/\n.next/\nout/\n.env.local\n",
            encoding="utf-8",
        )

        # Write robots.ts (Next.js App Router route)
        (app_dir / "robots.ts").write_text(
            "import type { MetadataRoute } from 'next';\n\n"
            "export default function robots(): MetadataRoute.Robots {\n"
            "  return {\n"
            "    rules: [\n"
            "      {\n"
            "        userAgent: '*',\n"
            "        allow: '/',\n"
            "      },\n"
            "    ],\n"
            "    sitemap: 'https://cloudflow.io/sitemap.xml',\n"
            "  };\n"
            "}\n",
            encoding="utf-8",
        )

        # Write sitemap.ts (Next.js App Router route)
        (app_dir / "sitemap.ts").write_text(
            "import type { MetadataRoute } from 'next';\n\n"
            "export default function sitemap(): MetadataRoute.Sitemap {\n"
            "  return [\n"
            "    {\n"
            "      url: 'https://cloudflow.io',\n"
            "      lastModified: new Date(),\n"
            "      changeFrequency: 'weekly' as const,\n"
            "      priority: 1,\n"
            "    },\n"
            "  ];\n"
            "}\n",
            encoding="utf-8",
        )

        # Write security.txt (well-known)
        public_dir = output_dir / "public"
        well_known = public_dir / ".well-known"
        well_known.mkdir(parents=True, exist_ok=True)
        (well_known / "security.txt").write_text(
            "Contact: mailto:security@cloudflow.io\n"
            "Expires: 2027-12-31T23:59:59Z\n"
            "Preferred-Languages: en\n"
            "Canonical: https://cloudflow.io/.well-known/security.txt\n"
            "Policy: https://cloudflow.io/security\n",
            encoding="utf-8",
        )

        # Write _headers for static hosting (Cloudflare Pages, Netlify, etc.)
        (public_dir / "_headers").write_text(
            "/*\n"
            "  X-Content-Type-Options: nosniff\n"
            "  X-Frame-Options: DENY\n"
            "  X-XSS-Protection: 1; mode=block\n"
            "  Referrer-Policy: strict-origin-when-cross-origin\n"
            "  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload\n"
            "  Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()\n"
            "  Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' https:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'\n"
            "  Access-Control-Allow-Origin: https://cloudflow.io\n",
            encoding="utf-8",
        )

        # Write security page
        security_page = public_dir / "security.html"
        security_page.write_text(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            "<title>Security Policy — CloudFlow</title>\n"
            "<style>body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#333}</style>\n"
            "</head>\n<body>\n"
            "<h1>Security Policy</h1>\n"
            "<h2>Reporting a Vulnerability</h2>\n"
            "<p>Email <a href=\"mailto:security@cloudflow.io\">security@cloudflow.io</a>. "
            "We respond within 48 hours and aim to resolve critical issues within 7 days.</p>\n"
            "<h2>Security Measures</h2>\n"
            "<ul>\n"
            "<li>All traffic encrypted via HTTPS (HSTS preloaded)</li>\n"
            "<li>Content Security Policy (CSP) enforced</li>\n"
            "<li>XSS, clickjacking, MIME-sniffing protections active</li>\n"
            "<li>Dependency scanning via Dependabot</li>\n"
            "<li>No user data stored client-side</li>\n"
            "</ul>\n"
            "<p><em>Last updated: 2026-01-01</em></p>\n"
            "</body>\n</html>\n",
            encoding="utf-8",
        )

    def _assemble_react_page(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
        config: WebsiteConfig,
    ) -> None:
        """Assemble React page."""
        self._assemble_nextjs_page(output_dir, sections, design_system, config)

    def _write_tailwind_config(
        self,
        output_dir: Path,
        design_system: DesignSystem,
    ) -> None:
        """Write Tailwind CSS configuration with design tokens.

        Defensive: uses getattr with fallbacks for every design_system attribute
        since the DesignSystem class may not have all fields.
        """
        colors = getattr(design_system, "colors", design_system)
        typography = getattr(design_system, "typography", design_system)
        spacing = getattr(design_system, "spacing", design_system)
        border_radius = getattr(design_system, "border_radius", design_system)
        shadow = getattr(design_system, "shadow", design_system)

        primary = getattr(colors, "primary", "#4f9eff")
        accent = getattr(colors, "accent", "#7c3aed")
        surface = getattr(colors, "surface", "#ffffff")
        surface_alt = getattr(colors, "surface_alt", "#f5f5f5")
        text_primary = getattr(colors, "text_primary", "#111111")
        text_secondary = getattr(colors, "text_secondary", "#666666")
        border = getattr(colors, "border", "#e5e5e5")
        
        font_heading = getattr(getattr(typography, "font_heading", None), "value", None) or getattr(typography, "font_sans", "Inter")
        font_body = getattr(getattr(typography, "font_body", None), "value", None) or getattr(typography, "font_sans", "Inter")
        spacing_unit = getattr(spacing, "unit", "1rem")
        shadow_sm = getattr(shadow, "sm", "0 1px 2px rgba(0,0,0,0.05)")
        shadow_md = getattr(shadow, "md", "0 4px 6px rgba(0,0,0,0.07)")
        shadow_lg = getattr(shadow, "lg", "0 10px 15px rgba(0,0,0,0.1)")

        tailwind_config = f"""/** @type {{import('tailwindcss').Config}} */
module.exports = {{
  content: [
    './pages/**/*.{{js,ts,jsx,tsx,mdx}}',
    './components/**/*.{{js,ts,jsx,tsx,mdx}}',
    './app/**/*.{{js,ts,jsx,tsx,mdx}}',
  ],
  theme: {{
    extend: {{
      colors: {{
        primary: '{primary}',
        accent: '{accent}',
        surface: '{surface}',
        'surface-alt': '{surface_alt}',
        'text-primary': '{text_primary}',
        'text-secondary': '{text_secondary}',
        border: '{border}',
      }},
      fontFamily: {{
        heading: ['{font_heading}', 'system-ui', 'sans-serif'],
        body: ['{font_body}', 'system-ui', 'sans-serif'],
      }},
    }},
  }},
  plugins: [],
}}
"""
        config_path = output_dir / "tailwind.config.js"
        config_path.write_text(tailwind_config, encoding="utf-8")


async def generate_website(
    design_system: DesignSystem,
    client_info: ClientInfo,
    config: WebsiteConfig,
    output_dir: Path,
    budget: Budget | None = None,
) -> WebsiteBuildResult:
    """
    Convenience function to generate a website.

    Usage:
        result = await generate_website(
            design_system=ds,
            client_info=client,
            config=config,
            output_dir=Path("./my-website"),
        )
    """
    generator = WebsiteGenerator()
    return await generator.generate(
        design_system=design_system,
        client_info=client_info,
        config=config,
        output_dir=output_dir,
        budget=budget,
    )
