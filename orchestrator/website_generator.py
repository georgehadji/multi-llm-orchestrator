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

from .component_registry import get_registry
from .design_system import (
    ContentBrief,
    DesignSystem,
    QualityReport,
)
from .models import Budget, ProjectState, Task, TaskType

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
        self._registry = get_registry()
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

        return result

    def _create_section_tasks(
        self,
        sections: list[str],
        components: list[ComponentSpec],
        design_system: DesignSystem,
        content_brief: ContentBrief,
        config: WebsiteConfig,
    ) -> list[Task]:
        """Create orchestration tasks for each section."""
        tasks = []

        for i, (section, component) in enumerate(zip(sections, components, strict=False)):
            prompt = self._build_section_prompt(
                section=section,
                component=component,
                design_system=design_system,
                content_brief=content_brief.get_section_content(section),
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
            )
            tasks.append(task)

        return tasks

    def _build_section_prompt(
        self,
        section: str,
        component: ComponentSpec,
        design_system: DesignSystem,
        content_brief: dict,
        config: WebsiteConfig,
    ) -> str:
        """Build prompt for generating a section."""
        return f"""
You are building a premium website section using design system-driven development.

{design_system.to_prompt_context()}

COMPONENT REFERENCE:
Name: {component.name}
Source: {component.source.value}
Category: {component.category}

DESCRIPTION:
{component.prompt_reference}

SECTION: {section}

CONTENT FOR THIS SECTION:
Headline: {content_brief.get('headline', '')}
CTA: {content_brief.get('cta', '')}
Pain Points: {', '.join(content_brief.get('pain_points', []))}

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
7. Maintain contrast ratio minimum {design_system.accessibility.min_contrast_ratio}:1.
8. Mobile-first responsive design.

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
            component_path = components_dir / f"{section}.tsx"
            component_path.write_text(
                f"""
// {section} component
// Generated with Design System: {design_system.tone.value}

export default function {section.title()}() {{
  return (
    <section className="{section}">
      <h2>{section.title()}</h2>
      <p>Content placeholder - implement with design system tokens</p>
    </section>
  )
}}
""",
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
        if config.framework == "next.js":
            self._assemble_nextjs_page(output_dir, sections, design_system, config)
        else:
            self._assemble_react_page(output_dir, sections, design_system, config)

    def _assemble_nextjs_page(
        self,
        output_dir: Path,
        sections: list[str],
        design_system: DesignSystem,
        config: WebsiteConfig,
    ) -> None:
        """Assemble Next.js page."""
        page_content = """
"use client"

import { useState, useEffect } from 'react'
"""
        # Import all components
        for section in sections:
            page_content += f"import {section.title()} from '@/components/{section}'\n"

        page_content += f"""

export default function HomePage() {{
  return (
    <main className="min-h-screen bg-[{design_system.colors.surface}]">
"""
        # Add all sections
        for section in sections:
            page_content += f"      <{section.title()} />\n"

        page_content += """
    </main>
  )
}
"""

        # Write page file
        if config.framework == "next.js":
            page_path = output_dir / "page.tsx"
        else:
            page_path = output_dir / "App.tsx"

        page_path.write_text(page_content, encoding="utf-8")

        # Write Tailwind config with design tokens
        self._write_tailwind_config(output_dir, design_system)

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
        """Write Tailwind CSS configuration with design tokens."""
        tailwind_config = f"""
/** @type {{import('tailwindcss').Config}} */
module.exports = {{
  content: [
    './pages/**/*{{js,ts,jsx,tsx,mdx}}',
    './components/**/*{{js,ts,jsx,tsx,mdx}}',
    './app/**/*{{js,ts,jsx,tsx,mdx}}',
  ],
  theme: {{
    extend: {{
      colors: {{
        primary: '{design_system.colors.primary}',
        accent: '{design_system.colors.accent}',
        surface: '{design_system.colors.surface}',
        'surface-alt': '{design_system.colors.surface_alt}',
        'text-primary': '{design_system.colors.text_primary}',
        'text-secondary': '{design_system.colors.text_secondary}',
        border: '{design_system.colors.border}',
        success: '{design_system.colors.success}',
        error: '{design_system.colors.error}',
      }},
      fontFamily: {{
        heading: ['{design_system.font_heading}', 'sans-serif'],
        body: ['{design_system.font_body}', 'sans-serif'],
      }},
      spacing: {{
        'unit': '{design_system.spacing.unit}',
      }},
      borderRadius: {{
        'sm': '{design_system.border_radius.sm}',
        'md': '{design_system.border_radius.md}',
        'lg': '{design_system.border_radius.lg}',
        'full': '{design_system.border_radius.full}',
      }},
      boxShadow: {{
        'sm': '{design_system.shadow.sm}',
        'md': '{design_system.shadow.md}',
        'lg': '{design_system.shadow.lg}',
      }},
      animation: {{
        'duration': '{design_system.animation.duration}',
        'easing': '{design_system.animation.easing}',
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
