"""
CLI Command for DSDG Website Generation
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Usage:
    python -m orchestrator --type website \
        --preset saas_modern \
        --industry "project_management" \
        --sections "hero,features,pricing,testimonials,faq,cta,footer" \
        --budget 3.0 \
        --output ./results/my-website
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .design_system import (
    BrandTone,
    DesignSystem,
)
from .website_generator import (
    ClientInfo,
    WebsiteConfig,
    WebsiteGenerator,
)

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger("orchestrator.cli_website")


def setup_website_parser(subparsers):
    """Setup website generation subcommand."""
    website_parser = subparsers.add_parser(
        "website",
        help="Generate a website using Design-System-Driven Generation (DSDG)",
        description="Generate premium websites with design system-driven approach",
    )

    # Design system options
    ds_group = website_parser.add_argument_group("Design System")
    ds_group.add_argument(
        "--preset",
        type=str,
        choices=[t.value for t in BrandTone],
        default="premium_minimal",
        help="Design system preset to use",
    )
    ds_group.add_argument(
        "--design-system",
        type=Path,
        help="Path to custom design_system.yaml file (overrides --preset)",
    )

    # Client options
    client_group = website_parser.add_argument_group("Client Information")
    client_group.add_argument(
        "--client-info",
        type=Path,
        help="Path to client_info.yaml file",
    )
    client_group.add_argument(
        "--industry",
        type=str,
        default="technology",
        help="Client industry (used for content research)",
    )
    client_group.add_argument(
        "--company-name",
        type=str,
        default="Client Company",
        help="Company/brand name",
    )

    # Website configuration
    config_group = website_parser.add_argument_group("Website Configuration")
    config_group.add_argument(
        "--page-type",
        type=str,
        choices=["landing", "saas", "portfolio", "ecommerce"],
        default="landing",
        help="Type of page to generate",
    )
    config_group.add_argument(
        "--sections",
        type=str,
        default="hero,features,pricing,testimonials,faq,cta,footer",
        help="Comma-separated list of sections to generate",
    )
    config_group.add_argument(
        "--framework",
        type=str,
        choices=["next.js", "react", "html"],
        default="next.js",
        help="Frontend framework to use",
    )
    config_group.add_argument(
        "--no-animations",
        action="store_true",
        help="Disable animations",
    )
    config_group.add_argument(
        "--no-dark-mode",
        action="store_true",
        help="Disable dark mode support",
    )
    config_group.add_argument(
        "--no-seo",
        action="store_true",
        help="Disable SEO optimization",
    )

    # Output options
    output_group = website_parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./results/website"),
        help="Output directory for generated website",
    )

    # Budget options
    budget_group = website_parser.add_argument_group("Budget")
    budget_group.add_argument(
        "--budget",
        type=float,
        default=3.0,
        help="Budget in USD for LLM API calls",
    )

    website_parser.set_defaults(func=run_website_generation)
    return website_parser


async def run_website_generation(args: argparse.Namespace) -> int:
    """Execute website generation."""
    from .budget import Budget

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    try:
        # 1. Load or create design system
        if args.design_system:
            logger.info(f"Loading design system from {args.design_system}")
            design_system = DesignSystem.from_yaml(args.design_system)
        else:
            logger.info(f"Using design system preset: {args.preset}")
            design_system = DesignSystem.from_preset(BrandTone(args.preset))
            design_system.brand_name = args.company_name
            design_system.industry = args.industry

        # 2. Load or create client info
        if args.client_info:
            logger.info(f"Loading client info from {args.client_info}")
            client_info = ClientInfo.from_yaml(args.client_info)
        else:
            client_info = ClientInfo(
                name=args.company_name,
                industry=args.industry,
            )

        # 3. Create website configuration
        sections = [s.strip() for s in args.sections.split(",")]
        config = WebsiteConfig(
            page_type=args.page_type,
            sections=sections,
            framework=args.framework,
            include_dark_mode=not args.no_dark_mode,
            include_animations=not args.no_animations,
            seo_optimized=not args.no_seo,
        )

        # 4. Create budget
        budget = Budget(max_usd=args.budget)

        # 5. Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 6. Generate website
        logger.info(f"Generating {args.page_type} website with {len(sections)} sections...")
        logger.info(f"  Design system: {design_system.tone.value}")
        logger.info(f"  Industry: {client_info.industry}")
        logger.info(f"  Framework: {config.framework}")
        logger.info(f"  Output: {output_dir}")

        generator = WebsiteGenerator()
        result = await generator.generate(
            design_system=design_system,
            client_info=client_info,
            config=config,
            output_dir=output_dir,
            budget=budget,
        )

        # 7. Report results
        if result.success:
            logger.info("✅ Website generation complete!")
            logger.info(f"  Components generated: {result.components_generated}")
            logger.info(f"  Total cost: ${result.total_cost:.2f}")
            logger.info(f"  Total time: {result.total_time_seconds:.1f}s")

            if result.quality_report:
                logger.info(f"  Quality score: {result.quality_report.score:.2f}")
                logger.info(f"  Lighthouse: {result.quality_report.lighthouse_score}/100")
                logger.info(f"  WCAG level: {result.quality_report.wcag_level}")
                logger.info(f"  SEO score: {result.quality_report.seo_score}/100")

            logger.info(f"\n📁 Output directory: {output_dir.absolute()}")
            logger.info("  Generated files:")
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output_dir)
                    logger.info(f"    {rel_path}")

            return 0
        else:
            logger.error("❌ Website generation failed:")
            for error in result.errors:
                logger.error(f"  - {error}")
            return 1

    except Exception as e:
        logger.error(f"Website generation failed with error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback

            traceback.print_exc()
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# Example Design System YAML Generator
# ═══════════════════════════════════════════════════════════════════════════════


def generate_design_system_yaml(preset: str, output_path: Path) -> None:
    """Generate a design system YAML file from a preset."""
    design_system = DesignSystem.from_preset(BrandTone(preset))

    yaml_content = f"""# Design System Configuration
# Generated from preset: {preset}

brand:
  name: "Your Brand Name"
  industry: "your_industry"
  tone: "{preset}"

typography:
  font_heading: "{design_system.font_heading}"
  font_body: "{design_system.font_body}"
  scale:
    xs: "{design_system.typography.xs}"
    sm: "{design_system.typography.sm}"
    base: "{design_system.typography.base}"
    lg: "{design_system.typography.lg}"
    xl: "{design_system.typography.xl}"
    2xl: "{design_system.typography.__dict__.get('2xl', '1.953rem')}"
    3xl: "{design_system.typography.__dict__.get('3xl', '2.441rem')}"
    4xl: "{design_system.typography.__dict__.get('4xl', '3.052rem')}"

colors:
  primary: "{design_system.colors.primary}"
  accent: "{design_system.colors.accent}"
  surface: "{design_system.colors.surface}"
  surface_alt: "{design_system.colors.surface_alt}"
  text_primary: "{design_system.colors.text_primary}"
  text_secondary: "{design_system.colors.text_secondary}"
  border: "{design_system.colors.border}"
  success: "{design_system.colors.success}"
  error: "{design_system.colors.error}"

spacing:
  unit: "{design_system.spacing.unit}"
  scale: {design_system.spacing.scale[:10]}

layout:
  max_width: "{design_system.layout.max_width}"
  columns: {design_system.layout.columns}
  gutter: "{design_system.layout.gutter}"
  breakpoints:
    sm: "{design_system.layout.breakpoints['sm']}"
    md: "{design_system.layout.breakpoints['md']}"
    lg: "{design_system.layout.breakpoints['lg']}"
    xl: "{design_system.layout.breakpoints['xl']}"

components:
  border_radius:
    sm: "{design_system.border_radius.sm}"
    md: "{design_system.border_radius.md}"
    lg: "{design_system.border_radius.lg}"
    full: "{design_system.border_radius.full}"
  shadow:
    sm: "{design_system.shadow.sm}"
    md: "{design_system.shadow.md}"
    lg: "{design_system.shadow.lg}"
  animation:
    duration: "{design_system.animation.duration}"
    easing: "{design_system.animation.easing}"

accessibility:
  min_contrast_ratio: {design_system.accessibility.min_contrast_ratio}
  focus_ring: {str(design_system.accessibility.focus_ring).lower()}
  reduced_motion_support: {str(design_system.accessibility.reduced_motion_support).lower()}
  semantic_html: {str(design_system.accessibility.semantic_html).lower()}
"""

    output_path.write_text(yaml_content, encoding="utf-8")
    logger.info(f"Generated design system YAML: {output_path}")
