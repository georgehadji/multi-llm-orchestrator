"""
Website generation command handler — extracted from cli.py.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..budget import Budget
from ..design_system import DesignSystem
from ..engine import Orchestrator
from ..generators.website_generator import ClientInfo, WebsiteConfig, WebsiteGenerator


def register(subparsers) -> None:
    """Register the 'website' subcommand arguments."""
    wp = subparsers.add_parser("website", help="Generate a website with design system")
    wp.add_argument("--description", "-d", required=True, help="Website description")
    wp.add_argument("--output-dir", "-o", default="outputs/website", help="Output directory")
    wp.add_argument(
        "--framework",
        "-f",
        default="html",
        choices=["html", "react", "next.js"],
        help="Target framework",
    )
    wp.add_argument(
        "--preset",
        default="modern",
        choices=["modern", "minimalist", "playful", "corporate", "luxury", "tech"],
        help="Design preset",
    )
    wp.add_argument(
        "--image-model",
        default="auto",
        help="OpenRouter image model (default: auto-select; use 'none' for SVG only)",
    )
    wp.add_argument(
        "--image-quality",
        default="balanced",
        choices=["draft", "balanced", "premium"],
        help="Image quality tier (draft=cheapest, balanced=best VFM, premium=best quality)",
    )
    wp.add_argument("--atelier-theme", default="", help="Atelier design theme")
    wp.add_argument(
        "--sections",
        "-s",
        default="hero,features,pricing,testimonials,faq,cta,footer",
        help="Comma-separated section names",
    )
    wp.add_argument("--company-name", default="", help="Brand/company name")
    wp.add_argument("--industry", default="technology", help="Client industry")
    wp.add_argument(
        "--page-type",
        default="landing",
        choices=["landing", "saas", "portfolio", "ecommerce", "agency", "editorial", "custom"],
        help="Type of page",
    )
    wp.add_argument("--deps", default="", help="Extra npm deps (comma-separated)")
    wp.add_argument(
        "--3d", dest="use_3d", action="store_true", default=False, help="3D FX shorthand"
    )
    wp.add_argument("--source-url", default="", help="Live URL to clone via Playwright")
    wp.set_defaults(func=execute)


def execute(args) -> None:
    """Execute website generation — LLM-powered with design-system-driven generation."""
    logger = logging.getLogger(__name__)

    print(f"\n>>> Generating '{args.preset}' LLM-powered website: {args.description[:80]}...")
    print(f"   Framework: {args.framework}")

    design_system = DesignSystem(tone=args.preset)

    company_name = args.company_name or (
        args.description.split()[0][:20] if args.description.split() else "Company"
    )
    sections = [s.strip() for s in args.sections.split(",")]
    extra_deps = [d.strip() for d in args.deps.split(",")] if args.deps else []
    if getattr(args, "use_3d", False):
        extra_deps.extend(["three", "@react-three/fiber", "@react-three/drei"])

    client_info = ClientInfo(
        name=company_name, industry=args.industry, description=args.description
    )
    config = WebsiteConfig(
        framework=args.framework,
        styling="tailwind" if args.framework != "html" else "css",
        page_type=args.page_type,
        sections=sections,
        image_model=args.image_model,
        atelier_theme=args.atelier_theme,
        description=args.description,
        brand_name=company_name,
        dependencies=["react", "react-dom"] + extra_deps,
        image_quality=getattr(args, "image_quality", "balanced"),
        source_url=args.source_url if hasattr(args, "source_url") else "",
    )
    from ._path_utils import resolve_allowed_path

    _outputs_root = (Path(__file__).parent.parent.parent / "outputs").resolve()
    output_dir = resolve_allowed_path(args.output_dir, outputs_root=_outputs_root)

    print(f"   Sections: {sections}   Page type: {args.page_type}")
    if config.source_url:
        print(f"   Source URL: {config.source_url}")

    engine = None
    try:
        engine = Orchestrator(budget=Budget(max_usd=3.0), max_concurrency=3)
        print("   Engine: LLM-powered (OpenRouter)")
    except Exception as e:
        print(f"   Engine: content-brief fallback ({e})")

    from ..domain.ports import TaskExecutorAdapter

    gen = WebsiteGenerator(
        executor=TaskExecutorAdapter(engine._execute_task) if engine else None,
        orchestrator_engine=engine,
    )

    async def _run():
        return await gen.generate(
            design_system=design_system,
            client_info=client_info,
            config=config,
            output_dir=output_dir,
        )

    result = asyncio.run(_run())

    if result.success:
        print(
            f"[OK] Website: {output_dir.resolve()}  Components: {result.components_generated}  Cost: ${result.total_cost:.4f}"
        )
    elif (output_dir / "package.json").exists():
        print(f"!  Fallback: cd {output_dir} && npm install && npm run dev")
    elif (output_dir / "index.html").exists():
        print(f"!  Fallback HTML: {(output_dir / 'index.html').stat().st_size} bytes")
    else:
        print(f"[FAIL] {result.errors}")
