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


def execute(args) -> None:
    """Execute website generation — LLM-powered with design-system-driven generation."""
    logger = logging.getLogger(__name__)

    print(f"\n>>> Generating '{args.preset}' LLM-powered website: {args.description[:80]}...")
    print(f"   Framework: {args.framework}")

    design_system = DesignSystem(tone=args.preset)

    company_name = args.company_name or args.description.split()[0][:20]
    sections = [s.strip() for s in args.sections.split(",")]
    extra_deps = [d.strip() for d in args.deps.split(",")] if args.deps else []
    if getattr(args, "use_3d", False):
        extra_deps.extend(["three", "@react-three/fiber", "@react-three/drei"])

    client_info = ClientInfo(
        name=company_name,
        industry=args.industry,
        description=args.description,
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
    output_dir = Path(args.output_dir)

    print(f"   Sections: {sections}")
    print(f"   Page type: {args.page_type}")
    if config.source_url:
        print(f"   Source URL: {config.source_url}")
    if extra_deps:
        print(f"   Extra deps: {extra_deps}")

    engine = None
    try:
        orchestrator = Orchestrator(budget=Budget(max_usd=3.0), max_concurrency=3)
        engine = orchestrator
        print("   Engine: LLM-powered (OpenRouter)")
    except Exception as e:
        print(f"   Engine: content-brief fallback ({e})")

    # Wrap engine as a TaskExecutorPort adapter
    class _ExecutorAdapter:
        def __init__(self, eng):
            self._eng = eng

        async def execute(self, task):
            return await self._eng._execute_task(task)

    generator = WebsiteGenerator(executor=_ExecutorAdapter(engine) if engine else None)

    async def _run():
        return await generator.generate(
            design_system=design_system,
            client_info=client_info,
            config=config,
            output_dir=output_dir,
        )

    result = asyncio.run(_run())

    has_index = (output_dir / "index.html").exists()
    has_nextjs = (output_dir / "package.json").exists()

    if result.success:
        print(f"[OK] LLM-powered website: {output_dir.resolve()}")
        print(f"   Components: {result.components_generated}")
        print(f"   Cost: ${result.total_cost:.4f}")
    elif has_nextjs:
        print("!  Content-brief fallback (engine not available)")
        print(f"[OK] Next.js + Tailwind: {output_dir.resolve()}")
        print(f"   Run: cd {output_dir} && npm install && npm run dev")
    elif has_index:
        print(f"!  Pipeline issues ({result.errors[0][:60]}...), but fallback HTML written")
        print(f"[OK] Fallback website: {output_dir.resolve() / 'index.html'}")
        print(f"   Size: {(output_dir / 'index.html').stat().st_size} bytes")
    else:
        print(f"[FAIL] Failed: {result.errors}")
