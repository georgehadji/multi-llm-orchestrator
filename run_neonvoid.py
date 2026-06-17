"""
Direct WebsiteGenerator invocation — NEONVOID ad agency site.
Bypasses CLI hardcoded ClientInfo/SaaS template.
"""
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from orchestrator.design_system import DesignSystem
from orchestrator.generators.website_generator import ClientInfo, WebsiteConfig, WebsiteGenerator
from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

async def main():
    design_system = DesignSystem(tone="luxury")

    client_info = ClientInfo(
        name="NEONVOID",
        industry="advertising",
        description=(
            "NEONVOID is a bold Los Angeles creative advertising agency specializing in "
            "immersive brand experiences. They craft cinematic campaigns for global brands "
            "with a signature dark-neon aesthetic. Their work combines 3D motion design, "
            "WebGL interactivity, and typography-forward layouts. Clients include Nike, "
            "Spotify, and Red Bull. The site should feel like an Awwwards-winning portfolio — "
            "experimental layouts, massive typography, smooth animations, glassmorphism UI, "
            "and a dark theme with electric blue (#0066ff) and magenta (#ff0066) accents."
        ),
        target_audience="Fortune 500 brand managers, CMOs, creative directors",
        competitors=["AKQA", "R/GA", "Huge", "Fantasy"],
        preferences={
            "vibe": "dark cinematic neon",
            "motion": "smooth and premium",
            "typography": "bold display headings, clean body",
        },
    )

    config = WebsiteConfig(
        framework="react",
        styling="tailwind",
        page_type="portfolio",
        sections=["hero", "work", "services", "about", "clients", "contact"],
        atelier_theme="midnight",
        include_dark_mode=True,
        include_animations=True,
        seo_optimized=True,
        performance_optimized=True,
    )

    output_dir = Path("outputs/neonvoid-v2")

    # Pre-create subdirs to avoid Windows permission races during image gen
    for sub in ["public/images", "public/.well-known", "components"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Building NEONVOID ad agency site...")
    print(f"   Sections: {config.sections}")
    print(f"   Theme: {config.atelier_theme}")
    print(f"   Framework: {config.framework}")
    print(f"   Output: {output_dir.resolve()}")
    print()

    orchestrator = Orchestrator(budget=Budget(max_usd=5.0), max_concurrency=3)
    generator = WebsiteGenerator(orchestrator_engine=orchestrator)

    result = await generator.generate(
        design_system=design_system,
        client_info=client_info,
        config=config,
        output_dir=output_dir,
    )

    print(f"\n{'[OK]' if result.success else '[FAIL]'} Complete: {output_dir.resolve()}")
    print(f"   Components: {result.components_generated}")
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Time: {result.total_time_seconds:.1f}s")
    if result.errors:
        for e in result.errors:
            print(f"   !  {e}")

asyncio.run(main())
