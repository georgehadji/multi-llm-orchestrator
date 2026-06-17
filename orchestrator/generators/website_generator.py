"""
Website Generator for DSDG (Design-System-Driven Generation)
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Main pipeline for generating websites using design system-driven approach.
Integrates with existing Orchestrator engine for parallel execution.
"""

from __future__ import annotations

import asyncio
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

from ..design_system import DesignSystem, QualityReport

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


logger = logging.getLogger(__name__)
from ..models import ProjectState, Task, TaskType

logger = logging.getLogger(__name__)


def _page_type_schema(page_type: str, site_name: str) -> tuple[str, str]:
    """Return (Schema.org @type, applicationCategory) for a given page type."""
    mapping = {
        "agency": ("Organization", ""),
        "portfolio": ("CreativeWork", ""),
        "editorial": ("WebSite", ""),
        "ecommerce": ("WebSite", ""),
        "saas": ("SoftwareApplication", "BusinessApplication"),
        "landing": ("SoftwareApplication", "BusinessApplication"),
    }
    return mapping.get(page_type, ("WebSite", ""))


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
    image_model: str = ""  # OpenRouter image model ID; empty = SVG placeholders
    atelier_theme: str = ""  # Atelier design theme slug (e.g. specimen, midnight)
    # ── Phase 2: user description + deps ──
    description: str = ""  # User's full project description (injected into LLM prompts)
    brand_name: str = ""  # Brand/company name for metadata and prompts
    dependencies: list[str] = field(default_factory=lambda: ["react", "react-dom"])  # npm deps
    image_quality: str = "balanced"  # "draft", "balanced", "premium"
    # ── Phase 3: URL source extraction ──
    source_url: str = ""  # Live URL to extract design tokens, fonts, and content from


@dataclass
class ExtractedSiteData:
    """Data extracted from a live website URL.

    Populated by WebsiteExtractor when source_url is provided.
    Maps directly to the cloner template's spec file pattern.
    """

    url: str = ""
    # Design tokens
    colors: dict[str, str] = field(default_factory=dict)  # CSS variable name → value
    fonts: list[dict] = field(default_factory=list)  # [{family, weights, style, url}]
    # Page topology
    sections: list[dict] = field(default_factory=list)  # [{name, selector, order}]
    # Computed CSS per section
    section_styles: dict[str, dict] = field(default_factory=dict)  # section → {selector → {prop → value}}
    # Extracted text content per section
    section_content: dict[str, str] = field(default_factory=dict)  # section → text
    # Assets
    images: list[dict] = field(default_factory=list)  # [{src, alt, width, height}]
    favicons: list[dict] = field(default_factory=list)
    # Full-page screenshots
    screenshots: list[str] = field(default_factory=list)
    # Global behaviors
    behaviors: list[dict] = field(default_factory=list)  # [{type, trigger, before, after}]
    # Raw extraction JS (for debugging / builder agent use)
    extraction_script: str = ""


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
    sanitization_fixes: int = 0  # number of sections fixed by _sanitize_output
    total_cost: float = 0.0
    total_time_seconds: float = 0.0


class ContentResearcher:
    """Integrated industry research -> content generation — now LLM-powered with template fallback."""

    def __init__(self, nexus_search=None):
        self.nexus_search = nexus_search

    async def generate_content_brief(
        self,
        client_info: ClientInfo,
        engine=None,
        config: "WebsiteConfig | None" = None,
    ) -> ContentBrief:
        """
        Generate content brief from client info.

        When an orchestrator engine is available, uses LLM to generate
        industry-specific content. Falls back to template-based generation
        if no engine is available or the LLM call fails.
        """
        # Try LLM-powered generation first
        if engine is not None:
            try:
                return await self._generate_brief_llm(client_info, engine, config)
            except Exception as e:
                logger.warning("LLM content brief failed, falling back to template: %s", e)

        # Fallback: template-based (industry-aware but generic)
        return self._generate_brief_template(client_info)

    async def _generate_brief_llm(
        self,
        client_info: ClientInfo,
        engine,
        config: "WebsiteConfig | None" = None,
    ) -> ContentBrief:
        """Use the orchestrator engine to generate an industry-specific content brief."""
        sections = getattr(config, "sections", ["hero", "features", "pricing"]) if config else ["hero", "features", "pricing"]
        page_type = getattr(config, "page_type", "landing") if config else "landing"

        prompt = f"""Generate a content brief for a {page_type} website.

Brand: {client_info.name}
Industry: {client_info.industry}
Description: {getattr(client_info, 'description', '') or 'A modern website'}
Target Audience: {getattr(client_info, 'target_audience', 'professionals')}
Competitors: {', '.join(getattr(client_info, 'competitors', [])) or 'industry leaders'}
Sections needed: {', '.join(sections)}

Return a JSON object with these keys:
- "headlines": dict mapping each section name to a compelling headline (tailored to this brand/industry)
- "ctas": dict mapping relevant sections to call-to-action text
- "value_props": list of 4-6 value propositions (industry-specific, not generic SaaS)
- "faqs": list of 3-5 {{"question": "...", "answer": "...", "section": "faq"}} objects (industry-relevant)
- "testimonials_angles": list of 4 angles for testimonials (specific to this industry)
- "pain_points": list of 4-6 pain points this industry's customers face
- "tagline": a short brand tagline
- "social_proof": list of 2-3 {{"name": "...", "role": "...", "quote": "..."}} sample testimonials

IMPORTANT: Do NOT use generic SaaS content. Tailor EVERYTHING to the {client_info.industry} industry.
For a portfolio/agency site, headlines should be creative and brand-forward, not "Why Choose Us".
For an ecommerce site, focus on products and shopping experience.
For an editorial site, focus on content and readership.

Return ONLY valid JSON, no markdown fences."""

        try:
            from ..models import Task, TaskType

            task = Task(
                id="content_brief_000",
                type=TaskType.CODE_GEN,
                prompt=prompt,
                max_output_tokens=2048,
                acceptance_threshold=0.7,
                max_iterations=1,
            )
            result = await engine._execute_task(task)
            if result and result.output:
                import json
                data = json.loads(result.output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                brief = ContentBrief(
                    headlines=data.get("headlines", {}),
                    ctas=data.get("ctas", {}),
                    value_props=data.get("value_props", []),
                    faqs=data.get("faqs", []),
                    testimonials_angles=data.get("testimonials_angles", []),
                    pain_points=data.get("pain_points", []),
                    tagline=data.get("tagline", ""),
                    social_proof=data.get("social_proof", []),
                )
                logger.info("LLM-generated content brief with %d sections", len(brief.headlines))
                return brief
        except Exception as e:
            logger.warning("Failed to parse LLM content brief JSON: %s", e)
            raise

        raise RuntimeError("LLM content brief generation returned no output")

    def _generate_brief_template(self, client_info: ClientInfo) -> ContentBrief:
        """Template-based fallback — industry-aware but generic."""
        industry = getattr(client_info, "industry", "technology")
        name = getattr(client_info, "name", "the company")

        brief = ContentBrief(
            headlines={
                "hero": f"Transform Your {industry.title()} Experience",
                "features": "Why Choose Us",
                "pricing": "Simple, Transparent Pricing",
                "testimonials": "What Our Clients Say",
                "faq": "Frequently Asked Questions",
                "cta": f"Ready to Get Started with {name}?",
                "work": "Our Work",
                "portfolio": "Selected Projects",
                "services": "What We Offer",
                "about": f"About {name}",
                "clients": "Trusted By",
                "contact": "Get In Touch",
            },
            ctas={
                "hero": "Get Started Free",
                "pricing": "Choose Your Plan",
                "cta": "Start Your Free Trial",
                "work": "View Our Work",
                "contact": "Contact Us",
            },
            faqs=[
                {
                    "question": "How do I get started?",
                    "answer": f"Simply reach out to our team and we'll set up a consultation to understand your needs.",
                    "section": "faq",
                },
                {
                    "question": f"What makes {name} different?",
                    "answer": f"We combine deep {industry} expertise with cutting-edge technology to deliver results that matter.",
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
                f"Complex {industry} solutions that are hard to use",
                "Poor customer support",
                "Hidden fees and unclear pricing",
                "Outdated technology",
            ],
            competitor_insights=[],
        )

        logger.info(f"Template content brief with {len(brief.headlines)} sections")
        return brief


class WebsiteExtractor:
    """
    Extract visual design tokens, content, and assets from a live website URL.

    Uses BrowserTester (Playwright) to navigate the target site and execute
    JavaScript extraction scripts — mirroring the cloner template's Phase 1-2
    reconnaissance methodology.

    Usage:
        extractor = WebsiteExtractor()
        data = await extractor.extract("https://example.com")
    """

    # ── CSS extraction JS (mirrors cloner template's getComputedStyle() extraction) ──
    _EXTRACT_CSS_JS = r"""
(async () => {
    const props = [
        'fontSize','fontWeight','fontFamily','lineHeight','letterSpacing','color',
        'textTransform','textDecoration','backgroundColor','background',
        'padding','paddingTop','paddingRight','paddingBottom','paddingLeft',
        'margin','marginTop','marginRight','marginBottom','marginLeft',
        'width','height','maxWidth','minWidth','maxHeight','minHeight',
        'display','flexDirection','justifyContent','alignItems','gap',
        'gridTemplateColumns','gridTemplateRows',
        'borderRadius','border','borderTop','borderBottom','borderLeft','borderRight',
        'boxShadow','overflow','overflowX','overflowY',
        'position','top','right','bottom','left','zIndex',
        'opacity','transform','transition','cursor',
        'objectFit','objectPosition','mixBlendMode','filter','backdropFilter',
        'whiteSpace','textOverflow','WebkitLineClamp'
    ];
    // 1. Extract global CSS custom properties
    const root = document.documentElement;
    const rootCS = getComputedStyle(root);
    const cssVars = {};
    for (let i = 0; i < rootCS.length; i++) {
        const name = rootCS[i];
        if (name.startsWith('--')) cssVars[name] = rootCS.getPropertyValue(name).trim();
    }
    // 2. Color palette: sample common elements
    const colorEls = ['h1','h2','p','a','button','header','footer','main','section','nav','body'];
    const colorMap = {};
    colorEls.forEach(tag => {
        const el = document.querySelector(tag);
        if (el) {
            const cs = getComputedStyle(el);
            colorMap[tag] = {
                color: cs.color,
                bg: cs.backgroundColor,
                font: cs.fontFamily,
                size: cs.fontSize
            };
        }
    });
    // 3. Font discovery
    const fontFamilies = [...new Set([...document.querySelectorAll('*')].slice(0, 200).map(el => getComputedStyle(el).fontFamily))];
    const fontLinks = [...document.querySelectorAll('link[rel="stylesheet"]')].map(l => l.href).filter(h => h.includes('fonts') || h.includes('google'));
    // 4. Favicons
    const favicons = [...document.querySelectorAll('link[rel*="icon"], link[rel="apple-touch-icon"]')].map(l => ({ rel: l.rel, href: l.href, sizes: l.sizes?.toString() }));
    // 5. Images
    const images = [...document.querySelectorAll('img[src]')].slice(0, 30).map(img => ({
        src: img.src || img.currentSrc,
        alt: img.alt,
        w: img.naturalWidth,
        h: img.naturalHeight
    }));
    // 6. Section discovery: heading-based page topology
    const sections = [];
    let idx = 0;
    document.querySelectorAll('section, div[class*="section"], div[class*="hero"], div[class*="feature"], div[class*="pricing"], div[class*="footer"], header, footer').forEach(el => {
        const heading = el.querySelector('h1, h2, h3, h4');
        const text = heading ? heading.textContent.trim() : (el.className?.split(' ')[0] || '');
        const rect = el.getBoundingClientRect();
        sections.push({
            order: idx++,
            name: text.slice(0, 60),
            tag: el.tagName.toLowerCase(),
            classes: (el.className || '').slice(0, 100),
            rect: { w: Math.round(rect.width), h: Math.round(rect.height), top: Math.round(rect.top) }
        });
    });
    return JSON.stringify({ cssVars, colorMap, fontFamilies, fontLinks, favicons, images: images.slice(0, 20), sections: sections.slice(0, 30), extractionScript: 'extracted' });
})();
"""

    async def extract(self, url: str, output_dir: Path | None = None) -> ExtractedSiteData:
        """
        Navigate to a URL and extract design tokens, topology, and assets.

        Args:
            url: The target website URL to extract from.
            output_dir: Optional directory to save screenshots and raw data.

        Returns:
            ExtractedSiteData with all discovered design tokens, sections, and assets.
        """
        logger.info("WebsiteExtractor: extracting from %s", url)

        from ..browser_testing import BrowserTester

        data = ExtractedSiteData(url=url)
        screenshots_dir = (output_dir / "screenshots" if output_dir else Path("outputs/extraction/screenshots"))
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        tester = BrowserTester(browser_type="chromium", headless=True, screenshot_on_failure=True)

        try:
            await tester.initialize()

            # Navigate to target URL
            logger.info("WebsiteExtractor: navigating to %s ...", url)
            await tester.page.goto(url, wait_until="networkidle")
            await tester.page.wait_for_timeout(1000)  # Let JS render settle

            # ── Full-page screenshots at 2 viewports ──
            for vp_name, vp_width, vp_height in [("desktop", 1440, 900), ("mobile", 390, 844)]:
                await tester.page.set_viewport_size({"width": vp_width, "height": vp_height})
                await tester.page.wait_for_timeout(500)
                shot_path = str(screenshots_dir / f"{vp_name}_{vp_width}x{vp_height}.png")
                await tester.page.screenshot(path=shot_path, full_page=True)
                data.screenshots.append(shot_path)
                logger.info("  Screenshot: %s", shot_path)

            # ── Run CSS extraction JS ──
            extraction_js = self._EXTRACT_CSS_JS
            raw = await tester.page.evaluate(extraction_js)
            import json

            extracted = json.loads(raw) if isinstance(raw, str) else raw
            data.extraction_script = extraction_js

            # Parse CSS custom properties → colors dict
            css_vars = extracted.get("cssVars", {})
            data.colors = css_vars

            # Parse font families — take first entry from each CSS font-stack
            font_families = extracted.get("fontFamilies", [])
            data.fonts = [
                {"family": f.split(",")[0].strip().strip("\"'"), "weights": [400], "style": "normal", "url": ""}
                for f in font_families if f and f.strip()
            ]

            # Parse favicons
            data.favicons = extracted.get("favicons", [])

            # Parse images
            data.images = [
                {"src": img.get("src", ""), "alt": img.get("alt", ""), "width": img.get("w", 0), "height": img.get("h", 0)}
                for img in extracted.get("images", [])
            ]

            # Parse section topology
            raw_sections = extracted.get("sections", [])
            data.sections = raw_sections

            # ── Extract text content per section (using Playwright arg-passing, safe) ──
            for sec in raw_sections[:10]:
                classes = sec.get('classes', '')
                tag = sec.get('tag', 'section')
                sel = f"{tag}.{classes.split(' ')[0]}" if classes else tag
                try:
                    text_content = await tester.page.evaluate(
                        """(sel) => {
                            const el = document.querySelector(sel);
                            return el ? el.innerText.slice(0, 2000) : '';
                        }""",
                        sel,
                    )
                    if text_content and isinstance(text_content, str) and text_content.strip():
                        data.section_content[sec.get("name", f"section_{sec.get('order', 0)}")] = text_content.strip()[:2000]
                except Exception:
                    pass

            logger.info("WebsiteExtractor: extracted %d CSS vars, %d fonts, %d sections, %d images",
                        len(data.colors), len(data.fonts), len(data.sections), len(data.images))

        except Exception as e:
            logger.error("WebsiteExtractor: extraction failed: %s", e)
            raise
        finally:
            await tester.close()

        return data


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

    @staticmethod
    def _sanitize_output(raw: str, section_name: str = "") -> tuple[str, list[str]]:
        """Clean and validate LLM-generated component output before writing to disk.

        Handles the four bug classes discovered in production:
        1. Markdown code fences (`` ```tsx `` / `` ``` ``)
        2. Truncated output (unclosed tags, unbalanced braces)
        3. SWC-incompatible casts (`` } as React.CSSProperties} ``)
        4. Template literals with oklch() number ambiguities

        Args:
            raw: Raw LLM output string.
            section_name: Section name for log context.

        Returns:
            (cleaned_output, warnings_list)
        """
        warnings: list[str] = []
        cleaned = raw.strip()

        # 1. Strip markdown code fences
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl > 0:
                cleaned = cleaned[first_nl + 1 :]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[: -3].rstrip()
            warnings.append("stripped markdown code fences")

        # 2. Fix SWC-incompatible casts (} as React.CSSProperties} → }})
        if "as React.CSSProperties}" in cleaned:
            cleaned = cleaned.replace("} as React.CSSProperties}", "}}")
            warnings.append("removed as React.CSSProperties cast(s)")

        # 3. Fix template literals with oklch() → number ambiguity
        #    Pattern: `${ink} 1px` where ink = "oklch(0.15 0.01 95)"
        #    SWC fails parsing `oklch(...)1px` — convert to string concat
        import re as _re

        tl_pattern = _re.compile(r"`([^`]*)\$\{([^}]+)\}([^`]*)`")
        fixed_tls = 0

        def _replace_tl(match):
            nonlocal fixed_tls
            before, expr, after = match.group(1), match.group(2).strip(), match.group(3)
            # Only fix if the expression is a simple variable name (not complex expressions)
            if _re.match(r"^[a-zA-Z_]\w*(\.\w+)*$", expr):
                fixed_tls += 1
                parts = []
                if before:
                    parts.append(repr(before))
                parts.append(expr)
                if after:
                    parts.append(repr(after))
                return " + ".join(parts)
            return match.group(0)

        cleaned = tl_pattern.sub(_replace_tl, cleaned)
        if fixed_tls > 0:
            warnings.append(f"converted {fixed_tls} template literal(s) to string concat")

        # 4. Detect truncation: check balanced braces and closing tag
        open_b = cleaned.count("{")
        close_b = cleaned.count("}")
        if open_b != close_b:
            warnings.append(f"unbalanced braces ({open_b} open, {close_b} close) — output may be truncated")

        # Check for trailing truncated patterns
        last_line = cleaned.rsplit("\n", 1)[-1].strip() if "\n" in cleaned else cleaned
        truncation_signs = [
            last_line.endswith("<") or last_line.endswith("</"),
            last_line.endswith("={") or last_line.endswith("=("),
            last_line.endswith("d=") or last_line.endswith("d='") or last_line.endswith('d="'),
            last_line == "exit",
            len(last_line) < 10 and ("<" in last_line or "{" in last_line),
        ]
        if any(truncation_signs):
            warnings.append("output appears truncated (incomplete last line)")

        # Check for minimum viable output
        if len(cleaned) < 200:
            warnings.append(f"output too short ({len(cleaned)} chars) — likely failed generation")

        if warnings:
            logger.warning("WebsiteGenerator._sanitize [%s]: %s", section_name, "; ".join(warnings))

        return cleaned, warnings

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
            content_brief = await self._researcher.generate_content_brief(
                client_info, engine=self._engine, config=config
            )

            # ── Step 1.5: URL source extraction (when source_url is set) ──
            extraction_data: ExtractedSiteData | None = None
            if config.source_url:
                logger.info("WebsiteGenerator: extracting from source URL %s ...", config.source_url)
                try:
                    extractor = WebsiteExtractor()
                    extraction_data = await extractor.extract(config.source_url, output_dir=output_dir)
                    # Override design system colors with extracted CSS custom properties
                    if extraction_data.colors:
                        for css_var, value in extraction_data.colors.items():
                            if value and not value.startswith("var("):
                                clean_val = value.strip()
                                # Map CSS var names to ColorTokens fields
                                # e.g. --primary → primary, --color-primary → primary
                                var_short = css_var.replace("--", "").replace("-", "_").removeprefix("color_")
                                if hasattr(design_system.colors, var_short):
                                    setattr(design_system.colors, var_short, clean_val)
                    # Override fonts with extracted font families
                    if extraction_data.fonts:
                        families = [f["family"] for f in extraction_data.fonts if f.get("family")]
                        if families:
                            design_system.typography.font_sans = families[0]
                            if len(families) > 1:
                                design_system.typography.font_mono = families[1]
                    # Update section list from discovered topology (if user didn't override)
                    if extraction_data.sections and config.sections == ["hero", "features", "pricing", "testimonials", "faq", "cta", "footer"]:
                        discovered = [s.get("name", "").lower().replace(" ", "_") or f"section_{s['order']}" for s in extraction_data.sections]
                        if discovered:
                            config.sections = discovered
                            logger.info("  Updated sections from extraction: %s", config.sections)
                    logger.info("  Extraction complete: %d CSS vars, %d fonts, %d sections",
                                len(extraction_data.colors), len(extraction_data.fonts), len(extraction_data.sections))
                except Exception as extract_err:
                    logger.warning("URL extraction failed, falling back to LLM generation: %s", extract_err)
                    extraction_data = None

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
                client_info=client_info,
                extraction_data=extraction_data,
            )

            # Step 4: Execute through orchestrator (if available)
            if self._engine:
                logger.info(f"LLM-powered generation: {len(tasks)} sections via orchestrator")
                max_concurrent = getattr(self._engine, "max_concurrency", 3)
                semaphore = asyncio.Semaphore(max_concurrent)
                sanitization_counts: list[int] = [0]

                async def _run_section(i: int, task: Task) -> tuple[int, bool]:
                    async with semaphore:
                        section_name = config.sections[i]
                        for attempt in range(2):
                            try:
                                component_result = await self._engine._execute_task(task)
                                if not component_result or not component_result.output:
                                    logger.warning(f"  ✗ {section_name}: empty LLM output")
                                    return i, False

                                cleaned, warnings_list = self._sanitize_output(
                                    component_result.output, section_name
                                )

                                # Retry once if output appears truncated
                                if any("truncated" in w for w in warnings_list) and attempt == 0:
                                    logger.info(f"  ↻ {section_name}: retrying with 2× tokens (truncated)")
                                    task.max_output_tokens = min(task.max_output_tokens * 2, 16384)
                                    continue

                                ext = ".html" if config.framework == "html" else ".tsx"
                                comp_path = output_dir / "components" / f"{section_name}{ext}"
                                comp_path.parent.mkdir(parents=True, exist_ok=True)
                                comp_path.write_text(cleaned, encoding="utf-8")
                                result.total_cost += getattr(component_result, "cost_usd", 0)

                                if warnings_list:
                                    sanitization_counts[0] += 1
                                status = "⚠" if warnings_list else "✓"
                                logger.info(
                                    f"  {status} {section_name}: {len(cleaned)} chars"
                                    f"{' (' + '; '.join(warnings_list) + ')' if warnings_list else ''}"
                                )
                                return i, True
                            except Exception as task_err:
                                if attempt == 0:
                                    logger.info(f"  ↻ {section_name}: retrying after error: {task_err}")
                                    continue
                                logger.warning(f"  ✗ {section_name}: {task_err}")
                                return i, False
                        return i, False

                results = await asyncio.gather(
                    *[_run_section(i, task) for i, task in enumerate(tasks)]
                )
                result.components_generated = sum(1 for _, ok in results if ok)
                result.sanitization_fixes = sanitization_counts[0]
                result.success = any(ok for _, ok in results)
            else:
                # Without engine, generate content from content brief
                logger.warning("No orchestrator engine available — using content brief")
                try:
                    self._create_content_from_brief(
                        output_dir=output_dir,
                        sections=config.sections,
                        design_system=design_system,
                        content_brief=content_brief,
                        config=config,
                    )
                    result.components_generated = len(config.sections)
                except Exception as brief_err:
                    logger.warning(f"Content-from-brief failed: {brief_err}")
                    self._create_content_from_brief(
                        output_dir=output_dir,
                        sections=config.sections,
                        design_system=design_system,
                        content_brief=None,
                        config=config,
                    )
                    result.components_generated = len(config.sections)
                result.components_generated = len(config.sections)

            # Step 5: Generate placeholder images (hero bg, OG, section images)
            logger.info("WebsiteGenerator: generating images...")
            await self._generate_images(output_dir, config, design_system)

            # Step 5.5: Install deps + build-verify + auto-fix errors
            if config.framework in ("next.js", "react"):
                logger.info("WebsiteGenerator: installing npm dependencies + build-verifying...")
                build_ok, build_log = await self._verify_and_fix_build(
                    output_dir, config, result
                )
                if not build_ok:
                    logger.warning("Build verification had issues (see log)")
                    result.errors.append("build-verify: " + build_log[-1] if build_log else "unknown build error")

            # Step 6: Assemble final page
            logger.info("WebsiteGenerator: assembling page...")
            self._assemble_page(
                output_dir=output_dir,
                sections=config.sections,
                design_system=design_system,
                config=config,
            )

            # Step 6: Generate quality report (optional — validator may be missing)
            logger.info("WebsiteGenerator: validating quality...")
            try:
                from .website_validator import WebsiteQualityValidator

                validator = WebsiteQualityValidator()
                quality_report = await validator.validate(output_dir)
                result.quality_report = quality_report
            except ImportError:
                logger.warning("WebsiteQualityValidator not available — skipping quality validation")
                quality_report = None

            result.success = True
            result.total_time_seconds = time.time() - start_time

            if quality_report is not None:
                logger.info(
                    f"WebsiteGenerator: complete in {result.total_time_seconds:.1f}s, "
                    f"quality score: {quality_report.score:.2f}"
                )
            else:
                logger.info(
                    f"WebsiteGenerator: complete in {result.total_time_seconds:.1f}s "
                    f"(quality validation skipped)"
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
        client_info=None,
        extraction_data: ExtractedSiteData | None = None,
    ) -> list[Task]:
        """Create orchestration tasks for each section."""
        tasks = []

        for i, section in enumerate(sections):
            component = components[i] if i < len(components) else None
            # Look up extracted data for this section (by index or name match)
            section_extraction = None
            if extraction_data and extraction_data.section_content:
                # Try name match
                for sec_name, sec_content in extraction_data.section_content.items():
                    if section.lower() in sec_name.lower() or sec_name.lower() in section.lower():
                        section_extraction = {
                            "text": sec_content[:1500],
                        }
                        break
            prompt = self._build_section_prompt(
                section=section,
                component=component or section,
                design_system=design_system,
                content_brief=content_brief,
                config=config,
                client_info=client_info,
                extraction_data=extraction_data,
                section_extraction=section_extraction,
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
                target_path=f"components/{section}{'.html' if config.framework == 'html' else '.tsx'}",
                tech_context=f"{config.framework} + {config.styling}",
                acceptance_threshold=0.85,
                max_iterations=3,
                max_output_tokens=8192,  # bumped from 4096 — complex 3D TSX needs headroom
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
        client_info=None,
        extraction_data: ExtractedSiteData | None = None,
        section_extraction: dict | None = None,
    ) -> str:
        """Build prompt for generating a section — now with project brief, section-type guidance, and library awareness."""
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

        # ── PROJECT BRIEF — the user's actual description ──
        project_brief = ""
        if client_info is not None:
            brief_parts = []
            if getattr(client_info, "name", ""):
                brief_parts.append(f"Brand/Company: {client_info.name}")
            if getattr(client_info, "industry", ""):
                brief_parts.append(f"Industry: {client_info.industry}")
            if getattr(client_info, "description", "") or config.description:
                desc_text = getattr(client_info, "description", "") or config.description
                brief_parts.append(f"Project Description: {desc_text}")
            if getattr(client_info, "target_audience", ""):
                brief_parts.append(f"Target Audience: {client_info.target_audience}")
            if getattr(client_info, "competitors", []):
                brief_parts.append(f"Competitors: {', '.join(client_info.competitors)}")
            if brief_parts:
                project_brief = "PROJECT BRIEF:\n" + "\n".join(brief_parts) + "\n\n"
                project_brief += "IMPORTANT: This is NOT a generic SaaS site. Tailor ALL content, tone, and design to this specific brand and industry.\n"

        # ── SECTION-TYPE GUIDANCE — tells LLM what each section should be ──
        section_guidance = self._get_section_guidance(section, config.page_type)

        # ── LIBRARY AWARENESS — tells LLM what 3D/animation libraries are available ──
        library_context = ""
        deps = getattr(config, "dependencies", []) or []
        _3d_deps = [d for d in deps if any(kw in d.lower() for kw in ("three", "react-three", "drei", "fiber", "cannon", "babylon"))]
        anim_deps = [d for d in deps if any(kw in d.lower() for kw in ("gsap", "framer-motion", "motion", "lenis", "spring"))]
        if _3d_deps:
            library_context += (
                f"\n3D LIBRARIES AVAILABLE: {', '.join(_3d_deps)}\n"
                "You CAN use these for immersive effects: Three.js scenes, 3D models, particle systems, "
                "post-processing, canvas-based backgrounds, WebGL effects. Include all necessary imports.\n"
                "IMPORTANT: Use 'use client' directive for any component using browser APIs or 3D libraries.\n"
            )
        if anim_deps:
            library_context += (
                f"\nANIMATION LIBRARIES AVAILABLE: {', '.join(anim_deps)}\n"
                "Use these for scroll-triggered animations, page transitions, hover effects, and micro-interactions.\n"
            )

        atelier_theme_context = ""
        if config.atelier_theme:
            try:
                from ..design.atelier.themes import get_theme, theme_to_prompt_context
                theme = get_theme(config.atelier_theme)
                if theme:
                    atelier_theme_context = theme_to_prompt_context(theme)
            except ImportError:
                pass

        return f"""
You are building a premium website section using design system-driven development.

{project_brief}
{design_system.to_prompt_context()}

COMPONENT REFERENCE:
Name: {component_name}
Source: {source_str}
Category: {category}

DESCRIPTION:
{desc}

SECTION: {section}
PAGE TYPE: {config.page_type}

{section_guidance}
{library_context}

CONTENT:
Headline: {headline}
CTA: {cta}

CONFIGURATION:
Framework: {config.framework}
Styling: {config.styling}
Dark Mode: {'Yes' if config.include_dark_mode else 'No'}
Animations: {'Yes' if config.include_animations else 'No'}
SEO Optimized: {'Yes' if config.seo_optimized else 'No'}
Atelier Theme: {"Yes" if config.atelier_theme else "None"}

{atelier_theme_context}

SOURCE URL EXTRACTION:
{'YES — This section is a CLONE of the source. Match the extracted values below exactly.' if extraction_data else 'No source URL — generate from description.'}

{'Extracted Colors (override design system with these exact values):' if extraction_data and extraction_data.colors else ''}
{chr(10).join(f'  {k}: {v}' for k, v in (extraction_data.colors or {}).items()) if extraction_data and extraction_data.colors else ''}

{'Extracted Fonts:' if extraction_data and extraction_data.fonts else ''}
{chr(10).join(f'  {f["family"]}' for f in (extraction_data.fonts or [])) if extraction_data and extraction_data.fonts else ''}

{'Extracted Content for this section (verbatim from source):' if section_extraction and section_extraction.get('text') else ''}
{section_extraction['text'] if section_extraction and section_extraction.get('text') else ''}

IMPORTANT: When source URL extraction is present, match the extracted values EXACTLY.
This is a clone, not an approximation. Use the extracted colors, fonts, and content verbatim.

ATELIER DESIGN RULES (when a theme is selected):
A. Structural variety — avoid centered heroes and 3-even-column grids.
B. Color discipline — use OKLCH-paired colors; no purple-to-cyan gradients.
C. Typography hierarchy — use paired heading/body fonts at multiple scales.
D. Motion character — match easing to the theme's motion direction.
E. Imagery restraint — no Unsplash people, undersea cables, or grey placeholders.
F. Interactive polish — 8-state buttons, focus-visible rings, prefers-reduced-motion.
G. No 'Inter' monoculture — use the specified font pairing only.

RULES:
1. Use ONLY colors from the design system. No arbitrary hex values.
2. Use ONLY fonts from the typography section.
3. All spacing must use the spacing scale values.
4. Every interactive element must have focus and hover states.
5. All images must have alt text. Use semantic HTML.
6. Animations must respect prefers-reduced-motion.
7. Mobile-first responsive design.
8. All <img> tags MUST include loading="lazy" and decoding="async".
9. NEVER embed API keys, secrets, or tokens in client-side code. All API calls
   requiring credentials MUST route through a backend API handler.
10. All contact forms and registration endpoints MUST include rate limiting by
    client IP. Include a rate-limit error state (429 Too Many Requests).
11. If the page contains auth/registration, include email verification flow:
    send a verification token after signup before allowing login.

OUTPUT: {'Complete HTML section with inlined CSS. Use semantic HTML5 elements. Return a plain HTML + CSS <style> block. No JavaScript framework, no JSX, no React.' if config.framework == 'html' else 'Complete React/Next.js component with Tailwind CSS. Export as default export. Include TypeScript types.'}
{'Only the HTML content (no ``` fences).' if config.framework == 'html' else 'Export as default export. Include TypeScript types.'}
"""

    @staticmethod
    def _get_section_guidance(section: str, page_type: str) -> str:
        """Return section-type-specific guidance for the LLM prompt."""
        section_lower = section.lower()

        guidance_map = {
            "hero": (
                "HERO SECTION: Design a dramatic above-the-fold section. "
                "Bold typography, clear value proposition, strong CTA. "
                "Use the brand name prominently. Consider geometric decoration, "
                "animated entrance effects, or a 3D/particle background if libraries are available. "
                "Avoid generic centered dark hero cliches."
            ),
            "work": (
                "WORK/PORTFOLIO SECTION: Showcase projects with visual cards. "
                "Use hover effects (scale, tilt, reveal), category filters, and project metadata. "
                "Each card should have a thumbnail, title, category, and description. "
                "Consider masonry grid, horizontal scroll, or staggered layout. "
                "NOT a generic features grid — these are REAL client projects."
            ),
            "portfolio": (
                "PORTFOLIO SECTION: Showcase creative work with large visuals. "
                "Use image-first layouts, hover reveals, project detail overlays. "
                "Consider horizontal scroll, filmstrip, or bento grid layout. "
                "Each piece should feel art-directed, not templated."
            ),
            "services": (
                "SERVICES SECTION: List what the company offers. "
                "Use icon + title + description cards with clear visual hierarchy. "
                "Include pricing tiers or service packages if relevant. "
                "Consider alternating layout, numbered list, or split-panel design."
            ),
            "about": (
                "ABOUT SECTION: Tell the company story. "
                "Include mission statement, founding story, team photos, timeline, values. "
                "Use asymmetric layout, pull quotes, stats, or a narrative flow. "
                "NOT a contact form — this is the brand story section."
            ),
            "clients": (
                "CLIENTS SECTION: Display client logos with subtle animations. "
                "Use a logo grid or marquee with grayscale-to-color hover effects. "
                "Include trust indicators (testimonial snippets, case study links)."
            ),
            "contact": (
                "CONTACT SECTION: A contact form with validation and rate limiting. "
                "Include office location, email, phone, social links. "
                "Consider a map embed, 3D globe (if libraries available), or split layout."
            ),
            "team": (
                "TEAM SECTION: Showcase team members with photos, names, roles. "
                "Use hover reveals for bios, social links. Consider carousel or 3D depth layout."
            ),
            "testimonials": (
                "TESTIMONIALS SECTION: Client quotes with attribution. "
                "Use card layouts with avatar, name, role, company. "
                "Consider carousel, 3D rotation, or masonry grid with varied card sizes."
            ),
            "features": (
                "FEATURES SECTION: Product/service capabilities with icons. "
                "Use grid layout with clear icon+title+description pattern. "
                "Consider bento grid, alternating rows, or numbered stepper."
            ),
            "pricing": (
                "PRICING SECTION: Pricing tiers or packages. "
                "Use comparison cards with feature checkmarks. Highlight recommended tier. "
                "Include FAQ toggle within or below pricing cards."
            ),
            "faq": (
                "FAQ SECTION: Expandable accordion with common questions. "
                "Use clean typography, smooth expand/collapse animations. "
                "Group questions by category if many."
            ),
            "cta": (
                "CTA SECTION: Strong call-to-action before footer. "
                "Bold headline, supporting text, primary button. "
                "Consider dramatic background treatment (gradient, geometric, 3D)."
            ),
            "footer": (
                "FOOTER SECTION: Site map, social links, copyright, newsletter signup. "
                "Use multi-column layout. Include back-to-top button."
            ),
        }

        for key, guidance in guidance_map.items():
            if key in section_lower:
                return guidance

        # Generic fallback based on page_type
        if page_type in ("portfolio", "agency"):
            return (
                f"SECTION '{section}': This is a creative {page_type} site. "
                "Design with visual impact — bold layouts, experimental typography, "
                "generous whitespace. Make it feel premium and art-directed."
            )
        elif page_type == "ecommerce":
            return (
                f"SECTION '{section}': This is an ecommerce site. "
                "Focus on product presentation, clear CTAs, trust signals. "
                "Use conversion-optimized layout patterns."
            )
        elif page_type == "editorial":
            return (
                f"SECTION '{section}': This is an editorial/publishing site. "
                "Typography-forward layout with generous whitespace. "
                "Serif headings, clean reading flow, newspaper-inspired grid."
            )
        else:
            return (
                f"SECTION '{section}': Design this section to match the {page_type} "
                "page type. Use the project brief for context about brand and audience."
            )

    def _create_content_from_brief(
        self,
        output_dir: Path,
        sections: list[str],
        design_system,
        content_brief,
        config,
    ) -> None:
        """Create component files with real content from the content brief."""
        components_dir = output_dir / "components"
        components_dir.mkdir(parents=True, exist_ok=True)

        # Extract content from brief
        headlines = getattr(content_brief, "headlines", {}) or {}
        value_props = getattr(content_brief, "value_props", []) or []
        ctas = getattr(content_brief, "ctas", []) or []
        tagline = getattr(content_brief, "tagline", "") or f"A premium {getattr(config, 'page_type', 'website')} experience"
        faqs = getattr(content_brief, "faqs", []) or []
        testimonials_data = getattr(content_brief, "social_proof", []) or []

        # Default value props if brief is sparse
        if not value_props:
            value_props = ["Lightning-fast performance", "Enterprise-grade security", "Real-time analytics", "Team collaboration", "API-first architecture", "24/7 support"]
        if not ctas:
            ctas = ["Get Started Free", "Schedule Demo", "Start Building"]
        if not testimonials_data:
            brand_ref = getattr(config, "brand_name", "") or "Our platform"
            testimonials_data = [
                {"name": "Sarah Chen", "role": "CTO, TechCorp", "quote": f"{brand_ref} transformed our workflow. 3x faster deployments and zero downtime."},
                {"name": "Marcus Rivera", "role": "VP Engineering, DataSync", "quote": "The best platform we've ever used. Intuitive, powerful, reliable."},
                {"name": "Aisha Patel", "role": "Founder, LaunchPad", "quote": f"From idea to production in hours. {brand_ref} is a game-changer."},
            ]
        if not faqs:
            faqs = [
                {"q": "How does the free trial work?", "a": "Start with full access for 14 days. No credit card required. Upgrade anytime."},
                {"q": "Can I integrate with existing tools?", "a": "Yes, we offer native integrations with Slack, GitHub, Jira, and 50+ other tools via our API."},
                {"q": "Is my data secure?", "a": "We use AES-256 encryption at rest and TLS 1.3 in transit. SOC 2 Type II certified."},
                {"q": "What kind of support do you offer?", "a": "All plans include email support. Pro and Enterprise plans get dedicated Slack support with < 1hr response time."},
            ]

        for section in sections:
            name = section.replace("-", " ").replace("_", " ").title().replace(" ", "")
            headline = headlines.get(section, section.title())
            component_path = components_dir / f"{section}{'.html' if config.framework == 'html' else '.tsx'}"

            if "hero" in section.lower():
                component_path.write_text(self._build_hero_component(name, headline, tagline, ctas, design_system), encoding="utf-8")
            elif "feature" in section.lower():
                component_path.write_text(self._build_features_component(name, headline, value_props[:4], design_system), encoding="utf-8")
            elif "pric" in section.lower():
                component_path.write_text(self._build_pricing_component(name, headline, design_system), encoding="utf-8")
            elif "testimonial" in section.lower() or "social" in section.lower():
                component_path.write_text(self._build_testimonials_component(name, headline, testimonials_data, design_system), encoding="utf-8")
            elif "faq" in section.lower():
                component_path.write_text(self._build_faq_component(name, headline, faqs, design_system), encoding="utf-8")
            elif "cta" in section.lower() or "call" in section.lower():
                component_path.write_text(self._build_cta_component(name, headline, ctas, design_system), encoding="utf-8")
            elif "contact" in section.lower():
                component_path.write_text(self._build_contact_form(name, headline, design_system), encoding="utf-8")
            elif "footer" in section.lower():
                component_path.write_text(self._build_footer_component(name, design_system), encoding="utf-8")
            elif any(kw in section.lower() for kw in ("auth", "login", "register", "signup")):
                component_path.write_text(self._build_auth_component(name, headline, design_system), encoding="utf-8")
            else:
                component_path.write_text(self._build_generic_section(name, headline, value_props[:3], design_system), encoding="utf-8")

        # Write design tokens (best-effort)
        try:
            tokens_path = output_dir / "design_system.json"
            import json
            tokens_path.write_text(json.dumps(
                {"name": getattr(design_system, "name", ""), "colors": getattr(design_system.colors, "__dict__", {})},
                indent=2,
            ), encoding="utf-8")
        except Exception:
            pass

    # ── Image cache ──────────────────────────────────────────────────────

    def _get_cache(self):
        """Lazy-init DiskCache for image generation caching."""
        if not hasattr(self, "_img_cache") or self._img_cache is None:
            try:
                from ..cache import DiskCache
                self._img_cache = DiskCache()
            except Exception:
                self._img_cache = None
        return self._img_cache

    # ── Per-task image model selection (VFM-optimized) ────────────────────

    # Best VFM model + 2 fallbacks per image type (no Gemini — avoid content=None bug)
    _IMAGE_MODEL_MAP: dict[str, list[str]] = {
        "favicon": [
            "recraft/recraft-v4-vector",               # $0.08/img — SVG native
            "black-forest-labs/flux.2-klein-4b",      # $0.014/img — cheapest
            "sourceful/riverflow-v2-fast",             # $0.02/img — fast
        ],
        "apple-touch-icon": [
            "recraft/recraft-v4-vector",               # $0.08/img — SVG native
            "black-forest-labs/flux.2-klein-4b",      # $0.014/img — cheapest
            "sourceful/riverflow-v2-fast",             # $0.02/img — fast
        ],
        "logo": [
            "recraft/recraft-v4.1-pro",               # $0.25/img — good quality
            "black-forest-labs/flux.2-pro",           # $0.03/img — high quality
            "bytedance-seed/seedream-4.5",            # $0.04/img — good quality
        ],
        "hero-bg": [
            "black-forest-labs/flux.2-pro",           # $0.03/img — high quality, cheap
            "black-forest-labs/flux.2-max",           # $0.07/img — best quality
            "bytedance-seed/seedream-4.5",            # $0.04/img — good quality
        ],
        "og-image": [
            "black-forest-labs/flux.2-max",           # $0.07/img — best quality
            "recraft/recraft-v4.1-pro",               # $0.25/img — good quality
            "bytedance-seed/seedream-4.5",            # $0.04/img — cheap
        ],
        "section": [
            "black-forest-labs/flux.2-klein-4b",      # $0.014/img — cheapest, fast
            "sourceful/riverflow-v2-fast",             # $0.02/img — fast
            "recraft/recraft-v4",                      # $0.04/img — good quality
        ],
    }

    # Quality-tier overrides: draft=cheapest, premium=best (no Gemini)
    _IMAGE_QUALITY_TIERS: dict[str, dict[str, list[str]]] = {
        "draft": {
            "favicon": ["sourceful/riverflow-v2-fast", "black-forest-labs/flux.2-klein-4b"],
            "logo": ["black-forest-labs/flux.2-pro", "sourceful/riverflow-v2-fast"],
            "hero-bg": ["black-forest-labs/flux.2-klein-4b", "sourceful/riverflow-v2-fast"],
            "og-image": ["sourceful/riverflow-v2-fast", "black-forest-labs/flux.2-klein-4b"],
            "section": ["sourceful/riverflow-v2-fast", "black-forest-labs/flux.2-klein-4b"],
        },
        "premium": {
            "favicon": ["recraft/recraft-v4-pro-vector", "recraft/recraft-v4-vector"],
            "logo": ["recraft/recraft-v4.1-pro", "black-forest-labs/flux.2-pro"],
            "hero-bg": ["black-forest-labs/flux.2-max", "black-forest-labs/flux.2-pro"],
            "og-image": ["black-forest-labs/flux.2-max", "bytedance-seed/seedream-4.5"],
            "section": ["black-forest-labs/flux.2-max", "black-forest-labs/flux.2-pro"],
        },
    }

    def _select_image_model(self, image_type: str, global_model: str) -> str:
        """Select the best VFM model for an image type, falling back through the chain.

        If the user specified a global model, use it for everything.
        Otherwise, pick the best VFM model for the image type based on quality tier.
        """
        if global_model and global_model not in ("auto", "none"):
            return global_model

        # Check quality tier override first
        quality = getattr(self, "_quality_tier", "balanced")
        if quality in self._IMAGE_QUALITY_TIERS:
            tier_map = self._IMAGE_QUALITY_TIERS[quality]
            if image_type in tier_map:
                return tier_map[image_type][0]

        model_list = self._IMAGE_MODEL_MAP.get(image_type, self._IMAGE_MODEL_MAP["section"])
        return model_list[0]

    async def _generate_images(self, output_dir, config, design_system) -> None:
        """
        Generate all website images with per-type VFM model selection.

        Each image type (favicon, logo, hero-bg, etc.) gets its own best-VFM
        model with 2 fallbacks. When --image-model is specified, it overrides
        all types. Falls back to SVG placeholders on failure.
        """
        global_model = config.image_model
        if not global_model or global_model == "auto":
            global_model = "auto"  # triggers per-type selection

        # Set quality tier for per-type model selection
        self._quality_tier = getattr(config, "image_quality", "balanced")

        if global_model == "none":
            logger.info("WebsiteGenerator: image generation disabled (--image-model none)")
            try:
                (output_dir / "public" / "images").mkdir(parents=True, exist_ok=True)
                from .image_generator import generate_images as _svg_fallback
                _svg_fallback(output_dir, config, design_system)
            except PermissionError:
                logger.warning("Permission denied creating public/images — skipping SVG placeholders")
            except Exception as e:
                logger.warning("SVG placeholder generation failed: %s", e)
            return

        try:
            from ..infrastructure.image_client import ImageGenClient
            client = ImageGenClient(cache=self._get_cache())
            success = await self._generate_images_llm(output_dir, config, design_system, client)
            if success:
                return
        except Exception as e:
            logger.warning("LLM image generation failed, falling back to SVG: %s", e)

        try:
            (output_dir / "public" / "images").mkdir(parents=True, exist_ok=True)
            from .image_generator import generate_images as _svg_fallback
            _svg_fallback(output_dir, config, design_system)
            logger.info("WebsiteGenerator: generated SVG placeholder images")
        except PermissionError:
            logger.warning("Permission denied creating public/images — skipping SVG fallback")
        except Exception as e:
            logger.warning("SVG fallback failed: %s", e)

    async def _generate_images_llm(
        self,
        output_dir: Path,
        config: WebsiteConfig,
        design_system: DesignSystem,
        client: Any,
    ) -> bool:
        """Generate images via OpenRouter image model."""

        model = config.image_model
        ds = design_system
        colors = getattr(ds, "colors", ds)
        primary = getattr(colors, "primary", "#4f9eff")
        accent = getattr(colors, "accent", "#7c3aed")
        site_name = getattr(config, "brand_name", "") or getattr(config, "client_name", "Site") or "Site"
        page_type = getattr(config, "page_type", "landing")
        desc_short = (getattr(config, "description", "") or "")[:100]

        img_dir = output_dir / "public" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Description-aware image prompts
        desc_hint = f" — {desc_short}" if desc_short else ""
        images = [
            {
                "name": "hero-bg",
                "prompt": (
                    f"Premium {page_type} website hero background for {site_name}{desc_hint}, "
                    f"abstract geometric composition with {accent} and {primary} tones, "
                    "cinematic lighting, subtle grain texture, no text, no letters"
                ),
                "width": 1440,
                "height": 900,
            },
            {
                "name": "og-image",
                "prompt": (
                    f"Social sharing card for {site_name}{desc_hint}, "
                    f"professional {page_type} brand image with {primary} and {accent} color scheme, "
                    "clean premium design, no text, no letters"
                ),
                "width": 1200,
                "height": 630,
            },
            # ── Brand assets ──
            {
                "name": "favicon",
                "prompt": (
                    f"Minimalist favicon for {site_name}, {page_type} brand. "
                    f"Use {primary} as the primary color on a {accent} or transparent background. "
                    "Geometric abstract mark, NO text, NO letters, NO words. "
                    "Simple bold shape — circle, triangle, hexagon, or abstract geometric symbol. "
                    "Must be recognizable at 16x16 pixels."
                ),
                "width": 128,
                "height": 128,
            },
            {
                "name": "apple-touch-icon",
                "prompt": (
                    f"iOS home screen icon for {site_name}. "
                    f"Rounded-square icon with {primary} and {accent} gradient background. "
                    "Clean geometric symbol in the center — same shape as the favicon. "
                    "NO text, NO letters, premium app-icon quality."
                ),
                "width": 180,
                "height": 180,
            },
            {
                "name": "logo",
                "prompt": (
                    f"Brand logo / wordmark for {site_name}, a {page_type} website in the {getattr(config, 'atelier_theme', 'modern')} design style. "
                    f"Use {primary} and {accent} as primary colors. "
                    "Clean geometric logo mark + stylized brand name text. "
                    "Minimalist, modern, suitable for a website header/navbar. "
                    "Dark background compatible."
                ),
                "width": 512,
                "height": 128,
            },
        ]

        # Section-specific thumbnails based on actual sections
        sections = getattr(config, "sections", [])
        for i, section in enumerate(sections[:4]):
            section_name = section.replace("-", " ").title()
            images.append({
                "name": f"section-{section}",
                "prompt": (
                    f"{section_name} section visual for {site_name} {page_type} website{desc_hint}, "
                    f"abstract representation, {primary} and {accent} color palette, no text"
                ),
                "width": 600,
                "height": 400,
            })

        success_count = 0
        total = len(images)
        models_used = set()

        # Parallel image generation with concurrency=3
        semaphore = asyncio.Semaphore(3)

        async def _gen_one(img):
            nonlocal model
            img_type = img["name"].split("-")[0]
            img_type = img_type if img_type in self._IMAGE_MODEL_MAP else "section"
            img_model = model if model != "auto" else self._select_image_model(img_type, "auto")
            models_used.add(img_model)
            ext = ".svg" if "recraft" in img_model and "vector" in img_model.lower() else ".png"
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        client.generate(
                            prompt=img["prompt"],
                            model=img_model,
                            width=img["width"],
                            height=img["height"],
                            output_path=img_dir / f"{img['name']}{ext}",
                        ),
                        timeout=20.0,
                    )
                    return (result, img["name"], img_model)
                except asyncio.TimeoutError:
                    logger.warning("Timeout generating %s (model=%s)", img["name"], img_model)
                    return (None, img["name"], img_model)
                except Exception as e:
                    logger.warning("Error generating %s: %s", img["name"], e)
                    return (None, img["name"], img_model)

        gen_results = await asyncio.gather(*[_gen_one(img) for img in images])
        for result, name, img_model in gen_results:
            if result and result.success:
                success_count += 1
                logger.debug("Generated: %s (%s via %s)", name, result.mime_type, img_model)
            elif result:
                logger.warning("Failed to generate %s: %s", name, result.error)

        # ── Optional WebP conversion (requires Pillow) ─────────────────────
        try:
            from PIL import Image as _PIL
            import io as _io

            for img in images:
                # Determine extension same way _gen_one does
                img_type = img["name"].split("-")[0]
                img_type = img_type if img_type in self._IMAGE_MODEL_MAP else "section"
                img_model = model if model != "auto" else self._select_image_model(img_type, "auto")
                img_ext = ".svg" if "recraft" in img_model and "vector" in img_model.lower() else ".png"

                src = img_dir / f"{img['name']}{img_ext}"
                webp = img_dir / f"{img['name']}.webp"
                if src.exists() and not webp.exists():
                    try:
                        data = src.read_bytes()
                        _pil_img = _PIL.open(_io.BytesIO(data))
                        _pil_img.save(webp, "WEBP", quality=85)
                        # Remove original for large images to save space
                        if src.stat().st_size > 100000 and src.suffix != ".svg":
                            src.unlink()
                    except Exception:
                        pass
        except ImportError:
            pass  # Pillow not installed — keep PNG originals

        logger.info(
            "WebsiteGenerator: LLM images %d/%d generated (models=%s)",
            success_count,
            total,
            ", ".join(sorted(models_used)),
        )
        return success_count > 0

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
        brand = getattr(config, "brand_name", "") or "Site"
        css_lines = [
            f"/* {brand} — Generated Styles */",
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
        js_lines = [f"// {brand} — Generated Scripts", "(function() {", "  'use strict';", ""]
        js_path = output_dir / "script.js"

        # Build index.html from sections with security headers + OG meta
        site_name = getattr(config, "brand_name", "") or getattr(config, "client_name", "Site") or "Site"
        page_type = getattr(config, "page_type", "landing") or "landing"
        page_desc = getattr(config, "description", "") or f"A premium {page_type} website"
        if len(page_desc) > 160:
            page_desc = page_desc[:157] + "..."
        site_url = getattr(config, "site_url", f"https://{site_name.lower().replace(' ', '')}.io") or f"https://{site_name.lower().replace(' ', '')}.io"
        og_image = getattr(config, "og_image", "/og-image.png") or "/og-image.png"

        # page_type-aware JSON-LD
        json_ld_type, json_ld_category = _page_type_schema(page_type, site_name)
        page_title = f"{site_name} — {page_type.title()}"

        page_lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="UTF-8">',
            '  <meta http-equiv="X-UA-Compatible" content="IE=edge">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            # ── Security headers ──
            '  <meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-inline\' \'unsafe-eval\' https:; style-src \'self\' \'unsafe-inline\' https:; img-src \'self\' data: https:; font-src \'self\' https:; connect-src \'self\' https:; worker-src \'self\' blob:; frame-ancestors \'none\'; base-uri \'self\'; form-action \'self\';">',
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
            f'  <link rel="canonical" href="{site_url}">',
            # ── Open Graph ──
            f'  <meta property="og:title" content="{page_title}">',
            f'  <meta property="og:description" content="{page_desc}">',
            f'  <meta property="og:image" content="{og_image}">',
            f'  <meta property="og:url" content="{site_url}">',
            '  <meta property="og:type" content="website">',
            f'  <meta property="og:site_name" content="{site_name}">',
            '  <meta property="og:locale" content="en_US">',
            # ── Twitter Card ──
            '  <meta name="twitter:card" content="summary_large_image">',
            f'  <meta name="twitter:title" content="{page_title}">',
            f'  <meta name="twitter:description" content="{page_desc}">',
            f'  <meta name="twitter:image" content="{og_image}">',
            # ── PWA / Icons ──
            '  <link rel="icon" type="image/png" sizes="32x32" href="/images/favicon.png">',
            '  <link rel="icon" type="image/svg+xml" href="/favicon.svg">',
            '  <link rel="apple-touch-icon" sizes="180x180" href="/images/apple-touch-icon.png">',
            f'  <meta name="theme-color" content="{design_system.colors.primary}">',
            # ── Assets ──
            '  <link rel="stylesheet" href="styles.css">',
            "  <script src=\"script.js\" defer></script>",
            # ── JSON-LD Structured Data (page_type-aware) ──
            '  <script type="application/ld+json">',
            "  {",
            '    "@context": "https://schema.org",',
            f'    "@type": "{json_ld_type}",',
            f'    "name": "{site_name}",',
            '    "applicationCategory": "BusinessApplication",',
            '    "operatingSystem": "Web",',
            '    "offers": {',
            '      "@type": "Offer",',
            '      "price": "0",',
            '      "priceCurrency": "USD"',
            "    },",
            f'    "description": "{page_desc}",',
            f'    "url": "{site_url}"',
            "  }",
            "  </script>",
            "</head>",
            "<body>",
        ]

        # Read each component file and inject into the page
        if components_dir.exists():
            for section_file in sorted(components_dir.glob("*.html")) + sorted(components_dir.glob("*.tsx")):
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

    async def _verify_and_fix_build(
        self,
        output_dir: Path,
        config: WebsiteConfig,
        result: WebsiteBuildResult,
        max_attempts: int = 3,
    ) -> tuple[bool, list[str]]:
        """Install deps, run build, catch errors, and auto-fix common issues.

        Args:
            output_dir: Project directory with package.json and components/.
            config: Website configuration.
            result: Build result (mutated — cost_usd updated for fix LLM calls).
            max_attempts: Maximum fix-retry cycles.

        Returns:
            (build_passed: bool, log_lines: list[str])
        """
        import asyncio
        import re
        import shutil
        log: list[str] = []

        npm_path = shutil.which("npm")
        if not npm_path:
            log.append("npm not found on PATH — skipping build verification")
            return False, log

        # ── Step 1: npm install ─────────────────────────────────────────
        logger.info("  npm install...")
        proc = await asyncio.create_subprocess_exec(
            npm_path, "install", "--legacy-peer-deps",
            cwd=str(output_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        install_ok = proc.returncode == 0
        install_output = (stdout + stderr).decode(errors="replace")
        log.append(f"npm install: {'OK' if install_ok else 'FAILED'} (exit {proc.returncode})")

        if not install_ok and "ERESOLVE" in install_output:
            # Peer dep conflicts — retry with --force
            logger.info("  npm install --force (ERESOLVE)...")
            proc = await asyncio.create_subprocess_exec(
                npm_path, "install", "--force",
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            install_ok = proc.returncode == 0
            log.append(f"npm install --force: {'OK' if install_ok else 'FAILED'}")

        build_ok = True

        # ── Step 2: Build-verify loop ───────────────────────────────────
        # Resolve next binary on both Windows (next.cmd) and Unix (next)
        npx_path = shutil.which("next", path=str(output_dir / "node_modules" / ".bin"))
        if not npx_path:
            npx_path = shutil.which("next.cmd", path=str(output_dir / "node_modules" / ".bin"))
        if not npx_path:
            npx_path = str(output_dir / "node_modules" / ".bin" / "next")  # Unix fallback
        for attempt in range(max_attempts):
            logger.info(f"  build-verify attempt {attempt + 1}/{max_attempts}...")

            try:
                proc = await asyncio.create_subprocess_exec(
                    npx_path, "build",
                    cwd=str(output_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                log.append("next binary not found after install — cannot verify")
                return False, log

            stdout, stderr = await proc.communicate()
            build_output = (stdout + stderr).decode(errors="replace")

            if proc.returncode == 0:
                log.append(f"Build attempt {attempt + 1}: PASSED")
                build_ok = True
                return build_ok, log

            log.append(f"Build attempt {attempt + 1}: FAILED (exit {proc.returncode})")

            # ── Parse errors for auto-fix ───────────────────────
            missing_modules = re.findall(
                r"Module not found: Can't resolve '([^']+)'",
                build_output,
            )
            missing_imports = set(missing_modules)

            if missing_imports:
                for mod in sorted(missing_imports):
                    logger.info(f"  ↻ auto-installing missing dep: {mod}")
                    proc = await asyncio.create_subprocess_exec(
                        npm_path, "install", mod, "--legacy-peer-deps",
                        cwd=str(output_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()
                    log.append(f"npm install {mod}: exit {proc.returncode}")
                continue  # retry build with new deps

            # Syntax errors — extract file + line, feed to LLM for fix
            syntax_errors = re.findall(
                r"\./(components/[a-zA-Z0-9_./-]+)\s*\n\s*.*?(\d+):(\d+):\s*\n\s*(.*?)(?:\n\n|\n\s*\n)",
                build_output,
                re.DOTALL,
            )
            if syntax_errors and self._engine:
                for file_path, line_no, col_no, details in syntax_errors[:3]:
                    # Validate path stays under output_dir (path-traversal prevention)
                    resolved = (output_dir / file_path).resolve()
                    if not str(resolved).startswith(str(output_dir.resolve())):
                        log.append(f"path traversal blocked: {file_path}")
                        continue
                    comp_name = Path(file_path).stem
                    logger.info(f"  ↻ auto-fixing syntax error in {comp_name} (line {line_no})...")
                    fixed = await self._llm_fix_syntax(
                        file_path=str(resolved),
                        error_line=int(line_no),
                        error_details=details.strip()[:500],
                        component_name=comp_name,
                        config=config,
                    )
                    if fixed:
                        log.append(f"LLM fix applied to {comp_name}")
                    else:
                        log.append(f"LLM fix failed for {comp_name}")
                continue  # retry build with fixed files

            # No actionable errors found — give up
            log.append(f"No auto-fixable errors detected. Last ~300 chars: {build_output[-300:]}")
            build_ok = False
            return build_ok, log

        return build_ok, log

    async def _llm_fix_syntax(
        self,
        file_path: str,
        error_line: int,
        error_details: str,
        component_name: str,
        config: WebsiteConfig,
    ) -> bool:
        """Feed a syntax error back to the LLM for a targeted fix."""
        try:
            source = Path(file_path).read_text(encoding="utf-8")

            prompt = (
                f"Fix the following build error in a {config.framework} + {config.styling} component.\n\n"
                f"ERROR at line {error_line}:\n{error_details}\n\n"
                f"CURRENT FILE ({component_name}.tsx):\n```tsx\n{source}\n```\n\n"
                "Return ONLY the complete fixed file content (no markdown fences, no explanation). "
                "Fix the specific error — change as little as possible."
            )

            from ..models import Task, TaskType

            task = Task(
                id=f"fix_syntax_{component_name}",
                type=TaskType.CODE_GEN,
                prompt=prompt,
                max_output_tokens=16384,
                acceptance_threshold=0.7,
                max_iterations=1,
            )
            component_result = await self._engine._execute_task(task)
            if component_result and component_result.output:
                cleaned, _ = self._sanitize_output(component_result.output, component_name)
                if len(cleaned) > 200:
                    Path(file_path).write_text(cleaned, encoding="utf-8")
                    logger.info(f"  ✓ {component_name}: LLM fix applied ({len(cleaned)} chars)")
                    return True
        except Exception as e:
            logger.warning(f"LLM syntax fix failed for {component_name}: {e}")
        return False

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
            "          { key: 'Content-Security-Policy', value: \"default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' https:; worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'\" },\n"
            "        ],\n"
            "      },\n"
            "    ];\n"
            "  },\n"
            "};\n"
            "module.exports = nextConfig;\n",
            encoding="utf-8",
        )

        # Build data-driven identifiers
        brand = getattr(config, "brand_name", "") or "site"
        pkg_name = brand.lower().replace(" ", "-")
        safe_brand = brand or "Site"

        # Write package.json with all deps (including custom 3D/animation deps)
        import json as _json

        pkg = {
            "name": pkg_name,
            "version": "0.1.0",
            "private": True,
            "scripts": {"dev": "next dev", "build": "next build", "start": "next start"},
            "dependencies": {
                "next": "^15.0.0",
                "react": "^19.0.0",
                "react-dom": "^19.0.0",
            },
            "devDependencies": {
                "@types/node": "^20.0.0",
                "@types/react": "^19.0.0",
                "@types/react-dom": "^19.0.0",
                "typescript": "^5.0.0",
            },
        }

        # Merge custom dependencies (format: "pkg", "pkg@version", "@scope/pkg", "@scope/pkg@version")
        custom_deps = getattr(config, "dependencies", []) or []
        for dep in custom_deps:
            if dep in ("react", "react-dom", "next"):
                continue  # already in base
            if dep.startswith("@") and dep.count("/") >= 1:
                # Scoped package: @scope/name or @scope/name@version
                name, at, ver = dep.rpartition("@")
                if dep.count("@") == 1:
                    # Just @scope/name, no version
                    name = dep
                    ver = "latest"
                else:
                    # @scope/name@version — rpartition gives name=@scope/name, ver
                    pass
            elif "@" in dep:
                # Unscoped package with version: pkg@version
                name, ver = dep.split("@", 1)
            else:
                name = dep
                ver = "latest"
            pkg["dependencies"][name] = ver

        (output_dir / "package.json").write_text(
            _json.dumps(pkg, indent=2) + "\n",
            encoding="utf-8",
        )

        # Check for 3D deps and update CSP if needed
        has_3d = any(
            kw in str(custom_deps).lower()
            for kw in ("three", "react-three", "fiber", "drei", "cannon", "babylon")
        )

        # No postcss or tailwind config — using CDN Tailwind in layout.tsx

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

        # Design tokens live in globals.css :root — no build-time Tailwind config needed

        # Use data-driven brand identifiers
        site_name = safe_brand
        site_url = getattr(config, "site_url", f"https://{pkg_name}.io") or f"https://{pkg_name}.io"
        
        # Data-driven metadata for layout.tsx
        page_type_title = getattr(config, "page_type", "Website").title()
        desc = getattr(config, "description", "") or ""
        meta_desc = desc[:150] if desc else f"A premium {getattr(config, 'page_type', 'website')} website"

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
            f"  metadataBase: new URL('{site_url}'),\n"
            "  title: {\n"
            f"    default: '{safe_brand} — {page_type_title}',\n"
            f"    template: '%s | {safe_brand}',\n"
            "  },\n"
            f"  description: '{meta_desc}',\n"
            f"  keywords: ['{getattr(config, 'page_type', 'web')}', '{pkg_name}'],\n"
            "  robots: { index: true, follow: true },\n"
            "  openGraph: {\n"
            "    type: 'website',\n"
            "    locale: 'en_US',\n"
            "    url: '/',\n"
            f"    siteName: '{safe_brand}',\n"
            f"    title: '{safe_brand} — {page_type_title}',\n"
            f"    description: '{meta_desc}',\n"
            f"    images: [{{ url: '/og-image.png', width: 1200, height: 630, alt: '{safe_brand}' }}],\n"
            "  },\n"
            "  twitter: {\n"
            "    card: 'summary_large_image',\n"
            f"    title: '{safe_brand} — {page_type_title}',\n"
            f"    description: '{meta_desc}',\n"
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
            f"    name: '{safe_brand}',\n"
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
            "        {/* Favicon + PWA icons */}\n"
            '        <link rel="icon" type="image/png" sizes="32x32" href="/images/favicon.png" />\n'
            '        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />\n'
            '        <link rel="apple-touch-icon" sizes="180x180" href="/images/apple-touch-icon.png" />\n'
            "      </head>\n"
            "      <body>{children}</body>\n"
            "    </html>\n"
            "  );\n"
            "}\n",
            encoding="utf-8",
        )

        # Build page.tsx — imports default exports from components
        # Component names: each section file exports a default component named {Section}Section
        section_imports = []
        section_jsx = []
        for s in sections:
            component_name = s.title().replace('-', '').replace('_', '').replace(' ', '')
            section_imports.append(f"import {component_name} from '@/components/{s}';")
            section_jsx.append(f"      <{component_name} />")

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
            f"<h1>{safe_brand}</h1>\n"
            f"<p>{getattr(config, 'page_type', 'Website').title()}</p>\n"
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
                f"      <h1 className=\"text-5xl md:text-7xl font-bold tracking-tight mb-6 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent\">\n"
                f"        {safe_brand}\n"
                "      </h1>\n"
                f"      <p className=\"text-xl md:text-2xl text-gray-300 max-w-2xl mb-8\">\n"
                f"        A premium {getattr(config, 'page_type', 'website')} experience.\n"
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
            f"    sitemap: '{site_url}/sitemap.xml',\n"
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
            f"      url: '{site_url}',\n"
            "      lastModified: new Date(),\n"
            "      changeFrequency: 'weekly' as const,\n"
            "      priority: 1,\n"
            "    },\n"
            "  ];\n"
            "}\n",
            encoding="utf-8",
        )

        # Write security.txt (well-known) — data-driven
        security_email = f"security@{pkg_name}.io"
        public_dir = output_dir / "public"
        well_known = public_dir / ".well-known"
        well_known.mkdir(parents=True, exist_ok=True)
        (well_known / "security.txt").write_text(
            f"Contact: mailto:{security_email}\n"
            "Expires: 2027-12-31T23:59:59Z\n"
            "Preferred-Languages: en\n"
            f"Canonical: {site_url}/.well-known/security.txt\n"
            f"Policy: {site_url}/security\n",
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
            f"  Access-Control-Allow-Origin: {site_url}\n",
            encoding="utf-8",
        )

        # Write security page
        security_page = public_dir / "security.html"
        security_page.write_text(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>Security Policy — {safe_brand}</title>\n"
            "<style>body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#333}</style>\n"
            "</head>\n<body>\n"
            "<h1>Security Policy</h1>\n"
            "<h2>Reporting a Vulnerability</h2>\n"
            f"<p>Email <a href=\"mailto:{security_email}\">{security_email}</a>. "
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

    def _build_hero_component(self, name, headline, tagline, ctas, ds):
        cta1 = (ctas[0] if ctas else "Get Started")
        cta2 = (ctas[1] if len(ctas) > 1 else "Learn More")
        return (
            f'"use client";\n\n'
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section className="relative flex flex-col items-center justify-center min-h-[90vh] px-6 text-center">\n'
            f'      <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 to-transparent pointer-events-none" />\n'
            f'      <h1 className="relative text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">\n'
            f"        {headline}\n"
            f"      </h1>\n"
            f'      <p className="relative text-xl md:text-2xl text-gray-300 max-w-3xl mb-10 leading-relaxed">\n'
            f"        {tagline}\n"
            f"      </p>\n"
            f'      <div className="relative flex flex-wrap gap-4 justify-center">\n'
            f'        <a href="#features" className="bg-indigo-500 hover:bg-indigo-600 px-8 py-4 rounded-xl font-semibold text-white text-lg transition-all shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40">\n'
            f"          {cta1}\n"
            f"        </a>\n"
            f'        <a href="#pricing" className="border border-gray-500 hover:border-gray-300 px-8 py-4 rounded-xl font-semibold text-gray-200 text-lg transition-all">\n'
            f"          {cta2}\n"
            f"        </a>\n"
            f"      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_features_component(self, name, headline, items, ds):
        cards = []
        icons = ["⚡", "🔒", "📊", "🤝"]
        for i, item in enumerate(items[:4]):
            icon = icons[i % len(icons)]
            cards.append(
                f'        <div className="bg-gray-800/40 hover:bg-gray-800/60 p-8 rounded-2xl transition-all border border-gray-700/50 hover:border-gray-600">\n'
                f'          <div className="text-3xl mb-4">{icon}</div>\n'
                f'          <h3 className="text-xl font-semibold mb-3">{item}</h3>\n'
                f'          <p className="text-gray-400 leading-relaxed">\n'
                f"            {item} for modern teams. Designed for scale, built for speed.\n"
                f"          </p>\n"
                f"        </div>"
            )
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section id="features" className="py-24 px-6 max-w-6xl mx-auto">\n'
            f'      <h2 className="text-3xl md:text-5xl font-bold text-center mb-6">{headline}</h2>\n'
            f'      <p className="text-gray-400 text-center max-w-2xl mx-auto mb-16 text-lg">\n'
            f"        Everything you need to build, deploy, and scale your SaaS application.\n"
            f"      </p>\n"
            f'      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">\n'
            + "\n".join(cards) +
            f"\n      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_pricing_component(self, name, headline, ds):
        plans = [
            {"name": "Starter", "price": "$9", "desc": "For small teams getting started", "features": ["Up to 5 users", "10GB storage", "Email support", "Basic analytics"]},
            {"name": "Pro", "price": "$29", "desc": "For growing businesses", "features": ["Up to 50 users", "100GB storage", "Priority support", "Advanced analytics", "Custom integrations"]},
            {"name": "Enterprise", "price": "$99", "desc": "For large organizations", "features": ["Unlimited users", "Unlimited storage", "Dedicated support", "SSO & SAML", "Custom SLA", "On-premise option"]},
        ]
        plan_cards = []
        for plan in plans:
            feats = "\n".join(f'              <li className="flex items-center gap-2"><span className="text-green-400">✓</span> {f}</li>' for f in plan["features"])
            plan_cards.append(
                f'        <div className="bg-gray-800/40 border border-gray-700/50 rounded-2xl p-8 flex flex-col">\n'
                f'          <h3 className="text-xl font-semibold mb-2">{plan["name"]}</h3>\n'
                f'          <div className="text-4xl font-bold mb-1">{plan["price"]}<span className="text-lg text-gray-400 font-normal">/mo</span></div>\n'
                f'          <p className="text-gray-400 mb-6">{plan["desc"]}</p>\n'
                f'          <ul className="space-y-2 mb-8 flex-1 text-sm">\n{feats}\n          </ul>\n'
                f'          <a href="#" className="bg-indigo-500 hover:bg-indigo-600 text-center py-3 rounded-xl font-semibold text-white transition-all mt-auto">Get Started</a>\n'
                f"        </div>"
            )
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section id="pricing" className="py-24 px-6 max-w-6xl mx-auto">\n'
            f'      <h2 className="text-3xl md:text-5xl font-bold text-center mb-6">{headline}</h2>\n'
            f'      <p className="text-gray-400 text-center max-w-2xl mx-auto mb-16 text-lg">Simple, transparent pricing. No hidden fees.</p>\n'
            f'      <div className="grid md:grid-cols-3 gap-6">\n'
            + "\n".join(plan_cards) +
            f"\n      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_testimonials_component(self, name, headline, items, ds):
        cards = []
        for t in items[:6]:
            person_name = t.get("name", "User") if isinstance(t, dict) else str(t)
            role = t.get("role", "") if isinstance(t, dict) else ""
            quote = t.get("quote", str(t)) if isinstance(t, dict) else str(t)
            cards.append(
                f'        <div className="bg-gray-800/40 border border-gray-700/50 rounded-2xl p-8">\n'
                f'          <p className="text-gray-300 italic mb-6 leading-relaxed">&ldquo;{quote}&rdquo;</p>\n'
                f'          <div className="flex items-center gap-3">\n'
                f'            <div className="w-10 h-10 rounded-full bg-indigo-500/30 flex items-center justify-center font-bold text-indigo-300">{person_name[0]}</div>\n'
                f'            <div><div className="font-semibold text-sm">{person_name}</div><div className="text-gray-500 text-xs">{role}</div></div>\n'
                f"          </div>\n"
                f"        </div>"
            )
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section id="testimonials" className="py-24 px-6 max-w-6xl mx-auto">\n'
            f'      <h2 className="text-3xl md:text-5xl font-bold text-center mb-16">{headline}</h2>\n'
            f'      <div className="grid md:grid-cols-3 gap-6">\n'
            + "\n".join(cards) +
            f"\n      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_faq_component(self, name, headline, items, ds):
        items_html = []
        for i, faq in enumerate(items[:8]):
            q = faq.get("q", str(faq)) if isinstance(faq, dict) else str(faq)
            a = faq.get("a", "") if isinstance(faq, dict) else ""
            items_html.append(
                f'        <details className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-6 group cursor-pointer">\n'
                f'          <summary className="text-lg font-semibold list-none flex justify-between items-center">\n'
                f'            {q}\n'
                f'            <span className="text-gray-500 group-open:rotate-180 transition-transform text-xl ml-4">▼</span>\n'
                f"          </summary>\n"
                f'          <p className="mt-4 text-gray-400 leading-relaxed">{a}</p>\n'
                f"        </details>"
            )
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section id="faq" className="py-24 px-6 max-w-3xl mx-auto">\n'
            f'      <h2 className="text-3xl md:text-5xl font-bold text-center mb-16">{headline}</h2>\n'
            f'      <div className="space-y-4">\n'
            + "\n".join(items_html) +
            f"\n      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_cta_component(self, name, headline, ctas, ds):
        cta_text = ctas[0] if ctas else "Get Started Free"
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <section className="py-24 px-6 text-center">\n'
            f'      <div className="max-w-3xl mx-auto bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border border-indigo-500/30 rounded-3xl p-16">\n'
            f'        <h2 className="text-3xl md:text-5xl font-bold mb-6">{headline}</h2>\n'
            f'        <p className="text-gray-300 text-lg mb-10 max-w-xl mx-auto">Join thousands of teams already using {safe_brand}. Start free, upgrade when you\'re ready.</p>\n'
            f'        <a href="#" className="bg-indigo-500 hover:bg-indigo-600 px-10 py-4 rounded-xl font-semibold text-white text-lg transition-all shadow-lg shadow-indigo-500/25">{cta_text}</a>\n'
            f"      </div>\n"
            f"    </section>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_footer_component(self, name, ds):
        return (
            f"export default function {name}() {{\n"
            f"  return (\n"
            f'    <footer className="border-t border-gray-800 py-16 px-6">\n'
            f'      <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">\n'
            f'        <div>\n'
            f'          <h4 className="font-bold text-lg mb-4">{safe_brand}</h4>\n'
            f'          <p className="text-gray-500 text-sm">Intelligent SaaS platform for modern teams.</p>\n'
            f"        </div>\n"
            f'        <div><h4 className="font-semibold mb-3">Product</h4><ul className="space-y-2 text-gray-400 text-sm"><li><a href="#features">Features</a></li><li><a href="#pricing">Pricing</a></li><li><a href="#">Integrations</a></li><li><a href="#">Changelog</a></li></ul></div>\n'
            f'        <div><h4 className="font-semibold mb-3">Company</h4><ul className="space-y-2 text-gray-400 text-sm"><li><a href="#">About</a></li><li><a href="#">Blog</a></li><li><a href="#">Careers</a></li><li><a href="/security">Security</a></li></ul></div>\n'
            f'        <div><h4 className="font-semibold mb-3">Legal</h4><ul className="space-y-2 text-gray-400 text-sm"><li><a href="#">Privacy</a></li><li><a href="#">Terms</a></li><li><a href="/security">Security</a></li></ul></div>\n'
            f"      </div>\n"
            f'      <div className="max-w-6xl mx-auto mt-12 pt-8 border-t border-gray-800 text-center text-gray-600 text-sm">\n'
            f"        &copy; 2026 {safe_brand}. All rights reserved.\n"
            f"      </div>\n"
            f"    </footer>\n"
            f"  );\n"
            f"}}\n"
        )

    def _build_generic_section(self, name, headline, items, ds):
        return self._build_features_component(name, headline, items, ds)

    def _build_contact_form(self, name, headline, ds) -> str:
        """Generate a contact form with rate-limiting and 429 error handling."""
        primary = getattr(getattr(ds, "colors", ds), "primary", "#4f9eff")
        surface_alt = getattr(getattr(ds, "colors", ds), "surface_alt", "#f5f5f5")
        border = getattr(getattr(ds, "colors", ds), "border", "#e5e5e5")
        font_body = (
            getattr(getattr(getattr(ds, "typography", ds), "font_body", None), "value", None)
            or getattr(getattr(ds, "typography", ds), "font_sans", "Inter")
        )
        from .templates.contact_form import CONTACT_FORM_TEMPLATE

        return CONTACT_FORM_TEMPLATE.format(
            headline=headline,
            primary=primary,
            surface_alt=surface_alt,
            border=border,
            font_body=font_body,
        )

    def _build_auth_component(self, name, headline, ds) -> str:
        """Generate auth/register page with email verification flow."""
        primary = getattr(getattr(ds, "colors", ds), "primary", "#4f9eff")
        surface_alt = getattr(getattr(ds, "colors", ds), "surface_alt", "#f5f5f5")
        border = getattr(getattr(ds, "colors", ds), "border", "#e5e5e5")
        font_body = (
            getattr(getattr(getattr(ds, "typography", ds), "font_body", None), "value", None)
            or getattr(getattr(ds, "typography", ds), "font_sans", "Inter")
        )
        from .templates.auth_page import AUTH_TEMPLATE

        is_register = any(kw in name.lower() for kw in ("register", "signup"))

        if is_register:
            page_title = "Create your account"
            page_subtitle = "Enter your details to get started."
            button_text = "Sign Up"
            endpoint = "/api/auth/register"
            success_state = "setState('check-email')"
            success_message = "Check your email to complete registration."
            verify_note = '<p className="text-xs text-center opacity-60">We will send a verification email to confirm your address.</p>'
            switch_message = 'Already have an account? <a href="/login" className="underline" style={{ color: primary }}>Log in</a>'
        else:
            page_title = "Welcome back"
            page_subtitle = "Log in to your account."
            button_text = "Log In"
            endpoint = "/api/auth/login"
            success_state = "setState('success')"
            success_message = "Login successful! Redirecting..."
            verify_note = ""
            switch_message = "Don't have an account? <a href=\"/register\" className=\"underline\" style={{ color: primary }}>Sign up</a>"

        return AUTH_TEMPLATE.format(
            component_name=name,
            page_type="Registration" if is_register else "Login",
            headline=headline,
            primary=primary,
            surface_alt=surface_alt,
            border=border,
            font_body=font_body,
            page_title=page_title,
            page_subtitle=page_subtitle,
            button_text=button_text,
            endpoint=endpoint,
            success_state=success_state,
            success_message=success_message,
            verify_note=verify_note,
            switch_message=switch_message,
        )


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
