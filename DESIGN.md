Framework: Design-System-Driven Generation (DSDG)

Φιλοσοφία

Phase 0: Design System Definition (νέο — δεν υπάρχει στο ANF)
yaml# design_system.yaml — Ο πυρήνας που λείπει εντελώς
brand:
  name: "ClientName"
  industry: "medical_spa"
  tone: "premium_minimal"  # premium_minimal | bold_creative | corporate_clean | playful_modern

typography:
  font_heading: "Plus Jakarta Sans"
  font_body: "Inter"
  scale: # Major Third (1.25)
    xs: "0.75rem"
    sm: "0.875rem"
    base: "1rem"
    lg: "1.25rem"
    xl: "1.563rem"
    2xl: "1.953rem"
    3xl: "2.441rem"
    4xl: "3.052rem"

colors:
  primary: "#1a1a2e"
  accent: "#e94560"
  surface: "#fafafa"
  surface_alt: "#f0f0f5"
  text_primary: "#1a1a2e"
  text_secondary: "#6b7280"
  border: "#e5e7eb"
  success: "#10b981"
  error: "#ef4444"

spacing:
  unit: "0.25rem"  # 4px base
  scale: [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

layout:
  max_width: "1280px"
  columns: 12
  gutter: "2rem"
  breakpoints:
    sm: "640px"
    md: "768px"
    lg: "1024px"
    xl: "1280px"

components:
  border_radius:
    sm: "0.375rem"
    md: "0.75rem"
    lg: "1rem"
    full: "9999px"
  shadow:
    sm: "0 1px 2px rgba(0,0,0,0.05)"
    md: "0 4px 6px rgba(0,0,0,0.07)"
    lg: "0 10px 15px rgba(0,0,0,0.1)"
  animation:
    duration: "300ms"
    easing: "cubic-bezier(0.4, 0, 0.2, 1)"

accessibility:
  min_contrast_ratio: 4.5
  focus_ring: true
  reduced_motion_support: true
  semantic_html: 
  

Τώρα κάθε component που γεννιέται πρέπει να ακολουθεί αυτό. Δεν χρειάζεται "normalize" βήμα γιατί τα constraints υπάρχουν από πριν.
Phase 1: Component Selection (βελτιωμένο Assemble)
Αντί manual browsing, ο orchestrator κάνει automated selection:
pythonclass ComponentRegistry:
    """Multi-source component library with quality scoring."""
    
    SOURCES = {
        "twentyfirst": "https://twentyfirst.dev",
        "shadcn": "https://ui.shadcn.com",
        "aceternity": "https://ui.aceternity.com",
        "magic_ui": "https://magicui.design",
        "curated_internal": "./components/curated/",
    }
    
    async def select_components(
        self,
        page_type: str,            # "landing" | "saas" | "portfolio" | "ecommerce"
        design_system: DesignSystem,
        sections_needed: list[str], # ["hero", "features", "pricing", "testimonials", "faq", "cta"]
    ) -> list[ComponentSpec]:
        """Select best-matching components from multiple sources."""
        selected = []
        
        for section in sections_needed:
            candidates = await self._fetch_candidates(section, self.SOURCES)
            
            # Score each candidate against design system compatibility
            scored = []
            for candidate in candidates:
                score = self._compatibility_score(candidate, design_system)
                scored.append((candidate, score))
            
            # Pick highest-scoring
            best = max(scored, key=lambda x: x[1])
            selected.append(best[0])
        
        return selected
    
    def _compatibility_score(
        self, component: ComponentSpec, ds: DesignSystem
    ) -> float:
        """Score how well a component matches the design system."""
        score = 0.0
        
        # Color proximity
        if self._color_distance(component.primary_color, ds.colors.primary) < 0.3:
            score += 0.25
        
        # Layout compatibility
        if component.layout_type in ("grid", "flex") and ds.layout.columns == 12:
            score += 0.2
        
        # Animation style match
        if component.animation_style == ds.brand.tone:
            score += 0.2
        
        # Accessibility compliance
        if component.has_aria_labels and component.keyboard_navigable:
            score += 0.2
        
        # Responsive readiness
        if component.responsive:
            score += 0.15
        
        return 
        

Phase 2: Constrained Generation (αντικαθιστά Assemble + Normalize)
Αντί "generate then normalize", generate within constraints from the start:
pythonGENERATION_PROMPT = """
You are building a premium website section.

DESIGN SYSTEM (MANDATORY — every value must come from this system):
{design_system_yaml}

COMPONENT REFERENCE (follow this structure and animation style):
{component_prompt}

SECTION: {section_type}
CONTENT BRIEF: {content_brief}

RULES:
1. Use ONLY colors from the design system. No #purple, no #blue unless specified.
2. Use ONLY fonts from the typography section. No system fonts, no fallbacks visible.
3. All spacing must use the spacing scale values. No magic numbers.
4. Every interactive element must have: focus state, hover state, aria-label.
5. All images must have alt text. All sections must have semantic HTML (section, nav, main, footer).
6. Animations must respect prefers-reduced-motion.
7. Contrast ratio minimum 4.5:1 for all text on backgrounds.
8. Mobile-first responsive: design for 375px first, then scale up.

OUTPUT: Complete React/Next.js component with Tailwind CSS using ONLY the design tokens above.
Do NOT use any Tailwind color that isn't mapped to the design system.
"""
Δεν χρειάζεται normalize step γιατί κάθε component γεννιέται ήδη unified.


Phase 3: Research-Driven Content (βελτιωμένο Fill)
Ενσωματωμένο στο pipeline, όχι ξεχωριστό session:
pythonclass ContentResearcher:
    """Integrated industry research → content generation."""
    
    async def generate_content_brief(
        self,
        client_info: ClientInfo,
        nexus_search: NexusSearchOrchestrator,  # Χρησιμοποιεί Nexus Search!
    ) -> ContentBrief:
        # 1. Industry research via Nexus Search
        competitors = await nexus_search.search(
            f"{client_info.industry} top websites {client_info.location}",
            sources=[SearchSource.WEB, SearchSource.TECH],
        )
        
        reviews = await nexus_search.search(
            f"{client_info.industry} customer complaints reviews",
            sources=[SearchSource.WEB],
        )
        
        trends = await nexus_search.research(
            f"{client_info.industry} design trends 2026",
        )
        
        # 2. LLM synthesizes into content brief
        brief = await self._synthesize_brief(
            client_info, competitors, reviews, trends
        )
        
        return brief  # Headlines, CTAs, FAQs, testimonial angles, pain points


Phase 4: Quality Validation (νέο — δεν υπάρχει στο ANF)
pythonclass WebsiteQualityValidator:
    """Validate generated website against quality standards."""
    
    async def validate(self, output_dir: Path) -> QualityReport:
        checks = []
        
        # 1. Accessibility (WCAG 2.1 AA)
        checks.append(await self._run_axe_audit(output_dir))
        
        # 2. Performance (Lighthouse)
        checks.append(await self._run_lighthouse(output_dir))
        
        # 3. Design System Compliance
        checks.append(await self._check_design_tokens(output_dir))
        
        # 4. Responsive (320px, 768px, 1024px, 1440px)
        checks.append(await self._check_responsive(output_dir))
        
        # 5. SEO basics
        checks.append(await self._check_seo(output_dir))
        
        # 6. Visual Regression (screenshot comparison)
        checks.append(await self._visual_regression(output_dir))
        
        # 7. Content Quality (no Lorem ipsum, no generic)
        checks.append(await self._content_quality_check(output_dir))
        
        return QualityReport(
            checks=checks,
            passed=all(c.passed for c in checks),
            score=mean(c.score for c in checks),
        )

Integration στον AI Orchestrator
Αυτό μετατρέπεται σε νέο project type στον orchestrator. Σήμερα ο orchestrator δέχεται --project "Build a FastAPI REST API". Η πρόταση:
bash# Νέα εντολή: website generation
python -m orchestrator \
  --type website \
  --design-system design_system.yaml \
  --client-info client.yaml \
  --page-type landing \
  --sections "hero,features,pricing,testimonials,faq,cta,footer" \
  --budget 3.0

# Ή SaaS app UI
python -m orchestrator \
  --type webapp \
  --design-system design_system.yaml \
  --pages "dashboard,settings,billing,onboarding" \
  --framework "next.js" \
  --budget 5.0
Αρχιτεκτονική Integration
pythonclass WebsiteProjectType:
    """New project type: Design-System-Driven Website Generation."""
    
    async def execute(
        self,
        design_system: Path,
        client_info: Path,
        sections: list[str],
        page_type: str,
    ) -> ProjectState:
        # 1. Load design system
        ds = DesignSystem.from_yaml(design_system)
        client = ClientInfo.from_yaml(client_info)
        
        # 2. Research content (uses existing Nexus Search)
        content_brief = await self.researcher.generate_content_brief(
            client, self.nexus_search
        )
        
        # 3. Select components (multi-source registry)
        components = await self.registry.select_components(
            page_type=page_type,
            design_system=ds,
            sections_needed=sections,
        )
        
        # 4. Generate each section (uses existing parallel execution)
        tasks = []
        for i, (section, component) in enumerate(zip(sections, components)):
            tasks.append(Task(
                id=f"section_{i:03d}_{section}",
                type=TaskType.CODE_GEN,
                prompt=self._build_section_prompt(
                    section, component, ds, content_brief
                ),
                dependencies=[f"section_{i-1:03d}_{sections[i-1]}"] if i > 0 else [],
            ))
        
        # 5. Execute through orchestrator engine
        # (gets parallel execution, self-healing, model routing, etc. for free)
        state = await self.engine.execute_all(tasks)
        
        # 6. Assemble final page
        assembled = await self.assembler.combine_sections(
            state.outputs, ds
        )
        
        # 7. Validate quality
        report = await self.validator.validate(assembled.output_dir)
        
        if not report.passed:
            # Auto-fix failing checks (uses existing self-healing loop)
            assembled = await self._repair_quality_issues(
                assembled, report.failures, ds
            )
        
        return state
    
    def _build_section_prompt(
        self,
        section: str,
        component: ComponentSpec,
        ds: DesignSystem,
        brief: ContentBrief,
    ) -> str:
        return f"""
        DESIGN SYSTEM:
        {ds.to_yaml()}
        
        COMPONENT REFERENCE:
        {component.prompt}
        
        SECTION: {section}
        
        CONTENT FOR THIS SECTION:
        {brief.get_section_content(section)}
        
        Generate a React/Next.js component with Tailwind CSS.
        Use ONLY design system tokens. Mobile-first responsive.
        Include: semantic HTML, aria-labels, focus states, reduced-motion support.
        """
Design System Presets
Για γρήγορο start χωρίς custom design system:
pythonDESIGN_PRESETS = {
    "premium_minimal": {
        "fonts": ("Plus Jakarta Sans", "Inter"),
        "primary": "#0f172a",
        "accent": "#6366f1",
        "style": "Clean lines, generous whitespace, subtle shadows",
    },
    "bold_startup": {
        "fonts": ("Cabinet Grotesk", "Satoshi"),
        "primary": "#18181b",
        "accent": "#f97316",
        "style": "High contrast, bold typography, dynamic animations",
    },
    "luxury_brand": {
        "fonts": ("Playfair Display", "Lora"),
        "primary": "#1c1917",
        "accent": "#d4a853",
        "style": "Serif elegance, dark backgrounds, refined spacing",
    },
    "medical_clean": {
        "fonts": ("DM Sans", "Inter"),
        "primary": "#0c4a6e",
        "accent": "#06b6d4",
        "style": "Trust-building, clean, accessible, calming colors",
    },
    "saas_modern": {
        "fonts": ("Geist", "Geist Mono"),
        "primary": "#020617",
        "accent": "#8b5cf6",
        "style": "Dark/light mode, glassmorphism, gradient accents",
    },
}
bash# Quick start with preset
python -m orchestrator \
  --type website \
  --preset luxury_brand \
  --industry "real_estate" \
  --sections "hero,listings,testimonials,about,contact" \
  --budget 2.0

Τελικό Output Example
bash$ python -m orchestrator --type website --preset saas_modern --industry "project_management" --budget 2.0

[orchestrator] Design system: saas_modern
[orchestrator] Researching: project management industry...
[nexus] Found 12 competitor sites, 45 customer reviews, 8 trend articles
[orchestrator] Content brief: 6 sections, 23 headlines, 12 CTAs, 8 FAQs
[orchestrator] Component selection: hero(aceternity), features(shadcn), pricing(twentyfirst)...
[orchestrator] Generating 6 sections (parallel, 3 workers)...
  ✅ hero_section: 0.92 quality, $0.08
  ✅ features_section: 0.89 quality, $0.06
  ✅ pricing_section: 0.91 quality, $0.07
  ✅ testimonials_section: 0.88 quality, $0.05
  ✅ faq_section: 0.94 quality, $0.04
  ✅ footer_section: 0.90 quality, $0.03
[orchestrator] Assembling page...
[validator] Lighthouse: 96/100 | WCAG: AA pass | Responsive: 4/4 | SEO: 95/100
[orchestrator] Output: ./results/pm-landing/
  📁 components/ (6 React components)
  📁 styles/ (Tailwind config with design tokens)
  📄 page.tsx (assembled page)
  📄 design_system.json (exported tokens)
  📄 quality_report.json
  Total: $0.33 | Time: 2m 14s | Quality: 0.91 avg