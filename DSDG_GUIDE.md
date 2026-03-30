# DSDG: Design-System-Driven Generation

**Version:** 1.0.0 | **Status:** Implemented | **Author:** Georgios-Chrysovalantis Chatzivantsidis

## Overview

DSDG (Design-System-Driven Generation) is a comprehensive framework for generating premium websites and web applications using AI. Instead of generating code and then normalizing it, DSDG generates code **within constraints from the start**, ensuring consistent design system compliance.

## Philosophy

> "Don't generate then normalize. Generate within constraints from the start."

Traditional AI code generation follows this pattern:
1. Generate code → 2. Normalize styles → 3. Fix inconsistencies

DSDG follows this pattern:
1. Define design system → 2. Generate within constraints → 3. Validate compliance

## Quick Start

### Generate a Website with Preset

```bash
# Generate a SaaS landing page with modern design
python -m orchestrator website \
    --preset saas_modern \
    --industry "project_management" \
    --sections "hero,features,pricing,testimonials,faq,cta,footer" \
    --budget 3.0 \
    --output ./results/my-website
```

### Generate with Custom Design System

```bash
# First, generate a design system YAML from preset
python -c "
from orchestrator.cli_website import generate_design_system_yaml
from pathlib import Path
generate_design_system_yaml('luxury_brand', Path('design_system.yaml'))
"

# Then generate website with custom design system
python -m orchestrator website \
    --design-system design_system.yaml \
    --client-info client.yaml \
    --page-type landing \
    --sections "hero,listings,testimonials,about,contact" \
    --budget 2.0
```

## Design System Presets

DSDG includes 8 pre-configured design system presets:

| Preset | Fonts | Primary | Accent | Style |
|--------|-------|---------|--------|-------|
| `premium_minimal` | Plus Jakarta Sans, Inter | #0f172a | #6366f1 | Clean lines, generous whitespace |
| `bold_creative` | Cabinet Grotesk, Satoshi | #18181b | #f97316 | High contrast, bold typography |
| `corporate_clean` | Inter, System UI | #1e40af | #3b82f6 | Professional, trustworthy |
| `playful_modern` | Poppins, Nunito | #7c3aed | #ec4899 | Vibrant colors, rounded corners |
| `luxury_elegant` | Playfair Display, Lora | #1c1917 | #d4a853 | Serif elegance, dark backgrounds |
| `medical_trust` | DM Sans, Inter | #0c4a6e | #06b6d4 | Trust-building, clean, accessible |
| `saas_modern` | Geist, Geist Mono | #020617 | #8b5cf6 | Dark/light mode, glassmorphism |
| `startup_bold` | Clash Display, Inter | #000000 | #22c55e | Bold typography, high contrast |

## Architecture

### Phase 0: Design System Definition

The design system is the **core** that drives all generation. Every component MUST use values from this system.

```yaml
# design_system.yaml
brand:
  name: "ClientName"
  industry: "medical_spa"
  tone: "premium_minimal"

typography:
  font_heading: "Plus Jakarta Sans"
  font_body: "Inter"
  scale:
    xs: "0.75rem"
    sm: "0.875rem"
    base: "1rem"
    lg: "1.25rem"
    xl: "1.563rem"

colors:
  primary: "#1a1a2e"
  accent: "#e94560"
  surface: "#fafafa"
  text_primary: "#1a1a2e"
  text_secondary: "#6b7280"

layout:
  max_width: "1280px"
  columns: 12
  gutter: "2rem"
  breakpoints:
    sm: "640px"
    md: "768px"
    lg: "1024px"

accessibility:
  min_contrast_ratio: 4.5
  focus_ring: true
  reduced_motion_support: true
```

### Phase 1: Component Selection

Instead of manual browsing, the orchestrator automatically selects the best-matching components from multiple sources:

- **TwentyFirst** (twentyfirst.dev)
- **Shadcn UI** (ui.shadcn.com)
- **Aceternity UI** (ui.aceternity.com)
- **Magic UI** (magicui.design)
- **Internal Curated** (./components/curated/)

Components are scored based on:
- Color proximity to design system
- Layout compatibility
- Animation style match
- Accessibility compliance (ARIA labels, keyboard navigation)
- Responsive readiness

### Phase 2: Constrained Generation

Every generated component follows these rules:

1. **Colors**: ONLY from design system palette
2. **Fonts**: ONLY from typography section
3. **Spacing**: ONLY from spacing scale
4. **Accessibility**: Focus states, hover states, aria-labels required
5. **Semantic HTML**: section, nav, main, footer elements
6. **Animations**: Respect prefers-reduced-motion
7. **Contrast**: Minimum 4.5:1 ratio
8. **Responsive**: Mobile-first (375px → up)

### Phase 3: Research-Driven Content

Content is generated based on industry research:
- Competitor website analysis
- Customer reviews/complaints
- Industry design trends

### Phase 4: Quality Validation

Every generated website is validated against:

| Check | Description | Target |
|-------|-------------|--------|
| Accessibility | WCAG 2.1 AA compliance | AA pass |
| Performance | Lighthouse-like metrics | 90+/100 |
| Design System | Color/token compliance | 100% |
| Responsive | 4 breakpoints tested | 4/4 pass |
| SEO | Meta tags, heading hierarchy | 90+/100 |
| Content | No Lorem ipsum, no TODOs | 100% clean |

## API Usage

### Python API

```python
from orchestrator import (
    DesignSystem,
    BrandTone,
    ClientInfo,
    WebsiteConfig,
    generate_website,
)
from pathlib import Path

# Create design system from preset
design_system = DesignSystem.from_preset(BrandTone.SAAS_MODERN)

# Create client info
client_info = ClientInfo(
    name="My Startup",
    industry="project_management",
)

# Configure website
config = WebsiteConfig(
    page_type="landing",
    sections=["hero", "features", "pricing", "testimonials", "faq", "cta", "footer"],
    framework="next.js",
    styling="tailwind",
    include_dark_mode=True,
)

# Generate
result = await generate_website(
    design_system=design_system,
    client_info=client_info,
    config=config,
    output_dir=Path("./my-website"),
)

print(f"Generated {result.components_generated} components")
print(f"Quality score: {result.quality_report.score:.2f}")
```

### Validation API

```python
from orchestrator import validate_website
from pathlib import Path

report = await validate_website(
    output_dir=Path("./my-website"),
)

print(f"Lighthouse: {report.lighthouse_score}/100")
print(f"WCAG Level: {report.wcag_level}")
print(f"SEO Score: {report.seo_score}/100")
```

## Output Structure

```
./results/my-website/
├── components/
│   ├── hero.tsx
│   ├── features.tsx
│   ├── pricing.tsx
│   ├── testimonials.tsx
│   ├── faq.tsx
│   ├── cta.tsx
│   └── footer.tsx
├── styles/
│   └── globals.css
├── page.tsx              # Assembled page
├── tailwind.config.js    # Design tokens
├── design_system.json    # Exported tokens
└── quality_report.json   # Validation results
```

## Integration with Orchestrator

DSDG integrates seamlessly with the existing Orchestrator engine:

- **Parallel Execution**: Sections are generated in parallel with configurable workers
- **Self-Healing**: Failed generations are automatically retried
- **Model Routing**: Optimal model selection per task type
- **Budget Enforcement**: Mid-task budget checks prevent overspending
- **Telemetry**: All operations are logged for cross-run learning

## Example Output

```bash
$ python -m orchestrator website --preset saas_modern --industry "project_management"

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
  Total: $0.33 | Time: 2m 14s | Quality: 0.91 avg
```

## Modules

- `design_system.py` - Core design system models and presets
- `component_registry.py` - Multi-source component library
- `website_generator.py` - Main generation pipeline
- `website_validator.py` - Quality validation
- `cli_website.py` - CLI commands

## Future Enhancements

- [ ] Live component preview dashboard
- [ ] A/B testing integration
- [ ] Multi-language support (i18n)
- [ ] E-commerce templates
- [ ] CMS integration (Sanity, Contentful)
- [ ] Analytics integration (Plausible, Google Analytics)

## References

- DESIGN.md - Original design document
- ARCHITECTURE_OVERVIEW.md - System architecture
- USAGE_GUIDE.md - General orchestrator usage
