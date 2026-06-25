# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Models](https://img.shields.io/badge/models-52-blue)]()
[![engine.py](https://img.shields.io/badge/engine.py-2.6k%20lines-green)]()
[![Import Contracts](https://img.shields.io/badge/import--linter-3%20contracts-brightgreen)]()

> **Autonomous Multi-Agent Software Development Platform**  
> Decomposes project specs into tasks, routes them to 52 LLMs across 15 providers, and executes generate→critique→revise→evaluate cycles with self-improving skill documents, 20+ ARA reasoning methods, and CI-quality gating.

---

## Quick Start

```bash
pip install -e ".[dev,security,tracing]"
cp .env.example .env          # add at least one API key
python -m orchestrator --project "Build a FastAPI todo app" --budget 2.0
```

### Other Commands

```bash
# Generate a website (design-system-driven, any page type)
python -m orchestrator website -d "A modern SaaS landing page" --framework next.js

# Plan only (no execution)
python -m orchestrator --project "Build a REST API" --criteria "All tests pass" --dry-run

# Resume previous project
python -m orchestrator --resume <project_id>

# Dashboard with cross-run metrics
python -m orchestrator dashboard --days 30

# Interactive REPL
python -m orchestrator slash
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--budget 5.0` | 8.0 | Max spend in USD |
| `--concurrency 3` | 3 | Parallel API calls |
| `--tdd-first` | off | Test-Driven Development mode |
| `--dry-run` | off | Plan only, no execution |
| `--tracing` | off | OpenTelemetry tracing |
| `--output-dir ./my-app` | auto | Output directory |

---

## Key Capabilities

| Capability | What It Does |
|------------|-------------|
| **Website Generator** | Design-system-driven site builder — any page type, 3D/WebGL support, 20 curated themes, LLM-powered content |
| **9 Specialized Agents** | Architect, Developer, Reviewer, Tester, DevOps, Researcher, PM, QA, User |
| **52 LLMs / 15 Providers** | GPT-5, Claude Sonnet 4.6, Gemini, DeepSeek, Qwen, Grok, and more |
| **SkillOpt** | Self-improving per-TaskType skill documents — optimizer LLM evolves prompts via trajectory feedback |
| **20+ ARA Reasoning Methods** | Debate, Jury, CoVE, SoT, ToT, Self-Discover, MAP-Elites |
| **Hallucination Defense** | 8-layer: PersuasionDefense + CoVE + cross-model review + self-consistency retry |
| **Dual Budget Enforcement** | Per-run `Budget` + cross-run `BudgetHierarchy` (org → team → job) |
| **Self-Correcting Agents** | 3-attempt retry loops with error feedback and dynamic replanning |
| **CI-Quality Gating** | Syntax validation, lint, security scan, hallucination detection |
| **Human-in-the-Loop** | Approval gates for architecture, security, and deployment |
| **Import Boundary Enforcement** | `import-linter` with 3 layer contracts (domain purity, app-no-infra) |

---

## Design Quality Gates (taste-skill)

The orchestrator integrates **taste-skill**, a suite of anti-slop design rules that prevent AI-generated frontend UIs from defaulting to generic patterns (Inter font, purple-blue gradients, pure black/white, uniform 3-column grids).

### Activate & Configure

**Feature flags** (default behavior shown):

| Flag | Default | Purpose |
|------|---------|---------|
| `ORCH_TASTE_SKILL_ENABLED` | `true` | Inject anti-slop rules into frontend code generation |
| `ORCH_IMAGE_REFERENCE_PIPELINE` | `false` | Pre-flight text visual-direction brief (optional, slower) |

**Tunable dials** (1–10, default 5 for all):

| Dial | Range | What It Controls |
|------|-------|------------------|
| `ORCH_DESIGN_VARIANCE` | 1–10 | Layout experimentation (1=centered/clean, 10=asymmetric/modern) |
| `ORCH_MOTION_INTENSITY` | 1–10 | Animation depth (1=hover-only, 10=scroll-triggered/magnetic) |
| `ORCH_VISUAL_DENSITY` | 1–10 | Information per viewport (1=spacious/minimal, 10=dense dashboard) |

### Design Variants

When generating frontend tasks, the orchestrator picks a design direction:

| Variant | When Used | Characteristic |
|---------|-----------|-----------------|
| **DEFAULT** | All frontend tasks | Core anti-slop rules (always applied) |
| **SOFT** | Explicit selection | Premium agency aesthetic (serifs, luxury spacing) |
| **MINIMALIST** | Explicit selection | Editorial/Notion style (high whitespace, typography-first) |
| **BRUTALIST** | Explicit selection | Swiss/Industrial (geometric, constrained palette) |
| **REDESIGN** | Tasks with "redesign"/"improve ui" in prompt | Triggers structured audit critique instead of generic review |

### Usage Examples

```bash
# Default — taste-skill enabled with standard dials (5/5/5)
python -m orchestrator --project "Build a SaaS landing page in HTML/CSS" --budget 2.0

# Max layout variance, minimal motion
ORCH_DESIGN_VARIANCE=9 ORCH_MOTION_INTENSITY=1 \
  python -m orchestrator --project "Build a brutalist portfolio site" --budget 2.0

# Redesign audit mode — evaluates against typography/color/layout/interactivity
python -m orchestrator --project "Redesign the pricing page to feel more premium" --budget 1.5

# Disable entirely (for Python-only projects)
ORCH_TASTE_SKILL_ENABLED=false python -m orchestrator ...

# Enable visual-context pre-flight (experimental, slower)
ORCH_IMAGE_REFERENCE_PIPELINE=true \
  python -m orchestrator --project "Build a portfolio site with a specific mood" --budget 3.0
```

### How It Works

1. **Generator stage**: If task is frontend + flag enabled, prepends taste-skill rules to LLM system prompt, parameterized by dials
2. **Critic stage**: If task variant is REDESIGN, uses structured design audit (typography→color→layout→interactivity→content) instead of generic code review
3. **Validator stage**: Anti-slop pattern detector (soft WARN-only, logs generic patterns found but never blocks)

taste-skill is **always optional** — disabled flag or non-frontend task → no overhead.

---

## Website Generator — Design-System-Driven Site Builder

The orchestrator includes a **universal website generator** that produces complete, production-ready sites using LLM-powered content generation and a curated design system. It supports any page type — landing pages, SaaS, portfolios, agency sites, editorial, ecommerce — with optional 3D/WebGL and animation library support.

### Quick Start

```bash
# SaaS landing page (default sections)
python -m orchestrator website -d "A modern task management SaaS" --output-dir ./outputs/my-saas

# Ad agency portfolio with 3D effects
python -m orchestrator website \
  -d "Bold LA ad agency with immersive brand experiences, dark neon aesthetic" \
  --company-name "NEONVOID" \
  --industry advertising \
  --page-type agency \
  --sections "hero,work,services,about,clients,contact" \
  --framework next.js \
  --preset luxury \
  --atelier-theme midnight \
  --3d \
  --output-dir ./outputs/neonvoid

# Editorial magazine with minimal design
python -m orchestrator website \
  -d "Long-form journalism publication with typography-forward layout" \
  --company-name "The Interval" \
  --industry publishing \
  --page-type editorial \
  --sections "hero,featured,articles,newsletter,footer" \
  --framework html \
  --preset minimalist \
  --output-dir ./outputs/the-interval
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--description`, `-d` | *(required)* | Website description — drives content, tone, and imagery |
| `--company-name` | inferred | Brand/company name for metadata and content |
| `--industry` | `technology` | Industry for content research (advertising, fashion, publishing, etc.) |
| `--page-type` | `landing` | `landing`, `saas`, `portfolio`, `ecommerce`, `agency`, `editorial`, `custom` |
| `--sections`, `-s` | `hero,features,pricing,testimonials,faq,cta,footer` | Comma-separated section names |
| `--framework`, `-f` | `html` | `html`, `react`, `next.js` |
| `--preset` | `modern` | Design tone: `modern`, `minimalist`, `playful`, `corporate`, `luxury`, `tech` |
| `--atelier-theme` | *(none)* | Curated theme: `specimen`, `midnight`, `brutal`, `manifesto`, `riso`, and 15 more |
| `--image-model` | *(none)* | OpenRouter image model ID; empty = SVG placeholders |
| `--3d` | off | Shorthand for `--deps three,@react-three/fiber,@react-three/drei` |
| `--deps` | *(none)* | Extra npm packages (e.g. `gsap,swiper,lenis`) |
| `--output-dir`, `-o` | `outputs/website` | Output directory |

### Pipeline

```
User description
       │
       ▼
ContentResearcher (LLM-powered) → industry-specific content brief
       │                          (headlines, CTAs, FAQs, testimonials, pain points)
       ▼
DesignSystem + Atelier Theme → color tokens, typography, spacing, motion
       │
       ▼
Section Tasks (parallel LLM) → each section gets a tailored prompt with:
       │                        - PROJECT BRIEF (brand, industry, description)
       │                        - Section-type guidance (hero→dramatic, work→portfolio, about→story)
       │                        - Library awareness (3D deps → "use Three.js/R3F")
       │                        - Atelier design rules (color discipline, typography hierarchy)
       ▼
Assembly → framework-specific page (Next.js + Tailwind, React, or vanilla HTML)
       │    - Data-driven metadata (brand name, page type, description)
       │    - page_type-aware JSON-LD (Organization, CreativeWork, SoftwareApplication)
       │    - Security headers, OG tags, sitemap, robots.txt
       │    - Custom npm dependencies merged into package.json
       │    - 3D-compatible CSP (worker-src blob:, unsafe-eval)
       ▼
Image Generation → description-aware hero/OG/section images (LLM or SVG fallback)
       │
       ▼
Quality Validation → Lighthouse score, WCAG level, SEO score
```

### Supported Section Types

Each section name maps to type-specific LLM guidance:

| Section | Guidance |
|---------|----------|
| `hero` | Dramatic above-the-fold, bold typography, strong CTA, optional 3D/particle bg |
| `work`, `portfolio` | Project showcase with hover effects, masonry/horizontal scroll, category filters |
| `services` | Service cards with icons, descriptions, pricing tiers |
| `about` | Brand story, mission, team photos, timeline — NOT a contact form |
| `clients` | Logo grid or marquee, grayscale-to-color hover, trust indicators |
| `contact` | Form with validation, rate limiting, map embed, optional 3D globe |
| `team` | Member cards with hover bios, carousel or 3D depth layout |
| `testimonials` | Quote cards with avatars, carousel or masonry |
| `features` | Capability grid with icons, bento or alternating rows |
| `pricing` | Tier comparison cards, feature checkmarks, highlighted tier |
| `faq` | Accordion with smooth expand/collapse, category grouping |
| `cta` | Strong call-to-action, dramatic background, primary button |
| `footer` | Multi-column sitemap, social links, newsletter, copyright |
| *unrecognized* | Falls back to page-type-aware generic layout |

### 3D / Animation Library Support

When `--3d` or `--deps` specifies 3D/animation libraries, the generator:

- **Adds them to `package.json`** — `three`, `@react-three/fiber`, `@react-three/drei`, `gsap`, etc.
- **Updates CSP** — `worker-src blob:` and `script-src 'unsafe-eval'` for Three.js workers
- **Guides the LLM** — prompts include available library names and encourage their use: "You CAN use these for immersive effects: Three.js scenes, 3D models, particle systems, post-processing. Include all necessary imports. Use 'use client' directive."
- **Sets `"use client"` directive** — required for R3F and browser API components

### Atelier Themes (20 curated design directions)

| Theme | Genre | Vibe |
|-------|-------|------|
| `specimen` | Editorial | High-contrast serif headings, newspaper grid |
| `midnight` | Modern Minimal | Dark mode, cool blue, geometric minimalism |
| `brutal` | Brutal | Raw contrast, condensed bold, no decoration |
| `manifesto` | Brutal | Dark, red accent, condensed power |
| `riso` | Playful | Risograph print, misregistration, neon accents |
| `craft` | Artisanal | Warm tones, serif display, process-forward |
| `almanac` | Editorial | Journalistic, newspaper masthead, serif-only |
| `catalog` | Editorial | Typographic hierarchy, numbered sections |
| …plus 12 more | | |

### Content Generation Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **LLM-powered** | Engine available (default) | ContentResearcher calls LLM for industry-specific headlines, CTAs, FAQs, testimonials |
| **Template fallback** | No engine or LLM fails | Industry-aware templates with expanded section coverage (work, about, clients, etc.) |
| **Content-brief only** | LLM sections fail | Assembles directly from content brief without per-section LLM generation |

### Generated Output Structure

```
outputs/<project>/
  app/
    layout.tsx          ← brand-aware metadata, OG tags, JSON-LD, security headers
    page.tsx            ← imports all section components
    globals.css         ← design system CSS custom properties
    robots.ts           ← dynamic sitemap URL
    sitemap.ts          ← canonical URL
  components/
    hero.tsx            ← LLM-generated section components
    work.tsx
    services.tsx
    about.tsx
    clients.tsx
    contact.tsx
  public/
    images/             ← generated or SVG placeholder images
    og-image.html       ← static OG fallback
  package.json          ← base deps + custom 3D/animation deps
  next.config.js        ← security headers + 3D-compatible CSP
  tsconfig.json
  .gitignore
```

---

## Architecture

```
CommandCenter ──→ OrchestratorFacade ──→ GoalDecomposer (HTN)
                                                 │
                           ┌─────────────────────┴─────────┐
                           ▼                               ▼
                    ProductManager                  ArchitectAgent
                           │                               │
                           └─────────────┬─────────────────┘
                                         ▼
                              AgentOrchestrator (parallel)
                    ┌──────────┬─────────┬─────────┬───────────┐
                    ▼          ▼         ▼         ▼           ▼
              Developer   Reviewer   Tester    DevOps     Researcher
                                         │
                                         ▼
                                  ProjectWorkspace
                                  (shared blackboard)
```

### Core Pipeline (per task)

```
GenerateStage ──→ PersuasionDefenseStage ──→ CritiqueStage ──→ EvaluateStage
      │                                                                │
      └──── SelfConsistencyStage ──── PreflightStage ──── ValidateStage ◄──┘
```

Skill documents are injected into `GenerateStage` via `PipelineContext.skill_prefix` when SkillOpt has trained a skill for the current TaskType.

---

## Project Structure

```
orchestrator/
├── domain/              — 16 Protocols (ports), models, config schemas
├── application/         — 17 extracted service classes
│   ├── vs_selector.py             — VS candidate selection with quality scoring
│   ├── skill_optimizer.py         — per-TaskType epoch loop
│   ├── skill_manager.py           — facade + epoch scheduling
│   ├── skill_store.py             — aiosqlite trajectory + skill persistence
│   ├── model_health_tracker.py    — circuit breaker record_success/record_failure
│   ├── resumption_service.py      — resume_project logic
│   ├── dashboard_bridge.py        — null-safe dashboard notifications
│   ├── git_bridge.py              — git commit dispatch
│   ├── project_runner.py          — run_project / run_job / dry_run
│   └── ...                        — evaluator, critique_cycle, decomposer, executor
├── services/            — Adapter layer between ports and infrastructure
│   └── scorers.py                — EvaluatorScorer, ProbabilityScorer (QualityScorer adapters)
├── engine.py            — Mediator facade (1,725 lines, −7% from 1,867 baseline)
├── engine_core/         — ServiceContainer, TaskPipeline, 7 pipeline stages
├── infrastructure/      — SQLite state, disk cache, LLM adapter
├── agents/              — 9 specialized agent implementations
└── events/              — Unified EventBus + SyncHookRegistry
```

---

## SkillOpt — Self-Improving Prompts

The orchestrator continuously improves its own per-TaskType system prompts through a text-space optimization loop — no fine-tuning required.

```
Task executes → Trajectory recorded (prompt, output, score, critique)
                       │
              Every epoch_size trajectories
                       │
                       ▼
          Optimizer LLM proposes structured patches
          (append / insert_after / replace / delete)
                       │
          Edit budget enforced (≤150 tokens total)
                       │
          Validation gate: held-out val split must improve
                       │
          Accepted → best_skill.md updated
          Rejected → negative-feedback buffer (informs next epoch)
                       │
          Every 5 epochs → slow update: rebuild ## Guidance block
```

**Off by default.** Activate with `ORCH_SKILL_OPTIMIZATION_ENABLED=true`.

---

## Pipeline Stages (7)

| Stage | What It Checks | Cost |
|-------|---------------|------|
| **Generate** | LLM code generation with optional skill injection | $0.01–0.05 |
| **PersuasionDefense** | Claim extraction + NLI verification | $0.05–0.10 |
| **Critique** | Cross-model review (different provider) | $0.02–0.08 |
| **Evaluate** | 2-pass self-consistency scoring | $0.02–0.06 |
| **Validate** | Syntax, bracket balance, ruff lint | $0.00 |
| **Preflight** | PASS/WARN/ENRICH/BLOCK quality gate | $0.00–0.03 |
| **SelfConsistency** | Score < 0.7 → retry with fallback model | $0.01–0.05 |

---

## Reranking (VS Selector + Two-Stage Knowledge Recall)

Quality-driven candidate selection and LLM-based knowledge reranking. **Both features default OFF** — opt-in only, bounded cost.

### VS Candidate Selector

Turn `list[VSCandidate]` → single best candidate using a swappable quality scorer (`QualityScorer` port). Free probability prefilter bounds LLM scoring cost.

```python
# Adapters in services/scorers.py
EvaluatorScorer(evaluator)    # LLM-based quality via EvaluatorService (consistency_runs=1)
ProbabilityScorer()           # Free fallback based on VSCandidate.probability
```

| Flag | Default | Purpose |
|------|---------|---------|
| `ORCH_VS_RERANKING_ENABLED` | `false` | Score top VS candidates via QualityScorer before selection |

### Two-Stage Knowledge Recall

`KnowledgeBase.find_similar()` can optionally enable a second LLM rerank stage:
1. **Stage 1** — cosine similarity (free, local)
2. **Stage 2** — LLM-based reranking refines ordering when `knowledge_rerank_enabled=true`

| Flag | Default | Purpose |
|------|---------|---------|
| `ORCH_KNOWLEDGE_RERANK_ENABLED` | `false` | Two-stage recall: cosine → rerank |

`KnowledgeBase.find_similar()` gains optional `rerank=True` stage:
1. **Stage 1** — cosine similarity (free, local)
2. **Stage 2** — LLM-based `Reranker` port refines ordering

| Flag | Default | Purpose |
|------|---------|---------|
| `ORCH_KNOWLEDGE_RERANK_ENABLED` | `false` | Enable rerank stage in knowledge queries |

### Architecture
```
QualityScorer (Protocol, domain/ports.py)  ←—  EvaluatorScorer, ProbabilityScorer (services/scorers.py)
Reranker (Protocol, domain/ports.py)       ←—  LLMReranker (infrastructure/reranker.py)

CandidateSelector (application/vs_selector.py)
  └─ select(task, candidates) → best candidate (prefilter + parallel scoring)

KnowledgeBase (knowledge_base.py)
  └─ find_similar(..., rerank=True) → 2-stage (cosine → rerank) results
```

---

## 52 Models / 15 Providers

**Free:** owl-alpha, deepseek-v4-flash:free, nemotron-3-nano-omni, poolside/laguna-m.1, poolside/laguna-xs.2

**Budget ($0.01–0.50):** ling-2.6-flash, granite-4.1-8b, deepseek-v4-flash, qwen-3.5-flash, codestral-2508, gemini-2.5-flash

**Standard ($0.50–2.00):** deepseek-reasoner, qwen-3.6-plus, kimi-k2.6, gpt-5.4-nano, gemini-2.5-pro, grok-4.20

**Premium ($2.00+):** gpt-5, claude-sonnet-4.6, qwen-3.7-max, o3, sonar-pro

---

## 20+ ARA Reasoning Methods

| Method | Best For |
|--------|----------|
| Multi-Perspective | Architecture decisions, code review |
| Debate | Design trade-offs (two models argue, judge decides) |
| Jury | Critical quality (3 generators, 3 critics) |
| CoVE | Factual accuracy (draft → verify → revise) |
| SoT | Complex multi-part problems (parallel sub-solves) |
| ToT | Strategic choices (branch → evaluate → backtrack) |
| MAP-Elites | Code optimization (3×3 grid evolution) |
| PersuasionDefense | Hallucination prevention (claim extraction + NLI) |
| Brainstorming, PreMortem, Bayesian, Delphi, Socratic, Research, and more |

---

## Testing

```bash
pytest tests/ -v                    # All tests
pytest -m "not slow"                # Skip slow integration tests
pytest tests/contracts/ -v          # Protocol contract tests
pytest tests/unit/ -v               # Unit tests only
```

---

## Configuration

```bash
# Required (at least one provider key)
OPENAI_API_KEY=sk-...

# Optional tuning
ORCH_MAX_CONCURRENCY=3               # Parallel API calls
ORCH_DEFAULT_BUDGET_USD=10.0         # Default per-run budget
ORCH_CONTEXT_COMPRESSION=true        # Enable context compression
ORCH_SKILL_OPTIMIZATION_ENABLED=false  # SkillOpt self-improving prompts
LOG_FORMAT=json                      # Structured JSON logging
LOG_LEVEL=DEBUG                      # Debug logging
```

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| `docs/CODEBASE_MINDMAP.md` | Complete architecture reference |
| `docs/REFACTORING_PLAN_V7.md` | Phases 0–5 refactoring roadmap |
| `docs/DEPENDENCY_POLICY.md` | Import boundary rules |

---

## License

MIT License — see [LICENSE](LICENSE).

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** 6.2.0  
**Last updated:** 2026-05-28
