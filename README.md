# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](python.org)
[![Tests](https://img.shields.io/badge/tests-191%2B%20passing-brightgreen.svg)]()
[![Models](https://img.shields.io/badge/models-50%2B-blue)]()
[![Agents](https://img.shields.io/badge/agents-9-gray)]()

> **Autonomous Multi-Agent Software Development Platform**  
> Decomposes project specs into tasks, routes them to 50+ LLMs across 15 providers, and executes generate-critique-revise-evaluate cycles with ARA reasoning, safety gates, and CI-quality gating.

---

## Quick Start

> **Γρήγορη εκκίνηση** / Quick Start

### Εγκατάσταση (Installation)

```bash
# Ανάπτυξη (development install)
pip install -e ".[dev,security,tracing]"

# Ή απλά με requirements
pip install -r requirements-dev.txt
```

### Ρύθμιση (Setup)

Αντέγραψε το `.env.example` → `.env` και πρόσθεσε τουλάχιστον ένα API key:

```bash
cp .env.example .env
# Έπειτα άνοιξε το .env και βάλε: OPENAI_API_KEY=sk-...  ή  DEEPSEEK_API_KEY=sk-...  κλπ.
```

### Εκτέλεση (Run)

```bash
# 🚀 Νέο project — περιγράφεις τι θες, το χτίζει
python -m orchestrator --project "Build a FastAPI todo app" --budget 2.0

# 📋 Μόνο πλάνο (χωρίς εκτέλεση) — dry-run
python -m orchestrator --project "Build a REST API" --criteria "All tests pass" --dry-run

# 🔄 Συνέχιση προηγούμενου project
python -m orchestrator --resume <project_id>

# 📄 Φόρτωση από YAML αρχείο
python -m orchestrator --file project.yml

# 📊 Λίστα όλων των projects
python -m orchestrator --list-projects
```

### Άλλες εντολές (Other Commands)

```bash
# 🔍 Ανάλυση υπάρχοντος codebase
python -m orchestrator analyze --path ./my-project --focus "architecture,security"

# 🏗️ Build με AppBuilder pipeline
python -m orchestrator build --description "Build a web app" --criteria "Works correctly"

# 🤖 NL → προδιαγραφές → ControlPlane
python -m orchestrator agent --intent "Build a user auth service" --interactive

# 💬 Interactive slash commands (/architect, /implement, /help)
python -m orchestrator slash

# 📈 Dashboard με cross-run metrics
python -m orchestrator dashboard --days 30

# 📦 Στατιστικά cache
python -m orchestrator cache-stats

# 🌐 Web search (Nexus)
python -m orchestrator nexus search "latest FastAPI patterns"

# ⚖️ Nash stability status
python -m orchestrator nash status

# 🔧 Τροποποίηση codebase
python -m orchestrator modify --repo ./my-project --objective "Add JWT authentication"

# 🎮 IDE Backend (WebSocket + API)
python -m orchestrator.ide_backend.server
```

### Βασικά flags

| Flag | Default | Περιγραφή |
|------|---------|-----------|
| `--budget 5.0` | 8.0 | Μέγιστο budget σε USD |
| `--time 5400` | 5400 | Μέγιστος χρόνος σε δευτερόλεπτα |
| `--concurrency 3` | 3 | Παράλληλα API calls |
| `--tdd-first` | off | Test-Driven Development mode |
| `--dry-run` | off | Μόνο πλάνο, χωρίς εκτέλεση |
| `--output-dir ./my-app` | auto | Φάκελος εξόδου |
| `--verbose` | off | Λεπτομερές logging |
| `--tracing` | off | OpenTelemetry tracing |
| `--new-project` | off | Πάντα νέο project (skip resume) |

### Pipeline εκτέλεσης

```
Περιγραφή Project → Resume Detection → Project Enhancer → Architecture Advisor
       │
       ▼
Decompose into Tasks (ProductManager + Architect agents)
       │
       ▼
Route (model selection) → Generate → Critique → Revise → Evaluate (cross-provider)
       │
       ▼
Deterministic Validation (python_syntax, pytest, ruff, json_schema)
       │
       ▼
Store Results + Telemetry + State Checkpoint (SQLite)
       │
       ▼
Output: structured files στο outputs/<project_id>/
```

---

## Key Capabilities

| Capability | What It Does |
|------------|-------------|
| **9 Specialized Agents** | Architect, Developer, Reviewer, Tester, DevOps, Researcher, User, PM, QA — cooperate via shared workspace |
| **52 LLMs / 15 Providers** | GPT-5, Claude Sonnet, Gemini, DeepSeek, Qwen, Kimi 2.6, Grok, MiMo, Llama, Mistral, and more |
| **20 ARA Reasoning Methods** | Debate, Jury, CoVE, SoT, ToT, Self-Discover, Brainstorming, PersuasionDefense, and 12 more |
| **PersuasionDefense Stage** | Claim extraction + NLI verification = blocks hallucinated code before delivery |
| **CoVE (Chain-of-Verification)** | Factored verification: each claim independently verified, cross-checked, revised |
| **Budget Enforcement** | Hierarchical budgets with mid-task enforcement and atomic reservation |
| **Self-Correcting Agents** | 3-attempt retry loops with error feedback; dynamic replanning on failure |
| **CI-Quality Gating** | Syntax validation, type checking, lint, test, security scan, hallucination detection — all automated |
| **Human-in-the-Loop** | Approval gates for architecture decisions, security-sensitive code, and production deploys |
| **Project Management** | Sprint plans, milestones, progress reports, product backlogs with P0-P3 priorities |
| **Knowledge Management** | Searchable cross-project knowledge base, auto-generated architecture docs |
| **UX Standards** | 20 WCAG standards enforced: responsive design, keyboard nav, color contrast, ARIA, dark mode |
| **Security Rules** | 22 OWASP Top 10 rules injected: CSP, SQLi prevention, CSRF, JWT best practices, secrets mgmt |
| **OpenGraph Generation** | Perfect social sharing meta tags: og:title, og:description, og:image 1200x630, JSON-LD |
| **Command Center** | Natural language REPL: "build a landing page" → decomposed, generated, validated, delivered |

---

## Architecture

```
User says: "Build a todo app"
       │
       ▼
Command Center ──→ AgentOrchestrator ──→ GoalDecomposer (HTN recursive)
       │                                        │
       │                                  ┌─────┴─────────┐
       │                                  ▼               ▼
       │                           ProductManager   ArchitectAgent
       │                           (generates        (designs system)
       │                            user stories)
       │                                  │               │
       │                                  ▼               ▼
       └──────────────────────────→  AgentOrchestrator dispatches
                                              │
                ┌────────────┬──────┬─────────┴──┬──────┬──────────┐
                ▼            ▼      ▼            ▼      ▼          ▼
          Developer    Reviewer   Tester      DevOps  Researcher  QCAgent
          (write)      (audit)    (test)      (build) (search)    (gate)
                │            │      │            │                    │
                └────────────┴──────┴────────────┴────────────────────┘
                                              │
                                              ▼
                                     ProjectWorkspace
                                     (shared blackboard)
                                              │
                                              ▼
                                     Delivery: tested, secure, deployed
```

---

## Pipeline Stages

Every code generation task goes through these stages:

```
GenerateStage → PersuasionDefenseStage → CritiqueStage → EvaluateStage
    │                                                              │
    ▼                                                              ▼
ValidateStage ←────── PreflightStage ←────── SelfConsistencyStage ←──┘
    │
    ▼
Delivery (or retry with fallback model)
```

| Stage | What It Checks | Cost |
|-------|---------------|------|
| **Generate** | Single LLM call for initial code | ~$0.01-0.05 |
| **PersuasionDefense** | Claim extraction → NLI verification → conflict detection | ~$0.05-0.10 |
| **Critique** | Cross-model review (different provider) | ~$0.02-0.08 |
| **Evaluate** | 2-pass self-consistency scoring | ~$0.02-0.06 |
| **Validate** | Syntax check, bracket balance, ruff lint | $0.00 |
| **Preflight** | PASS/WARN/ENRICH/BLOCK quality gate | $0.00-0.03 |
| **SelfConsistency** | If score < 0.7: retry with fallback model | ~$0.01-0.05 |

**8 hallucination layers:** PersuasionDefense + CoVE + cross-model critique + self-consistency retry + preflight gate + syntax validation + developer self-correction + knowledge graph confidence scoring.

---

## The 9 Agents

| Agent | Budget Model | Premium Model | Responsibility |
|-------|-------------|--------------|----------------|
| **Architect** | `DEEPSEEK_REASONER` ($0.70/M) | `GPT_5` ($1.25/M) | System design, framework selection, trade-off analysis |
| **Developer** | `CODESTRAL_2508` ($0.30/M) | `GPT_5` ($1.25/M) | Code generation with self-correcting loop |
| **Reviewer** | `QWEN_3_6_PLUS` ($0.33/M) | `CLAUDE_SONNET_4_6` ($3.00/M) | Security audit, code quality |
| **Tester** | `QWEN_3_CODER_NEXT` ($0.11/M) | `GPT_5` ($1.25/M) | Test generation with coverage tracking |
| **DevOps** | `DEEPSEEK_V4_FLASH` ($0.10/M) | `GPT_5_2_CODEX` ($1.75/M) | Docker, CI/CD, infra-as-code |
| **Researcher** | `SONAR_PRO` ($3.00/M) | `SONAR_DEEP_RESEARCH` ($2.00/M) | Web search, library research |
| **Product Manager** | `DEEPSEEK_REASONER` ($0.70/M) | `GPT_5` ($1.25/M) | User stories, requirements, priorities |
| **QA** | `CODESTRAL_2508` ($0.30/M) | `CLAUDE_SONNET_4_6` ($3.00/M) | CI pipeline, quality reports, regression |
| **User** | `OWL_ALPHA` (**free**) | `CLAUDE_SONNET_4_6` ($3.00/M) | Talks to the user, presents plans |

---

## 52 Models / 15 Providers

| Provider | Models Available |
|----------|----------------|
| OpenRouter | deepseek, openai, anthropic, google, mistralai, qwen |
| Meta | llama-4-maverick, llama-3.1-70b |
| xAI | grok-4.20, grok-4.3, grok-3, grok-3-mini |
| Moonshot (Kimi) | kimi-k2.6, kimi-k2.5 |
| MiniMax | minimax-m2.5, minimax-m2.1, minimax-m1 |
| Zhipu (GLM) | glm-5.1, glm-4.6, glm-4.5-air |
| Baidu | ernie-4.5, ernie-4.5-thinking, ernie-4.5-300b, cobuddy |
| Xiaomi | mimo-v2.5-pro, mimo-v2-flash |
| InclusionAI | ling-2.6-flash ($0.01/M), ring-2.6-1t |
| Perplexity | sonar-pro, sonar-deep-research |
| NVIDIA | nemotron-3-nano-omni (**free**) |
| IBM | granite-4.1-8b ($0.05/M) |
| Poolside | laguna-m.1 (**free**), laguna-xs.2 (**free**) |
| Owl | owl-alpha (**free**) |
| OpenRouter Auto | Dynamic model selection |

**Free models:** owl-alpha, deepseek-v4-flash:free, nemotron-nano-omni:free, poolside/laguna-m.1:free, poolside/laguna-xs.2:free

---

## 20 ARA Reasoning Methods

| Method | Purpose | Best For |
|--------|---------|----------|
| **Multi-Perspective** | 4-angle analysis | Architecture decisions, code review |
| **Debate** | Two models argue, third judges | Design trade-offs |
| **Jury** | 3 independent scores, meta-evaluation | Critical quality assessment |
| **CoVE** | Chain-of-Verification: draft → verify → revise | Factual accuracy, hallucination reduction |
| **SoT** | Skeleton-of-Thought: parallel sub-problems | Complex multi-part problems |
| **ToT** | Tree-of-Thoughts: explore decision branches | Strategic choices |
| **Self-Discover** | Meta-reasoning: select + adapt reasoning modules | Novel problems |
| **PersuasionDefense** | Claim extraction + NLI conflict detection | Hallucination prevention |
| **Brainstorming** | Divergent idea generation | Creative writing |
| **Research** | Iterative web search | Documentation, library research |
| **PreMortem** | "Assume it failed — why?" | Failure anticipation |
| **Dialectical** | Thesis → Antithesis → Synthesis | Architecture decisions |
| **Bayesian** | Prior → Evidence → Posterior | Uncertainty quantification |
| **Delphi** | 4 experts → aggregate → revise | Consensus building |
| **MAP-Elites** | Population evolution across 3x3 grid | Code optimization (#21) |

---

## 4 Pillars Management

| Pillar | Tool | What It Provides |
|--------|------|-----------------|
| Project Management | `SprintPlanner`, `ProgressReporter` | Milestones, deadlines, completion % |
| Product Management | `ProductBacklog`, `UserStory` | P0-P3 priorities, acceptance criteria |
| Knowledge Management | `KnowledgeBase`, `DocsGenerator` | Searchable decisions, ARCHITECTURE.md |
| Quality Control | `QualityReport`, `RegressionDetector` | Score 0-10, PASS/REVISE/BLOCK |

---

## Security & UX

### Security (22 OWASP Rules)

| Category | Critical | High |
|----------|----------|------|
| Web | CSP, HSTS, X-Frame-Options | X-Content-Type-Options, Referrer-Policy |
| API | SQL injection, input validation, HTTPS | Rate limiting, CORS |
| Auth | bcrypt/argon2, HttpOnly cookies | JWT RS256, CSRF tokens |
| Data | Env vars, AES-256 at rest | TLS 1.3 |

### UX Standards (20 Rules)

| Category | Standards |
|----------|----------|
| Layout | Responsive, mobile-first, spacing scale, visual hierarchy |
| Accessibility | WCAG AA, keyboard nav, ARIA labels, screen reader, touch targets |
| Typography | Font pairing, line length (60-80 chars), type scale |
| Color | Design system, dark mode, limited palette |
| Interaction | Micro-animations (150-300ms), loading states, error handling |

### OpenGraph (Every Page)

```html
<meta property="og:title" content="..." />
<meta property="og:description" content="..." />
<meta property="og:image" content="..." />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<script type="application/ld+json">{ "@context": "https://schema.org" }</script>
```

---

## Testing

```bash
# All tests
pytest

# Core engine tests (72 tests, ~1s)
pytest tests/test_god_file_refactoring.py tests/test_decomposer.py \
       tests/test_validator.py tests/test_pipeline.py

# Agentic system (22 tests)
pytest tests/test_agentic_system.py

# End-to-end suite (38 tests)
pytest tests/test_e2e_full_suite.py

# All new capabilities (154 tests)
pytest tests/test_phase6_10_comprehensive.py tests/test_capabilities_5_10.py \
       tests/test_planning.py tests/test_optimizations.py
```

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| `docs/MASTER_ARCHITECTURE_ENHANCEMENT_PLAN.md` | ARA, Protocols, DI, mypy (10 phases) |
| `docs/AGENTIC_SYSTEM_IMPLEMENTATION_PLAN.md` | 10 capabilities: multi-agent, workspace |
| `docs/AGENTIC_SYSTEM_OPTIMIZATION_PLAN.md` | 12 optimizations: self-correction, metrics |
| `docs/FOUR_PILLARS_ENHANCEMENT_PLAN.md` | Project/Product/Knowledge/Quality management |
| `docs/CYCLIC_IMPORT_RESOLUTION_PLAN.md` | 7 import chain fixes |
| `docs/CODEBASE_AWARE_RESEARCH.md` | Codebase reader, context, writer architecture |
| `docs/ARA_INTEGRATION_ANALYSIS.md` | ARA method placement analysis |
| `docs/TWO_TIER_MODEL_ROUTING.md` | Budget/premium model routing |
| `docs/AGENT_MODEL_ASSIGNMENTS.md` | Per-agent model assignments with fallbacks |
| `docs/OPENCYCLICAL_EVOLUTION_PLAN.md` | MAP-Elites, diff mutations, island migration |

---

## Configuration

```bash
# Required (at least one)
OPENAI_API_KEY="sk-..."
DEEPSEEK_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."
ANTHROPIC_API_KEY="sk-ant-..."

# Orchestrator
ORCH_MAX_CONCURRENCY=3
ORCH_DEFAULT_BUDGET_USD=10.0

# Cache
CACHE_TTL_HOURS=48
SEMANTIC_CACHE_THRESHOLD=0.85

# Dashboard
DASHBOARD_PORT=8000
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** 6.0.0  
**Last Updated:** 2026-05-25
