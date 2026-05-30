# Enhancement Plan: Lovable-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Lovable and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements `REPLIT_INSPIRED_ENHANCEMENTS.md`

---

## Overview

Four unique enhancements identified from Lovable's capabilities that are distinct from the Replit-inspired features. These focus on **persistent context management**, **reusable instruction workflows**, **cross-project knowledge sharing**, and **project-level design systems**.

```
Phase 7: Knowledge System (Workspace + Project Context)     (Highest ROI, 2-3 days)
Phase 8: Skills System (On-Demand Workflow Playbooks)       (High ROI, 2-3 days)
Phase 9: Cross-Project Referencing                          (Medium ROI, 2-3 days)
Phase 10: Design Systems as Projects                         (Lower ROI, 3-4 days)
```

---

## Phase 7: Knowledge System (Workspace + Project Context)

### Objective

Persistent instruction files at two levels — workspace-wide rules that apply to every project, and project-specific context. Always included in the LLM's context window for every generation call.

### Current State

- `orchestrator/architecture_rules.py` — generates `.orchestrator-rules.yml` once, not continuously referenced
- `SystemPrompt.build()` — builds prompts from templates
- No persistent "always-on" context injection mechanism
- `AGENTS.md` exists in the repo but is not read by the orchestrator at runtime

### Implementation

#### 7.1 — Create `orchestrator/knowledge.py`

```python
@dataclass
class WorkspaceKnowledge:
    """Workspace-level rules applied to all projects."""
    coding_standards: str  # Markdown text
    preferred_libraries: str
    architectural_conventions: str
    testing_requirements: str
    general_rules: str
    max_chars: int = 10_000

    @classmethod
    def from_file(cls, path: Path) -> "WorkspaceKnowledge":
        """Load from .orchestrator/knowledge/workspace.md."""

    def to_context(self) -> str:
        """Render as context injection for LLM prompts."""

@dataclass
class ProjectKnowledge:
    """Project-specific context injected into every call."""
    project_purpose: str
    user_personas: str
    database_schema: str
    architecture_decisions: str
    domain_terminology: str
    design_guidelines: str
    external_references: str
    max_chars: int = 10_000

    @classmethod
    def from_file(cls, path: Path) -> "ProjectKnowledge":
        """Load from .orchestrator/knowledge/project.md."""

    def to_context(self) -> str:
        """Render as context injection for LLM prompts."""

class KnowledgeInjector:
    """Injects workspace + project knowledge into every generation call.

    Priority: project knowledge overrides workspace knowledge when
    the same rule appears in both.
    """

    def __init__(
        self,
        workspace: WorkspaceKnowledge | None = None,
        project: ProjectKnowledge | None = None,
    ):
        self._workspace = workspace
        self._project = project

    def inject(self, prompt: str) -> str:
        """Prepend knowledge context to any generation prompt."""
        parts = []
        if self._workspace:
            parts.append(self._workspace.to_context())
        if self._project:
            parts.append(self._project.to_context())
        if not parts:
            return prompt
        context = "\n\n---\n\n".join(parts)
        return f"{context}\n\n---\n\n## TASK\n{prompt}"

    def inject_system(self, system_prompt: str) -> str:
        """Merge knowledge into a system prompt."""
        parts = []
        if self._workspace:
            parts.append(self._workspace.to_context())
        if self._project:
            parts.append(self._project.to_context())
        if not parts:
            return system_prompt
        return f"{system_prompt}\n\n## PERSISTENT KNOWLEDGE\n{' '.join(parts)}"
```

**File structure:**
```
.orchestrator/
├── knowledge/
│   ├── workspace.md    ← Coding standards, preferred libraries, rules
│   └── project.md      ← App purpose, DB schema, domain terms
├── rules.yml           ← Architecture rules (existing)
└── checkpoints/        ← Checkpoint snapshots (Phase 1)
```

**Workspace knowledge example (`workspace.md`):**
```markdown
# Workspace Knowledge

## Coding Standards
- Always enable Python type hints (strict mode)
- Use Pydantic v2 for data validation
- Prefer async/await over threading
- Maximum function complexity: 10

## Preferred Libraries
- FastAPI for web APIs
- SQLAlchemy 2.0 for ORM
- pytest for testing
- ruff for linting

## Architecture
- Hexagonal architecture with ports/adapters
- Domain layer has zero external dependencies
- All external I/O through infrastructure adapters
```

**Project knowledge example (`project.md`):**
```markdown
# Project: Inventory Manager

## Purpose
B2B SaaS for restaurant managers tracking food inventory across locations.

## Users
- Primary: Restaurant managers (quick stock visibility)
- Secondary: Staff (log inventory changes)

## Database
- inventory_items (id, name, category, quantity, unit, location_id)
- locations (id, name, workspace_id)
- transactions (id, item_id, quantity_change, type, created_at)

## Domain Terms
- "Inventory item" = tracked ingredient or product
- "Transaction" = change in inventory quantity

## Architecture Decisions
- Store monetary values in cents as integers
- Use optimistic updates for all mutations
```

**File changes:**
- NEW: `orchestrator/knowledge.py` (~250 lines)
- MODIFY: `orchestrator/engine.py` — load knowledge at init, inject into every prompt
- MODIFY: `orchestrator/prompt_builder.py` — accept `KnowledgeInjector` parameter
- NEW: `.orchestrator/knowledge/workspace.md` — template file

#### 7.2 — Integration Points

Every code generation call passes through the knowledge injector:
```
Task prompt → KnowledgeInjector.inject() → enriched_prompt → LLM
```

The injector is wired once in `Orchestrator.__init__` and used by every service that builds prompts.

#### Verification Gate

```bash
# Create workspace.md and project.md
# Run project generation
# Verify: generated code follows workspace coding standards
# Verify: generated code uses project domain terminology
```

---

## Phase 8: Skills System (On-Demand Workflow Playbooks)

### Objective

Named, portable playbooks that the orchestrator loads when a task matches the skill's description. Each skill has a trigger description, markdown instructions, and optional bundled files. Invocable via `/skill-name` syntax in task descriptions.

### Current State

- No skills/reusable workflow system
- `SystemPrompt.build()` is static per task type
- `PromptBuilder` has `CritiquePrompt`, `DecompositionPrompt`, `DeltaPrompt`, `RevisionPrompt`, `SystemPrompt` — all hardcoded

### Implementation

#### 8.1 — Create `orchestrator/skills.py`

```python
@dataclass
class Skill:
    """A reusable, on-demand workflow playbook.

    Compatible with the Anthropic/Claude SKILL.md format for portability.
    """
    name: str  # lowercase, hyphens only, e.g. "launch-checklist"
    description: str  # Starts with "Use when..." — LLM uses this to decide trigger
    instructions: str  # Markdown body — steps, constraints, examples
    bundled_files: dict[str, str]  # filename → content

    @classmethod
    def from_skill_md(cls, path: Path) -> "Skill":
        """Parse a SKILL.md file — same format as Anthropic Claude skills."""

    @classmethod
    def from_directory(cls, path: Path) -> "Skill":
        """Load SKILL.md + bundled files from a directory."""

    def to_context(self) -> str:
        """Render skill instructions as context for LLM."""

class SkillRegistry:
    """Manages workspace skills — load, match, inject."""

    def __init__(self, skills_dir: Path):
        self._skills: dict[str, Skill] = {}
        self._load_all(skills_dir)

    def _load_all(self, skills_dir: Path) -> None:
        """Load all SKILL.md files from skills_dir."""

    def match(self, task_description: str) -> list[Skill]:
        """Return skills whose descriptions match the task.
        
        Uses keyword matching against the skill's description field.
        The description should start with "Use when..." for reliable matching.
        """

    def get(self, name: str) -> Skill | None:
        """Get a skill by name for explicit invocation."""

    def inject(self, prompt: str, task_description: str) -> str:
        """Match skills against task and inject matching instructions."""

    def parse_skill_tags(self, prompt: str) -> tuple[list[str], str]:
        """Extract /skill-name tags from prompt. Returns (skill_names, cleaned_prompt)."""
```

**Skills directory structure:**
```
.orchestrator/
├── skills/
│   ├── launch-checklist/
│   │   ├── SKILL.md
│   │   ├── seo-checklist.md
│   │   └── accessibility-checklist.md
│   ├── release-notes/
│   │   └── SKILL.md
│   └── security-audit/
│       └── SKILL.md
```

**Example SKILL.md:**
```markdown
# Launch Checklist

## Description
Use when I say I'm about to launch, ship, share, or release a project, or when
I ask whether it is ready to go live. Run the checklist below before giving a
go/no-go. Not for incremental development checks.

## Instructions

### Account flows
- Sign-up, sign-in, sign-out, password reset wired end-to-end
- Signed-out users cannot reach authenticated pages
- Signed-in users land on correct page after login

### Data and permissions
- Database tables with user data have row-level security
- One user cannot read another's data
- Destructive actions ask for confirmation

### Content and trust
- App name, favicon, social preview image, meta description set
- Footer links to working privacy policy and terms page
- No placeholder copy ("Lorem ipsum", "TODO") in UI

### Environment
- No dev URLs, localhost references, or test API keys in production
- Required environment variables set
- Analytics events fire on expected actions

### Final step
- Summarize failures and manual checks in priority order
- Do not say "ready" unless every item passes or is explicitly waived
```

**Invocation patterns:**
```
User prompt: "I'm about to ship the inventory app"
→ Skills matched: launch-checklist
→ Instructions injected into context

User prompt: "/launch-checklist Check the auth flow before Friday's release"
→ Explicit invocation: launch-checklist loaded
→ Cleaned prompt: "Check the auth flow before Friday's release"
```

**File changes:**
- NEW: `orchestrator/skills.py` (~300 lines)
- NEW: `.orchestrator/skills/` — directory with example skills
- MODIFY: `orchestrator/engine.py` — load `SkillRegistry` at init
- MODIFY: `orchestrator/prompt_builder.py` — accept skills injection
- MODIFY: `orchestrator/cli.py` — add `skill` command for managing skills

#### 8.2 — CLI Skill Management

```bash
# List available skills
python -m orchestrator skill list

# Create a new skill
python -m orchestrator skill create launch-checklist

# Import from Anthropic Claude format
python -m orchestrator skill import ./path/to/SKILL.md

# Export a skill
python -m orchestrator skill export launch-checklist
```

#### Verification Gate

```bash
# Create a skill, invoke via /skill-name, verify instructions are in context
# Create a skill, describe a matching task, verify auto-match
# Import a Claude SKILL.md, verify it works identically
```

---

## Phase 9: Cross-Project Referencing

### Objective

Allow projects to reference and reuse implementations from other projects in the same workspace. Read-only access to code, assets, chat history, and architecture decisions.

### Current State

- Each `Orchestrator` instance is independent
- No mechanism to share outputs between projects
- `DependencyResolver` handles intra-project task dependencies only
- No multi-project awareness

### Implementation

#### 9.1 — Create `orchestrator/cross_project.py`

```python
@dataclass
class ProjectReference:
    project_id: str
    project_path: Path
    name: str
    description: str
    key_files: list[str]
    architecture_style: str
    generated_at: str

class CrossProjectRegistry:
    """Tracks all projects in a workspace for cross-referencing."""

    def __init__(self, workspace_dir: Path):
        self._workspace_dir = workspace_dir
        self._projects: dict[str, ProjectReference] = {}

    def register(self, project_id: str, output_dir: Path) -> None:
        """Register a project for cross-referencing."""

    def list_projects(self) -> list[ProjectReference]:
        """List all available projects."""

    def get_project(self, project_id: str) -> ProjectReference | None:
        """Get a project by ID."""

    def search(self, query: str) -> list[ProjectReference]:
        """Search projects by name or description."""

    def read_files(
        self, project_id: str, paths: list[str]
    ) -> dict[str, str]:
        """Read specific files from a referenced project (read-only)."""

class CrossProjectContextBuilder:
    """Builds context from referenced projects for inclusion in prompts."""

    def __init__(self, registry: CrossProjectRegistry):
        self._registry = registry

    def parse_references(self, prompt: str) -> list[str]:
        """Extract @ProjectName references from prompt."""

    def build_context(
        self, project_ids: list[str], max_chars: int = 8_000
    ) -> str:
        """Build context string from referenced projects' key files."""

    def inject(self, prompt: str) -> str:
        """Parse @references, build context, inject into prompt."""
```

**Invocation patterns:**
```
User prompt: "Build this using the same auth setup as @InventoryApp"
→ Registry looks up InventoryApp → reads auth-related files → injects into context

User prompt: "Use the logo and color palette from @BrandSite"
→ Reads brand assets from BrandSite → injects into context
```

**Workspace structure:**
```
workspace/
├── .orchestrator/
│   ├── registry.json         ← Cross-project registry
│   ├── knowledge/
│   │   └── workspace.md      ← Shared workspace knowledge
│   └── skills/               ← Shared workspace skills
├── inventory-app/
│   ├── .orchestrator/
│   │   └── knowledge/project.md
│   └── src/
├── brand-site/
│   ├── .orchestrator/
│   └── src/
└── new-project/
    └── src/
```

**File changes:**
- NEW: `orchestrator/cross_project.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — register projects on completion
- MODIFY: `orchestrator/prompt_builder.py` — accept cross-project context

#### Verification Gate

```bash
# Build two projects, reference one from the other via @ProjectName
# Verify: referenced project's code is read-only, injected into context
# Verify: non-existent project reference produces helpful error
```

---

## Phase 10: Design Systems as Projects

### Objective

A dedicated project that defines a component library, styling guidelines, and setup instructions. Other projects connect to it and automatically receive design updates.

### Current State

- `UXDesignEnhancer` — 20 generic WCAG/UX standards, no component library awareness
- `scaffold/` directories — project templates, not design systems
- `DesignSystemConfig` (Phase 6) — brand tokens, not component libraries

### Implementation

#### 10.1 — Create `orchestrator/design_system_project.py`

```python
@dataclass
class DesignSystemProject:
    """A dedicated project that serves as the design system source of truth."""

    project_id: str
    components: dict[str, str]  # component_name → specification
    styling: dict[str, str]  # token_name → value
    guidelines: str  # Usage patterns, code conventions
    setup_instructions: str  # npm install, imports, configuration
    rules: dict[str, str]  # Individual rule files

    @classmethod
    def from_project(cls, project_path: Path) -> "DesignSystemProject":
        """Load from .orchestrator/design-system/ directory."""

    def to_context(self) -> str:
        """Render all design system rules as context for connected projects."""

    def to_rule_files(self) -> dict[str, str]:
        """Export individual rule files for the .orchestrator/rules/ directory."""

class DesignSystemConnector:
    """Connects projects to design systems and propagates updates."""

    def __init__(self):
        self._connections: dict[str, list[str]] = {}  # project_id → [ds_project_ids]

    def connect(self, project_id: str, ds_project_id: str, priority: int = 0) -> None:
        """Connect a project to a design system. Priority determines order."""

    def disconnect(self, project_id: str, ds_project_id: str) -> None:
        """Remove a design system connection from a project."""

    def get_active_systems(self, project_id: str) -> list[DesignSystemProject]:
        """Get all active design systems for a project, ordered by priority."""

    def inject_all(self, project_id: str, prompt: str) -> str:
        """Inject all connected design systems' context into a prompt."""
```

**Design system project structure:**
```
design-system/
├── .orchestrator/
│   └── design-system/
│       ├── system.md           ← Core rules always loaded
│       └── rules/
│           ├── components/
│           │   ├── button.md
│           │   ├── input.md
│           │   └── modal.md
│           ├── patterns/
│           │   ├── forms.md
│           │   └── navigation.md
│           └── styling/
│               ├── colors.md
│               └── typography.md
└── src/                        ← Reference implementation
```

**Connection model:**
```
Project A ──connect──→ Design System 1 (priority 0)
Project A ──connect──→ Design System 2 (priority 1)
Project B ──connect──→ Design System 1 (priority 0)

When Project A generates code:
  → Design System 1 rules injected first (highest priority)
  → Design System 2 rules injected second (falls back to DS1 on conflicts)
```

**File changes:**
- NEW: `orchestrator/design_system_project.py` (~350 lines)
- MODIFY: `orchestrator/ux/design_enhancer.py` — call `DesignSystemConnector.inject_all()`
- MODIFY: `orchestrator/engine.py` — load design system connections at project init
- NEW: `.orchestrator/design-system/system.md` — template file

#### Verification Gate

```bash
# Create a design system project with component rules
# Create a new project connected to the design system
# Generate UI code — verify it uses DS components and tokens
# Update the design system — verify connected project picks up changes
```

---

## Integration with Replit-Inspired Features

```
Phase 1-6 (Replit): Checkpoints, Plan Workflow, Self-Review, Sandbox Tasks, Browser Testing, Design System Injection
Phase 7-10 (Lovable): Knowledge System, Skills System, Cross-Project Referencing, Design Systems as Projects

Combined dependency map:

Phase 1 (Checkpoints) ────────────────────────────────────────────────────────┐
Phase 2 (Plan Workflow) ──────────────────────────────────────────────────────┤
Phase 3 (Self-Review) ────────────────────────────────────────────────────────┤
Phase 7 (Knowledge System) ───────────────────────────────────────────────────┤
    │  No dependencies                                                         │
Phase 8 (Skills System) ──────────────────────────────────────────────────────┤
    │  Depends on Phase 7 (knowledge provides fallback context)                │
Phase 9 (Cross-Project) ──────────────────────────────────────────────────────┤
    │  Depends on Phase 1 (checkpoints for project snapshots)                  │
    │  Depends on Phase 7 (workspace knowledge shared across projects)         │
Phase 4 (Sandbox Tasks) ──────────────────────────────────────────────────────┤
Phase 5 (Browser Testing) ────────────────────────────────────────────────────┤
Phase 6 (Design System Injection) ────────────────────────────────────────────┤
Phase 10 (Design System Projects) ─────────────────────────────────────────────┤
    │  Depends on Phase 6 (basic design tokens)                                │
    │  Depends on Phase 9 (cross-project connections)                          │
    └──────────────────────────────────────────────────────────────────────────┘
```

## Lovable-Specific Effort Estimate

| Phase | New Files | Modified Files | Est. Lines | Est. Days |
|-------|-----------|---------------|------------|-----------|
| 7. Knowledge System | 1 | 3 | ~250 | 2-3 |
| 8. Skills System | 1 | 3 | ~300 | 2-3 |
| 9. Cross-Project Referencing | 1 | 2 | ~300 | 2-3 |
| 10. Design System Projects | 1 | 2 | ~350 | 3-4 |
| **Lovable Subtotal** | **4** | **10** | **~1,200** | **9-13** |

## Combined Totals (Replit + Lovable)

| Source | Phases | New Files | Modified Files | Est. Lines | Est. Days |
|--------|--------|-----------|---------------|------------|-----------|
| Replit | 1-6 | 5 | 14 | ~1,430 | 13-18 |
| Lovable | 7-10 | 4 | 10 | ~1,200 | 9-13 |
| **Total** | **10** | **9** | **24** | **~2,630** | **22-31** |
