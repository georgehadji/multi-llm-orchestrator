# Enhancement Plan: Base44-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Base44 Developer Platform and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements Replit, Lovable, Newly, and UI enhancement plans

---

## Overview

Base44 is a backend-as-a-service (BaaS) platform designed for AI agents. Unlike the other three apps (which are code generation tools), Base44 provides a managed platform where AI agents work with **configuration files** (YAML/JSON) rather than generating raw code. This is a fundamentally different architecture model — config-driven rather than code-generation-driven.

Seven enhancements identified, focused on **configuration-as-code output**, **dynamic type generation**, and **automation/scheduling patterns** for generated projects.

```
Phase B1: Configuration-as-Code Output (Entity/Auth/Agent schemas)  (Highest ROI, 2-3 days)
Phase B2: Dynamic TypeScript/Python Type Generation                    (High ROI, 1-2 days)
Phase B3: Automations & Scheduling System for Generated Projects       (High ROI, 2-3 days)
Phase B4: Push/Pull Configuration Workflow                             (Medium ROI, 2-3 days)
Phase B5: Entity Schema Validation + Row-Level Security Rules         (Medium ROI, 2-3 days)
Phase B6: Local Development Server for Generated Projects             (Lower ROI, 3-4 days)
Phase B7: Skills System for External AI Coding Agents                  (Lower ROI, 2-3 days)
```

---

## What Base44 Has — Quick Reference

| Base44 Feature | Description | Translates? |
|---------------|-------------|:-----------:|
| **Config-as-Code** | Entities, auth, agents, connectors defined as JSON/YAML | ✅ Phase B1 |
| **Dynamic Types** | TypeScript types auto-generated from entity schemas | ✅ Phase B2 |
| **Automations** | Cron, schedules, entity events, connector webhooks | ✅ Phase B3 |
| **Push/Pull Config** | CLI syncs local config ↔ remote backend | ✅ Phase B4 |
| **Entity RLS** | Row-level + field-level security rules | ✅ Phase B5 |
| **Local Dev Server** | `base44 dev` with in-memory DB + auto-reload | ✅ Phase B6 |
| **AI Agent Skills** | Reusable instructions for external AI tools | ✅ Phase B7 |
| **MCP Server** | AI assistants create/manage projects via MCP | Partial — orchestrator has MCP |
| **Enterprise APIs** | Audit logs + monitoring API | ✗ (Platform-specific) |
| **Managed Backend** | Deno functions, MongoDB, OAuth connectors | ✗ (Platform-specific) |

---

## Phase B1: Configuration-as-Code Output (Entity/Auth/Agent Schemas)

### Objective

When the orchestrator generates a project, produce structured YAML/JSON configuration files alongside the code — entity schemas, authentication config, agent definitions, and connector settings. These can be pushed to any backend that accepts configuration-as-code (Base44, Supabase, custom backends).

### Current State

- `orchestrator/scaffold/` — project templates (FastAPI, Next.js, React, CLI)
- `orchestrator/app_assembler.py` — writes task outputs to files
- `ArchitectureAnalyzer._detect_database()` — detects database type from project description
- No structured configuration output format — only raw code files
- Templates are static, not dynamically generated from analysis

### Implementation

#### B1.1 — Create `orchestrator/config_generator.py`

```python
@dataclass
class EntitySchema:
    """Represents a data model entity as a JSON Schema."""
    name: str  # e.g., "tasks", "users"
    fields: dict[str, FieldSchema]
    security_rules: SecurityRules | None = None
    indexes: list[str] = field(default_factory=list)

@dataclass
class FieldSchema:
    type: str  # "string", "number", "boolean", "date", "array"
    required: bool
    default: Any | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class SecurityRules:
    """Row-level and field-level security."""
    read: str  # Expression: "true", "user.id == record.user_id"
    create: str
    update: str
    delete: str

@dataclass
class AuthConfig:
    providers: list[str]  # "email", "google", "github", etc.
    session_type: str  # "jwt", "cookie", "oauth"
    password_policy: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfig:
    name: str
    description: str
    instructions: str  # System prompt
    model: str  # "openai/gpt-4o", "anthropic/claude-sonnet-4"
    entity_permissions: dict[str, list[str]]  # entity_name → ["read", "create", ...]
    function_permissions: list[str]  # Function names agent can call

@dataclass
class AppConfig:
    """Complete app configuration as structured data."""
    app_name: str
    entities: list[EntitySchema]
    auth: AuthConfig
    agents: list[AgentConfig]
    integrations: list[str]  # "gmail", "slack", "stripe", etc.
    environment_variables: dict[str, str]

class ConfigGenerator:
    """Generates configuration-as-code from project analysis.

    Outputs:
    - .orchestrator/config/entities/*.jsonc  (entity schemas)
    - .orchestrator/config/auth.jsonc        (authentication config)
    - .orchestrator/config/agents/*.jsonc    (AI agent definitions)
    - .orchestrator/config/integrations.jsonc (third-party connectors)
    """

    def __init__(self, architecture_analyzer: ArchitectureAnalyzer):
        self._analyzer = architecture_analyzer

    async def generate_from_plan(
        self, plan: list[PlanTask], architecture: ProjectRules
    ) -> AppConfig:
        """Generate full app configuration from the execution plan."""

    def to_files(self, config: AppConfig, output_dir: Path) -> None:
        """Write configuration to files in the project directory."""

    def to_base44_format(self, config: AppConfig) -> dict:
        """Convert to Base44-compatible config structure."""

    def to_supabase_format(self, config: AppConfig) -> dict:
        """Convert to Supabase migration format."""
```

**Example output — `entities/tasks.jsonc`:**
```jsonc
{
  "name": "tasks",
  "description": "Task management entity for tracking work items",
  "fields": {
    "id": { "type": "string", "format": "uuid", "required": true },
    "title": { "type": "string", "required": true, "maxLength": 200 },
    "description": { "type": "string", "maxLength": 2000 },
    "priority": {
      "type": "string",
      "enum": ["low", "medium", "high", "critical"],
      "default": "medium"
    },
    "status": {
      "type": "string",
      "enum": ["todo", "in_progress", "done", "blocked"],
      "default": "todo"
    },
    "due_date": { "type": "string", "format": "date-time" },
    "assigned_to": { "type": "string", "format": "uuid" },
    "created_by": { "type": "string", "format": "uuid" },
    "created_at": { "type": "string", "format": "date-time", "readonly": true },
    "updated_at": { "type": "string", "format": "date-time", "readonly": true }
  },
  "security": {
    "read": "user.id == record.created_by OR record.assigned_to == user.id",
    "create": "user.is_authenticated == true",
    "update": "user.id == record.created_by",
    "delete": "user.id == record.created_by AND record.status != 'done'"
  },
  "indexes": ["assigned_to", "status", "due_date"]
}
```

**File changes:**
- NEW: `orchestrator/config_generator.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — call `ConfigGenerator` during project decomposition
- MODIFY: `orchestrator/scaffold/` — add `config/` template to each scaffold
- NEW: `orchestrator/scaffold/templates/config/` — template config files

#### Verification Gate

```bash
# Run a project, check output directory
# Expect: .orchestrator/config/ with entities, auth, agents config files
# Expect: config matches the architecture analysis (correct entity fields, auth providers)
```

---

## Phase B2: Dynamic TypeScript/Python Type Generation

### Objective

Generate TypeScript types and Python type stubs from the entity schemas produced in Phase B1. Types stay in sync with the backend through a `types generate` command.

### Current State

- Scaffold templates generate static code with typed Pydantic models for Python
- No dynamic type generation from entity schemas
- No TypeScript type generation for generated frontend projects
- No mechanism to regenerate types after schema changes

### Implementation

#### B2.1 — Create `orchestrator/type_generator.py`

```python
class TypeGenerator:
    """Generates TypeScript and Python types from entity schemas.

    Uses JSON Schema as the intermediate representation.
    """

    def generate_typescript(self, entities: list[EntitySchema]) -> dict[str, str]:
        """Generate TypeScript interfaces/types from entity schemas.

        Returns: {filename: content} mapping.
        """
        types = {}
        for entity in entities:
            interface = self._schema_to_ts_interface(entity)
            types[f"{entity.name}.ts"] = interface

        # Generate an index file
        types["index.ts"] = self._generate_ts_index(entities)

        return types

    def generate_python(self, entities: list[EntitySchema]) -> dict[str, str]:
        """Generate Pydantic v2 models from entity schemas.

        Returns: {filename: content} mapping.
        """
        models = {}
        for entity in entities:
            model = self._schema_to_pydantic(entity)
            models[f"{entity.name}.py"] = model

        models["__init__.py"] = self._generate_py_init(entities)
        return models

    def _schema_to_ts_interface(self, entity: EntitySchema) -> str:
        """Convert entity schema to TypeScript interface."""
        lines = [f"// Auto-generated from {entity.name} entity schema"]
        lines.append(f"// Regenerate: orchestrator types generate")
        lines.append(f"export interface {self._pascal(entity.name)} {{")

        for field_name, field in entity.fields.items():
            ts_type = self._json_type_to_ts(field.type, field.required, field.validation)
            optional = "?" if not field.required else ""
            if field.description:
                lines.append(f"  /** {field.description} */")
            lines.append(f"  {field_name}{optional}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

    def _json_type_to_ts(
        self, json_type: str, required: bool, validation: dict
    ) -> str:
        """Map JSON Schema types to TypeScript types."""
        type_map = {
            "string": "string",
            "number": "number",
            "boolean": "boolean",
            "date": "string",  # ISO 8601 string
            "array": "any[]",  # Could be narrowed from validation
            "object": "Record<string, unknown>",
        }

        base = type_map.get(json_type, "unknown")

        # Handle enums
        if "enum" in validation:
            options = ", ".join(f"'{v}'" for v in validation["enum"])
            return options

        # Handle nullable
        if not required:
            base = f"{base} | null"

        return base

    def _schema_to_pydantic(self, entity: EntitySchema) -> str:
        """Convert entity schema to Pydantic v2 model."""
        lines = ["# Auto-generated from entity schema", "# Regenerate: orchestrator types generate"]
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("from datetime import datetime")
        lines.append("from uuid import UUID")
        lines.append("")
        lines.append("from pydantic import BaseModel, Field")
        lines.append("")
        lines.append(f"class {self._pascal(entity.name)}(BaseModel):")
        if entity.fields:
            for field_name, field in entity.fields.items():
                py_type = self._json_type_to_py(field.type, field.required)
                field_args = []
                if field.description:
                    field_args.append(f'description="{field.description}"')
                if not field.required:
                    field_args.append(f"default=None")
                if "pattern" in field.validation:
                    field_args.append(f'pattern=r"{field.validation["pattern"]}"')
                if "min" in field.validation:
                    field_args.append(f"ge={field.validation['min']}")
                if "max" in field.validation:
                    field_args.append(f"le={field.validation['max']}")

                field_def = f"    {field_name}: {py_type}"
                if field_args:
                    field_def += f" = Field({', '.join(field_args)})"
                lines.append(field_def)
        else:
            lines.append("    pass")

        return "\n".join(lines) + "\n"
```

**CLI command:**
```bash
# Generate types from entity schemas
python -m orchestrator types generate

# Output:
# types/
#   entities/
#     tasks.ts
#     users.ts
#     index.ts
#   python/
#     tasks.py
#     users.py
#     __init__.py
```

**File changes:**
- NEW: `orchestrator/type_generator.py` (~350 lines)
- MODIFY: `orchestrator/cli.py` — add `types generate` command
- MODIFY: `orchestrator/config_generator.py` — call `TypeGenerator` after config generation
- DEPENDS ON: Phase B1 (Config-as-Code — entity schemas required)

#### Verification Gate

```bash
# Generate types from entity schemas
# Verify: TypeScript interfaces include all entity fields with correct types
# Verify: Pydantic models include validation rules from entity schema
# Change an entity, regenerate — verify types update (not drift)
```

---

## Phase B3: Automations & Scheduling System for Generated Projects

### Objective

When the orchestrator generates a project that needs background tasks, scheduled jobs, or event-driven automations, include the automation configuration and scaffold code. Support cron, simple schedules, database triggers, and webhook handlers.

### Current State

- No background task/automation generation in scaffolds
- No cron/scheduling support in generated projects
- `AsyncOrchestrator` uses `asyncio.gather()` for parallel execution but this is the orchestrator's own execution, not generated project automation
- `PhaseAwareModelSelector` has scheduling concepts but for orchestrator's own model selection, not for generated projects

### Implementation

#### B3.1 — Create `orchestrator/automation_generator.py`

```python
@dataclass
class AutomationConfig:
    name: str
    type: str  # "scheduled_cron", "scheduled_simple", "entity_event", "webhook"
    schedule: str | None = None  # Cron expression or "every_30_minutes"
    entity_name: str | None = None  # For entity event automations
    event_types: list[str] | None = None  # ["create", "update", "delete"]
    webhook_url: str | None = None
    function_name: str  # The handler function
    function_args: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

class AutomationGenerator:
    """Generates automation configuration and scaffold code for projects.

    Supports:
    - Cron-based scheduled tasks (via APScheduler or node-cron)
    - Entity event triggers (via database hooks)
    - Webhook endpoint handlers
    - Background task queues (via Celery/Redis or Bull/Redis)
    """

    def __init__(self, technology_stack: TechnologyStack):
        self._stack = technology_stack

    def generate_python_automations(
        self, automations: list[AutomationConfig]
    ) -> dict[str, str]:
        """Generate Python automation setup (APScheduler + Celery).

        Returns: {filename: content} mapping.
        """
        files = {}

        # APScheduler config
        cron_jobs = [a for a in automations if a.type.startswith("scheduled")]
        if cron_jobs:
            files["automations/scheduler.py"] = self._generate_apscheduler(cron_jobs)

        # Celery task definitions
        entity_jobs = [a for a in automations if a.type == "entity_event"]
        if entity_jobs:
            files["automations/tasks.py"] = self._generate_celery_tasks(entity_jobs)

        # Webhook handlers
        webhook_jobs = [a for a in automations if a.type == "webhook"]
        if webhook_jobs:
            files["automations/webhooks.py"] = self._generate_webhook_handlers(webhook_jobs)

        # Config file
        files["automations/config.yml"] = self._generate_config_yml(automations)

        return files

    def generate_typescript_automations(
        self, automations: list[AutomationConfig]
    ) -> dict[str, str]:
        """Generate TypeScript automation setup (node-cron + Bull).

        Returns: {filename: content} mapping.
        """
        # Similar pattern to Python but using bull, node-cron, etc.
        ...

    def _generate_apscheduler(self, jobs: list[AutomationConfig]) -> str:
        """Generate APScheduler background scheduler setup."""
        ...

    def _generate_celery_tasks(self, entity_jobs: list[AutomationConfig]) -> str:
        """Generate Celery task definitions for entity event automations."""
        ...

    def _generate_webhook_handlers(self, webhook_jobs: list[AutomationConfig]) -> str:
        """Generate webhook endpoint handlers."""
        ...

    def _generate_config_yml(self, automations: list[AutomationConfig]) -> str:
        """Generate YAML configuration file for all automations."""
        ...
```

**Example generated `automations/scheduler.py`:**
```python
# Auto-generated automation scheduler
# Regenerate: orchestrator scaffold regenerate --automations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .tasks import send_daily_report, cleanup_old_records

scheduler = AsyncIOScheduler()

# Daily report at midnight
scheduler.add_job(
    send_daily_report,
    trigger=CronTrigger(hour=0, minute=0),
    kwargs={"mode": "full_sync"},
    id="daily_midnight_report",
    name="Daily Report Generator",
)

# Cleanup every 30 minutes
scheduler.add_job(
    cleanup_old_records,
    trigger=CronTrigger(minute="*/30"),
    id="cleanup_every_30min",
    name="Old Record Cleanup",
)

def start_scheduler():
    scheduler.start()

def stop_scheduler():
    scheduler.shutdown()
```

**File changes:**
- NEW: `orchestrator/automation_generator.py` (~350 lines)
- MODIFY: `orchestrator/scaffold/fastapi.py` — include `automations/` directory
- MODIFY: `orchestrator/scaffold/nextjs.py` — include automation setup
- MODIFY: `orchestrator/engine.py` — call `AutomationGenerator` during project decomposition

#### Verification Gate

```bash
# Generate a project that describes "send daily email reports and clean up old records"
# Verify: automations/scheduler.py, automations/tasks.py generated
# Verify: APScheduler config includes both jobs
```

---

## Phase B4: Push/Pull Configuration Workflow

### Objective

Add CLI commands to sync the orchestrator's generated configuration (entities, auth, agents, automations) with a remote backend. This enables a "configuration source of truth" model where the orchestrator generates config, and the backend applies it.

### Current State

- `orchestrator/cli.py` — `run`, `plan`, `resume` commands
- No configuration sync commands
- `orchestrator/app_assembler.py` writes files to disk only
- No remote deployment capability for configuration files

### Implementation

#### B4.1 — Add sync commands to CLI

```python
@cli.group()
def config():
    """Manage project configuration."""

@config.command("push")
@click.option("--target", type=click.Choice(["base44", "supabase", "local"]), default="local")
@click.option("--dry-run", is_flag=True)
async def config_push(target: str, dry_run: bool):
    """Push local configuration to remote backend.

    Supported targets:
    - local: Write to .orchestrator/config/ (default)
    - base44: Push via Base44 API
    - supabase: Push via Supabase migrations
    """

@config.command("pull")
@click.option("--target", type=click.Choice(["base44", "supabase"]), default="base44")
async def config_pull(target: str):
    """Pull remote configuration to local project."""

@config.command("diff")
@click.option("--target", type=click.Choice(["base44", "supabase"]), default="base44")
async def config_diff(target: str):
    """Show differences between local and remote configuration."""
```

**Backend adapters:**
```python
class ConfigBackend(ABC):
    """Abstract interface for pushing/pulling configuration."""

    @abstractmethod
    async def push(self, config: AppConfig) -> PushResult:
        """Push configuration to backend."""

    @abstractmethod
    async def pull(self) -> AppConfig:
        """Pull configuration from backend."""

    @abstractmethod
    async def diff(self, local: AppConfig) -> DiffResult:
        """Show differences between local and remote."""

class LocalBackend(ConfigBackend):
    """Write to .orchestrator/config/ directory."""

class Base44Backend(ConfigBackend):
    """Push via Base44 CLI or API."""

class SupabaseBackend(ConfigBackend):
    """Push via Supabase migrations API."""
```

**Usage:**
```bash
# Generate config and push to local directory
python -m orchestrator config push --target local

# Push entity schemas to a Base44 backend
python -m orchestrator config push --target base44

# Pull changes made in the Base44 dashboard back to local
python -m orchestrator config pull --target base44

# See what would change before pushing
python -m orchestrator config diff --target supabase
```

**File changes:**
- NEW: `orchestrator/config_backends/` — `base.py`, `local_backend.py`, `base44_backend.py`, `supabase_backend.py`
- MODIFY: `orchestrator/cli.py` — add `config` command group
- DEPENDS ON: Phase B1 (Config-as-Code generation)

#### Verification Gate

```bash
# Generate a project, push config to local, verify files exist
# Modify config, push again, verify diff detection
```

---

## Phase B5: Entity Schema Validation + Row-Level Security (RLS)

### Objective

Validate generated entity schemas for correctness and auto-generate row-level security rules based on entity relationships and architecture analysis.

### Current State

- `orchestrator/validators.py` — validates Python syntax, ruff, pytest, JSON Schema, tool safety
- No entity schema validation
- No security rules generation

### Implementation

#### B5.1 — Create `orchestrator/entity_validator.py`

```python
class EntityValidator:
    """Validates entity schemas and generates security rules.

    Validation checks:
    - Required fields present
    - Types match supported types
    - Relationship integrity (foreign keys exist)
    - No circular dependencies in security rules
    - Indexes reference existing fields
    """

    def validate(self, entities: list[EntitySchema]) -> list[ValidationError]:
        """Validate entity schemas and return errors."""

    def validate_relationships(
        self, entities: list[EntitySchema]
    ) -> list[RelationshipError]:
        """Verify foreign keys reference existing entities."""

    def suggest_security_rules(
        self, entity: EntitySchema, auth_config: AuthConfig
    ) -> SecurityRules:
        """Auto-generate RLS rules based on entity purpose and auth config.

        Rules:
        - "owned" entities (created_by field): RLS restricts to owner
        - "public" entities (no created_by): RLS allows read for all auth users
        - "admin" entities: RLS restricts to admin role
        - Mixed: combine conditions
        """

    def validate_security_rules(
        self, rules: SecurityRules, entity: EntitySchema
    ) -> list[str]:
        """Validate security rules reference valid fields."""
```

**Example generated RLS:**
```json
{
  "read": "user.id == record.created_by OR user.role == 'admin'",
  "create": "user.is_authenticated == true",
  "update": "user.id == record.created_by OR user.role == 'admin'",
  "delete": "user.id == record.created_by"
}
```

**File changes:**
- NEW: `orchestrator/entity_validator.py` (~250 lines)
- MODIFY: `orchestrator/config_generator.py` — validate entities after generation
- MODIFY: `orchestrator/validators.py` — register `validate_entity_schemas` in the validator chain

#### Verification Gate

```bash
# Generate entities, validate them, verify no structural errors
# Entity with foreign key to non-existent entity → error flagged
# Entity with circular RLS rules → warning flagged
```

---

## Phase B6: Local Development Server for Generated Projects

### Objective

Add a `dev` CLI command that starts a local development server for generated projects — auto-reload on file changes, in-memory database, local auth, and real-time console output.

### Current State

- `orchestrator/app_verifier.py` — startup checks (npm build, Docker build, process alive)
- `validate_pytest()` — subprocess test runner
- No local dev server for generated projects
- Scaffold templates include `README.md` with manual setup instructions

### Implementation

#### B6.1 — Add `dev` command to CLI

```python
@cli.command()
@click.option("--project-dir", default=".", type=click.Path(exists=True))
@click.option("--port", default=8000, type=int)
@click.option("--hot-reload", is_flag=True, default=True)
def dev(project_dir: str, port: int, hot_reload: bool):
    """Start a local development server for the generated project.

    For Python/FastAPI projects: starts uvicorn with --reload
    For Next.js projects: starts next dev
    For React projects: starts vite dev server
    """
```

**Auto-detect project type and start appropriate dev server:**
```python
class DevServer:
    """Starts and manages local development servers for generated projects."""

    async def start(self, project_dir: Path) -> DevServerProcess:
        """Detect project type and start appropriate dev server.

        Detection order:
        1. Look for pyproject.toml → uvicorn/FastAPI
        2. Look for package.json + next.config → next dev
        3. Look for package.json + vite.config → vite dev
        4. Look for Dockerfile → docker-compose up
        """

    async def stream_output(self, process: subprocess.Popen) -> AsyncIterator[str]:
        """Stream stdout/stderr from dev server to terminal."""

    async def stop(self):
        """Gracefully stop the dev server."""

    def detect_project_type(self, project_dir: Path) -> str:
        """Detect project type from configuration files."""
```

**File changes:**
- NEW: `orchestrator/dev_server.py` (~250 lines)
- MODIFY: `orchestrator/cli.py` — add `dev` command
- MODIFY: `orchestrator/scaffold/` — add dev scripts to each template

#### Verification Gate

```bash
# Generate a FastAPI project, run dev, verify server starts
# Make a code change, verify auto-reload
# Check console output for errors
```

---

## Phase B7: Skills System for External AI Coding Agents

### Objective

Package orchestrator capabilities as reusable skill files (SKILL.md format) that external AI coding agents (Claude, Cursor, Copilot) can install and use to work with orchestrator-generated projects.

### Current State

- Orchestrator has its own agent system (`orchestrator/agents/`)
- No skills system for external AI tools
- `.claude/skills/mindmap/SKILL.md` exists in the repo for the mindmap skill
- Phase 8 (Skills System) in the Lovable plan adds internal skill loading
- No export mechanism for skills

### Implementation

#### B7.1 — Create skill packages for external AI tools

```
.orchestrator/skills/
├── orchestrator-code/           ← For working with generated code
│   └── SKILL.md
├── orchestrator-config/         ← For managing entity/auth/agent config
│   └── SKILL.md
├── orchestrator-deploy/         ← For deployment and CI/CD
│   └── SKILL.md
└── orchestrator-troubleshoot/   ← For debugging generated projects
    └── SKILL.md
```

**Example `orchestrator-code/SKILL.md`:**
```markdown
# Orchestrator Code Skill

## Description
Use when working with code generated by the Multi-LLM Orchestrator. The
project follows the architecture specified in .orchestrator-rules.yml.

## Instructions

### Code structure
The project follows Clean Architecture with these layers:
- `domain/` — Business logic, entities, value objects (zero external deps)
- `application/` — Use cases, DTOs, service interfaces
- `infrastructure/` — Database, external APIs, file system
- `presentation/` — FastAPI routes, request/response models

### Validation
Before submitting changes:
1. Run `pytest` — all tests must pass
2. Run `ruff check .` — no lint errors
3. Run `mypy src/` — strict type checking
4. Check `.orchestrator-rules.yml` for forbidden patterns

### Authorization
- All database queries must include row-level security checks
- JWT tokens include user role — check in endpoints
- Admin operations require `role='admin'` in JWT

### Common patterns
- Pydantic v2 for all data validation
- Async/await for all I/O operations
- Repository pattern for data access
- Dependency injection via FastAPI's Depends()
```

**Installation (for external agents):**
```bash
# Install orchestrator skills for Claude/Cursor/Copilot
npx skills add orchestrator/skills -g

# Install to a specific project
npx skills add orchestrator/skills
```

**File changes:**
- NEW: `.orchestrator/skills/orchestrator-code/SKILL.md`
- NEW: `.orchestrator/skills/orchestrator-config/SKILL.md`
- NEW: `.orchestrator/skills/orchestrator-deploy/SKILL.md`
- NEW: `.orchestrator/skills/orchestrator-troubleshoot/SKILL.md`
- MODIFY: `orchestrator/engine.py` — generate skills with every project

#### Verification Gate

```bash
# Create a project, verify skills directory exists
# Install skills in Claude, describe a task, verify agent uses orchestrator patterns
```

---

## Integration with Existing Plans

```
Phase B1 (Config-as-Code) ───────────────────────────────────────────────────┐
    │  No dependencies — pure output format enhancement                       │
Phase B2 (Dynamic Types) ─────────────────────────────────────────────────────┤
    │  Depends on Phase B1 (entity schemas required for type generation)      │
Phase B3 (Automation Generator) ──────────────────────────────────────────────┤
    │  No dependencies — independent scaffold enhancement                     │
Phase B4 (Push/Pull Config) ──────────────────────────────────────────────────┤
    │  Depends on Phase B1 (config-as-code output format)                     │
Phase B5 (Entity RLS) ────────────────────────────────────────────────────────┤
    │  Depends on Phase B1 (entity schemas required)                          │
Phase B6 (Local Dev Server) ──────────────────────────────────────────────────┤
    │  No dependencies — independent CLI enhancement                          │
Phase B7 (Skills System) ─────────────────────────────────────────────────────┘
    Depends on Phase 8 (Skills System from Lovable — internal skill registry)
    Also depends on Phase B1 (config-as-code — skills reference config format)
```

## Base44-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| B1 | Config-as-Code Output | 2 | 3 | ~400 | 2-3 |
| B2 | Dynamic Type Generation | 1 | 2 | ~350 | 1-2 |
| B3 | Automation Generator | 1 | 3 | ~350 | 2-3 |
| B4 | Push/Pull Config Workflow | 4 | 1 | ~300 | 2-3 |
| B5 | Entity Validator + RLS | 1 | 2 | ~250 | 2-3 |
| B6 | Local Dev Server | 1 | 2 | ~250 | 3-4 |
| B7 | Skills for External Agents | 4 | 1 | ~200 | 2-3 |
| **Base44 Subtotal** | **7 phases** | **14** | **14** | **~2,100** | **14-21** |

## Combined Grand Total (All Five Sources)

| Source | Phases | New Files | Modified Files | Est. Days |
|--------|--------|-----------|---------------|-----------|
| Replit | 1-6 | 5 | 14 | 13-18 |
| Lovable | 7-10 | 4 | 10 | 9-13 |
| UI | U1-U7 | 20 | 15 | 17-23 |
| Newly | N1-N7 | 2 | 21 | 10-14 |
| Base44 | B1-B7 | 14 | 14 | 14-21 |
| **Grand Total** | **31** | **45** | **74** | **63-89** |
