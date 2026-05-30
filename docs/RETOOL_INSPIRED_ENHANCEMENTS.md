# Enhancement Plan: Retool-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Retool and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements Replit, Lovable, Newly, Base44, v0, and UI enhancement plans

---

## Overview

Retool is an internal tool builder (not an AI code generator), with a fundamentally different architecture: 100+ prebuilt UI components, drag-and-drop IDE, real-time multiplayer collaboration, and managed app releases. While many features are platform-specific, Retool's **module system**, **app-level release management**, and **data source integration patterns** offer unique value for the orchestrator's generated project output.

Seven enhancements identified, focused on **modular project generation**, **managed releases**, **rich component libraries**, and **query generation**.

```
Phase R1: Module System for Generated Projects (Reusable Features)  (Highest ROI, 3-4 days)
Phase R2: App-Level Release Management (Semantic Versioning)          (High ROI, 2-3 days)
Phase R3: AI-Assisted Query Generation (SQL, JS, GraphQL)            (High ROI, 2-3 days)
Phase R4: Rich Component Library Generation (100+ Components)        (Medium ROI, 4-5 days)
Phase R5: Data Source Integration Templates                           (Medium ROI, 2-3 days)
Phase R6: Internationalization (i18n) for Generated Projects         (Lower ROI, 2-3 days)
Phase R7: App Documentation Generation (README + User Guide)         (Lower ROI, 1-2 days)
```

---

## What Retool Has — Quick Reference

| Retool Feature | Description | Translates? |
|---------------|-------------|:-----------:|
| **Module System** | Reusable component+query packages shared across apps | ✅ Phase R1 |
| **App Releases** | Semantic versioning, draft/published, diff, revert | ✅ Phase R2 |
| **Ask AI for Queries** | Generate, edit, explain, fix SQL/JS/GraphQL inline | ✅ Phase R3 |
| **100+ Components** | Table, Chart, Form, Map, Calendar, Agent Chat, etc. | ✅ Phase R4 |
| **Data Source Integration** | PostgreSQL, MongoDB, REST, GraphQL, 50+ integrations | ✅ Phase R5 |
| **Internationalization** | Built-in i18n for apps | ✅ Phase R6 |
| **AI-Generated README** | LLM evaluates app purpose, writes documentation | ✅ Phase R7 |
| **Multiplayer** | CRDT-based real-time collaborative editing | ✗ (Platform-specific) |
| **Drag-and-Drop IDE** | Visual app building interface | ✗ (Platform-specific) |
| **Embedded/External Apps** | Embed apps in external applications | ✗ (Platform-specific) |
| **Observability** | Datadog, Sentry, Fullstory integrations | Partial — telemetry exists |
| **App Themes** | Organization and app-level themes | Partial — Phase V1 (Design Registry) |

---

## Phase R1: Module System for Generated Projects (Reusable Features)

### Objective

When the orchestrator generates a project, decompose complex features into reusable **modules** — self-contained packages of components, queries, and logic that can be shared across multiple apps. Each module has defined inputs (data passed in) and outputs (data passed out), making them composable.

### Current State

- `orchestrator/scaffold/` — project templates generate monolithic codebases
- `CodebaseAnalyzer` — finds imports via AST but doesn't model component reusability
- No concept of reusable, composable feature packages
- Dependency resolution (`application/dependency_resolver.py`) handles task-level dependencies, not component-level

### Implementation

#### R1.1 — Create `orchestrator/module_generator.py`

```python
"""
Module Generator — generate reusable, composable feature packages.
===================================================================

A module is a self-contained package of:
- Components (UI + logic)
- Queries (database, API)
- Configuration (inputs/outputs)
- Tests

Modules are shared across generated projects via a module registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModuleInput:
    """Data or query input that a parent app passes into a module."""
    name: str
    type: str  # "string", "number", "boolean", "enum", "query", "any"
    description: str = ""
    default: Any = None
    required: bool = True
    enum_options: list[str] | None = None
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleOutput:
    """Data that a module exposes to its parent app."""
    name: str
    value_expression: str  # e.g., "{{ textInput1.value }}"
    description: str = ""
    type: str = "string"


@dataclass
class ModuleSpec:
    """Complete specification for a reusable module."""

    name: str  # e.g., "auth", "search", "notification", "payment"
    version: str  # Semantic version
    description: str
    category: str  # "auth", "data", "ui", "integration", "utility"

    # Inputs (what the parent app provides)
    inputs: list[ModuleInput] = field(default_factory=list)

    # Outputs (what the module exposes to the parent app)
    outputs: list[ModuleOutput] = field(default_factory=list)

    # Components (UI + logic files)
    components: dict[str, str] = field(default_factory=dict)  # filename → source

    # Queries (database/API queries)
    queries: dict[str, str] = field(default_factory=dict)  # name → SQL/JS/GraphQL

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Dependencies on other modules
    module_dependencies: list[str] = field(default_factory=list)

    # Package dependencies (npm/pip)
    package_dependencies: list[str] = field(default_factory=list)


class ModuleGenerator:
    """Generates reusable modules for projects.

    Common module templates:
    - auth/login — User authentication with login/signup forms
    - auth/permissions — Role-based access control
    - data/table — Sortable, filterable, paginated data table
    - data/chart — Chart generation from query results
    - data/search — Full-text search with autocomplete
    - data/export — CSV/PDF export
    - ui/navigation — Navigation bar with responsive menu
    - ui/notifications — Toast notifications
    - integration/email — Email sending (SendGrid, SMTP)
    - integration/payment — Payment processing (Stripe)
    - integration/storage — File uploads/downloads
    - utility/logging — Structured logging
    - utility/caching — Response caching
    """

    BUILTIN_MODULES: dict[str, ModuleSpec] = {
        "auth/login": ModuleSpec(
            name="auth/login",
            version="1.0.0",
            description="User authentication with login, signup, and password reset",
            category="auth",
            inputs=[
                ModuleInput(name="authConfig", type="any",
                    description="Authentication configuration (JWT secret, expiry, etc.)"),
            ],
            outputs=[
                ModuleOutput(name="currentUser", value_expression="{{ user }}",
                    description="Currently authenticated user object"),
                ModuleOutput(name="isAuthenticated", value_expression="{{ !!user }}",
                    description="Whether a user is currently authenticated"),
            ],
            components={
                "LoginForm.tsx": "// Login form component with email/password fields",
                "SignupForm.tsx": "// Signup form with validation",
                "AuthProvider.tsx": "// Context provider for auth state",
                "useAuth.ts": "// Hook for accessing auth state",
            },
            queries={
                "loginUser": "// POST /api/auth/login",
                "registerUser": "// POST /api/auth/register",
                "refreshToken": "// POST /api/auth/refresh",
            },
            package_dependencies=["jsonwebtoken", "bcrypt"],
        ),

        "data/table": ModuleSpec(
            name="data/table",
            version="1.0.0",
            description="Sortable, filterable, server-side paginated data table",
            category="data",
            inputs=[
                ModuleInput(name="dataSource", type="query",
                    description="Query that returns tabular data"),
                ModuleInput(name="columns", type="any",
                    description="Column definitions with sort/filter configuration"),
            ],
            outputs=[
                ModuleOutput(name="selectedRow", value_expression="{{ selectedItem }}",
                    description="Currently selected row data"),
                ModuleOutput(name="filteredData", value_expression="{{ filteredResults }}",
                    description="Filtered and sorted dataset"),
            ],
            components={
                "DataTable.tsx": "// Full-featured data table",
                "TableColumns.tsx": "// Column configuration",
                "TableFilters.tsx": "// Filter UI",
                "Pagination.tsx": "// Server-side pagination",
            },
            queries={
                "fetchPage": "// SELECT * FROM ? LIMIT ? OFFSET ?",
            },
        ),

        "integration/payment": ModuleSpec(
            name="integration/payment",
            version="1.0.0",
            description="Payment processing with Stripe integration",
            category="integration",
            inputs=[
                ModuleInput(name="stripeConfig", type="any",
                    description="Stripe API keys and configuration"),
                ModuleInput(name="products", type="any",
                    description="Product/price definitions"),
            ],
            outputs=[
                ModuleOutput(name="paymentStatus", value_expression="{{ status }}",
                    description="Current payment status"),
            ],
            components={
                "CheckoutForm.tsx": "// Stripe Elements checkout form",
                "PaymentHistory.tsx": "// Payment history table",
                "SubscriptionManager.tsx": "// Subscription management UI",
            },
            queries={
                "createPaymentIntent": "// POST /api/payments/create-intent",
                "confirmPayment": "// POST /api/payments/confirm",
            },
            package_dependencies=["stripe"],
        ),
    }

    def generate_module(self, spec: ModuleSpec, output_dir: Path) -> None:
        """Generate a module's source code to the project directory.

        Structure:
        modules/{category}/{name}/
            index.ts          (or __init__.py)
            components/       (UI components)
            queries/          (database/API queries)
            config.json       (module metadata)
            README.md         (usage documentation)
            tests/            (module tests)
        """

    def compose_project(self, modules: list[str], output_dir: Path) -> None:
        """Generate a project composed of multiple modules.

        Resolves module dependencies and wire inputs/outputs.
        """

    def discover_modules(self, project_description: str) -> list[str]:
        """Use the LLM to discover which modules the project needs.

        Example output for "Build an e-commerce admin dashboard":
        ["auth/login", "data/table", "data/chart", "data/search",
         "ui/navigation", "utility/logging"]
        """

    def wire_modules(self, module_specs: list[ModuleSpec]) -> dict[str, str]:
        """Wire module inputs and outputs together.

        Example: auth/login.currentUser → data/table.dataSource (filtered by user)
        """
```

**Module directory structure in generated project:**
```
my-project/
├── modules/
│   ├── auth/
│   │   └── login/
│   │       ├── components/
│   │       │   ├── LoginForm.tsx
│   │       │   ├── SignupForm.tsx
│   │       │   └── AuthProvider.tsx
│   │       ├── queries/
│   │       │   └── authQueries.ts
│   │       ├── hooks/
│   │       │   └── useAuth.ts
│   │       ├── config.json
│   │       └── README.md
│   ├── data/
│   │   ├── table/
│   │   └── chart/
│   └── integration/
│       └── payment/
├── src/
│   ├── app.tsx              ← Wires modules together
│   └── module-registry.ts   ← Module import/export registry
└── .orchestrator/
    └── modules.yml           ← Module manifest
```

**File changes:**
- NEW: `orchestrator/module_generator.py` (~500 lines)
- NEW: `orchestrator/modules/` — built-in module templates
- MODIFY: `orchestrator/engine.py` — call `ModuleGenerator` during project composition
- MODIFY: `orchestrator/scaffold/` — include module-based project structure

#### Verification Gate

```bash
# Generate a project that needs auth + data table + charts
# Verify: modules/auth/login, modules/data/table, modules/data/chart directories exist
# Verify: Module inputs/outputs are wired correctly
# Verify: project builds and runs with all modules integrated
```

---

## Phase R2: App-Level Release Management (Semantic Versioning)

### Objective

Add semantic versioning support for generated projects — create releases with MAJOR.MINOR.PATCH versioning, draft/published state, diff comparison between releases, and revert capability. Extends the Version Manager (Phase V6) with release management concepts.

### Current State

- Phase V6 (Versions) — auto-version every code change with diffs and reverts
- `GitIntegration` — git-based branching for milestones
- No concept of "releases" — just development versions
- No semantic versioning for generated projects

### Implementation

#### R2.1 — Extend `VersionManager` with release support

```python
@dataclass
class Release:
    """A published version of the generated project."""
    version: str  # Semantic version: "1.2.3"
    version_type: str  # "major", "minor", "patch"
    description: str
    author: str
    published_at: float
    is_live: bool  # Currently published to users
    state: str  # "draft" or "published"
    diff_from_previous: str | None  # Diff since last release
    metadata: dict[str, Any] = field(default_factory=dict)

class ReleaseManager:
    """Manages semantic versioning and releases for generated projects.

    Features:
    - Semantic versioning (MAJOR.MINOR.PATCH)
    - Draft releases (create, test, then publish)
    - Release comparison (diff between any two releases)
    - Release reverting (unpublish, revert to previous)
    - Release URL access (_releaseVersion=1.2.3)
    - Release history timeline
    - Release notes generation (AI-powered changelog)
    """

    def __init__(self, output_dir: Path):
        self._releases: list[Release] = []
        self._current_live: Release | None = None
        self._drafts: list[Release] = []

    def create_draft(
        self,
        version_type: str = "patch",
        description: str = "",
    ) -> Release:
        """Create a draft release (not yet published)."""

    def publish(self, release: Release) -> None:
        """Publish a release — becomes the live version for all users."""

    def unpublish(self, release: Release) -> None:
        """Unpublish a release — restores latest dev version as live."""

    def compare(self, base: str, compare: str) -> str:
        """Compare two releases and return the diff."""

    def revert_to(self, version: str) -> None:
        """Revert the working version to a previous release state."""

    def generate_release_notes(self, release: Release, client: UnifiedClient) -> str:
        """Generate human-readable release notes using the LLM.

        Analyzes the diff and produces a changelog:
        - New features
        - Bug fixes
        - Breaking changes
        - Dependency updates
        """

    def timeline(self) -> list[dict]:
        """Get release timeline for UI display."""

    def increment_version(
        self, current: str, increment_type: str
    ) -> str:
        """Increment semantic version.
        
        major: 1.2.3 → 2.0.0
        minor: 1.2.3 → 1.3.0
        patch: 1.2.3 → 1.2.4
        """
        major, minor, patch = map(int, current.split("."))
        if increment_type == "major":
            return f"{major + 1}.0.0"
        elif increment_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"
```

**CLI commands:**
```bash
# Create a release
python -m orchestrator release create --type minor --description "Added search feature"

# Publish a release
python -m orchestrator release publish 1.3.0

# Compare two releases
python -m orchestrator release diff 1.2.0 1.3.0

# List releases
python -m orchestrator release list

# Generate release notes
python -m orchestrator release notes 1.3.0

# Revert to a previous release
python -m orchestrator release revert 1.2.0

# Run project at a specific release version (deployment artifact)
python -m orchestrator release serve 1.3.0
```

**File changes:**
- NEW: `orchestrator/release_manager.py` (~350 lines)
- MODIFY: `orchestrator/version_manager.py` — integrate release tracking
- MODIFY: `orchestrator/cli.py` — add `release` command group
- DEPENDS ON: Phase V6 (Versions — version infrastructure)

#### Verification Gate

```bash
# Create 3 releases, publish v1.1.0, compare with v1.0.0
# Verify: semantic versioning, diff generation, draft workflow
# Generate release notes, verify AI-produced changelog
```

---

## Phase R3: AI-Assisted Query Generation (SQL, JS, GraphQL)

### Objective

When the orchestrator generates database queries, API calls, or GraphQL operations, use an LLM (with access to the database schema) to generate, explain, and validate queries. Similar to Retool's "Ask AI" for queries.

### Current State

- `ArchitectureAnalyzer._detect_database()` — detects database type
- Scaffold templates generate static queries (hardcoded SQL patterns)
- No LLM-assisted query generation from natural language
- No query validation against schema

### Implementation

#### R3.1 — Create `orchestrator/query_generator.py`

```python
"""
AI-Assisted Query Generator — natural language → SQL, JS, GraphQL.
====================================================================

Generates database queries, API calls, and GraphQL operations from
natural language descriptions, with schema awareness for accuracy.
"""

@dataclass
class QuerySpec:
    language: str  # "sql", "javascript", "graphql", "python"
    operation: str  # "SELECT", "INSERT", "UPDATE", "DELETE", "FUNCTION", "QUERY", "MUTATION"
    description: str  # Natural language description
    schema_context: str  # Relevant schema (tables, types, endpoints)
    parameters: list[dict] | None = None  # Query parameters

@dataclass
class GeneratedQuery:
    code: str  # The generated query code
    explanation: str  # Human-readable explanation
    validation: QueryValidation  # Validation results
    estimated_cost: float  # LLM cost for generation

@dataclass
class QueryValidation:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    schema_checks: dict[str, bool]  # table/field existence checks


class QueryGenerator:
    """Generates and validates queries using AI.

    Capabilities:
    - Generate: Natural language → query code
    - Edit: Modify existing query from instructions
    - Explain: Add comments explaining query logic
    - Fix: Identify and resolve query errors
    - Optimize: Suggest performance improvements
    """

    def __init__(self, client: UnifiedClient):
        self._client = client
        self._schema_cache: dict[str, str] = {}

    async def generate(
        self,
        spec: QuerySpec,
        language: str = "sql",
    ) -> GeneratedQuery:
        """Generate a query from a natural language description.

        Args:
            spec: Query specification with description and schema context
            language: Target language (sql, javascript, graphql, python)

        Returns:
            GeneratedQuery with code, explanation, and validation
        """

    async def edit(
        self,
        existing_query: str,
        instruction: str,
        language: str = "sql",
    ) -> GeneratedQuery:
        """Edit an existing query based on natural language instructions."""

    async def explain(self, query: str, language: str = "sql") -> str:
        """Add comments explaining what each part of the query does."""

    async def fix(self, query: str, error: str, language: str = "sql") -> GeneratedQuery:
        """Fix a query that has errors."""

    async def optimize(
        self, query: str, language: str = "sql"
    ) -> list[str]:
        """Suggest performance optimizations for a query."""

    def _build_schema_context(self, tables: list[str]) -> str:
        """Build schema context from table definitions.

        Includes: column names, types, constraints, indexes, relationships.
        """

    def _validate_against_schema(
        self, query: str, schema: dict
    ) -> QueryValidation:
        """Validate a query against the schema.

        Checks: table existence, column existence, type compatibility,
        constraint violations, missing indexes.
        """
```

**Example usage during project generation:**
```python
# The orchestrator detects the database from the project description
schema_context = QueryGenerator._build_schema_context([
    "users (id, email, name, role, created_at)",
    "orders (id, user_id, total, status, created_at)",
    "order_items (id, order_id, product_id, quantity, price)",
    "products (id, name, category, price, stock)",
])

# Generate queries for the project
queries = {
    "getOrdersByUser": await query_gen.generate(
        QuerySpec(
            language="sql",
            operation="SELECT",
            description="Get all orders for a specific user, sorted by date, with items",
            schema_context=schema_context,
            parameters=[{"name": "user_id", "type": "uuid"}],
        )
    ),
    "getTopProducts": await query_gen.generate(
        QuerySpec(
            language="sql",
            operation="SELECT",
            description="Get top 10 products by total revenue, with category information",
            schema_context=schema_context,
        )
    ),
}

# Each query is validated against the schema and includes explanations
for name, query in queries.items():
    if not query.validation.is_valid:
        print(f"Query {name} has errors: {query.validation.errors}")
    else:
        print(f"Generated: {query.code}")
        print(f"Explanation: {query.explanation}")
```

**File changes:**
- NEW: `orchestrator/query_generator.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — generate queries during project composition
- MODIFY: `orchestrator/scaffold/` — use generated queries instead of static templates
- DEPENDS ON: Phase B1 (Config-as-Code) for entity schemas
- DEPENDS ON: Phase B2 (Dynamic Types) for type-aware query generation

#### Verification Gate

```bash
# Generate a project with "user orders" and "top products"
# Verify: generated SQL queries are valid against the schema
# Verify: queries include explanatory comments
# Verify: query generation cost is logged
```

---

## Phase R4: Rich Component Library Generation (100+ Components)

### Objective

Generate a rich library of 100+ UI components tailored to the project's technology stack and design system. Components include data display (Table, Chart, Calendar, Timeline), forms (all input types, validation), navigation, feedback, and integration components.

### Current State

- Scaffold templates generate basic components (Button, Input, Card)
- No component catalog — each project generates components from scratch
- Phase V1 (Design Registry) provides component specifications
- Phase V7 (Templates) provides page templates

### Implementation

#### R4.1 — Create component catalog

```python
"""
Component Library — 100+ prebuilt UI components for generated projects.
========================================================================

Components are organized by category and generated with:
- Framework-specific code (React, Vue, Svelte)
- CSS framework (Tailwind, CSS Modules, styled-components)
- Design system tokens (from Phase V1 registry)
- TypeScript types (from Phase B2 type generation)
- Accessibility (ARIA attributes, keyboard navigation)
"""

class ComponentCatalog:
    """Catalog of 100+ components for various stacks.

    Categories:
    - buttons: Button, ButtonGroup, DropdownButton, SplitButton, ToggleButton, Link, LinkList
    - charts: BarChart, LineChart, PieChart, ScatterChart, BubbleChart, HeatMap,
              FunnelChart, SankeyChart, Sparkline, Treemap, WaterfallChart
    - containers: Card, Container, Stack, Tabs, Accordion, Collapsible, SteppedContainer
    - data: DataTable, JSONExplorer, KeyValue, ReorderableList, Filter
    - forms: Form, JSONSchemaForm, TextInput, NumberInput, Currency, Percent,
             Select, MultiSelect, Checkbox, Radio, Switch, Slider, Rating,
             DatePicker, DateRange, TimePicker, FileUpload, ColorPicker
    - feedback: Alert, Toast, Modal, Drawer, Tooltip, Popover, Badge,
                ProgressBar, Skeleton, Spinner, EmptyState
    - navigation: Navbar, Sidebar, Breadcrumbs, Pagination, Steps, Tabs, Wizard
    - presentation: Avatar, AvatarGroup, Image, Icon, Divider, Statistic,
                    Timeline, Calendar, Map, Video, QRCode
    - integration: AuthButton, StripeForm, PaymentHistory, ChatWidget,
                   SearchBar, ExportButton, ImportButton
    - utility: CopyToClipboard, KeyboardShortcuts, CommandPalette,
               ThemeSwitcher, LanguageSwitcher
    """

    COMPONENT_CATALOG: dict[str, dict] = {
        # Data display components
        "DataTable": {
            "category": "data",
            "features": ["sort", "filter", "paginate", "select", "export", "inline-edit"],
            "variants": ["basic", "server-paginated", "virtualized", "tree"],
            "frameworks": ["react", "vue", "svelte"],
            "dependencies": ["@tanstack/react-table"],
        },
        "Chart": {
            "category": "charts",
            "features": ["bar", "line", "pie", "scatter", "area", "mixed"],
            "variants": ["recharts", "chartjs", "echarts"],
            "frameworks": ["react", "vue"],
            "dependencies": ["recharts"],
        },

        # Input components
        "AutoComplete": {
            "category": "forms",
            "features": ["multi-select", "search", "debounce", "async-options"],
            "variants": ["basic", "multi", "creatable", "async"],
            "frameworks": ["react", "vue"],
            "dependencies": [],
        },

        # Feedback components
        "CommandPalette": {
            "category": "utility",
            "features": ["search", "keyboard-navigation", "categories", "recent"],
            "variants": ["basic", "with-actions"],
            "frameworks": ["react"],
            "dependencies": ["cmdk"],
        },

        # ... 96+ more component definitions
    }

    def generate_component(
        self,
        component_name: str,
        framework: str = "react",
        css_framework: str = "tailwind",
        design_tokens: DesignTokens | None = None,
        variant: str = "basic",
    ) -> dict[str, str]:
        """Generate a single component with framework-specific code.

        Returns: {filename: source_code} mapping.
        """

    def generate_full_library(
        self,
        framework: str = "react",
        selected_components: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Generate all selected components (or entire catalog).

        Returns: {component_name: {filename: source_code}} mapping.
        """

    def generate_storybook(
        self,
        components: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        """Generate Storybook stories for all components."""

    def discover_required_components(
        self, project_description: str, client: UnifiedClient
    ) -> list[str]:
        """Use LLM to discover which components the project needs.

        Example for "Build an e-commerce dashboard":
        → ["DataTable", "LineChart", "PieChart", "Filter",
           "SearchInput", "Modal", "Toast", "Statistic",
           "DateRange", "ExportButton", "Pagination"]
        """
```

**Generated component example (`DataTable.tsx`):**
```tsx
// Auto-generated DataTable component
// Category: data | Framework: react | CSS: tailwind | Tokens: acme-corp

import { useMemo, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';

interface DataTableProps<T> {
  /** Array of data objects to display */
  data: T[];
  /** Column definitions with sort and filter configuration */
  columns: ColumnDef<T>[];
  /** Enable row selection with checkboxes */
  selectable?: boolean;
  /** Enable CSV/JSON export */
  exportable?: boolean;
  /** Enable inline row editing */
  editable?: boolean;
  /** Server-side pagination config */
  pagination?: {
    pageIndex: number;
    pageSize: number;
    pageCount: number;
    onPageChange: (page: number) => void;
  };
}

export function DataTable<T>({
  data,
  columns,
  selectable = false,
  exportable = false,
  editable = false,
  pagination,
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    ...(pagination ? {
      getPaginationRowModel: getPaginationRowModel(),
      manualPagination: true,
      pageCount: pagination.pageCount,
    } : {}),
    state: { sorting },
    onSortingChange: setSorting,
  });

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <table className="w-full">
        <thead>
          {/* Column headers with sort indicators */}
        </thead>
        <tbody>
          {/* Data rows */}
        </tbody>
      </table>
      {pagination && <PaginationControls {...pagination} />}
    </div>
  );
}

export { DataTable, type DataTableProps };
```

**File changes:**
- NEW: `orchestrator/component_catalog.py` (~600 lines)
- NEW: `orchestrator/components/` — component source templates
- MODIFY: `orchestrator/scaffold/` — generate from catalog instead of static templates
- MODIFY: `orchestrator/type_generator.py` — generate TypeScript types for components
- DEPENDS ON: Phase V1 (Design Registry — design tokens for styling)
- DEPENDS ON: Phase B2 (Dynamic Types — component prop types)

#### Verification Gate

```bash
# Generate a dashboard project, verify 15+ components generated
# Check: DataTable with sort/filter/paginate, Charts, Forms, Navigation
# Verify: components use correct design system tokens
# Verify: Storybook stories generated for each component
```

---

## Phase R5: Data Source Integration Templates

### Objective

When the orchestrator detects database or API usage in the project description, generate connection code and configuration for the appropriate data source (PostgreSQL, MongoDB, REST APIs, GraphQL, Redis, S3, etc.) with best-practice patterns.

### Current State

- `ArchitectureAnalyzer._detect_database()` — detects database type
- `ArchitectureAnalyzer._detect_integration()` — detects integration keywords
- Scaffolds generate static connection code
- No connection pooling, retry logic, or health checks in generated code

### Implementation

#### R5.1 — Create data source integration templates

```python
"""
Data Source Integration — generate connection code for 50+ data sources.
==========================================================================

Each integration includes:
- Connection setup with pooling and retry logic
- Health check endpoint
- Query/operation patterns
- Environment variable configuration
- Documentation
"""

DATA_SOURCE_TEMPLATES = {
    "postgresql": {
        "python": {
            "driver": "asyncpg",
            "connection_pool": "asyncpg.create_pool()",
            "orm": "SQLAlchemy 2.0 (async)",
            "migrations": "Alembic",
            "patterns": ["repository", "unit-of-work", "connection-pool"],
        },
        "typescript": {
            "driver": "pg",
            "connection_pool": "pg.Pool",
            "orm": "Drizzle ORM",
            "migrations": "Drizzle Kit",
            "patterns": ["repository", "connection-pool"],
        },
    },
    "mongodb": {
        "python": {
            "driver": "motor (async)",
            "odm": "Beanie",
            "patterns": ["repository", "aggregation-pipeline"],
        },
        "typescript": {
            "driver": "mongodb",
            "odm": "Mongoose",
            "patterns": ["repository", "schema-validation"],
        },
    },
    "redis": {
        "both": {
            "driver": "ioredis (TS) / redis-py (Python)",
            "patterns": ["cache-aside", "rate-limiter", "session-store", "pub-sub"],
        },
    },
    "s3": {
        "both": {
            "driver": "@aws-sdk/client-s3 / boto3",
            "patterns": ["presigned-urls", "multipart-upload", "cdn-integration"],
        },
    },
    "rest-api": {
        "both": {
            "driver": "fetch / aiohttp",
            "patterns": ["api-client", "retry-policy", "circuit-breaker", "cache"],
        },
    },
    "graphql": {
        "both": {
            "driver": "@apollo/client / gql",
            "patterns": ["query-batching", "fragment-colocation", "optimistic-updates"],
        },
    },
    # ... 44+ more integration templates
}

class IntegrationGenerator:
    """Generates data source integration code for projects."""

    def generate_connection(
        self,
        data_source: str,
        framework: str = "python",
    ) -> dict[str, str]:
        """Generate connection setup code for a data source.

        Includes:
        - Connection pool configuration
        - Retry/backoff policy
        - Health check
        - Environment variables (.env.example)
        - Docker Compose (if applicable)
        """

    def detect_required_integrations(
        self, project_description: str
    ) -> list[str]:
        """Detect which data sources the project needs.

        Uses keyword matching against the project description.
        """

    def generate_integration_docs(
        self, data_source: str
    ) -> str:
        """Generate documentation for using the data source.

        Includes: setup steps, connection string format, common patterns,
        troubleshooting, security best practices.
        """
```

**Generated connection code example (Python + PostgreSQL):**
```python
# Auto-generated PostgreSQL integration
# Data source: PostgreSQL | Driver: asyncpg | Pattern: connection pool

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from asyncpg import Pool

# Connection configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/dbname"
)

POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))

# Global pool instance
_pool: Pool | None = None


async def create_pool() -> Pool:
    """Create a connection pool with retry logic."""
    global _pool
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_TIMEOUT,
        command_timeout=60,
        max_inactive_connection_lifetime=300,
    )
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    """Get a connection from the pool with automatic release."""
    pool = await create_pool()
    async with pool.acquire() as conn:
        try:
            yield conn
        except asyncpg.PostgresError as e:
            # Log the error, re-raise for caller handling
            raise


async def health_check() -> bool:
    """Check database connectivity."""
    try:
        async with get_connection() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
```

**File changes:**
- NEW: `orchestrator/integration_generator.py` (~400 lines)
- NEW: `orchestrator/integrations/` — 50+ data source templates
- MODIFY: `orchestrator/scaffold/` — use generated integration code
- MODIFY: `orchestrator/engine.py` — detect and generate integrations during project analysis

#### Verification Gate

```bash
# Generate a project with PostgreSQL + Redis + S3
# Verify: connection pool code, health check, env config, Docker Compose
# Verify: each integration has documentation
```

---

## Phase R6: Internationalization (i18n) for Generated Projects

### Objective

When the orchestrator detects a need for multi-language support, generate i18n infrastructure — translation files, language detection, locale switching components, and number/date formatting.

### Current State

- No i18n support in generated projects
- All generated UI text is hardcoded in the source language

### Implementation

```python
class I18nGenerator:
    """Generates internationalization infrastructure for projects.

    Features:
    - Translation key extraction from UI components
    - Translation file generation (JSON format)
    - Language detection (browser, URL, user preference)
    - Locale-specific formatting (dates, numbers, currencies)
    - RTL language support (Arabic, Hebrew, etc.)
    - Translation management UI (optional)
    """

    SUPPORTED_LOCALES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh",
        "ja", "ko", "ar", "he", "hi", "tr", "nl", "pl",
    ]

    def generate_i18n_setup(self, locales: list[str]) -> dict[str, str]:
        """Generate i18n infrastructure for specified locales.

        Generates:
        - i18n/config.ts — configuration
        - i18n/locales/{locale}.json — translation files
        - i18n/useTranslation.ts — translation hook
        - i18n/LanguageSwitcher.tsx — locale switching component
        - i18n/formatters.ts — date/number/currency formatters
        """

    def extract_translation_keys(
        self, components: dict[str, str]
    ) -> list[str]:
        """Extract translatable text from component source code.

        Patterns to detect:
        - Text content: <button>Login</button>
        - Labels: aria-label="Close menu"
        - Placeholders: placeholder="Enter email"
        - Alt text: alt="User avatar"
        """

    async def translate_keys(
        self, keys: list[str], target_locale: str, client: UnifiedClient
    ) -> dict[str, str]:
        """Use LLM to translate keys to a target locale.

        Provides context for accurate translations (e.g., "Save" in a
        button vs. "Save" in a file dialog).
        """
```

**File changes:**
- NEW: `orchestrator/i18n_generator.py` (~250 lines)
- MODIFY: `orchestrator/scaffold/` — include i18n infrastructure
- MODIFY: `orchestrator/component_catalog.py` — use translation keys in components

#### Verification Gate

```bash
# Generate a project with i18n enabled (es, fr, de)
# Verify: translation files exist for all locales
# Verify: components use translation keys, not hardcoded text
# Switch locale, verify UI text changes
```

---

## Phase R7: App Documentation Generation (README + User Guide)

### Objective

Generate comprehensive documentation for generated projects — a README with architecture overview, setup instructions, and deployment guide, plus user-facing documentation. Uses the LLM to evaluate the app's purpose and functionality.

### Current State

- `orchestrator/scaffold/` templates include basic README.md
- No AI-generated documentation
- No user-facing documentation for generated apps
- `PreflightValidator` checks for missing documentation but doesn't generate it

### Implementation

#### R7.1 — Create `orchestrator/doc_generator.py`

```python
"""
Documentation Generator — AI-powered README and user guides.
==============================================================

Generates:
- README.md — architecture, setup, deployment, API reference
- USER_GUIDE.md — user-facing documentation with screenshots
- API_REFERENCE.md — endpoint documentation
- CHANGELOG.md — release history
- ARCHITECTURE.md — design decisions and rationale
"""

class DocGenerator:
    """Generates comprehensive documentation for projects.

    Uses the LLM with full project context to produce accurate,
    well-structured documentation.
    """

    async def generate_readme(
        self, project_description: str, architecture: ProjectRules,
        tasks: list[Task], results: list[TaskResult],
    ) -> str:
        """Generate a comprehensive README.md.

        Sections:
        - Project overview (from project description)
        - Architecture (from ProjectRules)
        - Technology stack (from detected stack)
        - Setup instructions (from scaffold)
        - Environment variables (from integration generator)
        - Development guide (from scaffold)
        - Deployment guide (from scaffold)
        - API reference (from generated endpoints)
        - Testing (from generated test suite)
        - Contributing (template)
        - License (template)
        """

    async def generate_user_guide(
        self, project_description: str, results: list[TaskResult],
    ) -> str:
        """Generate a user-facing guide.

        Sections:
        - Getting started
        - Key features (from task descriptions)
        - How to use each feature
        - FAQs
        - Troubleshooting
        """

    async def generate_api_reference(
        self, endpoints: list[dict],
    ) -> str:
        """Generate API reference documentation.

        Sections:
        - Authentication
        - Endpoints (method, path, description, parameters, response)
        - Error codes
        - Rate limiting
        - Examples
        """

    async def generate_changelog(
        self, releases: list[Release],
    ) -> str:
        """Generate a changelog from releases.

        Format: Keep a Changelog (https://keepachangelog.com)
        """

    async def generate_all(
        self, context: ProjectContext,
    ) -> dict[str, str]:
        """Generate all documentation files."""
        return {
            "README.md": await self.generate_readme(...),
            "docs/USER_GUIDE.md": await self.generate_user_guide(...),
            "docs/API_REFERENCE.md": await self.generate_api_reference(...),
            "CHANGELOG.md": await self.generate_changelog(...),
            "ARCHITECTURE.md": context.architecture_doc,
        }
```

**File changes:**
- NEW: `orchestrator/doc_generator.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — generate docs after project completion
- MODIFY: `orchestrator/scaffold/` — include docs/ directory structure

#### Verification Gate

```bash
# Generate a full-stack project, verify documentation completeness
# Check: README.md has all sections, USER_GUIDE.md is user-friendly
# Check: API reference matches generated endpoints
# Check: Changelog matches release history
```

---

## Integration with Existing Plans

```
Phase R1 (Module System) ─────────────────────────────────────────────────────┐
    │  No dependencies — independent scaffold enhancement                      │
Phase R2 (Release Management) ─────────────────────────────────────────────────┤
    │  Depends on Phase V6 (Versions — version infrastructure)                 │
Phase R3 (Query Generation) ───────────────────────────────────────────────────┤
    │  Depends on Phase B1 (Config-as-Code — entity schemas for context)       │
    │  Depends on Phase B2 (Dynamic Types — type-aware generation)             │
Phase R4 (Component Library) ──────────────────────────────────────────────────┤
    │  Depends on Phase V1 (Design Registry — design tokens for styling)       │
    │  Depends on Phase B2 (Dynamic Types — component prop types)              │
Phase R5 (Data Source Integration) ────────────────────────────────────────────┤
    │  No dependencies — independent scaffold enhancement                      │
Phase R6 (Internationalization) ───────────────────────────────────────────────┤
    │  Depends on Phase R4 (Component Library — components to translate)       │
Phase R7 (Documentation) ──────────────────────────────────────────────────────┘
    No dependencies — uses existing project context
```

## Retool-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| R1 | Module System | 2 | 3 | ~500 | 3-4 |
| R2 | Release Management | 1 | 2 | ~350 | 2-3 |
| R3 | Query Generation | 1 | 2 | ~400 | 2-3 |
| R4 | Component Library | 2 | 3 | ~600 | 4-5 |
| R5 | Data Source Integration | 2 | 2 | ~400 | 2-3 |
| R6 | Internationalization | 1 | 2 | ~250 | 2-3 |
| R7 | Documentation | 1 | 2 | ~300 | 1-2 |
| **Retool Subtotal** | **7 phases** | **10** | **16** | **~2,800** | **16-23** |

## Combined Grand Total (All Seven Sources)

| # | Source | Phases | New Files | Modified Files | Est. Days |
|---|--------|--------|-----------|---------------|-----------|
| 1 | Replit | 1-6 | 5 | 14 | 13-18 |
| 2 | Lovable | 7-10 | 4 | 10 | 9-13 |
| 3 | UI | U1-U7 | 20 | 15 | 17-23 |
| 4 | Newly | N1-N7 | 2 | 21 | 10-14 |
| 5 | Base44 | B1-B7 | 14 | 14 | 14-21 |
| 6 | v0 | V1-V7 | 12 | 19 | 18-26 |
| 7 | Retool | R1-R7 | 10 | 16 | 16-23 |
| **Grand Total** | **45** | **67** | **109** | **97-138** |
