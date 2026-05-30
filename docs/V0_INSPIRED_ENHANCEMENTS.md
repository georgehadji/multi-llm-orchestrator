# Enhancement Plan: v0-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between v0 by Vercel and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements Replit, Lovable, Newly, Base44, and UI enhancement plans

---

## Overview

v0 by Vercel is the most complete AI development platform we've analyzed — combining code generation, visual design, agent capabilities, design systems, and deployment. Its core differentiator is the **design system registry** (shadcn/ui format) and **visual design mode** with fine-grained element controls. It also has the most advanced agent autonomy of any platform, with browser use, web search, automatic error fixing, and MCP integrations.

Seven enhancements identified, focused on **structured design systems**, **visual design controls**, **agent autonomy**, and **version management**.

```
Phase V1: Structured Design System Registry (shadcn/ui format)   (Highest ROI, 3-4 days)
Phase V2: Visual Design Panel Controls (typography, layout, etc)   (High ROI, 3-4 days)
Phase V3: Browser-Use Agent (autonomous app testing + debugging)   (High ROI, 3-4 days)
Phase V4: Permission Modes (Ask/Auto/Full for tool execution)      (High ROI, 1-2 days)
Phase V5: Auto-Error Fix Button (one-click fix from error logs)    (Medium ROI, 2-3 days)
Phase V6: Versions as First-Class Concept (diff, review, revert)   (Medium ROI, 3-4 days)
Phase V7: Templates + Registry Marketplace                         (Lower ROI, 3-5 days)
```

---

## What v0 Has — Quick Reference

| v0 Feature | Description | Translates? |
|-----------|-------------|:-----------:|
| **Design Registry** | shadcn/ui format for sharing components, blocks, tokens with AI | ✅ Phase V1 |
| **Design Mode** | Visual element panel: typography, color, layout, border, shadow | ✅ Phase V2 |
| **Browser Use Agent** | Opens app, uses it, debugs, sends screenshots | ✅ Phase V3 |
| **Permission Modes** | Ask/Auto/Full for terminal commands | ✅ Phase V4 |
| **Fix with v0** | One-click error fix from deployment logs | ✅ Phase V5 |
| **Versions** | Diff, review, revert for every change | ✅ Phase V6 |
| **Templates** | Ready-made components + full-page designs | ✅ Phase V7 |
| **Isolated Sandbox** | Per-chat VM with Node.js, dev server, terminal | Partial — Phase 4 (Sandbox Tasks) |
| **Web Search Agent** | Real-time web search with citation links | Partial — Nexus Search exists |
| **MCP Integrations** | Marketplace + custom MCP servers | Partial — MCP server exists |
| **Real-time Feedback** | Progress indicators, screenshots, citations | Partial — Phase U6 (Generation Progress) |
| **Figma Import** | Direct Figma → code conversion | ✗ (Platform-specific) |
| **Vercel Deploy** | One-click production deployment | ✗ (Platform-specific) |
| **Pre-installed Agents** | Slack, other first-class integrations | ✗ (Platform-specific) |

---

## Phase V1: Structured Design System Registry (shadcn/ui Format)

### Objective

Adopt the shadcn/ui registry format for design systems — a structured distribution specification that passes design context (components, blocks, tokens) to AI models. This is the most advanced design system approach across all 5 platforms analyzed.

### Current State

- `orchestrator/ux/design_enhancer.py` — 20 generic WCAG/UX standards
- Phase 6 (Design System Injection) — `.design-system.yml` with brand tokens
- Phase 10 (Design System Projects) — design system as a dedicated project
- `orchestrator/scaffold/` — project templates with static code
- No registry format for sharing components/blocks/tokens with AI models

### Implementation

#### V1.1 — Create `orchestrator/design_registry.py`

```python
"""
Design System Registry — shadcn/ui-compatible component distribution.
========================================================================

Adopts the shadcn/ui registry format for distributing design systems:
- registry.json — core manifest of components, blocks, and tokens
- Token system — CSS variables for colors, typography, spacing, shadows
- Block system — prebuilt page compositions (dashboard, store, landing)
- Component system — UI primitives (Button, Input, Card, Modal, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RegistryItem:
    """A single item in the design registry.

    Compatible with the shadcn/ui registry-item specification.
    """
    name: str
    type: str  # "registry:component", "registry:block", "registry:ui", "registry:lib"
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    registryDependencies: list[str] = field(default_factory=list)
    files: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_registry_json(self) -> dict:
        """Serialize to registry-item format."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "dependencies": self.dependencies,
            "registryDependencies": self.registryDependencies,
            "files": self.files,
            "meta": self.meta,
        }


@dataclass
class DesignTokens:
    """CSS variable-based design tokens (shadcn/ui format)."""

    # Colors (HSL values)
    background: str = "0 0% 100%"       # --background
    foreground: str = "0 0% 3.9%"       # --foreground
    primary: str = "0 0% 9%"            # --primary
    primary_foreground: str = "0 0% 98%"  # --primary-foreground
    secondary: str = "0 0% 96.1%"       # --secondary
    secondary_foreground: str = "0 0% 9%"  # --secondary-foreground
    muted: str = "0 0% 96.1%"           # --muted
    muted_foreground: str = "0 0% 45.1%"  # --muted-foreground
    accent: str = "0 0% 96.1%"          # --accent
    accent_foreground: str = "0 0% 9%"  # --accent-foreground
    destructive: str = "0 84.2% 60.2%"  # --destructive
    destructive_foreground: str = "0 0% 98%"  # --destructive-foreground
    border: str = "0 0% 89.8%"          # --border
    input: str = "0 0% 89.8%"           # --input
    ring: str = "0 0% 3.9%"             # --ring

    # Border radius
    radius: str = "0.5rem"              # --radius

    # Typography
    font_sans: str = "Inter, ui-sans-serif, system-ui, sans-serif"
    font_mono: str = "JetBrains Mono, ui-monospace, monospace"
    font_serif: str = "Georgia, ui-serif, serif"

    # Spacing scale
    spacing_scale: list[str] = field(default_factory=lambda: [
        "0.25rem", "0.5rem", "0.75rem", "1rem", "1.5rem",
        "2rem", "2.5rem", "3rem", "4rem", "6rem"
    ])

    def to_css_variables(self) -> str:
        """Generate CSS variable declarations."""
        lines = ["@layer base {", "  :root {"]
        color_vars = [
            ("--background", self.background),
            ("--foreground", self.foreground),
            ("--primary", self.primary),
            ("--primary-foreground", self.primary_foreground),
            ("--secondary", self.secondary),
            ("--secondary-foreground", self.secondary_foreground),
            ("--muted", self.muted),
            ("--muted-foreground", self.muted_foreground),
            ("--accent", self.accent),
            ("--accent-foreground", self.accent_foreground),
            ("--destructive", self.destructive),
            ("--destructive-foreground", self.destructive_foreground),
            ("--border", self.border),
            ("--input", self.input),
            ("--ring", self.ring),
            ("--radius", self.radius),
        ]
        for name, value in color_vars:
            lines.append(f"    {name}: {value};")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_tailwind_config(self) -> dict:
        """Generate Tailwind CSS configuration."""
        return {
            "colors": {
                "border": "hsl(var(--border))",
                "input": "hsl(var(--input))",
                "ring": "hsl(var(--ring))",
                "background": "hsl(var(--background))",
                "foreground": "hsl(var(--foreground))",
                "primary": {
                    "DEFAULT": "hsl(var(--primary))",
                    "foreground": "hsl(var(--primary-foreground))",
                },
                "secondary": {
                    "DEFAULT": "hsl(var(--secondary))",
                    "foreground": "hsl(var(--secondary-foreground))",
                },
                "destructive": {
                    "DEFAULT": "hsl(var(--destructive))",
                    "foreground": "hsl(var(--destructive-foreground))",
                },
                "muted": {
                    "DEFAULT": "hsl(var(--muted))",
                    "foreground": "hsl(var(--muted-foreground))",
                },
                "accent": {
                    "DEFAULT": "hsl(var(--accent))",
                    "foreground": "hsl(var(--accent-foreground))",
                },
            },
            "borderRadius": {
                "lg": "var(--radius)",
                "md": "calc(var(--radius) - 2px)",
                "sm": "calc(var(--radius) - 4px)",
            },
            "fontFamily": {
                "sans": [self.font_sans],
                "mono": [self.font_mono],
                "serif": [self.font_serif],
            },
        }


@dataclass
class DesignRegistry:
    """Complete design system registry — shadcn/ui compatible.

    Supports:
    - Components (UI primitives like Button, Input, Card)
    - Blocks (prebuilt page compositions)
    - Design tokens (colors, typography, spacing, shadows)
    - Registry manifest (registry.json)
    """

    name: str
    description: str = ""
    tokens: DesignTokens = field(default_factory=DesignTokens)
    items: list[RegistryItem] = field(default_factory=list)
    base_url: str = ""

    def add_component(self, name: str, files: list[dict], **kwargs) -> None:
        """Register a UI component in the registry."""
        item = RegistryItem(
            name=name,
            type="registry:component",
            files=files,
            **kwargs,
        )
        self.items.append(item)

    def add_block(self, name: str, files: list[dict], **kwargs) -> None:
        """Register a prebuilt block/page in the registry."""
        item = RegistryItem(
            name=name,
            type="registry:block",
            files=files,
            **kwargs,
        )
        self.items.append(item)

    def to_registry_json(self) -> dict:
        """Generate registry.json manifest."""
        return {
            "name": self.name,
            "description": self.description,
            "homepage": self.base_url,
            "items": [item.to_registry_json() for item in self.items],
        }

    def write_to_directory(self, output_dir: Path) -> None:
        """Write the complete design registry to a directory.

        Outputs:
        - registry.json          — manifest
        - tokens.css             — CSS variables
        - tailwind.config.ts     — Tailwind configuration
        - components/*.tsx       — UI primitives
        - blocks/*.tsx           — Page compositions
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write registry manifest
        (output_dir / "registry.json").write_text(
            json.dumps(self.to_registry_json(), indent=2)
        )

        # Write design tokens
        (output_dir / "tokens.css").write_text(self.tokens.to_css_variables())

        # Write Tailwind config
        (output_dir / "tailwind.config.json").write_text(
            json.dumps(self.tokens.to_tailwind_config(), indent=2)
        )

    def to_context_for_ai(self) -> str:
        """Generate context string for LLM injection.

        Tells the AI what design system is in use, what components
        are available, and what tokens to use.
        """
        lines = [
            f"## Design System: {self.name}",
            f"{self.description}",
            "",
            "### Available Components",
        ]
        for item in self.items:
            if item.type == "registry:component":
                lines.append(f"- `<{item.name}/>` — {item.description}")
                if item.dependencies:
                    lines.append(f"  Dependencies: {', '.join(item.dependencies)}")

        blocks = [i for i in self.items if i.type == "registry:block"]
        if blocks:
            lines.append("")
            lines.append("### Available Blocks (prebuilt page compositions)")
            for item in blocks:
                lines.append(f"- `{item.name}` — {item.description}")

        lines.append("")
        lines.append("### Design Tokens")
        lines.append(f"- Primary color: `hsl({self.tokens.primary})`")
        lines.append(f"- Font: `{self.tokens.font_sans}`")
        lines.append(f"- Border radius: `{self.tokens.radius}`")
        lines.append(f"- Spacing scale: {', '.join(self.tokens.spacing_scale[:6])}")

        return "\n".join(lines)
```

**Example registry.json snippet:**
```json
{
  "name": "acme-corp",
  "description": "Acme Corporation design system with shadcn/ui components",
  "items": [
    {
      "name": "acme-login-form",
      "type": "registry:block",
      "description": "Branded login form with Acme colors and validation",
      "registryDependencies": ["input", "button", "card", "label"],
      "files": [
        { "path": "blocks/acme-login-form.tsx", "target": "components/acme-login-form.tsx" }
      ]
    },
    {
      "name": "acme-dashboard",
      "type": "registry:block",
      "description": "Full admin dashboard with charts, tables, and navigation",
      "registryDependencies": ["card", "table", "dropdown-menu"],
      "files": [
        { "path": "blocks/acme-dashboard/page.tsx", "target": "app/dashboard/page.tsx" },
        { "path": "blocks/acme-dashboard/layout.tsx", "target": "app/dashboard/layout.tsx" }
      ]
    }
  ]
}
```

**File changes:**
- NEW: `orchestrator/design_registry.py` (~500 lines)
- MODIFY: `orchestrator/ux/design_enhancer.py` — load registry, inject tokens as system context
- MODIFY: `orchestrator/engine.py` — load design registry at project init
- NEW: `.orchestrator/design-system/registry.json` — template manifest
- DEPENDS ON: Phase 6 (Design System Injection) + Phase 10 (Design System Projects)

#### Verification Gate

```bash
# Create a design system with 5 components + 2 blocks
# Verify: registry.json, tokens.css, tailwind.config.json generated
# Generate UI code — verify it uses registered components and design tokens
# Run: python -m orchestrator design-system export --format registry
```

---

## Phase V2: Visual Design Panel Controls

### Objective

Extend the Element Picker (UI Phase U4) with v0-style visual design panel controls — typography (font, size, weight, line height, letter spacing, alignment), color (text + background), layout (margin, padding), border (color, style, width), appearance (opacity, border radius), and shadow. Combine visual tweaks with natural-language instructions.

### Current State

- Phase U4 (Element Picker) — click elements in preview, see source code + properties
- Current `PropertyPanel` implementation: tag, text, color, size — minimal controls
- No typography, layout, border, shadow, or appearance controls
- No combine visual tweaks + natural language instructions on same element

### Implementation

#### V2.1 — Extend the PropertyPanel with full design controls

```typescript
// ide_frontend/src/components/PropertyPanel.tsx

interface ElementProperties {
  // Selection
  tag: string;
  text: string;
  sourceLocation: { file: string; line: number } | null;

  // Typography
  fontFamily: string;
  fontSize: string;
  fontWeight: string;
  lineHeight: string;
  letterSpacing: string;
  textAlign: string;
  textDecoration: string;

  // Color
  color: string;
  backgroundColor: string;

  // Layout
  margin: { top: string; right: string; bottom: string; left: string };
  padding: { top: string; right: string; bottom: string; left: string };

  // Border
  borderColor: string;
  borderStyle: string;
  borderWidth: string;

  // Appearance
  opacity: string;
  borderRadius: string;

  // Shadow
  boxShadow: string;

  // Content
  editableText: string;
}

interface PendingEdit {
  elementId: string;
  property: string;
  oldValue: string;
  newValue: string;
}

function PropertyPanel({ element, onApply, onReset }: Props) {
  const [pendingEdits, setPendingEdits] = useState<PendingEdit[]>([]);
  const [naturalInstruction, setNaturalInstruction] = useState("");
  const [showBeforePreview, setShowBeforePreview] = useState(false);

  const applyEdits = () => {
    // Serialize all pending visual edits + any natural instruction
    const payload = {
      element,
      visualEdits: pendingEdits,
      instruction: naturalInstruction || null,
    };
    onApply(payload);
    setPendingEdits([]);
    setNaturalInstruction("");
  };

  const undoLastEdit = () => {
    setPendingEdits(prev => prev.slice(0, -1));
  };

  const resetAll = () => {
    setPendingEdits([]);
    setNaturalInstruction("");
  };

  return (
    <div className="property-panel">
      {/* Inspect / Interact toggle */}
      <ToggleMode />

      {/* Natural language instruction */}
      <TextArea
        placeholder='e.g. "make this a 3-column grid" or "match the card above"'
        value={naturalInstruction}
        onChange={setNaturalInstruction}
      />

      {/* Typography section */}
      <Section title="Typography" icon="type">
        <FontFamilyControl value={element.fontFamily} ... />
        <NumberControl label="Size" value={element.fontSize} unit="px" ... />
        <SelectControl label="Weight" value={element.fontWeight} options={["400", "500", "600", "700", "800", "900"]} ... />
        <NumberControl label="Line Height" value={element.lineHeight} ... />
        <NumberControl label="Letter Spacing" value={element.letterSpacing} unit="px" ... />
        <SegmentControl label="Align" value={element.textAlign} options={["left", "center", "right", "justify"]} ... />
        <DecorToggle label="Underline" value={element.textDecoration} ... />
      </Section>

      {/* Color section */}
      <Section title="Color" icon="palette">
        <ColorPicker label="Text" value={element.color} ... />
        <ColorPicker label="Background" value={element.backgroundColor} ... />
      </Section>

      {/* Layout section */}
      <Section title="Layout" icon="layout">
        <SpacingControl label="Margin" values={element.margin} ... />
        <SpacingControl label="Padding" values={element.padding} ... />
      </Section>

      {/* Border section */}
      <Section title="Border" icon="border-all">
        <ColorPicker label="Color" value={element.borderColor} ... />
        <SelectControl label="Style" value={element.borderStyle} options={["solid", "dashed", "dotted", "none"]} ... />
        <NumberControl label="Width" value={element.borderWidth} unit="px" ... />
      </Section>

      {/* Appearance section */}
      <Section title="Appearance" icon="eye">
        <SliderControl label="Opacity" value={element.opacity} min={0} max={100} unit="%" ... />
        <NumberControl label="Radius" value={element.borderRadius} unit="px" ... />
      </Section>

      {/* Shadow section */}
      <Section title="Shadow" icon="layers">
        <ShadowPresets value={element.boxShadow} onChange={...} />
        <ShadowCustomizer value={element.boxShadow} onChange={...} />
      </Section>

      {/* Content section (for text elements) */}
      {element.tag.match(/^(h[1-6]|p|span|button|a|label)$/) && (
        <Section title="Content" icon="text">
          <ContentEditor value={element.editableText} onChange={...} />
        </Section>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 mt-4 pt-4 border-t">
        <button onClick={undoLastEdit}>↩ Undo</button>
        <button onClick={resetAll}>↺ Reset</button>
        <TogglePreview
          showBefore={showBeforePreview}
          onToggle={setShowBeforePreview}
        />
        <button onClick={applyEdits} className="apply-btn">
          Apply ({pendingEdits.length} changes)
        </button>
      </div>
    </div>
  );
}
```

**Backend serialization:**
```python
# orchestrator/design_mode.py

@dataclass
class DesignModeEdits:
    element_selector: str  # CSS selector or React component name
    visual_edits: list[VisualEdit]
    natural_instruction: str | None = None
    screenshot_base64: str | None = None

@dataclass
class VisualEdit:
    property: str  # "fontSize", "color", "margin.top", "borderRadius"
    old_value: str
    new_value: str

class DesignModeProcessor:
    """Processes design-mode edits and generates code changes."""

    def to_llm_prompt(self, edits: DesignModeEdits) -> str:
        """Convert visual edits to an LLM prompt for code generation."""

    def to_code_diff(self, edits: DesignModeEdits) -> str:
        """Convert visual edits to a code diff (for Tailwind class changes)."""

    def is_tailwind_applicable(self, edit: VisualEdit) -> bool:
        """Check if an edit maps to a Tailwind utility class."""
```

**File changes:**
- MODIFY: `ide_frontend/src/components/PropertyPanel.tsx` — full design controls
- NEW: `ide_frontend/src/components/design-controls/` — Typography, Color, Layout, Border, Shadow, Content
- NEW: `orchestrator/design_mode.py` — backend processor for design-mode edits
- MODIFY: `orchestrator/ide_backend/websocket/handlers.py` — handle `design:apply` events
- DEPENDS ON: Phase U4 (Element Picker — requires element selection)

#### Verification Gate

```bash
# Open a generated app in preview, enter design mode
# Select a button, change font size, color, padding, border radius
# Apply edits — verify code changes (Tailwind class updates)
# Add natural instruction: "make this a 3-column grid" — verify combined with visual edits
# Undo/redo, compare before/after — verify all work
```

---

## Phase V3: Browser-Use Agent (Autonomous App Testing + Debugging)

### Objective

An autonomous agent that opens the generated app in a browser, interacts with it (click buttons, fill forms, navigate), critiques the design, debugs complex flows, and sends screenshots back. This goes beyond browser testing (Phase 5) — it's an agent that actively uses the app as a user would.

### Current State

- `BrowserTester` (Phase 5) — test scenarios with Playwright, video recording
- `CodebaseAnalyzer.debug()` — text-based debugging
- No autonomous agent that opens the app and interacts as a user

### Implementation

#### V3.1 — Create `orchestrator/browser_agent.py`

```python
"""
Browser-Use Agent — autonomous agent that opens and uses generated apps.
========================================================================

Unlike BrowserTester (which runs predefined test scenarios), the browser-use
agent opens the app, explores it as a user would, and reports findings:

1. Design critique — "the button color doesn't match the palette"
2. Flow debugging — "signup form fails silently on empty email"
3. UX issues — "navigation menu doesn't close on mobile after selection"
4. Screenshot capture — sends annotated screenshots back to the chat

The agent uses Playwright for browser control and a vision-capable LLM
for understanding what it sees.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page


@dataclass
class BrowserSession:
    """A single browser-use exploration session."""

    app_url: str
    actions: list[dict] = field(default_factory=list)  # What the agent did
    screenshots: list[str] = field(default_factory=list)  # base64 screenshots
    observations: list[str] = field(default_factory=list)  # What the agent observed
    issues: list[dict] = field(default_factory=list)  # Problems found
    fixes_needed: list[str] = field(default_factory=list)  # Suggested fixes


class BrowserUseAgent:
    """Autonomous agent that opens and uses a generated app.

    Capabilities:
    - Explore the app organically (clicking buttons, filling forms, navigating)
    - Critique design decisions against the design system registry
    - Debug frontend flows end-to-end (signup, login, checkout, etc.)
    - Capture and annotate screenshots of issues
    - Generate fix suggestions with file locations
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        max_actions: int = 50,
    ):
        self._headless = headless
        self._timeout = timeout_ms
        self._max_actions = max_actions
        self._session: BrowserSession | None = None

    async def explore(
        self,
        app_url: str,
        design_registry: DesignRegistry | None = None,
        focus_areas: list[str] | None = None,
    ) -> BrowserSession:
        """Open the app and explore it organically.

        Args:
            app_url: URL of the deployed/development app
            design_registry: Optional design system to validate against
            focus_areas: Optional areas to focus on (e.g., ["signup", "checkout"])

        Returns:
            BrowserSession with actions, screenshots, observations, issues
        """

    async def critique_design(
        self, page: Page, registry: DesignRegistry
    ) -> list[dict]:
        """Critique the UI against the design system registry.

        Checks:
        - Are component colors within the design token palette?
        - Are font families, sizes, weights consistent?
        - Is the spacing scale respected?
        - Are registered components used where applicable?
        """

    async def debug_flow(
        self, page: Page, flow_description: str
    ) -> dict:
        """Debug a specific user flow end-to-end.

        Example: "Signup flow from landing page to dashboard"
        - Navigate to starting point
        - Execute the flow step-by-step
        - Capture screenshots at each step
        - Report any errors, UX issues, or unexpected behavior
        """

    async def capture_screenshot(
        self, page: Page, annotation: str = ""
    ) -> str:
        """Capture a screenshot with an optional annotation."""
        screenshot = await page.screenshot(type="png", full_page=True)
        return base64.b64encode(screenshot).decode("utf-8")

    def generate_fix_prompt(self, session: BrowserSession) -> str:
        """Generate a fix prompt from the session's findings.

        The prompt can be sent back to the critique cycle for automatic fixing.
        """
        prompt = "The following issues were found during browser exploration:\n\n"
        for issue in session.issues:
            prompt += f"### {issue['title']}\n"
            prompt += f"- Observation: {issue['observation']}\n"
            prompt += f"- Suggested fix: {issue['fix']}\n"
            if issue.get("screenshot"):
                prompt += f"- Screenshot attached\n"
            prompt += "\n"
        return prompt
```

**Agent workflow:**
```
1. Agent starts browser session → opens app at URL
2. Agent explores:
   - Clicks all visible buttons, links, navigation items
   - Fills forms with test data
   - Tests responsive layout at 3+ breakpoints
   - Screenshots key pages
3. Agent critiques design (if registry provided):
   - Checks colors against tokens
   - Checks typography consistency
   - Checks component usage
4. Agent reports findings:
   - Screenshots attached to issues
   - File locations where fixes are needed
   - Concrete fix suggestions
5. Agent feeds findings into CritiqueCycle → auto-fix generation
```

**File changes:**
- NEW: `orchestrator/browser_agent.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — wire `BrowserUseAgent` into post-execution phase
- MODIFY: `requirements.txt` — add `playwright`
- DEPENDS ON: Phase 5 (Browser Testing) for Playwright infrastructure
- DEPENDS ON: Phase V1 (Design Registry) for design critique
- DEPENDS ON: Phase N3 (Screenshot Debugger) for vision-based analysis

#### Verification Gate

```bash
# Generate a full-stack app with signup flow
# Deploy locally, run BrowserUseAgent.explore()
# Expect: screenshots of all pages, list of UI issues, specific fix suggestions
# Expect: design critique if registry is configured
```

---

## Phase V4: Permission Modes (Ask/Auto/Full for Tool Execution)

### Objective

Three permission modes for terminal commands, file writes, and external API calls — Ask (always prompts for permission), Auto (auto-approves safe operations), Full (executes without asking). Gives the user granular control over agent autonomy.

### Current State

- No permission system for tool execution
- CLI runs with full user permissions
- File writes happen without confirmation (unless running with `--dry-run`)
- Terminal commands in the orchestrator are manual, not agent-driven

### Implementation

#### V4.1 — Create `orchestrator/permission_manager.py`

```python
"""
Permission Manager — Ask/Auto/Full modes for agent tool execution.
===================================================================

Three permission modes control how much autonomy the agent has:

Ask  — The agent always asks before executing a tool.
       Safest mode. Best for production or shared environments.

Auto — The agent auto-approves "safe" operations (reads, syntax checks,
       formatting) and asks for "unsafe" ones (writes, shell commands,
       external API calls).

Full — The agent executes all tools without asking.
       Best for trusted, isolated sandbox environments.
"""

from enum import Enum
from typing import Callable, Any


class PermissionMode(str, Enum):
    ASK = "ask"    # Always prompt for permission
    AUTO = "auto"  # Auto-approve safe, prompt for unsafe
    FULL = "full"  # Execute without asking


class PermissionManager:
    """Controls agent tool execution permissions.

    Safe operations (always allowed in Auto mode):
    - File reads (read_file, list_dir, grep_files, file_search)
    - Git reads (git_status, git_diff, git_show, git_log, git_blame)
    - Web fetches (fetch_url, web_search)
    - Diagnostics (diagnostics)

    Unsafe operations (require permission in Ask and Auto modes):
    - File writes (write_file, edit_file, apply_patch)
    - Shell execution (exec_shell, task_shell_start)
    - Sub-agent spawns (agent_spawn)
    - External tool calls (MCP integrations, API calls)
    """

    SAFE_TOOLS: set[str] = {
        "read_file", "list_dir", "grep_files", "file_search",
        "git_status", "git_diff", "git_show", "git_log", "git_blame",
        "fetch_url", "web_search", "diagnostics",
        "task_list", "task_read",
        "agent_list", "agent_wait", "agent_result",
    }

    UNSAFE_TOOLS: set[str] = {
        "write_file", "edit_file", "apply_patch",
        "exec_shell", "task_shell_start", "task_shell_wait",
        "agent_spawn", "delegate_to_agent", "agent_send_input",
        "task_create", "task_cancel",
    }

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.ASK,
        on_permission_required: Callable | None = None,
    ):
        self.mode = mode
        self._on_permission_required = on_permission_required
        self._auto_approved: list[dict] = []
        self._permission_denied: list[dict] = []

    def requires_permission(self, tool_name: str) -> bool:
        """Check if a tool requires permission based on current mode."""
        if self.mode == PermissionMode.FULL:
            return False
        if self.mode == PermissionMode.ASK:
            return tool_name not in self.SAFE_TOOLS or tool_name in self.UNSAFE_TOOLS
        if self.mode == PermissionMode.AUTO:
            return tool_name in self.UNSAFE_TOOLS
        return True

    async def request_permission(
        self, tool_name: str, description: str, risk_level: str = "low"
    ) -> bool:
        """Request permission to execute a tool.

        Args:
            tool_name: The tool being requested
            description: Human-readable description of the action
            risk_level: "low", "medium", "high" — affects urgency of prompt

        Returns:
            True if permission granted
        """
        if not self.requires_permission(tool_name):
            self._auto_approved.append({
                "tool": tool_name,
                "description": description,
                "timestamp": asyncio.get_event_loop().time(),
            })
            return True

        if self._on_permission_required:
            return await self._on_permission_required(tool_name, description, risk_level)

        # Default: deny if no callback configured
        self._permission_denied.append({
            "tool": tool_name,
            "description": description,
            "reason": "No permission handler configured",
        })
        return False

    def set_mode(self, mode: PermissionMode) -> None:
        """Change permission mode."""
        self.mode = mode

    def get_dashboard_data(self) -> dict:
        """Return dashboard display data."""
        return {
            "mode": self.mode.value,
            "auto_approved_count": len(self._auto_approved),
            "denied_count": len(self._permission_denied),
            "recent_requests": [
                *self._auto_approved[-5:],
                *self._permission_denied[-5:],
            ],
        }
```

**Integration into Orchestrator:**
```python
class Orchestrator:
    def __init__(
        self,
        ...,
        permission_mode: PermissionMode = PermissionMode.ASK,
    ):
        self._permissions = PermissionManager(mode=permission_mode)

    async def _execute_task(self, task: Task) -> TaskResult:
        # Check permission before each write/shell operation
        if self._permissions.requires_permission("write_file"):
            granted = await self._permissions.request_permission(
                "write_file",
                f"Write output for task {task.id} ({task.description[:80]})",
                risk_level="medium",
            )
            if not granted:
                return self._failed_result(task, "Permission denied")
        ...
```

**CLI flags:**
```bash
# Ask mode (default) — always prompt
python -m orchestrator run --project "..." --permission ask

# Auto mode — approve safe ops, prompt for unsafe
python -m orchestrator run --project "..." --permission auto

# Full mode — no prompts, full autonomy
python -m orchestrator run --project "..." --permission full
```

**File changes:**
- NEW: `orchestrator/permission_manager.py` (~200 lines)
- MODIFY: `orchestrator/engine.py` — integrate PermissionManager into tool execution
- MODIFY: `orchestrator/cli.py` — add `--permission` flag
- MODIFY: `orchestrator/app_assembler.py` — check permissions before file writes

#### Verification Gate

```bash
# Run in Ask mode — verify permission prompt for every write
# Run in Auto mode — verify reads auto-approve, writes prompt
# Run in Full mode — verify no prompts
```

---

## Phase V5: Auto-Error Fix Button (One-Click from Error Logs)

### Objective

When a task fails, validation fails, or tests fail, add a "Fix" button that sends the error logs back to the LLM for automatic diagnosis and repair. Instead of manually describing the error, the LLM sees the exact error output.

### Current State

- `CritiqueCycle` handles revision after low scores — but no one-click fix from error logs
- Failed tasks return `TaskResult` with error messages
- User must manually describe the error to trigger a fix
- No automatic error → fix pipeline

### Implementation

#### V5.1 — Create `orchestrator/auto_fixer.py`

```python
"""
Auto-Fixer — one-click error fix from failure logs.
====================================================

When a task fails or tests fail, the AutoFixer sends the error logs
to an LLM for diagnosis and repair, then re-runs the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Task, TaskResult


@dataclass
class FixAttempt:
    attempt_number: int
    error_type: str  # "validation", "test", "syntax", "build", "runtime"
    error_log: str
    diagnosis: str
    fix_code: str
    fix_applied: bool
    fix_successful: bool
    cost: float


class AutoFixer:
    """One-click fix for failed tasks.

    Workflow:
    1. Detect failure (validation, test, syntax, build, runtime)
    2. Extract error logs
    3. Send to LLM: "Here's the error. Diagnose and fix."
    4. Apply the fix
    5. Re-run the failed step
    6. Report success/failure
    """

    def __init__(self, client: UnifiedClient, max_attempts: int = 3):
        self._client = client
        self._max_attempts = max_attempts
        self._history: list[FixAttempt] = []

    async def fix(
        self,
        task: Task,
        result: TaskResult,
        error_log: str,
        error_type: str = "validation",
    ) -> tuple[TaskResult, list[FixAttempt]]:
        """Attempt to fix a failed task.

        Args:
            task: The failed task
            result: The failed result
            error_log: Full error output (traceback, test output, lint output)
            error_type: Type of error for context-aware fixing

        Returns:
            Tuple of (final_result, fix_history)
        """
        attempts = []
        current_result = result

        for i in range(self._max_attempts):
            # Generate fix
            fix_prompt = self._build_fix_prompt(
                task=task,
                error_log=error_log,
                error_type=error_type,
                previous_attempts=attempts,
            )

            response = await self._client.call(
                model=self._get_fixer_model(),
                prompt=fix_prompt,
                max_tokens=2000,
                temperature=0.2,
            )

            diagnosis, fix_code = self._parse_fix_response(response.text)

            attempt = FixAttempt(
                attempt_number=i + 1,
                error_type=error_type,
                error_log=error_log,
                diagnosis=diagnosis,
                fix_code=fix_code,
                fix_applied=False,
                fix_successful=False,
                cost=response.cost_usd,
            )

            # Apply the fix
            success = await self._apply_fix(fix_code, task, current_result)
            attempt.fix_applied = success

            if not success:
                attempts.append(attempt)
                continue

            # Re-run validation
            current_result = await self._rerun_validation(task)
            attempt.fix_successful = current_result.status in (
                TaskStatus.COMPLETED, TaskStatus.DEGRADED
            )

            attempts.append(attempt)

            if attempt.fix_successful:
                break

        self._history.extend(attempts)
        return current_result, attempts

    def _build_fix_prompt(
        self,
        task: Task,
        error_log: str,
        error_type: str,
        previous_attempts: list[FixAttempt],
    ) -> str:
        """Build the fix prompt with error context."""

        base = (
            f"Fix this error and return ONLY the corrected code.\n\n"
            f"## Task\n{task.prompt}\n\n"
            f"## Error Type\n{error_type}\n\n"
            f"## Error Log\n```\n{error_log}\n```\n\n"
        )

        if task.output:
            base += f"## Current Code\n```\n{task.output[:3000]}\n```\n\n"

        if previous_attempts:
            base += "## Previous Fix Attempts (failed)\n"
            for attempt in previous_attempts:
                base += f"Attempt {attempt.attempt_number}: {attempt.diagnosis[:200]}\n"

        base += (
            "## Instructions\n"
            "1. Identify the root cause of the error\n"
            "2. Provide a fix that addresses the root cause\n"
            "3. Return ONLY the corrected code (no explanation)\n"
            "4. If the error is in test expectations, fix the code, not the tests\n"
        )

        return base
```

**Fix types and their triggers:**
```python
FIX_HANDLERS = {
    "syntax": {
        "trigger": "validate_python_syntax fails",
        "prompt_template": "Fix the Python syntax error. Only return corrected code.",
    },
    "test": {
        "trigger": "validate_pytest fails",
        "prompt_template": "Fix the code so all tests pass. Do NOT modify test files.",
    },
    "lint": {
        "trigger": "validate_ruff fails",
        "prompt_template": "Apply ruff lint fixes. Only return corrected code.",
    },
    "build": {
        "trigger": "npm build / pip install fails",
        "prompt_template": "Fix the build error. Check imports, dependencies, and syntax.",
    },
    "runtime": {
        "trigger": "App fails to start or crashes at runtime",
        "prompt_template": "Fix the runtime error. Check imports, initialization, and logic.",
    },
    "security": {
        "trigger": "validate_tool_safety fails",
        "prompt_template": "Replace the unsafe pattern with a secure alternative.",
    },
}
```

**UI integration:**
```
┌──────────────────────────────────────────────────────────┐
│  ❌ Task "auth.py" failed                               │
│                                                          │
│  Error: validate_pytest failed                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ FAILED tests/test_auth.py::test_login_redirect   │   │
│  │ AssertionError: Expected 302, got 401            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  [🔧 Fix with AI]  (20 free fixes remaining today)     │
└──────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/auto_fixer.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — expose `AutoFixer` on `Orchestrator`, wire into `_execute_task()`
- MODIFY: `orchestrator/validators.py` — return structured error logs for AutoFixer consumption
- MODIFY: `orchestrator/ide_backend/websocket/handlers.py` — handle `fix:request` event

#### Verification Gate

```bash
# Generate a task with a deliberate Python syntax error
# Run, click Fix, verify error is resolved
# Generate a task where tests fail, click Fix, verify tests pass
```

---

## Phase V6: Versions as First-Class Concept

### Objective

Every code-generating action produces a version — diffable, reviewable, revertable. This is already partially covered by Checkpoints (Phase 1) and Diff View (Phase U3), but should be elevated to a first-class concept where every change is automatically versioned.

### Current State

- Phase 1 (Checkpoints) — manual snapshot points
- Phase 2 (Prompt Restore Points - Newly) — per-prompt restore
- Phase U3 (Diff View) — visual side-by-side diff
- `GitIntegration` — milestone-based commits
- No automatic versioning of every code change

### Implementation

#### V6.1 — Create `orchestrator/version_manager.py`

```python
"""
Version Manager — automatic versioning for every code change.
==============================================================

Every code-generating action creates a version with:
- Diff (unified format)
- Version metadata (timestamp, task, model, score)
- Parent version reference (for chain navigation)
- Revert capability
"""

@dataclass
class Version:
    id: str  # UUID
    parent_id: str | None
    task_id: str | None
    timestamp: float
    description: str  # Auto-generated from task description + diff
    diff: str  # Unified diff
    file_manifest: dict[str, str]  # path → sha256
    model_used: str
    cost: float
    score: float

    def to_dict(self) -> dict:
        """Serializable version for UI display."""

class VersionManager:
    """Manages project version history.

    Features:
    - Automatic version creation on every code change
    - Version chain navigation (forward/backward)
    - Diff viewing between any two versions
    - Selective revert (revert single file or full version)
    - Version comparison for context injection
    """

    def __init__(self, output_dir: Path, max_versions: int = 100):
        self._output_dir = output_dir
        self._versions: list[Version] = []
        self._current: Version | None = None
        self._max_versions = max_versions

    async def create_version(
        self,
        task: Task | None = None,
        description: str | None = None,
    ) -> Version:
        """Create a new version from the current state.
        
        Called automatically after every _execute_task().
        """

    def get_current(self) -> Version | None:
        """Get the current (latest) version."""

    def get_version(self, version_id: str) -> Version | None:
        """Get a specific version by ID."""

    async def diff(
        self, version_a: str, version_b: str | None = None
    ) -> str:
        """Get diff between two versions (or between version and current)."""

    async def revert(self, version_id: str, files: list[str] | None = None) -> bool:
        """Revert to a specific version (optionally specific files only)."""

    def timeline(self, limit: int = 20) -> list[dict]:
        """Get version timeline for UI display."""

    async def compare_with(self, version_id: str, prompt: str) -> str:
        """Ask the LLM to compare current code with a past version.
        
        Useful for: "What changed between v3 and v7 that broke the login?"
        """
```

**Version display in UI:**
```
┌──────────────────────────────────────────────────────────┐
│  📜 Versions                          [Compare] [Revert] │
│                                                          │
│  ◉ v14  14:35  ✅ feat(auth): Add JWT authentication    │
│    │     Score: 0.94  Model: GPT-4.1  Cost: $0.023      │
│    │     +87 -3 lines in 3 files                        │
│    │                                                     │
│  ◉ v13  14:32  ✅ feat(db): Create user model schema    │
│    │     Score: 0.96  Model: Qwen 2.5  Cost: $0.004     │
│    │     +45 lines in 2 files                            │
│    │                                                     │
│  ◉ v12  14:30  ❌ fix(api): Attempted rate limiter      │
│    │     Score: 0.45  Model: Qwen 2.5  Cost: $0.003     │
│    │     Reverted                                        │
│    │                                                     │
│  ◉ v11  14:28  ✅ feat(api): Add CRUD endpoints         │
│          Score: 0.92  Model: DeepSeek V4  Cost: $0.018  │
│          +128 lines in 2 files                           │
└──────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/version_manager.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — auto-create version after each `_execute_task()`
- MODIFY: `ide_frontend/src/components/VersionTimeline.tsx` — UI component
- DEPENDS ON: Phase 1 (Checkpoints) for file snapshot infrastructure
- DEPENDS ON: Phase U3 (Diff View) for visual diff rendering

#### Verification Gate

```bash
# Run 5 tasks, verify 5 versions created
# Compare v3 with v5, verify diff
# Revert to v2, verify code is restored
# Check version timeline for metadata (model, cost, score)
```

---

## Phase V7: Templates + Registry Marketplace

### Objective

Build a template gallery with ready-made components, blocks, and full-page designs. Users select a template, the orchestrator generates a project from it. Templates are versioned and shareable via registry URLs.

### Current State

- `orchestrator/scaffold/` — project templates (FastAPI, Next.js, React, CLI) — code templates, not design templates
- No UI component templates
- No block/page templates
- No registry URL for sharing templates

### Implementation

#### V7.1 — Create template registry

```python
"""
Template Registry — ready-made components, blocks, and pages.
==============================================================

Templates are stored in .orchestrator/templates/ and referenced
by registry.json. External templates can be imported via URLs.
"""

class TemplateRegistry:
    """Manages project and component templates.

    Sources:
    - Local: .orchestrator/templates/
    - Remote: registry URLs (shadcn/ui format)
    - Community: GitHub-based template repositories
    """

    def __init__(self, workspace_dir: Path):
        self._templates_dir = workspace_dir / ".orchestrator" / "templates"
        self._templates: dict[str, Template] = {}
        self._load_local_templates()

    def _load_local_templates(self):
        """Load templates from .orchestrator/templates/."""
        for template_dir in self._templates_dir.iterdir():
            if not template_dir.is_dir():
                continue
            config_file = template_dir / "template.json"
            if config_file.exists():
                template = Template.from_directory(template_dir)
                self._templates[template.name] = template

    async def import_remote(self, registry_url: str) -> Template:
        """Import a template from a remote registry URL."""

    def list_templates(self, category: str | None = None) -> list[Template]:
        """List all templates, optionally filtered by category."""

    def get_template(self, name: str) -> Template | None:
        """Get a template by name."""

    async def generate_from_template(
        self, template_name: str, options: dict = None
    ) -> str:
        """Generate a project from a template.

        Prompt: "Use the {template_name} template and customize it for: {description}"
        """
```

**Template categories:**
```
.orchestrator/templates/
├── registry.json               ← Main manifest
├── landing-pages/
│   ├── saas-landing/            ← SaaS landing page template
│   ├── portfolio/               ← Portfolio site template
│   ├── waitlist/                ← Waitlist/coming soon template
│   └── agency/                  ← Agency website template
├── dashboards/
│   ├── analytics/               ← Analytics dashboard
│   ├── ecommerce/               ← E-commerce dashboard
│   └── crm/                     ← CRM dashboard
├── apps/
│   ├── todo/                    ← Full-stack todo app
│   ├── chat/                    ← Real-time chat app
│   └── blog/                    ← Blog with CMS
├── components/
│   ├── hero-section/            ← Hero section variants
│   ├── pricing-table/           ← Pricing table variants
│   ├── navigation/              ← Nav bar variants
│   └── footer/                  ← Footer variants
└── blocks/
    ├── signup-form/             ← Signup form block
    ├── search-bar/              ← Search bar block
    └── data-table/              ← Data table block
```

**CLI commands:**
```bash
# List available templates
python -m orchestrator template list

# List by category
python -m orchestrator template list --category dashboards

# Import from remote registry
python -m orchestrator template import https://your-registry.com/r/registry.json

# Generate from template
python -m orchestrator new --template saas-landing --project "My SaaS"
```

**File changes:**
- NEW: `orchestrator/template_registry.py` (~350 lines)
- NEW: `.orchestrator/templates/` — template directory with example templates
- MODIFY: `orchestrator/cli.py` — add `template` command group
- MODIFY: `orchestrator/engine.py` — support `--template` parameter
- DEPENDS ON: Phase V1 (Design Registry) for registry format
- DEPENDS ON: Phase 10 (Design System Projects) for design system templates

#### Verification Gate

```bash
# List templates, pick one, generate project
# Verify: generated project uses template components and styles
# Import a remote registry, verify templates are available
```

---

## Integration with Existing Plans

```
Phase V1 (Design Registry) ───────────────────────────────────────────────────┐
    │  Depends on Phase 6 (Design System Injection)                           │
    │  Depends on Phase 10 (Design System Projects)                           │
Phase V2 (Visual Design Panel) ───────────────────────────────────────────────┤
    │  Depends on Phase U4 (Element Picker — requires element selection)      │
    │  Depends on Phase V1 (Design Registry — validates against tokens)       │
Phase V3 (Browser-Use Agent) ─────────────────────────────────────────────────┤
    │  Depends on Phase 5 (Browser Testing — Playwright infrastructure)      │
    │  Depends on Phase V1 (Design Registry — design critique)                │
    │  Depends on Phase N3 (Screenshot Debugger — vision model)               │
Phase V4 (Permission Modes) ──────────────────────────────────────────────────┤
    │  No dependencies — standalone safety enhancement                        │
Phase V5 (Auto-Error Fix) ────────────────────────────────────────────────────┤
    │  No dependencies — uses existing validators + CritiqueCycle             │
Phase V6 (Versions) ───────────────────────────────────────────────────────────┤
    │  Depends on Phase 1 (Checkpoints — file snapshot infrastructure)        │
    │  Depends on Phase U3 (Diff View — visual diff rendering)                │
Phase V7 (Templates) ──────────────────────────────────────────────────────────┘
    Depends on Phase V1 (Design Registry — registry format)
    Depends on Phase 10 (Design System Projects)
```

## v0-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| V1 | Design Registry | 1 | 3 | ~500 | 3-4 |
| V2 | Visual Design Panel | 6 | 2 | ~400 | 3-4 |
| V3 | Browser-Use Agent | 1 | 2 | ~400 | 3-4 |
| V4 | Permission Modes | 1 | 3 | ~200 | 1-2 |
| V5 | Auto-Error Fix | 1 | 3 | ~300 | 2-3 |
| V6 | Versions | 1 | 3 | ~300 | 3-4 |
| V7 | Templates + Registry | 1 | 3 | ~350 | 3-5 |
| **v0 Subtotal** | **7 phases** | **12** | **19** | **~2,450** | **18-26** |

## Combined Grand Total (All Six Sources)

| Source | Phases | New Files | Modified Files | Est. Days |
|--------|--------|-----------|---------------|-----------|
| Replit | 1-6 | 5 | 14 | 13-18 |
| Lovable | 7-10 | 4 | 10 | 9-13 |
| UI | U1-U7 | 20 | 15 | 17-23 |
| Newly | N1-N7 | 2 | 21 | 10-14 |
| Base44 | B1-B7 | 14 | 14 | 14-21 |
| v0 | V1-V7 | 12 | 19 | 18-26 |
| **Grand Total** | **38** | **57** | **93** | **81-115** |
