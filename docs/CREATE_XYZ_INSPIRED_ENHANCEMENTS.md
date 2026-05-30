# Enhancement Plan: Create.xyz (Anything)-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Create.xyz (Anything) and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements all 9 prior enhancement plans

---

## Overview

Create.xyz ("Anything") is an AI agent that builds mobile and web apps. Its key differentiator is **Max Mode** — a fully autonomous agent that builds, tests in a real browser, uses the app like a person, fixes issues, and repeats until the goal is complete. Combined with **parallel autonomous agents**, **100+ built-in integrations via slash commands**, and a **template marketplace**, this is the most autonomous execution model across all 10 platforms.

Five enhancements identified, focused on **autonomous execution**, **parallel agent orchestration**, and **integration discovery**.

```
Phase X1: Autonomous Agent Loop (Max Mode) — Test → Fix → Repeat    (Highest ROI, 4-5 days)
Phase X2: Parallel Autonomous Agents (Multi-Max Concurrency)          (High ROI, 3-4 days)
Phase X3: Slash Command Integration Discovery (100+ Built-in)         (High ROI, 2-3 days)
Phase X4: Multi-Mode Agent Selector (Auto/Discuss/Plan/Think/Fast)   (Medium ROI, 2-3 days)
Phase X5: Template Marketplace (Sell/Buy Templates for Credits)       (Lower ROI, 3-5 days)
```

---

## What Create.xyz Has — Quick Reference

| Create.xyz Feature | Description | Translates? |
|-------------------|-------------|:-----------:|
| **Max Mode** | Autonomous: build → test in browser → fix → repeat | ✅ Phase X1 |
| **Parallel Agents** | Multiple Max instances running simultaneously | ✅ Phase X2 |
| **100+ Integrations via /commands** | Slash commands for AI models, UI, maps, data, etc. | ✅ Phase X3 |
| **Multi-Mode Selector** | Auto/Discuss/Plan/Think/Fast/Max modes | ✅ Phase X4 |
| **Template Marketplace** | Sell templates for credits, reviewed & rated | ✅ Phase X5 |
| **Threads** | New thread per feature, parallel work | Partial — Phase D1 (Multiple Contexts) |
| **Model Switcher** | Override agent's model choice | Partial — Phase W1 (Agent Profiles) |
| **Browser Testing** | Real browser interaction | Partial — Phase V3 (Browser-Use Agent) |
| **Device Capabilities** | Camera, location, sensors for mobile | ✗ (Platform-specific) |
| **Ads Monetization** | Google AdMob + AdSense | ✗ (Platform-specific) |
| **Experts Network** | One-on-one help from experienced builders | ✗ (Platform-specific) |

---

## Phase X1: Autonomous Agent Loop (Max Mode) — Test → Fix → Repeat

### Objective

A fully autonomous agent that takes a goal, generates code, opens the app in a real browser (Playwright), uses it like a user would (click, type, navigate), spots visual/logical issues, generates fixes, re-tests, and repeats until the goal is achieved. This is "Max Mode" — the most advanced autonomous execution model across all 10 platforms.

### Current State

- `CritiqueCycle` — generate → critique → revise but no browser interaction
- `BrowserTester` (Phase 5) — test scenarios with Playwright
- `BrowserUseAgent` (Phase V3) — opens app, explores, critiques design
- `AutoFixer` (Phase V5) — one-click error fix
- These are separate tools — no unified autonomous loop combining all three

### Implementation

#### X1.1 — Create `orchestrator/autonomous_agent.py`

```python
"""
Autonomous Agent (Max Mode) — build, test, fix, repeat.
=========================================================

Combines code generation + browser testing + error fixing into an
autonomous loop that runs until the goal is achieved.

Goal → generate → test in browser → find issues → fix → retest → ✅
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .browser_agent import BrowserUseAgent
    from .auto_fixer import AutoFixer
    from .engine import Orchestrator


class LoopStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"  # Goal achieved
    FAILED = "failed"    # Unresolvable issues
    TIMEOUT = "timeout"  # Exceeded max iterations or time
    BLOCKED = "blocked"  # External dependency issue


@dataclass
class LoopStep:
    """One step in the autonomous loop."""
    step_number: int
    action: str  # "generate", "test", "analyze", "fix", "retest"
    description: str
    cost: float
    duration_ms: float
    browser_screenshot: str | None = None
    issues_found: list[str] = field(default_factory=list)
    issues_fixed: list[str] = field(default_factory=list)
    success: bool = True


@dataclass
class AutonomousSession:
    """Complete autonomous agent session."""
    goal: str
    status: LoopStatus
    steps: list[LoopStep] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    final_screenshot: str | None = None
    summary: str = ""


class AutonomousAgent:
    """Fully autonomous agent — goal-driven, self-fixing.

    Workflow:
    1. Parse the goal into actionable tasks
    2. Generate code for the first task
    3. Open the app in a real browser
    4. Interact with it as a user would (click, type, navigate)
    5. Take screenshots and analyze for issues
    6. If issues found: generate fixes
    7. Re-test after fixes
    8. Repeat until goal achieved or timeout

    Configuration:
        max_iterations: Maximum loop iterations (default: 50)
        max_duration_seconds: Maximum wall-clock time (default: 1800 = 30 min)
        max_cost_usd: Maximum cost ceiling (default: $10.00)
        headless: Whether to run browser headless (default: True)
        record_video: Whether to record browser sessions (default: False)
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        browser_agent: BrowserUseAgent,
        auto_fixer: AutoFixer,
        max_iterations: int = 50,
        max_duration_seconds: int = 1800,  # 30 minutes
        max_cost_usd: float = 10.0,
        headless: bool = True,
        record_video: bool = False,
    ):
        self._orch = orchestrator
        self._browser = browser_agent
        self._fixer = auto_fixer

        self._max_iterations = max_iterations
        self._max_duration = max_duration_seconds
        self._max_cost = max_cost_usd
        self._headless = headless
        self._record_video = record_video

    async def execute(self, goal: str) -> AutonomousSession:
        """Execute the autonomous agent loop.

        The loop runs until:
        - Goal is achieved (browser verification passes)
        - Max iterations exceeded
        - Max duration exceeded
        - Max cost exceeded
        - Unresolvable issue encountered

        Args:
            goal: Natural language description of the task
                   e.g. "Fix checkout so it stops failing on mobile"
                   e.g. "Build a teams feature with invite flow and role management"

        Returns:
            AutonomousSession with complete execution history
        """
        session = AutonomousSession(goal=goal, status=LoopStatus.RUNNING)
        t0 = time.time()

        for iteration in range(self._max_iterations):
            # Cost check
            if session.total_cost >= self._max_cost:
                session.status = LoopStatus.FAILED
                session.summary = f"Cost limit exceeded (${self._max_cost})"
                break

            # Time check
            elapsed = time.time() - t0
            if elapsed >= self._max_duration:
                session.status = LoopStatus.TIMEOUT
                session.summary = f"Time limit exceeded ({self._max_duration}s)"
                break

            # === STEP 1: Generate/Execute ===
            step = LoopStep(
                step_number=iteration + 1,
                action="generate",
                description=f"Executing: {goal[:100]}",
            )

            result = await self._orch._execute_task(self._build_task(goal, iteration))
            step.cost += result.cost_usd
            session.total_cost += result.cost_usd

            # === STEP 2: Test in Browser ===
            step.action = "test"
            browser_session = await self._browser.explore(
                app_url=self._get_app_url(),
                focus_areas=self._extract_focus_areas(goal),
            )

            if browser_session.screenshots:
                step.browser_screenshot = browser_session.screenshots[0]

            step.cost += browser_session.total_cost
            session.total_cost += browser_session.total_cost

            # === STEP 3: Analyze Issues ===
            step.action = "analyze"
            issues = browser_session.issues

            if not issues:
                # No issues found — goal achieved!
                step.issues_found = []
                step.success = True
                session.steps.append(step)
                session.status = LoopStatus.SUCCESS
                session.summary = f"Goal achieved in {iteration + 1} steps"
                session.final_screenshot = browser_session.screenshots[-1] if browser_session.screenshots else None
                break

            step.issues_found = [i["title"] for i in issues]
            step.success = False

            # === STEP 4: Fix Issues ===
            step.action = "fix"
            for issue in issues:
                fix_result = await self._fixer.fix(
                    task=self._build_task(issue["description"], iteration),
                    error_log=issue.get("observation", ""),
                    error_type=issue.get("type", "runtime"),
                )

                fixed = fix_result[0].status == TaskStatus.COMPLETED
                if fixed:
                    step.issues_fixed.append(issue["title"])
                step.cost += sum(a.cost for a in fix_result[1])

            session.total_cost += step.cost
            session.steps.append(step)

            # === STEP 5: Retest ===
            step.action = "retest"
            retest_session = await self._browser.explore(
                app_url=self._get_app_url(),
                focus_areas=[i["title"] for i in issues],
            )

            if not retest_session.issues:
                session.status = LoopStatus.SUCCESS
                session.summary = f"Goal achieved in {iteration + 1} steps — {len(issues)} issues fixed"
                session.final_screenshot = retest_session.screenshots[-1] if retest_session.screenshots else None
                break

        # Calculate total duration
        session.total_duration_ms = (time.time() - t0) * 1000
        return session

    def get_progress(self, session: AutonomousSession) -> dict:
        """Get current progress for UI display."""
        return {
            "goal": session.goal,
            "status": session.status.value,
            "steps_completed": len(session.steps),
            "total_cost": session.total_cost,
            "total_duration_ms": session.total_duration_ms,
            "latest_screenshot": session.steps[-1].browser_screenshot if session.steps else None,
        }
```

**UI display during autonomous execution:**
```
┌──────────────────────────────────────────────────────────────┐
│  🤖 Max Agent — "Fix checkout on mobile"                    │
│                                                              │
│  Step 8/50  ⏱️ 4:23 elapsed  💰 $0.89 spent                 │
│                                                              │
│  ► Step 1: generate  ✅  2.1s                               │
│  ► Step 2: test      ⚠️  3 found                            │
│       • Checkout button off-screen on mobile                 │
│       • Payment confirmation not showing                     │
│       • Error state missing for declined cards               │
│  ► Step 3: fix       🔧 fixing...                          │
│       Fixed: Checkout button                                │
│       Fixed: Payment confirmation                           │
│  ● Step 4: retest    🔄 testing in browser...               │
│                                                              │
│  [⏸️ Pause]  [⏹️ Stop]  [📋 View steps]                    │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/autonomous_agent.py` (~500 lines)
- MODIFY: `orchestrator/engine.py` — add `run_max()` method, wire AutonomousAgent
- MODIFY: `orchestrator/cli.py` — add `max` command
- DEPENDS ON: Phase V3 (Browser-Use Agent — browser interaction)
- DEPENDS ON: Phase V5 (AutoFixer — error fixing)
- DEPENDS ON: Phase 5 (Browser Testing — Playwright infrastructure)
- DEPENDS ON: Phase N3 (Screenshot Debugger — vision-based issue detection)

#### Verification Gate

```bash
# Run Max mode on a checkout bug
python -m orchestrator max --goal "Fix checkout so it stops failing on mobile"
# Verify: agent generates fix, tests in browser, finds issue, fixes, retests
# Verify: session report with step-by-step history, cost, duration
```

---

## Phase X2: Parallel Autonomous Agents (Multi-Max Concurrency)

### Objective

Run multiple autonomous agent instances simultaneously, each working on a separate task. One agent fixing auth bugs while another builds a settings page while a third QA's signup flow. All operate on the same codebase with conflict detection.

### Current State

- `Orchestrator._execute_all()` supports parallel task execution via `asyncio.gather()`
- Phase 4 (Sandbox Tasks) isolates individual tasks
- Phase D1 (Multiple Contexts) allows separate execution contexts
- No concurrent autonomous agent instances
- No conflict detection for simultaneous modifications

### Implementation

#### X2.1 — Create `orchestrator/parallel_agents.py`

```python
"""
Parallel Autonomous Agents — multiple Max instances, one codebase.
====================================================================

Run up to N autonomous agent instances simultaneously, each with
its own goal. Agents share the codebase but have isolated contexts.

Conflict detection: if two agents modify the same file, the first
one wins and the second one is notified of the conflict.
"""

@dataclass
class AgentSlot:
    """A slot running one autonomous agent instance."""
    slot_id: int
    goal: str
    agent: AutonomousAgent
    task: asyncio.Task | None = None
    session: AutonomousSession | None = None
    status: str = "idle"  # "idle", "running", "completed", "failed"


class ParallelAgentOrchestrator:
    """Orchestrates multiple autonomous agents in parallel.

    Configuration:
        max_concurrent: Maximum simultaneous agents (default: 3)
        conflict_strategy: "first_wins" or "sequential" or "isolate"
        auto_start: Whether to auto-start agents when slots open
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        max_concurrent: int = 3,
        conflict_strategy: str = "isolate",
    ):
        self._orch = orchestrator
        self._max_concurrent = max_concurrent
        self._conflict_strategy = conflict_strategy

        self._slots: list[AgentSlot] = []

        # Conflict tracking: file → [agent_slot_ids]
        self._file_owners: dict[str, set[int]] = {}

        # Queue of pending goals
        self._queue: list[str] = []

    async def start_agent(self, goal: str) -> int:
        """Start a new autonomous agent on a goal.

        If all slots are full, queues the goal.

        Returns:
            Slot ID (0-based index)
        """
        # Find available slot
        slot = self._find_available_slot()
        if slot is None:
            self._queue.append(goal)
            return -1  # Queued

        # Create agent
        agent = AutonomousAgent(
            orchestrator=self._orch,
            browser_agent=self._orch._browser_agent,
            auto_fixer=self._orch._auto_fixer,
            max_iterations=50,
            max_duration_seconds=1800,
            max_cost_usd=10.0,
        )

        # Run in background
        task = asyncio.create_task(agent.execute(goal))
        slot.agent = agent
        slot.goal = goal
        slot.task = task
        slot.status = "running"

        # Add completion callback
        task.add_done_callback(lambda t: self._on_agent_complete(slot.slot_id, t))

        return slot.slot_id

    async def stop_agent(self, slot_id: int) -> bool:
        """Stop a running agent."""
        ...

    def get_status(self) -> list[dict]:
        """Get status of all agent slots."""
        return [
            {
                "slot_id": s.slot_id,
                "goal": s.goal,
                "status": s.status,
                "steps_completed": len(s.session.steps) if s.session else 0,
                "total_cost": s.session.total_cost if s.session else 0,
            }
            for s in self._slots
        ]

    def _find_available_slot(self) -> AgentSlot | None:
        """Find an idle slot or create a new one."""
        for slot in self._slots:
            if slot.status in ("idle", "completed", "failed"):
                slot.status = "idle"
                return slot

        if len(self._slots) < self._max_concurrent:
            new_slot = AgentSlot(slot_id=len(self._slots))
            self._slots.append(new_slot)
            return new_slot

        return None  # All slots full

    def _on_agent_complete(self, slot_id: int, task: asyncio.Task):
        """Handle agent completion — update slot, start queued goal."""
        slot = self._slots[slot_id]
        slot.session = task.result()
        slot.status = "completed" if slot.session.status == LoopStatus.SUCCESS else "failed"

        # Release file locks
        if slot_id in self._file_owners:
            del self._file_owners[slot_id]

        # Start next queued goal
        if self._queue:
            next_goal = self._queue.pop(0)
            asyncio.create_task(self.start_agent(next_goal))

    def detect_conflicts(
        self, agent_id: int, modified_files: list[str]
    ) -> list[str]:
        """Detect if another agent is modifying the same files.

        Returns list of conflicting files.
        """
        conflicts = []
        for file in modified_files:
            owners = self._file_owners.get(file, set())
            if owners and agent_id not in owners:
                conflicts.append(file)
            else:
                owners.add(agent_id)
                self._file_owners[file] = owners
        return conflicts
```

**UI display:**
```
┌──────────────────────────────────────────────────────────────┐
│  🤖 Parallel Agents  [3/3 slots]                             │
│                                                              │
│  Slot 1 ● running                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 🛡️ "Fix auth bugs"                    Step 12/50    │   │
│  │ ⏱️ 8:23  💰 $1.34  🔴 2 issues found               │   │
│  │ [⏸️ Pause]  [⏹️ Stop]                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Slot 2 ● running                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ⚙️ "Build settings page"               Step 5/50    │   │
│  │ ⏱️ 2:51  💰 $0.45  🟢 clean sweep                   │   │
│  │ [⏸️ Pause]  [⏹️ Stop]                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Slot 3 ○ completed ✅                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 🧪 "QA signup flow"              Done in 15 steps   │   │
│  │ ⏱️ 12:03  💰 $2.10  0 issues remaining              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Queue: "Test Stripe webhook handling"                     │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/parallel_agents.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — add `run_parallel_max()` method
- MODIFY: `orchestrator/cli.py` — add `max --parallel` flag
- DEPENDS ON: Phase X1 (Autonomous Agent — AutonomousAgent base)
- DEPENDS ON: Phase D1 (Multiple Contexts — context isolation)

#### Verification Gate

```bash
# Run 3 Max agents in parallel on different goals
python -m orchestrator max --parallel 3 --goals "fix auth","build settings","qa signup"
# Verify: all 3 run simultaneously, no file conflicts, each completes independently
```

---

## Phase X3: Slash Command Integration Discovery (100+ Built-in)

### Objective

Add slash command (`/`) integration discovery in the chat interface. Type `/ChatGPT`, `/Google Maps`, `/PDF Generation`, `/Stripe`, etc. to instantly wire up integrations. The orchestrator recognizes the command, fetches the integration template, and generates code to connect it.

### Current State

- `ArchitectureAnalyzer._detect_integration()` — keyword detection
- `IntegrationGenerator` (Phase R5) — 50+ data source templates
- No slash command interface for integration discovery
- No 100+ built-in integration catalog

### Implementation

#### X3.1 — Create integration slash command system

```python
"""
Slash Command Integration System — /command → integration code.
==================================================================

Type /<integration> in a prompt to wire up integrations instantly.
"""


@dataclass
class IntegrationCommand:
    """A slash-command integration."""
    name: str  # "chatgpt", "google-maps", "pdf-generation"
    display_name: str  # "ChatGPT", "Google Maps", "PDF Generation"
    category: str  # "ai", "maps", "data", "files", "media", "payments", "devtools"
    description: str
    example_prompts: list[str]
    code_template: str  # Python/TypeScript connection code
    requires_api_key: bool = False
    requires_env_vars: list[str] = field(default_factory=list)
    credit_cost: float = 0.0  # Additional credit cost per use

    def to_slash_command(self) -> str:
        """Format for slash command hint display."""
        return f"/{self.name} — {self.description}"


class IntegrationRegistry:
    """Registry of 100+ built-in integrations.

    Categories:
    - AI Models: ChatGPT, Claude, Gemini, Kimi, GPT-4 Vision, DALL-E, Flux
    - UI Libraries: shadcn/ui, Chakra UI, Radix UI, Headless UI
    - Location/Maps: Google Maps, Mapbox, Weather, Place Autocomplete
    - Data: Airtable, Supabase, PostgreSQL, MongoDB, Redis
    - Files: PDF Generation, PDF Parser, File Converter, File Upload
    - Media: Image Generation, Audio Transcription, Text-to-Speech, Charts
    - Communication: Resend, Slack, Discord, X (Twitter)
    - Payments: Stripe, RevenueCat, PayPal
    - Search: Google Search, Web Scraper, Exa, Perplexity
    - Dev Tools: Code Runner, QR Code, Domain Inspector, Validate Emails
    """

    BUILTIN_INTEGRATIONS: dict[str, IntegrationCommand] = {
        "chatgpt": IntegrationCommand(
            name="chatgpt",
            display_name="ChatGPT",
            category="ai",
            description="OpenAI's ChatGPT for text generation and conversation",
            example_prompts=[
                "Add an AI chatbot to my app using /chatgpt",
                "Use /chatgpt to analyze user feedback",
            ],
            code_template="{{orchestrator}}/integrations/openai/chat_completion.py",
            requires_api_key=True,
            requires_env_vars=["OPENAI_API_KEY"],
        ),
        "google-maps": IntegrationCommand(
            name="google-maps",
            display_name="Google Maps",
            category="maps",
            description="Embed interactive maps with markers and routes",
            example_prompts=[
                "Show store locations on /google-maps",
                "Build a delivery tracker with /google-maps",
            ],
            code_template="{{orchestrator}}/integrations/google/maps.tsx",
            requires_api_key=True,
            requires_env_vars=["GOOGLE_MAPS_API_KEY"],
        ),
        "pdf-generation": IntegrationCommand(
            name="pdf-generation",
            display_name="PDF Generation",
            category="files",
            description="Create and download PDFs programmatically",
            example_prompts=[
                "Let users download invoices as PDFs using /pdf-generation",
                "Build a report generator with /pdf-generation",
            ],
            code_template="{{orchestrator}}/integrations/pdf/generate.py",
            requires_api_key=False,
        ),
        "stripe": IntegrationCommand(
            name="stripe",
            display_name="Stripe",
            category="payments",
            description="Accept payments with Stripe checkout and subscriptions",
            example_prompts=[
                "Add a checkout flow with /stripe",
                "Create a subscription management page using /stripe",
            ],
            code_template="{{orchestrator}}/integrations/stripe/checkout.py",
            requires_api_key=True,
            requires_env_vars=["STRIPE_SECRET_KEY", "STRIPE_PUBLISHABLE_KEY"],
        ),
        "shadcn-ui": IntegrationCommand(
            name="shadcn-ui",
            display_name="shadcn/ui",
            category="ui",
            description="Modern React component library with clean design system",
            example_prompts=[
                "Style my dashboard with /shadcn-ui components",
                "Build a settings panel using /shadcn-ui",
            ],
            code_template="{{orchestrator}}/integrations/shadcn/install.sh",
            requires_api_key=False,
        ),
        # ... 95+ more integration definitions
    }

    def resolve_command(self, command_text: str) -> IntegrationCommand | None:
        """Resolve a slash command to its integration.

        Handles: "/chatgpt", "/Google Maps", "/pdf generation", etc.
        """
        normalized = command_text.lstrip("/").lower().replace(" ", "-")
        return self.BUILTIN_INTEGRATIONS.get(normalized)

    def suggest_integrations(self, prompt: str) -> list[IntegrationCommand]:
        """Suggest integrations based on prompt content.

        Uses keyword matching to recommend relevant integrations.
        """
        keywords = {
            "chat": ["chatgpt", "claude", "ai"],
            "map": ["google-maps", "mapbox", "place-autocomplete"],
            "pdf": ["pdf-generation", "pdf-parser"],
            "pay": ["stripe", "revenuecat"],
            "email": ["resend", "sendgrid"],
            "style": ["shadcn-ui", "chakra-ui"],
            "search": ["google-search", "web-scraper", "perplexity"],
        }

        suggestions = []
        prompt_lower = prompt.lower()
        for keyword, integrations in keywords.items():
            if keyword in prompt_lower:
                for name in integrations:
                    cmd = self.BUILTIN_INTEGRATIONS.get(name)
                    if cmd:
                        suggestions.append(cmd)

        return suggestions[:5]  # Top 5 suggestions

    def search_integrations(self, query: str) -> list[IntegrationCommand]:
        """Search the integration registry."""
        query_lower = query.lower()
        return [
            cmd for cmd in self.BUILTIN_INTEGRATIONS.values()
            if query_lower in cmd.name.lower()
            or query_lower in cmd.display_name.lower()
            or query_lower in cmd.description.lower()
        ]

    def generate_integration_code(
        self, command: IntegrationCommand, context: dict
    ) -> str:
        """Generate integration code from a template.

        Substitutes {{variables}} with context values.
        """
        template_path = Path(command.code_template.replace("{{orchestrator}}", str(self._templates_dir)))
        template = template_path.read_text()

        # Substitute variables
        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))

        return template
```

**Chat UI with slash commands:**
```
┌──────────────────────────────────────────────────────────────┐
│  💬 Ask anything...                                          │
│                                                              │
│  /cha█                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ /chatgpt — OpenAI's ChatGPT                    AI    │   │
│  │ /charts — Interactive data visualizations   Media   │   │
│  │ /chakra-ui — Accessible React components       UI    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Suggestions for "payments app":                             │
│  /stripe  /revenuecat  /pdf-generation  /resend             │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/integration_registry.py` (~400 lines)
- NEW: `orchestrator/integrations/slash/` — 100+ integration templates
- MODIFY: `ide_frontend/src/components/ChatInput.tsx` — add slash command autocomplete
- MODIFY: `orchestrator/engine.py` — resolve slash commands in prompts
- DEPENDS ON: Phase R5 (Data Source Integration — integration template format)

#### Verification Gate

```bash
# Type /stripe in a prompt, verify integration code generated
# Type /chatgpt, verify AI model wired with streaming + structured outputs
# Search integrations, verify results filter by category
```

---

## Phase X4: Multi-Mode Agent Selector (Auto/Discuss/Plan/Think/Fast)

### Objective

Add a mode selector with 6 modes: Auto (picks best), Discussion (chat only, no code), Plan (implementation plan first), Thinking (accurate, slower), Fast (quick, fewer credits), and Max (autonomous). Each mode has different iteration strategies, model tiers, and cost profiles.

### Current State

- Phase N1 (Ask Mode) — query-only mode
- Phase N4 (Brainstorming) — plan with clarifying questions
- Phase W1 (Agent Profiles) — 5 profiles (Standard, Max, Creative, Conservative, Research)
- No unified mode selector combining all these

### Implementation

#### X4.1 — Create unified mode selector

```python
"""
Multi-Mode Agent Selector — 6 execution modes with different strategies.
==========================================================================

Modes (increasing autonomy + cost):
1. Discussion — Chat only, no code generation, 0 executions
2. Plan — Create implementation plan, no code, ~1 LLM call
3. Fast — Quick generation, 1 iteration, budget models, ~$0.01
4. Thinking — Accurate generation, 3 iterations, premium models, ~$0.05
5. Auto — Picks best mode for the task, adaptive
6. Max — Autonomous loop, test→fix→repeat, ~$1.00

Each mode inherits from a profile (Phase W1) but adds execution strategy.
"""

class AgentMode(str, Enum):
    DISCUSSION = "discussion"  # Chat only
    PLAN = "plan"             # Plan, no execute
    FAST = "fast"             # Quick, 1 iteration
    THINKING = "thinking"     # Accurate, 3 iterations
    AUTO = "auto"             # Adaptive
    MAX = "max"               # Autonomous

@dataclass
class ModeConfig:
    mode: AgentMode
    display_name: str
    description: str
    icon: str

    # Inherits from profile
    profile: AgentProfile  # From Phase W1

    # Execution strategy
    generates_code: bool
    max_iterations: int
    requires_browser: bool = False
    runs_in_background: bool = False
    credit_multiplier: float = 1.0

    # Best for
    best_for: list[str] = field(default_factory=list)


MODE_CONFIGS: dict[AgentMode, ModeConfig] = {
    AgentMode.DISCUSSION: ModeConfig(
        mode=AgentMode.DISCUSSION,
        display_name="Discussion",
        description="Chat without making changes",
        icon="💬",
        profile=AgentProfile.STANDARD,
        generates_code=False,
        max_iterations=0,
        best_for=["Asking questions", "Getting feedback", "Brainstorming ideas"],
    ),
    AgentMode.PLAN: ModeConfig(
        mode=AgentMode.PLAN,
        display_name="Plan",
        description="Create a plan before building",
        icon="📋",
        profile=AgentProfile.STANDARD,
        generates_code=False,
        max_iterations=1,
        best_for=["Complex features", "Want to review approach", "Architecture decisions"],
    ),
    AgentMode.FAST: ModeConfig(
        mode=AgentMode.FAST,
        display_name="Fast",
        description="Quick generation, fewer credits",
        icon="⚡",
        profile=AgentProfile.STANDARD,
        generates_code=True,
        max_iterations=1,
        credit_multiplier=0.5,
        best_for=["Simple tasks", "UI changes", "Quick edits"],
    ),
    AgentMode.THINKING: ModeConfig(
        mode=AgentMode.THINKING,
        display_name="Thinking",
        description="More accurate, slightly slower",
        icon="🧠",
        profile=AgentProfile.MAX,
        generates_code=True,
        max_iterations=3,
        credit_multiplier=1.0,
        best_for=["Most development tasks", "Bug fixes", "Feature additions"],
    ),
    AgentMode.AUTO: ModeConfig(
        mode=AgentMode.AUTO,
        display_name="Auto",
        description="Picks the best mode automatically",
        icon="🎯",
        profile=AgentProfile.STANDARD,
        generates_code=True,
        max_iterations=3,
        best_for=["When unsure which mode to use"],
    ),
    AgentMode.MAX: ModeConfig(
        mode=AgentMode.MAX,
        display_name="Max",
        description="Autonomous: builds, tests, fixes on its own",
        icon="🚀",
        profile=AgentProfile.MAX,
        generates_code=True,
        max_iterations=50,
        requires_browser=True,
        runs_in_background=True,
        credit_multiplier=5.0,
        best_for=["Complex goals", "Pre-launch testing", "Autonomous debugging"],
    ),
}
```

**Mode selector UI:**
```
┌──────────────────────────────────────────────────────────────┐
│  💬 Chatbox                                     Mode: [▼]   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ● Auto    🎯  Picks best mode automatically        │   │
│  │ ○ Discuss  💬  Chat without making changes         │   │
│  │ ○ Plan    📋  Create a plan before building        │   │
│  │ ○ Fast    ⚡  Quick, fewer credits (0.5x)          │   │
│  │ ○ Think   🧠  More accurate, slower (1x)           │   │
│  │ ○ Max     🚀  Autonomous: test→fix→repeat (5x)     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/agent_modes.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — integrate mode selector, execute based on mode
- MODIFY: `orchestrator/cli.py` — add `--mode` flag to all commands
- MODIFY: `orchestrator/agent_profiles.py` — map modes to profiles
- DEPENDS ON: Phase W1 (Agent Profiles — profile definitions)
- DEPENDS ON: Phase X1 (Autonomous Agent — Max mode implementation)

#### Verification Gate

```bash
# Run in Discussion mode, verify no code generated
# Run in Plan mode, verify plan returned without execution
# Run in Fast mode, verify 1 iteration, budget models
# Run in Thinking mode, verify 3 iterations, premium models
# Run in Auto mode, verify adaptive mode selection
```

---

## Phase X5: Template Marketplace (Sell/Buy Templates for Credits)

### Objective

A marketplace where creators can publish project templates for credits, with review, rating, and search. Buyers browse, purchase, and fork templates into new projects. Different from the existing TemplateRegistry (Phase V7) which is just a local catalog.

### Current State

- Phase V7 (Templates) — local template registry with categories
- No marketplace concept
- No credit/payment system for templates
- No review/rating system
- No creator/buyer distinction

### Implementation

#### X5.1 — Extend TemplateRegistry with marketplace support

```python
@dataclass
class TemplateListing:
    """A template available in the marketplace."""
    id: str
    name: str
    creator_id: str
    creator_name: str
    description: str
    category: str
    tags: list[str]
    price_credits: int  # 0 = free
    screenshots: list[str]  # Base64 or URLs
    rating: float = 0.0
    review_count: int = 0
    download_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "draft"  # "draft", "submitted", "approved", "rejected", "delisted"
    review_notes: str = ""
    template_config: dict = field(default_factory=dict)  # From Phase V7


class TemplateMarketplace:
    """Marketplace where creators sell and buyers purchase templates.

    Features:
    - Browse by category, trending, highest rated, newest
    - Purchase with credits (free = instant)
    - Submit template for review (1-2 business days)
    - Leave reviews and ratings (one per user per template)
    - Seller dashboard (sales, earnings, manage templates)
    - Edit listing (title, description, price, tags, screenshots)
    - Delist/remove templates
    """

    def __init__(self, credit_manager, review_service):
        self._listings: dict[str, TemplateListing] = {}
        self._reviews: dict[str, list[dict]] = {}
        self._credit_mgr = credit_manager
        self._review_service = review_service

    def browse(
        self,
        category: str | None = None,
        sort: str = "trending",
        query: str | None = None,
        limit: int = 20,
    ) -> list[TemplateListing]:
        """Browse available templates."""

    def purchase(self, template_id: str, buyer_id: str) -> str:
        """Purchase a template — deducts credits, forks into new project.

        Returns: New project ID
        """

    def submit_for_review(self, template_id: str) -> bool:
        """Submit a template for marketplace review."""

    def review(self, template_id: str, user_id: str, rating: int, text: str = "") -> None:
        """Leave a review and rating on a template."""

    def get_seller_dashboard(self, seller_id: str) -> dict:
        """Get seller's sales, earnings, and template listings."""
```

**Marketplace UI:**
```
┌──────────────────────────────────────────────────────────────┐
│  🛒 Template Marketplace                                     │
│                                                              │
│  🔍 [________________]  [Category ▼]  [Sort: Trending ▼]    │
│                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐           │
│  │ 📱 E-commerce App   │ │ 📊 Analytics Dash   │           │
│  │ by @alice           │ │ by @bob             │           │
│  │ ⭐ 4.8 (124)        │ │ ⭐ 4.5 (89)         │           │
│  │ 2,340 downloads     │ │ 1,892 downloads     │           │
│  │ 💰 50 credits       │ │ 🆓 Free             │           │
│  │ [Preview] [Use]     │ │ [Preview] [Use]     │           │
│  └─────────────────────┘ └─────────────────────┘           │
│                                                              │
│  Your Templates  |  Sales Dashboard  |  Submit Template     │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/template_marketplace.py` (~400 lines)
- MODIFY: `orchestrator/template_registry.py` — add marketplace integration
- MODIFY: `orchestrator/credits.py` — add credit deduction for template purchases
- MODIFY: `orchestrator/cli.py` — add `marketplace` command group
- MODIFY: `ide_frontend/src/components/Marketplace.tsx` — UI component
- DEPENDS ON: Phase V7 (Templates — TemplateRegistry base)

#### Verification Gate

```bash
# Browse marketplace, purchase a template, verify forked project
# Submit a template for review, verify approval workflow
# Leave a review, verify rating updates
```

---

## Integration with Existing Plans

```
Phase X1 (Autonomous Agent) ──────────────────────────────────────────────────┐
    │  Depends on Phase V3 (Browser-Use Agent), V5 (AutoFixer),               │
    │  Phase 5 (Browser Testing), N3 (Screenshot Debugger)                    │
Phase X2 (Parallel Agents) ────────────────────────────────────────────────────┤
    │  Depends on Phase X1 (Autonomous Agent), D1 (Multiple Contexts)         │
Phase X3 (Slash Commands) ─────────────────────────────────────────────────────┤
    │  Depends on Phase R5 (Data Source Integration templates)                 │
Phase X4 (Multi-Mode Selector) ────────────────────────────────────────────────┤
    │  Depends on Phase W1 (Agent Profiles), X1 (Autonomous Agent)            │
Phase X5 (Template Marketplace) ────────────────────────────────────────────────┘
    Depends on Phase V7 (Templates — TemplateRegistry base)
```

## Create.xyz-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| X1 | Autonomous Agent | 1 | 3 | ~500 | 4-5 |
| X2 | Parallel Agents | 1 | 3 | ~400 | 3-4 |
| X3 | Slash Commands | 2 | 3 | ~400 | 2-3 |
| X4 | Multi-Mode Selector | 1 | 3 | ~300 | 2-3 |
| X5 | Template Marketplace | 1 | 3 | ~400 | 3-5 |
| **Create.xyz Subtotal** | **5 phases** | **6** | **15** | **~2,000** | **14-20** |

## Combined Grand Total (All Ten Sources)

| # | Source | Phases | New Files | Modified Files | Est. Days |
|---|--------|--------|-----------|---------------|-----------|
| 1 | Replit | 1-6 | 5 | 14 | 13-18 |
| 2 | Lovable | 7-10 | 4 | 10 | 9-13 |
| 3 | UI | U1-U7 | 20 | 15 | 17-23 |
| 4 | Newly | N1-N7 | 2 | 21 | 10-14 |
| 5 | Base44 | B1-B7 | 14 | 14 | 14-21 |
| 6 | v0 | V1-V7 | 12 | 19 | 18-26 |
| 7 | Retool | R1-R7 | 10 | 16 | 16-23 |
| 8 | Dyad | D1-D6 | 4 | 17 | 11-17 |
| 9 | Bolt.new | W1-W5 | 2 | 15 | 10-15 |
| 10 | Create.xyz | X1-X5 | 6 | 15 | 14-20 |
| **Grand Total** | **61** | **79** | **156** | **132-190** |
