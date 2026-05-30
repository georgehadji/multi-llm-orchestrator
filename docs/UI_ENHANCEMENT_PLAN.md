# UI Enhancement Plan: Replit & Lovable-Inspired Interface Features

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** UI gap analysis — Replit Agent, Lovable, and existing AI Orchestrator UI  
> **Status:** Draft — pending prioritisation

---

## Existing UI: What We Already Have

### Core Infrastructure
| Component | Technology | Capabilities |
|-----------|-----------|--------------|
| **IDE Backend** | FastAPI + WebSockets | Real-time per-session connections, event broadcasting, orchestrator bridge, REST API |
| **Mission Control** | Plugin-based DashboardCore | Gamification (XP/levels), real-time task tracking, model status, budget gauge, event log, toasts, confetti |
| **Frontend** | Vite + React (`ide_frontend/`) | SPA with WebSocket subscriptions, component library, dist build |
| **MCP Server** | stdio + HTTP mode | IDE integration via MCP protocol |

### Dashboard Views
- **Mission Control** (default): Gamified project overview with XP, achievements, budget ring
- **Ant Design View**: Alternative theme (unified under dashboard_core)
- **Enhanced Dashboard**: Extended metrics
- **Optimized Dashboard**: Performance-focused variant

### WebSocket Event Flow
```
Orchestrator → EventBus → DashboardCore → WebSocket broadcast
                                           ↓
                                    UI clients receive:
                                    - TASK_STARTED / TASK_COMPLETED / TASK_FAILED
                                    - PROJECT_COMPLETED (+ confetti)
                                    - BUDGET_WARNING
```

---

## Gap Analysis: What Replit & Lovable Have

### Replit UI Features

| Feature | Description | Existing? |
|---------|-------------|:---------:|
| **Plan Review Panel** | Interactive task list with approve/reject/refine before execution | ✗ |
| **App Preview Panel** | Embedded browser showing running app with video recording | ✗ |
| **Diff View** | Side-by-side code diff with approve/reject buttons | ✗ |
| **Checkpoint Timeline** | Visual checkpoint history with rollback arrows | ✗ |
| **Code Editor** | Monaco-based inline code editor with syntax highlighting | ✗ |
| **Console Output** | Real-time stdout/stderr streaming from running processes | Partial — event log exists but no streaming stdout |
| **Status Bar** | Bottom bar showing model, cost, tokens, latency per operation | Partial — budget gauge exists but not per-operation |

### Lovable UI Features

| Feature | Description | Existing? |
|---------|-------------|:---------:|
| **Visual Edit (Element Picker)** | Click UI elements in preview to select + edit them | ✗ |
| **Design Selector** | Pick from visual design previews (themes/components) before building | ✗ |
| **Project Comments** | Annotations on specific UI elements + code locations | ✗ |
| **Workspace Switcher** | Navigate between projects in a workspace | ✗ |
| **Knowledge Panel** | Sidebar showing workspace + project context being injected | ✗ |
| **Skills Panel** | Sidebar listing available skills and trigger descriptions | ✗ |
| **Design System Browser** | Browse connected design system components and tokens | ✗ |
| **Chat Input** | Prompt input with / commands, @ references, file attachments | ✗ |
| **Generation Progress** | Streaming token output with a progress bar per generation | ✗ |
| **Cost Breakdown** | Per-generation cost with model, tokens in, tokens out | Partial — budget tracking exists but not per-call |

---

## Phase U1: Visual Plan Review Panel (3-4 days)

### Objective
Extend Mission Control with an interactive plan review mode where tasks are displayed before execution, with approve/reject/refine/prioritize controls.

### Implementation

#### U1.1 — New Dashboard View: `PlanReviewView`

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 Plan: "Build Inventory REST API"                  [Build] │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Phase 1  │→│ Phase 2  │→│ Phase 3  │→│ Phase 4  │         │
│  │ Init     │  │ Auth     │  │ API      │  │ Tests    │         │
│  │ $0.12    │  │ $0.45    │  │ $0.89    │  │ $0.23    │         │
│  │ ✓        │  │ ✓        │  │ ⏳       │  │ ○        │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│                                                              │
│  Tasks                                          ▸ Details    │
│  ┌──────────────────────────────────────────┐   ┌─────────┐ │
│  │ ✓ setup.py — FastAPI scaffold     $0.12  │   │ Model:   │ │
│  │ ✓ models.py — Pydantic schemas   $0.08  │   │ Qwen 2.5 │ │
│  │ ⏳ auth.py — JWT middleware       $0.45  │   │ Est:     │ │
│  │   ○ add tests for auth endpoints  $0.15  │   │ $1.69    │ │
│  │   ✗ rate limiter (low priority)   $0.05  │   │          │ │
│  └──────────────────────────────────────────┘   │ [Approve]│ │
│                                                  │ [Refine] │ │
│  [+ Add Task]  [Reorder]  [Set Budget ▼ $2.00]  └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/dashboard_core/plan_review.py` (~400 lines)
- MODIFY: `orchestrator/dashboard_core/core.py` — register `PlanReviewView`
- MODIFY: `ide_frontend/src/components/` — add PlanMode component
- MODIFY: `orchestrator/ide_backend/websocket/handlers.py` — handle plan events

**WebSocket events:**
```
Client → Server:  plan:approve (task_ids)
                  plan:reject (task_ids, reason)
                  plan:reorder (task_ids[])
                  plan:add_task (task_json)
                  plan:set_budget (amount)
Server → Client:  plan:updated (full_plan)
                  plan:execution_started
                  plan:execution_complete
```

**Verification Gate:** Run a project in plan mode, approve 3 of 5 tasks, verify only 3 execute.

---

## Phase U2: App Preview + Console Panel (4-5 days)

### Objective
Embed an iframe-based app preview alongside real-time console output. Capture screenshots and video recordings of browser tests. Stream stdout/stderr from running processes.

### Implementation

#### U2.1 — Split-Pane Layout

```
┌──────────────────────────────────────────────────────────────┐
│  [Plan] [Code] [Preview] [Console]              ⚡ Connected │
├──────────────────────────────────────────────────────────────┤
│                    │                                          │
│   Code Editor      │  ┌────────────────────────┐            │
│   (Monaco)         │  │  App Preview            │            │
│                    │  │  http://localhost:3000  │            │
│                    │  │  ┌──────────────────┐  │            │
│                    │  │  │  Inventory App    │  │            │
│                    │  │  │  ┌────────────┐   │  │            │
│                    │  │  │  │ Add Item   │   │  │            │
│                    │  │  │  │ [_______]  │   │  │            │
│                    │  │  │  │ [Submit]   │   │  │            │
│                    │  │  │  └────────────┘   │  │            │
│                    │  │  └──────────────────┘  │            │
│                    │  │  [🔄 Refresh] [📹 Record] [📷 Snap] │
│                    │  └────────────────────────┘            │
│                    │  ┌────────────────────────┐            │
│                    │  │  Console Output         │            │
│                    │  │  > pytest tests/        │            │
│                    │  │  ...F..F...              │            │
│                    │  │  FAILED test_auth.py    │            │
│                    │  │  FAILED test_api.py     │            │
│                    │  └────────────────────────┘            │
├──────────────────────────────────────────────────────────────┤
│  Status: 🟢 GPT-4.1 | $0.0234 | 1.2k tokens | 340ms        │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `ide_frontend/src/components/AppPreview.tsx` (~200 lines)
- NEW: `ide_frontend/src/components/ConsolePanel.tsx` (~150 lines)
- NEW: `ide_frontend/src/components/CodeEditor.tsx` (~200 lines) — Monaco wrapper
- MODIFY: `ide_frontend/src/App.tsx` — split-pane layout with resizable panels
- MODIFY: `orchestrator/ide_backend/websocket/handlers.py` — console streaming endpoint
- NEW: `orchestrator/ide_backend/console_streamer.py` — subprocess stdout capture + WS pipe

**Integration with Phase 5 (Browser Tester):**
```
BrowserTester.test_app() → screenshots → display in preview
BrowserTester.test_app() → video → play in preview panel
BrowserTester.test_app() → errors → highlight in code editor
```

**Dependencies:**
- `monaco-editor` npm package for code editor
- Phase 5 (Browser Tester) for video/screenshot rendering

**Verification Gate:** Start a generated app, see it in the preview panel, run a test, see console output streaming live.

---

## Phase U3: Diff View + Checkpoint Timeline (2-3 days)

### Objective
Show visual diffs when sandbox tasks complete. Display a timeline of checkpoints with rollback controls.

### Implementation

#### U3.1 — Diff View Component

```
┌──────────────────────────────────────────────────────────────┐
│  ✅ Task "auth.py" completed in sandbox                      │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │ main (before)        │  │ sandbox (after)      │           │
│  ├─────────────────────┤  ├─────────────────────┤           │
│  │ def login(user, pw): │  │ def login(user, pw): │           │
│  │     return True      │  │     import bcrypt    │           │
│  │                      │  │     hash = bcrypt... │           │
│  │                      │  │     if not hash:     │           │
│  │                      │  │         raise 401    │           │
│  └─────────────────────┘  └─────────────────────┘           │
│                                                              │
│  Changes: +4 lines, -1 line, 1 file modified                │
│  Tests: 12/12 passed  ✅                                    │
│                                                              │
│  [Approve & Merge]  [Reject]  [View Full Diff]              │
└──────────────────────────────────────────────────────────────┘
```

#### U3.2 — Checkpoint Timeline

```
┌──────────────────────────────────────────────────────────────┐
│  📍 Checkpoints                          [Create Checkpoint] │
│                                                              │
│  ◉ 14:32  after:auth.py           ⬅ current                │
│  │  ○ 14:28  after:models.py                                │
│  │  ○ 14:25  pre_retry:setup.py                             │
│  │  ○ 14:22  project_started                                │
│  │                                                           │
│  [Rollback to selected]  [Diff with current]  [Branch here] │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `ide_frontend/src/components/DiffView.tsx` (~200 lines) — uses `react-diff-viewer`
- NEW: `ide_frontend/src/components/CheckpointTimeline.tsx` (~150 lines)
- MODIFY: `orchestrator/ide_backend/api/routes.py` — add GET /api/checkpoints, GET /api/diff
- DEPENDS ON: Phase 1 (Checkpoints) and Phase 4 (Sandbox Tasks)

**Verification Gate:** Complete a task, see the diff in the UI, approve the merge, verify files are updated. Create a checkpoint, see it on the timeline, rollback, verify state is restored.

---

## Phase U4: Visual Edit + Element Picker (4-5 days)

### Objective
Click elements in the app preview to select them, see their source code location, and edit them inline. Inspired by Lovable's visual editing.

### Implementation

#### U4.1 — Element Picker Mode

```
┌──────────────────────────────────────────────────────────────┐
│  [Plan] [Code] [Preview 🖱️ Pick] [Console]                  │
├──────────────────────────────────────────────────────────────┤
│                    │                                          │
│   Code Editor      │  ┌────────────────────────┐            │
│                    │  │  App Preview            │            │
│                    │  │  ┌──────────────┐       │            │
│                    │  │  │ ┌──────────┐ │       │            │
│                    │  │  │ │ ▓▓▓▓▓▓▓▓▓▓│ │◀── ● │            │
│                    │  │  │ │ Login     │ │       │            │
│                    │  │  │ └──────────┘ │       │            │
│                    │  │  └──────────────┘       │            │
│                    │  └────────────────────────┘            │
│                    │  ┌────────────────────────┐            │
│  src/auth.py:42    │  │  Element: Button        │            │
│                    │  │  Tag: <button>          │            │
│  def login():      │  │  Text: "Login"          │            │
│      return (      │  │  Color: #2563EB         │            │
│          <button   │  │  Size: 120×40px         │            │
│  >>>      class={  │  │                          │            │
│            "btn"   │  │  [✏️ Edit Text]          │            │
│          }>        │  │  [🎨 Change Color]       │            │
│          Login     │  │  [📏 Adjust Size]        │            │
│        </button>   │  │  [🗑️ Remove]             │            │
│                    │  └────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

**How it works:**
1. Preview is an iframe loading the generated app
2. When "Pick" mode is on, a content script is injected that attaches click handlers to all elements
3. Clicking an element sends `postMessage(data)` to the parent frame
4. Parent frame maps the element to its source file location (via a mapping generated during build)
5. Code editor scrolls to and highlights the relevant source code
6. Property panel shows editable properties of the selected element

**Technical approach:**
```javascript
// Injected into preview iframe
document.addEventListener('click', (e) => {
    e.preventDefault();
    const el = e.target;
    const rect = el.getBoundingClientRect();
    const computedStyles = window.getComputedStyle(el);
    
    // Find source mapping (injected during build)
    const sourceMap = window.__AI_SOURCE_MAP__;
    const source = sourceMap?.findLocation(el.tagName, el.className, el.id);
    
    window.parent.postMessage({
        type: 'element:selected',
        element: {
            tag: el.tagName.toLowerCase(),
            text: el.textContent?.trim().substring(0, 100),
            attributes: Array.from(el.attributes).reduce((a, attr) => 
                ({ ...a, [attr.name]: attr.value }), {}),
            styles: {
                color: computedStyles.color,
                backgroundColor: computedStyles.backgroundColor,
                fontSize: computedStyles.fontSize,
                fontFamily: computedStyles.fontFamily,
            },
            boundingBox: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
            sourceLocation: source,
        }
    }, '*');
    
    // Draw highlight overlay
    drawHighlight(rect);
});
```

**File changes:**
- NEW: `ide_frontend/src/components/ElementPicker.tsx` (~250 lines)
- NEW: `ide_frontend/src/components/PropertyPanel.tsx` (~200 lines)
- NEW: `ide_frontend/src/components/SourceMap.ts` (~150 lines) — element-to-fileLocation mapping
- MODIFY: `ide_frontend/src/components/AppPreview.tsx` — add pick mode toggle
- NEW: `ide_frontend/public/picker-script.js` — injected content script
- NEW: `orchestrator/source_mapper.py` — generates `__AI_SOURCE_MAP__` during build (~100 lines)

**Dependencies:**
- Phase U2 (App Preview Panel) — picker needs the iframe preview
- Phase 5 (Browser Tester) — picks up the running app URL

**Verification Gate:** Load generated app in preview, enter pick mode, click a button, see it selected in the property panel, edit the text property, see the change in the code editor.

---

## Phase U5: Knowledge + Skills + Design Panels (2-3 days)

### Objective
Add collapsible sidebar panels showing active knowledge context, available skills, and connected design systems. Inspired by Lovable's persistent context visibility.

### Implementation

#### U5.1 — Sidebar Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  🏠 Workspace: My Projects                                         │
│                                                                    │
│  📂 Projects                        ┌─────────────────────────┐   │
│  ├── @inventory-app (active)        │  App Preview             │   │
│  ├── @brand-site                     │                         │   │
│  ├── @design-system                  │                         │   │
│  └── [+ New Project]                 │                         │   │
│                                      └─────────────────────────┘   │
│  🧠 Knowledge                                                      │
│  ┌──────────────────────────────┐                                  │
│  │ Workspace:                    │                                  │
│  │ • Use FastAPI + SQLAlchemy    │                                  │
│  │ • Hexagonal architecture      │                                  │
│  │                               │                                  │
│  │ Project:                      │                                  │
│  │ • B2B inventory SaaS          │                                  │
│  │ • Store money in cents (int)  │                                  │
│  │ • Users: managers + staff     │                                  │
│  └──────────────────────────────┘                                  │
│                                                                    │
│  🎯 Skills                                                         │
│  ┌──────────────────────────────┐                                  │
│  │ 🔍 /launch-checklist         │                                  │
│  │    Use when about to ship    │                                  │
│  │ 📝 /release-notes            │                                  │
│  │    Generate changelog        │                                  │
│  │ 🛡️ /security-audit           │                                  │
│  │    Scan for vulnerabilities  │                                  │
│  └──────────────────────────────┘                                  │
│                                                                    │
│  🎨 Design System (inventory-ds)                                  │
│  ┌──────────────────────────────┐                                  │
│  │ Colors:  #2563EB #7C3AED     │                                  │
│  │ Fonts:   Inter 16px          │                                  │
│  │ Radius:  8px                 │                                  │
│  │ Library: Tailwind            │                                  │
│  │ Components: Button, Input... │                                  │
│  └──────────────────────────────┘                                  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 💬 Ask anything...  [/skills] [@projects] [📎 attach]        │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `ide_frontend/src/components/KnowledgePanel.tsx` (~150 lines)
- NEW: `ide_frontend/src/components/SkillsPanel.tsx` (~150 lines)
- NEW: `ide_frontend/src/components/DesignSystemPanel.tsx` (~150 lines)
- NEW: `ide_frontend/src/components/WorkspaceSwitcher.tsx` (~100 lines)
- NEW: `ide_frontend/src/components/ChatInput.tsx` (~200 lines) — with /, @, 📎 support
- MODIFY: `ide_frontend/src/App.tsx` — add collapsible sidebar
- MODIFY: `orchestrator/ide_backend/api/routes.py` — GET /api/knowledge, /api/skills, /api/design-system
- DEPENDS ON: Phase 7 (Knowledge), Phase 8 (Skills), Phase 9 (Cross-Project), Phase 10 (Design System Projects)

**Verification Gate:** Open sidebar, see active knowledge context, invoke a skill via /, see it loading.

---

## Phase U6: Real-Time Generation Progress (1-2 days)

### Objective
Show streaming token output with a progress bar, per-generation cost, and model information while tasks are executing.

### Implementation

#### U6.1 — Generation Progress Card

```
┌──────────────────────────────────────────────────────────────┐
│  🔄 Generating: auth.py                      Model: Qwen 2.5 │
│                                                              │
│  ████████████████████░░░░░░  68%                            │
│                                                              │
│  Streaming output:                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ def login(request: LoginRequest) -> TokenResponse:  │   │
│  │     """Authenticate user and return JWT token."""    │   │
│  │     user = await db.authenticate(                   │   │
│  │         request.email,                               │   │
│  │         request.password  ▓                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  📊 Tokens: 342 in | 218 out    ⏱️ Latency: 1.2s             │
│  💰 Cost: $0.0023                🔄 Retry: 1/3               │
│                                                              │
│  [Cancel]                                                    │
└──────────────────────────────────────────────────────────────┘
```

**WebSocket events:**
```
Server → Client:  generation:streaming (chunk, bytes_so_far, total_estimated)
                  generation:token_update (tokens_in, tokens_out, cost)
                  generation:complete (final_output, total_cost, model)
                  generation:retry (attempt_number, reason)
```

**File changes:**
- NEW: `ide_frontend/src/components/GenerationProgress.tsx` (~200 lines)
- MODIFY: `ide_frontend/src/App.tsx` — add progress panel at bottom of code editor
- MODIFY: `orchestrator/ide_backend/integration/orchestrator_bridge.py` — stream token events
- MODIFY: `orchestrator/api_clients.py` — emit streaming events during generation

**Verification Gate:** Run a task, see the progress bar filling, see streaming output updating in real time, see cost and tokens updating.

---

## Phase U7: Status Bar with Per-Operation Metrics (1 day)

### Objective
Bottom status bar showing active model, current cost, tokens consumed, last operation latency, and health indicators.

### Implementation

```
┌──────────────────────────────────────────────────────────────┐
│  🟢 gpt-4.1  │  💰 $0.0234  │  📊 1.2k/4.5k  │  ⏱️ 340ms   │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `ide_frontend/src/components/StatusBar.tsx` (~100 lines)
- MODIFY: `ide_frontend/src/App.tsx` — mount at bottom

---

## Implementation Order & Dependencies

```
Phase U1 (Plan Review) ──────────────────────────────────────────────────┐
    │  No UI dependencies — standalone dashboard view                      │
    │  Depends on: Phase 2 (Plan Workflow)                                 │
    │                                                                      │
Phase U2 (App Preview + Console) ─────────────────────────────────────────┤
    │  Depends on: Phase 5 (Browser Testing) — provides app URL            │
    │                                                                      │
Phase U3 (Diff View + Timeline) ──────────────────────────────────────────┤
    │  Depends on: Phase 1 (Checkpoints), Phase 4 (Sandbox Tasks)          │
    │                                                                      │
Phase U4 (Visual Edit + Element Picker) ──────────────────────────────────┤
    │  Depends on: Phase U2 (App Preview)                                   │
    │                                                                      │
Phase U5 (Knowledge + Skills + Design Panels) ────────────────────────────┤
    │  Depends on: Phase 7-10 (Lovable backend features)                   │
    │                                                                      │
Phase U6 (Generation Progress) ───────────────────────────────────────────┤
    │  No strong dependencies — event stream already exists                │
    │                                                                      │
Phase U7 (Status Bar) ────────────────────────────────────────────────────┘
    No dependencies — standalone component
```

## Parallelisation Strategy

Phases U1, U6, and U7 have no UI dependencies and can run in parallel.
Phases U2-U5 depend on backend feature phases (1-10).

## Total Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Days |
|-------|---------|-----------|----------------|-----------|
| U1 | Plan Review Panel | 2 | 3 | 3-4 |
| U2 | App Preview + Console | 4 | 3 | 4-5 |
| U3 | Diff View + Timeline | 2 | 1 | 2-3 |
| U4 | Visual Edit + Picker | 5 | 2 | 4-5 |
| U5 | Knowledge/Skills Sidebar | 5 | 2 | 2-3 |
| U6 | Generation Progress | 1 | 3 | 1-2 |
| U7 | Status Bar | 1 | 1 | 1 |
| **Total** | **7 phases** | **20** | **15** | **17-23** |

## Combined Grand Total

| Category | Phases | New Files | Modified Files | Est. Days |
|----------|--------|-----------|---------------|-----------|
| Backend (Replit) | 1-6 | 5 | 14 | 13-18 |
| Backend (Lovable) | 7-10 | 4 | 10 | 9-13 |
| UI (This Plan) | U1-U7 | 20 | 15 | 17-23 |
| **Grand Total** | **17** | **29** | **39** | **39-54** |
