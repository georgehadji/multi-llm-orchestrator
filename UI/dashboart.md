Design: Dark theme με JetBrains Mono + DM Sans typography, accent glows, monochrome surfaces με color accents μόνο σε data points — αποφεύγει εντελώς τα generic purple-gradient AI aesthetics.
Sections:

KPI Row: Budget, Tasks, Quality, Repair Cycles, Cache Hit Rate — live values με trend badges
Cost & Quality Timeline: 24h rolling AreaChart (cost + quality overlaid)
Cost by Model: PieChart + breakdown list — βλέπεις ακριβώς πού πάνε τα χρήματα
Agent Activity Log: Real-time feed με color-coded levels (Supervisor, Verifier, Chairman, Router, Circuit Breaker)
Task Execution Table: Status dots (animated pulse για running), model, score, cost, time, repairs
Agent Hierarchy: D3 tree visualization (Supervisor → Specialists → Tasks)
Circuit Breakers: Per-provider health cards (CLOSED/OPEN/HALF_OPEN) με health bars
Latency Distribution: BarChart per model
ARA Pipeline Methods: Usage + avg score per reasoning method

Τεχνικά: React + Recharts + D3, auto-refresh κάθε 3s, sticky header, animated transitions, responsive grid layout. Plug-and-play — αντικαθιστά τα 9 legacy dashboards σε ένα unified view.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DASHBOARD 2

full IDE-style interface που αντιστοιχεί σε ό,τι κάνουν τα Lovable, Replit, Base44, Bolt, Emergent — αλλά tailored στον orchestrator σου. Ακολουθεί breakdown:

Layout (3-Panel IDE Pattern)
Left Panel (420px): Chat Interface

Conversation history με user/assistant messages
Assistant "thinking" mode με progress steps (Analyzing → Architecture → Decomposing → Routing)
File badges στα responses (clickable → ανοίγει code view)
Cost + quality badges per response
Input area με: text area, Enter/Shift+Enter, @mention support
Context buttons: Attach file, URL, Git commit, Screenshot, Enhance prompt
Keyboard shortcuts display, budget tracker

Center Panel (200px, collapsible): File Tree

Hierarchical file explorer με folder toggle
Selected file highlighting
File type icons

Right Panel (flex): Preview / Code / Terminal

Preview tab: Live app preview (Swagger UI simulation)
Code tab: Syntax-highlighted code editor with line numbers
Terminal: Collapsible, shows pytest output, ruff checks, uvicorn server
URL bar με status dot, refresh, external link, copy buttons

Top Bar

Project name + branch selector
Mode selector: Build, Plan, Chat, Debug (αντιστοιχεί στα modes του orchestrator)
Model picker dropdown: Auto (tiered), Opus, Sonnet, DeepSeek, GPT-5.4, Gemini
Autonomy selector dropdown: Lite → Standard → Autonomous → Max (με color coding)
Deploy button (primary CTA)

Bottom Status Bar

Connection status, mode, model, autonomy
Budget tracking ($0.33 / $5.00)
Time elapsed, tasks progress, quality score
Cache hit rate, Python version, encoding

Design Decisions

Obsidian dark theme — zinc-based neutrals, not purple defaults
Berkeley Mono / JetBrains Mono for code, Inter for UI — developer-grade typography
No gradients, no glow excess — clean surfaces, subtle borders, precision spacing
Lucide icons throughout (consistent, lightweight)
Dropdown menus with click-away dismiss, hover states, active indicators