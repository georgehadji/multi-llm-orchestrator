# Hermes Agent — Deep Research Report

**Source:** https://github.com/nousresearch/hermes-agent  
**Version:** 0.14.0  
**Build:** Nous Research, MIT license  
**File counts:** 1,868 Python files · 1,034 Markdown files · ~17,000 tests across ~900 test files

---

## 1. What It Is

Hermes Agent is a **self-improving AI agent** built by Nous Research. Unlike a thin chat wrapper, it has a closed learning loop: it creates skills from experience, improves them during use, persists knowledge across sessions via curated memory, and can search its own past conversations. It's designed to run anywhere — a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle.

---

## 2. Core Architecture

### Agent Loop (`run_agent.py`)

The heart is the `AIAgent` class in `run_agent.py` (~12k LOC, being actively refactored into the `agent/` package). The conversation loop in `run_conversation()` is:

```
while (api_call_count < max_iterations AND budget.remaining > 0) OR grace_call:
    response = client.chat.completions.create(model, messages, tools)
    if tool_calls:
        for each tool_call:
            result = handle_function_call(name, args)
            append_tool_result(messages)
    else:
        return response.content
```

Key properties of the loop:
- **Entirely synchronous** — no async. Subagent parallelism runs via threads + `concurrent.futures`.
- **Budget-gated** — `max_iterations` (default 90) shared with subagents; per-turn `IterationBudget` reset.
- **Grace call** — one free iteration when budget is exhausted, so the model can wrap up.
- **Interrupt-aware** — checks `_interrupt_requested` before every iteration; gateways can interrupt mid-turn.
- **Multi-model fallback** — `classify_api_error()` routes 429/401/500 to fallback models with jittered backoff.
- **Context compression** — preflight and mid-loop, summarises middle turns when nearing context limits.

### System Prompt Caching

The system prompt is built once per session, persisted in SQLite, and reused across turns. For Anthropic paths this means the prompt prefix cache survives across gateway messages (which create a fresh `AIAgent` per turn). Prompt rebuilds only happen after compression events.

### Provider Adapters

The `agent/` directory has adapters for:
- **Chat Completions** (OpenAI-compatible) — default, works with 200+ models
- **Anthropic Messages** — native Anthropic SDK
- **Codex Responses** — OpenAI Responses API / GitHub Copilot
- **Bedrock** — AWS Bedrock
- **Gemini** — native Google adapter + CloudCode
- **LM Studio** — local model server

All model-provider plugins live under `plugins/model-providers/` with lazy discovery via `providers/__init__.py`.

---

## 3. CLI Architecture (`cli.py` + `hermes_cli/`)

`cli.py` (~11k LOC) is the main CLI orchestrator. Built with:
- **Rich** — banner/panels
- **prompt_toolkit** — input with autocomplete via `SlashCommandCompleter`
- **KawaiiSpinner** (`agent/display.py`) — animated faces during API calls

### Slash Command Registry (`hermes_cli/commands.py`)

Single source of truth: `COMMAND_REGISTRY` is a list of `CommandDef` objects. Every downstream consumer derives from it:
- CLI `process_command()` dispatches via `resolve_command()`
- Gateway uses `GATEWAY_KNOWN_COMMANDS` frozenset
- Telegram generates BotCommand menu
- Slack generates `/hermes` subcommand routing
- Autocomplete feeds from `COMMANDS` flat dict

Categories: Session, Configuration, Tools & Skills, Info, Exit.

### TUI (`ui-tui/` + `tui_gateway/`)

Alternative to the classic CLI, activated via `hermes --tui`:
- **Ink (React)** renders the screen — transcript, composer, prompts
- **Python `tui_gateway/`** owns sessions, tools, model calls, slash command logic
- **Transport:** newline-delimited JSON-RPC over stdio
- **Dashboard embeds TUI** via `pty_bridge.py` + WebSocket — not a rewrite of the chat surface

---

## 4. Tool System (`tools/`)

### Discovery

- `tools/registry.py` — module-level `Registry` singleton
- Any `tools/*.py` with a top-level `registry.register()` call is auto-imported — no manual import list
- Plugin tools discovered from `~/.hermes/plugins/<name>/` + pip entry points via `PluginManager`

### Tool Wiring

```
tools/registry.py            (no deps — imported by all tool files)
       ↑
tools/*.py                  (each calls registry.register() at import time)
       ↑
model_tools.py              (imports tools/registry + triggers discovery)
       ↑
run_agent.py, cli.py,       (consumers)
batch_runner.py, environments
```

### Toolsets (`toolsets.py`)

Tools are grouped into toolsets defined in a single `TOOLSETS` dict:
`browser`, `code_execution`, `delegation`, `file`, `terminal`, `memory`, `web`, `vision`, `image_gen`, `search`, `todo`, `cronjob`, `kanban`, `skills`, `tts`, `video`, `discord`, `homeassistant`, etc.

Each platform adapter picks a base toolset; `_HERMES_CORE_TOOLS` is the default bundle.

### Key Tools

| Tool | Purpose |
|------|---------|
| `terminal` | Shell execution with multiple backends (local, Docker, SSH, Modal, Daytona, Singularity, Vercel) |
| `delegate_task` | Spawn isolated subagents (sync, cap at 3 concurrent) |
| `read_file` / `patch` / `write_file` | File operations |
| `web_search` / `web_extract` | Web access |
| `vision_analyze` | Image understanding |
| `memory` / `session_search` | Cross-session recall |
| `cronjob` | Schedule recurring work |
| `skill_manage` | Create/improve skills |
| `browser_navigate` | Cloud browser via Browser Use |
| `execute_code` | Sandboxed Python execution |

---

## 5. Gateway (`gateway/`)

Multi-platform messaging gateway. Single `gateway/run.py` process connects to:

**Platform adapters** (`gateway/platforms/`): Telegram, Discord, Slack, WhatsApp, Signal, Matrix, DingTalk, WeChat (wecom/weixin), Feishu, QQ Bot, BlueBubbles, Email, SMS, Home Assistant, Webhook, API server, Yuanbao.

Architecture:
```
[Telegram] [Discord] [WhatsApp] ...
       \       |       /
        gateway/run.py
              |
        gateway/session.py
              |
        AIAgent (shared)
```

Each platform has a `base.py` adapter with message queuing, approval gating, and session lifecycle.

### Cron Scheduler (`cron/`)

Built-in cron with 3-minute hard interrupt, catchup windows, file locking. Supports duration ("30m", "2h"), every-phrase ("every monday 9am"), cron expressions, and ISO timestamps.

### Kanban Board (`plugins/kanban/`)

SQLite-backed multi-agent work queue. Profiles/workers collaborate on shared tasks. Dispatcher runs inside the gateway by default.

---

## 6. Learning Loop

### Memory System (`agent/memory_manager.py`)

- **Built-in store:** FTS5 session search with LLM summarization for cross-session recall
- **Pluggable providers:** Honcho, Mem0, Supermemory, ByteRover, Hindsight, Holographic, OpenViking, RetainDB
- **Periodic nudges:** `memory.nudge_interval` prompts the agent to review and consolidate
- **Prefetch:** External providers prefetch relevant context at conversation start

### Skills System

Two parallel surfaces:
- `skills/` — built-in skills, active by default, organized by category (github, mlops, etc.)
- `optional-skills/` — niche/heavy skills shipped but inactive, installed via `hermes skills install`

Skill creation: agents autonomously create skills after complex tasks via `skill_manage(action="create")`. Skills self-improve during use — tracked in `~/.hermes/skills/.usage.json` with per-skill `use_count`, `view_count`, `patch_count`.

### Curator (`agent/curator.py`)

Background skill lifecycle manager:
- Tracks usage on agent-created skills
- Auto-archives stale skills (never deletes — archives go to `.archive/`)
- Pinned skills exempt from auto-transitions
- Only touches `created_by: "agent"` provenance

---

## 7. Security Model

- **Plugin sandboxing:** `PLUGIN_SANDBOX_ENABLED=true`
- **Command approval:** per-command gate with `/approve`/`/deny`
- **DM pairing:** gateway platforms require session pairing
- **Container isolation:** Docker and Modal backends for code execution
- **Secret scanning:** tools are env-gated (`requires_env` in registry)
- **Credential pools:** automatic rotation on 429 with cooldown

---

## 8. Development Stats

| Metric | Value |
|--------|-------|
| Python files | 1,868 |
| Markdown files | 1,034 |
| Test files | ~900 |
| Total tests | ~17,000 |
| Test runner | pytest via `scripts/run_tests.sh` |
| Test isolation | subprocess-per-test (spawn, not fork) |
| Config format | YAML + `.env` for secrets |
| Package manager | uv (setuptools build) |
| Python | 3.11+ (requires >=3.11) |
| Linting | ruff (only PLW1514 encoding rule) |
| Type checking | `ty` (minimal, not strict) |

### Testing philosophy

- `scripts/run_tests.sh` enforces CI parity (unset credentials, TZ=UTC, LANG=C.UTF-8, `-n auto` xdist)
- Subprocess-per-test isolation via `tests/_isolate_plugin.py` — prevents state leakage between tests
- No change-detector tests — assert invariants ("all models have context lengths"), not snapshots ("model X is in the list")

---

## 9. Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Synchronous agent loop | Simpler debugging, no asyncio complexity; parallelism via threads for subagents |
| Single-file `COMMAND_REGISTRY` | All downstream consumers (CLI, gateway, Telegram menu, Slack, autocomplete) derive from one source |
| System prompt cached in SQLite | Preserves Anthropic prefix cache across gateway messages (fresh AIAgent per turn) |
| Lazy import of optional deps | Reduces blast radius of supply-chain attacks; `tools/lazy_deps.py` installs on first use |
| Upper-bounded dependencies | Hardened after litellm and mistralai supply-chain compromises |
| Profiles (multi-instance) | Fully isolated `HERMES_HOME` directories — separate config, memory, sessions, skills, gateway |
| Plugin system > core edits | PR #5295 removed 95 lines of hardcoded plugin argparse from `main.py` |
| TUI not rewritten in dashboard | Dashboard embeds real `hermes --tui` via pty — extends, doesn't replace |

---

## 10. Graphify Pipeline Status

The `graphify-out/` directory in this workspace was created fresh for this research. The full graphify pipeline was not run on the entire 3,705-file corpus (1,868 Python + 1,034 Markdown files) because:

1. **Scale:** Semantic extraction would require ~85 subagent chunks (20-25 files each), costing significant tokens
2. **Precision:** AST extraction + manual reading of 19 core architecture files yields higher-quality understanding than batching 2,900 files through LLM extraction
3. **No existing graph:** The `graphify-out/` directory was empty before this session

If you want to run the graphify pipeline specifically on this repo (or a subset of it), I can execute it — but for a single-research question, the structural analysis above is more efficient and accurate.
