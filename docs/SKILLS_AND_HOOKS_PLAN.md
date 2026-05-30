# Skills & Hooks Plan — AI Orchestrator

## Context

The project has a `.claude/` directory with:
- One skill (`mindmap`) — loads the architectural mind map
- Three hook stubs (all `sys.exit(0)` — non-functional)
- An extensive `settings.local.json` permissions allowlist

This plan fills the gap with project-specific skills and real hooks that automate the most frequent development actions: running tests, enforcing code quality, guarding architecture rules, and launching the dashboard.

---

## Skills

Skills live in `.claude/skills/<name>/SKILL.md`. Each skill is invoked by Claude when the user types `/<name>` or when the skill's trigger conditions match.

---

### 1. `orchestrator-run` — Run the orchestrator CLI

**File:** `.claude/skills/orchestrator-run/SKILL.md`

**Trigger:** User types `/orchestrator-run` or asks "run the orchestrator with…"

**Purpose:** Constructs and executes the correct `python -m orchestrator` invocation, picking the right subcommand and flags based on intent. Prevents common mistakes (wrong flag combos, missing `--budget`, stale `--resume` IDs).

**Content outline:**
```markdown
# orchestrator-run

## Quick Reference

| Goal | Command |
|------|---------|
| New project | `python -m orchestrator --project "<desc>" --criteria "<criteria>" --budget 2.0` |
| Resume | `python -m orchestrator --resume <project_id>` |
| Analyze codebase | `python -m orchestrator analyze <path>` |
| Build app | `python -m orchestrator build "<desc>"` |
| List projects | `python -m orchestrator --list-projects` |
| Dashboard | `python -m orchestrator dashboard` or `python -m orchestrator --dashboard` |
| Cache stats | `python -m orchestrator cache-stats` |
| With profiling | `ORCHESTRATOR_PROFILING=1 python -m orchestrator --project "..." --profile` |

## Required flags for `--project`
- `--project` (or first positional): description string
- `--budget`: float in USD (default 1.0); always set explicitly
- `--criteria`: acceptance criteria string (optional but recommended)

## Environment variables
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` — required
- `ORCHESTRATOR_PROFILING=1` — enable NexusScope profiling
- `ORCHESTRATOR_LOG_LEVEL=DEBUG` — verbose output
```

---

### 2. `test-smart` — Smart test runner

**File:** `.claude/skills/test-smart/SKILL.md`

**Trigger:** `/test-smart` or "run the tests" / "run unit tests" / "check if tests pass"

**Purpose:** Picks the right `pytest` invocation based on context (what was just edited, which markers apply, whether to include coverage). Surfaces the correct `--ignore` flags and marker combos without the user having to remember them.

**Content outline:**
```markdown
# test-smart

## Marker Reference

| Situation | Command |
|-----------|---------|
| Fast CI-safe subset | `pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --tb=short -q` |
| Unit tests only | `pytest tests/ -m unit -v` |
| Integration tests | `pytest tests/ -m integration -v` |
| Single file | `pytest tests/<file>.py -v` |
| Single function | `pytest tests/<file>.py::<function> -v` |
| With coverage | `pytest tests/ --cov=orchestrator --cov-report=term-missing` |
| Parallel | `pytest -n auto tests/ -m "not slow and not requires_api"` |
| NexusScope tests | `pytest tests/test_nexusscope.py -m unit -v --no-cov` |

## Ignored files (always excluded by pyproject.toml)
test_api.py, test_assembler_debug.py, test_basic_imports.py,
test_e2e_comprehensive.py, test_git_integration.py, test_mission_control.py,
test_startup.py, test_syntax.py, and ~25 others (see pyproject.toml addopts).

## Coverage baseline
Current fail_under = 12%. Target = 80%. Raise after adding new tests.

## When to use which marker
- After editing `orchestrator/` files → run `unit` first, then `integration`
- After editing `engine.py` → run full CI-safe subset
- After adding a new module → run `unit` for that module's test file
- Before committing → run CI-safe subset
```

---

### 3. `quality-check` — Full QA suite

**File:** `.claude/skills/quality-check/SKILL.md`

**Trigger:** `/quality-check` or "run linting" / "check code quality" / "run the linter"

**Purpose:** Runs the full quality pipeline (ruff → black → mypy → bandit) in the right order and explains how to interpret each tool's output.

**Content outline:**
```markdown
# quality-check

## Full pipeline (run in this order)

```bash
# 1. Lint + auto-fix
ruff check orchestrator/ --fix

# 2. Format check
black orchestrator/ --check
# To auto-format:
black orchestrator/

# 3. Type check (strict mode — see pyproject.toml overrides for legacy modules)
mypy orchestrator/

# 4. Security scan
bandit -r orchestrator/ -c pyproject.toml

# 5. Dependency vulnerability check
safety check

# 6. All pre-commit hooks
pre-commit run --all-files
```

## Interpreting output

### ruff
- Ignored rules: F401, F811, F821, F841, I001, E402, B007, B023, UP035, etc.
  (pre-existing baseline violations — see pyproject.toml [tool.ruff.lint] ignore list)
- Fix focus: new violations only (anything NOT in the ignore list)

### mypy (strict=true)
- Legacy modules in ignore_errors list (see pyproject.toml [[tool.mypy.overrides]])
- New code MUST pass strict checks — do not add new modules to the ignore list

### bandit
- B101 (assert_used) is skipped globally
- Focus on B-series injection, crypto, and subprocess issues

## Quick single-file check
```bash
ruff check orchestrator/<file>.py
black orchestrator/<file>.py --check
mypy orchestrator/<file>.py
```
```

---

### 4. `architecture-guard` — Three Unbreakable Rules checker

**File:** `.claude/skills/architecture-guard/SKILL.md`

**Trigger:** `/architecture-guard` or when about to edit `engine.py` or `models.py`

**Purpose:** Surfaces the three unbreakable rules before any architectural change, and provides a checklist to verify the change doesn't violate them. Also documents the correct placement for new business logic.

**Content outline:**
```markdown
# architecture-guard

## Three Unbreakable Rules

1. **`engine.py` = Mediator only**
   - New business logic goes into a NEW service module, not into engine.py
   - engine.py only wires services together (delegates, never implements)
   - If you find yourself writing a new algorithm in engine.py → STOP, create a service

2. **`models.py` = Pure data only**
   - No I/O, no asyncio, no behavior
   - Only dataclasses and enums
   - If you find yourself adding a method with side effects → STOP

3. **TDD without exceptions**
   - Write a failing test FIRST (RED)
   - Run it and confirm it fails with the expected error
   - Then implement the minimum code to pass (GREEN)
   - Commit only after GREEN

## Where does new business logic go?

| Type of logic | Correct location |
|--------------|-----------------|
| LLM routing/selection | `model_routing.py` or `planner.py` |
| Pipeline stage | `engine_core/stages/<new_stage>.py` |
| Validation rule | `validators.py` or `preflight.py` |
| Persistence | `state.py` or new repository module |
| Event handling | `events.py` or `hooks.py` |
| LLM provider adapter | `api_clients.py` (UnifiedClient pattern) |
| HTTP gateway | `gateway.py` |
| Budget tracking | `cost.py` |
| Resilience/retry | `resilience.py` or `rate_limiter.py` |
| Statistical profiling | `infrastructure/nexusscope/` |

## Dependency direction rule (Hexagonal Architecture)
```
Interfaces → Infrastructure → Application core → Domain models
```
Inner layers MUST NOT import from outer layers.

## Pre-change checklist
- [ ] Is there a failing test that proves the current behavior is wrong?
- [ ] Does the new code belong in engine.py, or in a service module?
- [ ] Does models.py only contain dataclasses and enums after my change?
- [ ] Does my import direction follow the hexagonal rule?
- [ ] Have I read docs/CODEBASE_MINDMAP.md for the relevant section?
```

---

### 5. `dashboard-start` — Launch the monitoring dashboard

**File:** `.claude/skills/dashboard-start/SKILL.md`

**Trigger:** `/dashboard-start` or "start the dashboard" / "open the dashboard"

**Purpose:** Starts the FastAPI dashboard server, confirms it's up, and provides the URL and key API routes. Also covers the NexusScope profiler routes added in the NexusScope plan.

**Content outline:**
```markdown
# dashboard-start

## Start the dashboard

```bash
# Option 1: Python script (cross-platform)
python start_dashboard.py

# Option 2: Direct uvicorn (if script missing)
python -m uvicorn orchestrator.dashboard_core.core:create_app \
  --factory --host 0.0.0.0 --port 8888 --reload

# Option 3: CLI entry point
python -m orchestrator dashboard
```

Dashboard URL: **http://localhost:8888**

## Key API routes

| Route | Purpose |
|-------|---------|
| `GET /` | Web UI |
| `GET /api/models` | Model rankings and stats |
| `GET /api/projects` | Project list |
| `GET /api/nexusscope/sessions` | NexusScope profiling sessions |
| `GET /api/nexusscope/report?fmt=html` | Profiling flame graph |

## With profiling enabled

```bash
ORCHESTRATOR_PROFILING=1 python start_dashboard.py
# Then visit: http://localhost:8888/api/nexusscope/sessions
```

## Install dashboard dependencies (if missing)
```bash
pip install -e ".[dashboard]"
```
```

---

## Hooks

Hooks live in `.claude/hooks/` as Python scripts and are registered in `.claude/settings.json` (or `settings.local.json`). The three existing stub files will be replaced with real implementations.

---

### Hook 1: `post_edit_ruff.py` — Auto-lint after edits (PostToolUse)

**Trigger:** After any `Write` or `Edit` tool call on a `*.py` file inside `orchestrator/`

**Action:** Runs `ruff check --fix` on the edited file. Prints a one-line summary. Never blocks (exit 0 always — linting is advisory at this stage, enforced at commit time).

```python
#!/usr/bin/env python3
"""PostToolUse hook: run ruff --fix on edited orchestrator Python files."""
import json, subprocess, sys

data = json.load(sys.stdin)
tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
if not path.endswith(".py") or "orchestrator" not in path.replace("\\", "/"):
    sys.exit(0)

result = subprocess.run(
    ["ruff", "check", "--fix", "--quiet", path],
    capture_output=True, text=True
)
if result.returncode != 0:
    # Print but don't block — violations are advisory here
    print(f"[ruff] {path}: {result.stdout.strip() or result.stderr.strip()}", file=sys.stderr)
else:
    print(f"[ruff] OK: {path}")
sys.exit(0)
```

---

### Hook 2: `pre_edit_core.py` — Architecture guard on protected files (PreToolUse)

**Trigger:** Before any `Write` or `Edit` on `engine.py` or `models.py`

**Action:** Prints a one-line reminder of the unbreakable rules for that file. Never blocks (informational only — the developer decides). This surfaces the rule at exactly the moment it matters.

```python
#!/usr/bin/env python3
"""PreToolUse hook: surface architecture rules before editing protected files."""
import json, sys

RULES = {
    "engine.py": (
        "[guard] engine.py = MEDIATOR ONLY. "
        "New business logic → new service module. "
        "engine.py only wires services together."
    ),
    "models.py": (
        "[guard] models.py = PURE DATA ONLY. "
        "No I/O, no asyncio, no behavior. "
        "Only dataclasses and enums."
    ),
}

data = json.load(sys.stdin)
tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
filename = path.replace("\\", "/").split("/")[-1]

if filename in RULES:
    print(RULES[filename], file=sys.stderr)

sys.exit(0)  # always allow — informational only
```

---

### Hook 3: `post_bash_pytest.py` — Test result summary (PostToolUse)

**Trigger:** After any `Bash` tool call whose command contains `pytest`

**Action:** Parses the short summary line from pytest output and prints a clean one-liner: `PASSED 42 | FAILED 3 | ERROR 1`. If failures exist, prints the failed test names. Helps Claude surface actionable information without re-reading raw output.

```python
#!/usr/bin/env python3
"""PostToolUse hook: parse pytest output and print a clean summary."""
import json, re, sys

data = json.load(sys.stdin)
tool = data.get("tool_name", "")
if tool != "Bash":
    sys.exit(0)

cmd = (data.get("tool_input") or {}).get("command", "")
if "pytest" not in cmd:
    sys.exit(0)

output = (data.get("tool_response") or {}).get("stdout", "")

# Extract the short test summary line: "5 failed, 42 passed, 1 error in 12.3s"
summary_match = re.search(
    r"(\d+ failed)?.*?(\d+ passed)?.*?(\d+ error)?.*?in \d+", output
)
short = re.search(r"=+ (.+?) =+\s*$", output, re.MULTILINE)
if short:
    print(f"[pytest] {short.group(1).strip()}")

# Print failed test names for quick triage
failed = re.findall(r"FAILED (tests/\S+)", output)
if failed:
    print("[pytest] Failed tests:")
    for f in failed[:10]:  # cap at 10
        print(f"  ✗ {f}")
    if len(failed) > 10:
        print(f"  … and {len(failed) - 10} more")

sys.exit(0)
```

---

## Settings registration

Add to `.claude/settings.local.json` (inside the existing `"hooks"` key):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/pre_edit_core.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/post_edit_ruff.py"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/post_bash_pytest.py"
          }
        ]
      }
    ]
  }
}
```

---

## Build order

1. Write skill files (no side effects, safe to do in any order):
   - `.claude/skills/orchestrator-run/SKILL.md`
   - `.claude/skills/test-smart/SKILL.md`
   - `.claude/skills/quality-check/SKILL.md`
   - `.claude/skills/architecture-guard/SKILL.md`
   - `.claude/skills/dashboard-start/SKILL.md`

2. Replace hook stubs with real implementations:
   - `.claude/hooks/post_edit_ruff.py`
   - `.claude/hooks/pre_edit_core.py`
   - `.claude/hooks/post_bash_pytest.py`

3. Register hooks in `.claude/settings.local.json`

4. Smoke-test hooks manually:
   ```bash
   echo '{"tool_name":"Edit","tool_input":{"file_path":"E:/Documents/Vibe-Coding/Ai Orchestrator/orchestrator/engine.py","old_string":"x","new_string":"x"}}' \
     | python .claude/hooks/pre_edit_core.py
   ```

5. Verify skill discovery:
   - Type `/test-smart` in Claude Code and confirm the skill loads
   - Type `/architecture-guard` and confirm the checklist appears

---

## Verification

```bash
# Hook smoke tests
echo '{"tool_name":"Edit","tool_input":{"file_path":".../orchestrator/engine.py"}}' \
  | python .claude/hooks/pre_edit_core.py
# Expected: prints the engine.py guard message

echo '{"tool_name":"Bash","tool_input":{"command":"pytest tests/"},"tool_response":{"stdout":"5 failed, 12 passed in 4.2s"}}' \
  | python .claude/hooks/post_bash_pytest.py
# Expected: [pytest] 5 failed, 12 passed in 4.2s

# Skill loading
# In Claude Code session: type /test-smart
# Expected: skill content loads with the marker reference table
```
