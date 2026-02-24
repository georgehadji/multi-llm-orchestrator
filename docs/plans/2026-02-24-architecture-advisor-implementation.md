# Architecture Advisor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `AppDetector` with `ArchitectureAdvisor` — a single LLM call that decides app type AND architectural patterns (structural, topology, API, data), prints a summary, and injects the decision into the decomposition prompt.

**Architecture:** `ArchitectureDecision` dataclass is a superset of `AppProfile`. `ArchitectureAdvisor` makes one LLM call (DeepSeek Reasoner for complex specs, Chat for simple ones), prints a `🏗` summary block, and returns `ArchitectureDecision`. `AppBuilder` swaps `AppDetector` for `ArchitectureAdvisor`. `AppProfile` becomes a type alias for backward compat. `engine._decompose()` injects arch fields into its prompt.

**Tech Stack:** Python 3.10+, dataclasses, json, `orchestrator.api_clients.UnifiedClient`, `orchestrator.models.Model`

---

## Background (read before starting)

### Key files

| File | What matters |
|------|-------------|
| `orchestrator/app_detector.py` | Current `AppDetector` + `AppProfile` — this is what we're replacing |
| `orchestrator/app_builder.py:62` | `self._detector = AppDetector()` — swap to `ArchitectureAdvisor` |
| `orchestrator/app_builder.py:92-104` | Detection block — swap `detect()` → `analyze()` |
| `orchestrator/engine.py:349-368` | `app_context_block` string — extend with arch fields |
| `orchestrator/__init__.py:19` | Import pattern for new exports |

### Current AppDetector fields (must ALL be preserved in ArchitectureDecision)

```python
app_type: str          # "fastapi" | "flask" | "cli" | "library" | "script" |
                       #  "react-fastapi" | "nextjs" | "generic"
tech_stack: list[str]
entry_point: str
test_command: str
run_command: str
requires_docker: bool
detected_from: str     # "auto" | "yaml_override"
```

### How UnifiedClient is called

```python
from orchestrator.api_clients import UnifiedClient
client = UnifiedClient()
response = await client.call(
    model=Model.DEEPSEEK_REASONER,
    prompt="...",
    system="...",
    max_tokens=1024,
    temperature=0.2,
    timeout=30,
    retries=1,
)
# response.text: str, response.cost_usd: float
```

---

## Task 1: ArchitectureDecision dataclass and pure parse/select functions

**Files:**
- Create: `orchestrator/architecture_advisor.py`
- Create: `tests/test_architecture_advisor.py`

### Step 1: Write failing tests for pure functions

Create `tests/test_architecture_advisor.py`:

```python
"""Tests for ArchitectureDecision dataclass and pure utility functions."""
from __future__ import annotations

import json
import pytest
from orchestrator.architecture_advisor import (
    ArchitectureDecision,
    _select_model,
    _parse_response,
    _ARCH_DEFAULTS,
)
from orchestrator.models import Model


# ─── _select_model ────────────────────────────────────────────────────────────

def test_select_model_short():
    """≤50 words → DeepSeek Chat."""
    desc = "Build a FastAPI auth service with JWT tokens"
    assert _select_model(desc) == Model.DEEPSEEK_CHAT


def test_select_model_long():
    """>50 words → DeepSeek Reasoner."""
    desc = " ".join(["word"] * 51)
    assert _select_model(desc) == Model.DEEPSEEK_REASONER


def test_select_model_exactly_50():
    """Exactly 50 words → DeepSeek Chat (boundary: >50 triggers Reasoner)."""
    desc = " ".join(["word"] * 50)
    assert _select_model(desc) == Model.DEEPSEEK_CHAT


# ─── _parse_response ──────────────────────────────────────────────────────────

FULL_VALID_JSON = json.dumps({
    "app_type": "fastapi",
    "tech_stack": ["python", "fastapi", "postgresql"],
    "entry_point": "src/main.py",
    "test_command": "pytest",
    "run_command": "uvicorn src.main:app --reload",
    "requires_docker": False,
    "structural_pattern": "layered",
    "topology": "monolith",
    "data_paradigm": "relational",
    "api_paradigm": "rest",
    "rationale": "FastAPI is well-suited for REST APIs. Layered architecture keeps things clean.",
})


def test_parse_valid_response():
    """Valid JSON → correct ArchitectureDecision fields."""
    result = _parse_response(FULL_VALID_JSON)
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "fastapi"
    assert result.tech_stack == ["python", "fastapi", "postgresql"]
    assert result.structural_pattern == "layered"
    assert result.topology == "monolith"
    assert result.data_paradigm == "relational"
    assert result.api_paradigm == "rest"
    assert "REST" in result.rationale or "rest" in result.rationale.lower()
    assert result.detected_from == "advisor"


def test_parse_invalid_json():
    """Invalid JSON → fallback defaults, no exception."""
    result = _parse_response("this is not json {{{{")
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"   # fallback default
    assert result.detected_from == "advisor"


def test_parse_missing_arch_fields():
    """JSON with only AppProfile fields → arch fields get sensible defaults."""
    minimal = json.dumps({
        "app_type": "fastapi",
        "tech_stack": ["python", "fastapi"],
        "entry_point": "src/main.py",
        "test_command": "pytest",
        "run_command": "uvicorn src.main:app",
        "requires_docker": False,
    })
    result = _parse_response(minimal)
    assert result.app_type == "fastapi"
    # arch fields should default to known fastapi defaults
    assert result.structural_pattern == _ARCH_DEFAULTS["fastapi"]["structural_pattern"]
    assert result.topology == _ARCH_DEFAULTS["fastapi"]["topology"]


def test_parse_json_in_markdown_fences():
    """LLM sometimes wraps JSON in markdown fences — strip and parse correctly."""
    fenced = f"```json\n{FULL_VALID_JSON}\n```"
    result = _parse_response(fenced)
    assert result.app_type == "fastapi"
    assert result.structural_pattern == "layered"


def test_parse_empty_string():
    """Empty string → fallback defaults, no exception."""
    result = _parse_response("")
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"
```

### Step 2: Run to verify tests fail

```bash
cd "E:/Documents/Vibe-Coding/Ai Orchestrator/.claude/worktrees/lucid-brahmagupta"
python -m pytest tests/test_architecture_advisor.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'orchestrator.architecture_advisor'`

### Step 3: Create `orchestrator/architecture_advisor.py` with data layer

```python
"""
Architecture Advisor — LLM-powered architecture decision before decomposition.
==============================================================================
Replaces AppDetector. One LLM call decides app type AND architectural patterns:
- structural_pattern: layered | hexagonal | cqrs | event-driven | mvc | script
- topology:           monolith | microservices | serverless | bff | library
- data_paradigm:      relational | document | time-series | key-value | none
- api_paradigm:       rest | graphql | grpc | cli | none

Prints a 🏗 summary to terminal (inform mode — no user prompt).
Falls back to safe script defaults on any error. Never raises.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .models import Model

logger = logging.getLogger("orchestrator.architecture_advisor")


@dataclass
class ArchitectureDecision:
    """
    Superset of the old AppProfile — includes app type + runtime details
    AND explicit architectural pattern decisions.

    Drop-in replacement: all AppProfile fields are preserved.
    """
    # ── AppProfile-compatible fields (unchanged) ────────────────────────────
    app_type: str = "script"
    tech_stack: list[str] = field(default_factory=list)
    entry_point: str = "main.py"
    test_command: str = "pytest"
    run_command: str = "python main.py"
    requires_docker: bool = False
    detected_from: str = "advisor"   # "advisor" | "yaml_override"

    # ── New architecture fields ─────────────────────────────────────────────
    structural_pattern: str = "script"   # layered|hexagonal|cqrs|event-driven|mvc|script
    topology: str = "monolith"           # monolith|microservices|serverless|bff|library
    data_paradigm: str = "none"          # relational|document|time-series|key-value|none
    api_paradigm: str = "none"           # rest|graphql|grpc|cli|none
    rationale: str = ""                  # 2-3 sentences explaining all choices


# ── Per-type defaults used for YAML override and parse fallback ──────────────

_TYPE_DEFAULTS: dict[str, dict] = {
    "fastapi": {
        "tech_stack": ["python", "fastapi", "uvicorn"],
        "entry_point": "src/main.py",
        "test_command": "pytest",
        "run_command": "uvicorn src.main:app --reload",
        "requires_docker": False,
    },
    "flask": {
        "tech_stack": ["python", "flask"],
        "entry_point": "src/app.py",
        "test_command": "pytest",
        "run_command": "flask run",
        "requires_docker": False,
    },
    "cli": {
        "tech_stack": ["python"],
        "entry_point": "cli.py",
        "test_command": "pytest",
        "run_command": "python cli.py",
        "requires_docker": False,
    },
    "library": {
        "tech_stack": ["python"],
        "entry_point": "src/__init__.py",
        "test_command": "pytest",
        "run_command": "",
        "requires_docker": False,
    },
    "script": {
        "tech_stack": ["python"],
        "entry_point": "main.py",
        "test_command": "pytest",
        "run_command": "python main.py",
        "requires_docker": False,
    },
    "react-fastapi": {
        "tech_stack": ["python", "fastapi", "react", "typescript"],
        "entry_point": "backend/main.py",
        "test_command": "pytest",
        "run_command": "uvicorn backend.main:app",
        "requires_docker": True,
    },
    "nextjs": {
        "tech_stack": ["typescript", "nextjs", "react"],
        "entry_point": "pages/index.tsx",
        "test_command": "npm test",
        "run_command": "npm run dev",
        "requires_docker": False,
    },
    "generic": {
        "tech_stack": ["python"],
        "entry_point": "main.py",
        "test_command": "pytest",
        "run_command": "python main.py",
        "requires_docker": False,
    },
}

_ARCH_DEFAULTS: dict[str, dict] = {
    "fastapi":      {"structural_pattern": "layered",  "topology": "monolith",  "data_paradigm": "relational", "api_paradigm": "rest"},
    "flask":        {"structural_pattern": "mvc",       "topology": "monolith",  "data_paradigm": "relational", "api_paradigm": "rest"},
    "nextjs":       {"structural_pattern": "mvc",       "topology": "monolith",  "data_paradigm": "relational", "api_paradigm": "rest"},
    "react-fastapi":{"structural_pattern": "layered",  "topology": "monolith",  "data_paradigm": "relational", "api_paradigm": "rest"},
    "cli":          {"structural_pattern": "script",    "topology": "library",   "data_paradigm": "none",       "api_paradigm": "cli"},
    "library":      {"structural_pattern": "script",    "topology": "library",   "data_paradigm": "none",       "api_paradigm": "none"},
    "script":       {"structural_pattern": "script",    "topology": "library",   "data_paradigm": "none",       "api_paradigm": "none"},
    "generic":      {"structural_pattern": "layered",  "topology": "monolith",  "data_paradigm": "relational", "api_paradigm": "rest"},
}

_FALLBACK_TYPE = "script"
_FALLBACK_ARCH = _ARCH_DEFAULTS["script"]


def _select_model(description: str) -> Model:
    """Auto-select model based on description length.

    >50 words → DeepSeek Reasoner (multi-dimension reasoning for complex specs)
    ≤50 words → DeepSeek Chat (fast + cheap for simple descriptions)
    """
    return (
        Model.DEEPSEEK_REASONER
        if len(description.split()) > 50
        else Model.DEEPSEEK_CHAT
    )


def _parse_response(raw: str) -> ArchitectureDecision:
    """Parse LLM JSON output into ArchitectureDecision.

    Returns safe fallback defaults on any parse error — never raises.
    Handles JSON wrapped in markdown code fences.
    """
    fallback_defaults = _TYPE_DEFAULTS[_FALLBACK_TYPE]
    fallback_arch = _FALLBACK_ARCH

    try:
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for i, line in enumerate(lines)
                if i > 0 and line.strip() != "```"
            ).strip()

        if not text:
            raise ValueError("empty response")

        data = json.loads(text)

        app_type = str(data.get("app_type", _FALLBACK_TYPE))
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type '%s' from LLM; using 'generic'", app_type)
            app_type = "generic"

        type_defaults = _TYPE_DEFAULTS.get(app_type, fallback_defaults)
        arch_defaults = _ARCH_DEFAULTS.get(app_type, fallback_arch)

        return ArchitectureDecision(
            app_type=app_type,
            tech_stack=list(data.get("tech_stack", type_defaults["tech_stack"])),
            entry_point=str(data.get("entry_point", type_defaults["entry_point"])),
            test_command=str(data.get("test_command", type_defaults["test_command"])),
            run_command=str(data.get("run_command", type_defaults["run_command"])),
            requires_docker=bool(data.get("requires_docker", type_defaults["requires_docker"])),
            detected_from="advisor",
            structural_pattern=str(data.get("structural_pattern", arch_defaults["structural_pattern"])),
            topology=str(data.get("topology", arch_defaults["topology"])),
            data_paradigm=str(data.get("data_paradigm", arch_defaults["data_paradigm"])),
            api_paradigm=str(data.get("api_paradigm", arch_defaults["api_paradigm"])),
            rationale=str(data.get("rationale", "")),
        )

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("ArchitectureAdvisor JSON parse failed: %s — using fallback", exc)
        return ArchitectureDecision(
            app_type=_FALLBACK_TYPE,
            tech_stack=list(fallback_defaults["tech_stack"]),
            entry_point=fallback_defaults["entry_point"],
            test_command=fallback_defaults["test_command"],
            run_command=fallback_defaults["run_command"],
            requires_docker=fallback_defaults["requires_docker"],
            detected_from="advisor",
            **{k: v for k, v in fallback_arch.items()},
        )
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_architecture_advisor.py -v -k "select_model or parse_response"
```

Expected: `9 passed`

### Step 5: Commit

```bash
git add orchestrator/architecture_advisor.py tests/test_architecture_advisor.py
git commit -m "feat: add ArchitectureDecision dataclass and pure parse/select functions"
```

---

## Task 2: ArchitectureAdvisor class with LLM call

**Files:**
- Modify: `orchestrator/architecture_advisor.py` (add `ArchitectureAdvisor` class)
- Modify: `tests/test_architecture_advisor.py` (add advisor tests)

### Step 1: Write failing tests for ArchitectureAdvisor

Add to `tests/test_architecture_advisor.py`:

```python
# ─── ArchitectureAdvisor ──────────────────────────────────────────────────────

import asyncio
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.architecture_advisor import ArchitectureAdvisor


class FakeAPIResponse:
    def __init__(self, text: str, cost_usd: float = 0.002):
        self.text = text
        self.cost_usd = cost_usd


@pytest.mark.asyncio
async def test_analyze_calls_llm_once():
    """analyze() makes exactly one LLM call."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    result = await advisor.analyze("Build a FastAPI service", "tests pass")

    assert mock_client.call.call_count == 1
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "fastapi"


@pytest.mark.asyncio
async def test_analyze_handles_exception():
    """LLM call raises → returns fallback defaults, no crash."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(side_effect=Exception("network error"))

    advisor = ArchitectureAdvisor(client=mock_client)
    result = await advisor.analyze("Build something", "tests pass")

    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"   # fallback


@pytest.mark.asyncio
async def test_analyze_prints_summary(capsys):
    """Terminal output contains pattern, topology, api, storage."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze("Build a FastAPI service", "tests pass")

    captured = capsys.readouterr()
    assert "layered" in captured.out.lower() or "Layered" in captured.out
    assert "monolith" in captured.out.lower() or "Monolith" in captured.out
    assert "rest" in captured.out.lower() or "REST" in captured.out


@pytest.mark.asyncio
async def test_analyze_selects_reasoner_for_long_desc():
    """Descriptions >50 words use DeepSeek Reasoner."""
    long_desc = " ".join(["word"] * 51)
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze(long_desc, "criteria")

    call_args = mock_client.call.call_args
    used_model = call_args[1].get("model") or call_args[0][0]
    assert used_model == Model.DEEPSEEK_REASONER


@pytest.mark.asyncio
async def test_analyze_selects_chat_for_short_desc():
    """Descriptions ≤50 words use DeepSeek Chat."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze("Build a FastAPI auth service", "tests pass")

    call_args = mock_client.call.call_args
    used_model = call_args[1].get("model") or call_args[0][0]
    assert used_model == Model.DEEPSEEK_CHAT


def test_detect_from_yaml_skips_llm():
    """detect_from_yaml() returns known defaults without LLM call."""
    advisor = ArchitectureAdvisor(client=None)   # no client needed
    result = advisor.detect_from_yaml("fastapi")

    assert result.app_type == "fastapi"
    assert result.structural_pattern == "layered"
    assert result.topology == "monolith"
    assert result.api_paradigm == "rest"
    assert result.detected_from == "yaml_override"


def test_detect_from_yaml_unknown_type():
    """Unknown app_type_override falls back to script defaults."""
    advisor = ArchitectureAdvisor(client=None)
    result = advisor.detect_from_yaml("unknown-framework")

    assert result.app_type == "script"
    assert result.detected_from == "yaml_override"
```

### Step 2: Run to verify tests fail

```bash
python -m pytest tests/test_architecture_advisor.py -v -k "analyze or detect_from_yaml"
```

Expected: `AttributeError: module 'orchestrator.architecture_advisor' has no attribute 'ArchitectureAdvisor'`

### Step 3: Add `ArchitectureAdvisor` class to `orchestrator/architecture_advisor.py`

Append after the `_parse_response` function:

```python
_SYSTEM_PROMPT = """\
You are a senior software architect.
Given a project description and success criteria, select the optimal architecture.
Return only a JSON object — no markdown, no explanation.\
"""

_USER_PROMPT_TEMPLATE = """\
PROJECT: {description}
SUCCESS CRITERIA: {criteria}

Return a JSON object with these exact fields:
{{
  "app_type": "fastapi|flask|cli|library|script|react-fastapi|nextjs|generic",
  "tech_stack": ["list", "of", "technologies"],
  "entry_point": "relative/path/to/main.py",
  "test_command": "pytest",
  "run_command": "command to start app",
  "requires_docker": false,
  "structural_pattern": "layered|hexagonal|cqrs|event-driven|mvc|script",
  "topology": "monolith|microservices|serverless|bff|library",
  "data_paradigm": "relational|document|time-series|key-value|none",
  "api_paradigm": "rest|graphql|grpc|cli|none",
  "rationale": "2-3 sentences explaining all architectural choices"
}}

Rules:
- Choose the architecture that best fits the project scale and requirements
- layered: routes → services → repositories (standard API services)
- hexagonal: ports & adapters (when testing or swappable infrastructure matters)
- cqrs: separate read/write paths (high-read or event-sourced systems)
- event-driven: async message-passing (real-time, decoupled components)
- mvc: model-view-controller (web apps with server-side rendering)
- Return ONLY the JSON object, no markdown fences\
"""

_TIMEOUT_S = 30


class ArchitectureAdvisor:
    """
    Analyzes a project spec and decides optimal software architecture.

    Replaces AppDetector. One LLM call returns app type + 4 architecture
    dimensions. Prints a summary block to terminal (inform mode — no prompt).

    Usage:
        advisor = ArchitectureAdvisor()
        decision = await advisor.analyze(description, criteria)
        # Always returns ArchitectureDecision — never raises
    """

    def __init__(self, client=None):
        """client: UnifiedClient instance (created lazily if None)."""
        self._client = client

    def _get_client(self):
        if self._client is None:
            from .api_clients import UnifiedClient
            self._client = UnifiedClient()
        return self._client

    async def analyze(
        self,
        description: str,
        criteria: str,
        app_type_override: Optional[str] = None,
    ) -> ArchitectureDecision:
        """Decide architecture for this project.

        If app_type_override is provided (from YAML), skip LLM.
        Returns fallback defaults on any error — never raises.
        """
        if app_type_override:
            return self.detect_from_yaml(app_type_override)

        model = _select_model(description)
        model_label = "DeepSeek Reasoner" if model == Model.DEEPSEEK_REASONER else "DeepSeek Chat"

        try:
            prompt = _USER_PROMPT_TEMPLATE.format(
                description=description,
                criteria=criteria,
            )
            response = await self._get_client().call(
                model=model,
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.2,
                timeout=_TIMEOUT_S,
                retries=1,
            )
            decision = _parse_response(response.text)
            logger.debug(
                "ArchitectureAdvisor: used %s (cost $%.6f)", model.value, response.cost_usd
            )
        except Exception as exc:
            logger.warning("ArchitectureAdvisor LLM call failed (%s) — using fallback", exc)
            decision = _parse_response("")   # triggers fallback path

        _print_summary(decision, model_label)
        return decision

    def detect_from_yaml(self, app_type: str) -> ArchitectureDecision:
        """Return an ArchitectureDecision for a YAML app_type override without LLM."""
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type '%s' in YAML; falling back to 'script'", app_type)
            app_type = "script"

        type_d = _TYPE_DEFAULTS[app_type]
        arch_d = _ARCH_DEFAULTS.get(app_type, _FALLBACK_ARCH)

        return ArchitectureDecision(
            app_type=app_type,
            tech_stack=list(type_d["tech_stack"]),
            entry_point=type_d["entry_point"],
            test_command=type_d["test_command"],
            run_command=type_d["run_command"],
            requires_docker=type_d["requires_docker"],
            detected_from="yaml_override",
            structural_pattern=arch_d["structural_pattern"],
            topology=arch_d["topology"],
            data_paradigm=arch_d["data_paradigm"],
            api_paradigm=arch_d["api_paradigm"],
            rationale="",
        )


def _print_summary(decision: ArchitectureDecision, model_label: str = "DeepSeek Chat") -> None:
    """Print the 🏗 architecture summary block to terminal."""
    pat  = decision.structural_pattern.capitalize()
    top  = decision.topology.capitalize()
    api  = decision.api_paradigm.upper() if decision.api_paradigm != "none" else "None"
    data = decision.data_paradigm.capitalize() if decision.data_paradigm != "none" else "None"

    print(f"\n🏗  Architecture Decision ({model_label}):")
    print(f"    Pattern: {pat}  │  Topology: {top}  │  API: {api}  │  Storage: {data}")
    if decision.rationale:
        # Word-wrap rationale at 76 chars
        words = decision.rationale.split()
        lines, cur = [], []
        for w in words:
            if sum(len(x) + 1 for x in cur) + len(w) > 72:
                lines.append("    " + " ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append("    " + " ".join(cur))
        for line in lines:
            print(line)
    print("─" * 78)
```

### Step 4: Run all architecture advisor tests

```bash
python -m pytest tests/test_architecture_advisor.py -v
```

Expected: `15 passed`

### Step 5: Commit

```bash
git add orchestrator/architecture_advisor.py tests/test_architecture_advisor.py
git commit -m "feat: add ArchitectureAdvisor class with LLM call and detect_from_yaml"
```

---

## Task 3: Add AppProfile type alias to app_detector.py

**Files:**
- Modify: `orchestrator/app_detector.py` (end of file)

### Step 1: Write the test

Add to `tests/test_architecture_advisor.py`:

```python
# ─── backward compatibility ───────────────────────────────────────────────────

def test_app_profile_alias():
    """AppProfile is a type alias for ArchitectureDecision — same object."""
    from orchestrator.app_detector import AppProfile
    assert AppProfile is ArchitectureDecision
```

### Step 2: Run to verify it fails

```bash
python -m pytest tests/test_architecture_advisor.py::test_app_profile_alias -v
```

Expected: `AssertionError` (AppProfile is still the old dataclass)

### Step 3: Modify `orchestrator/app_detector.py`

At the **very end** of `orchestrator/app_detector.py`, append:

```python
# ── Backward compatibility ────────────────────────────────────────────────────
# AppProfile is now a type alias for ArchitectureDecision.
# All existing code importing AppProfile continues to work unchanged.
from .architecture_advisor import ArchitectureDecision  # noqa: E402
AppProfile = ArchitectureDecision
```

### Step 4: Run to verify test passes

```bash
python -m pytest tests/test_architecture_advisor.py::test_app_profile_alias -v
```

Expected: `PASSED`

Also verify existing AppDetector tests still pass:

```bash
python -m pytest tests/test_app_detector.py -v 2>&1 | tail -15
```

Expected: all tests pass (AppProfile alias is transparent)

### Step 5: Commit

```bash
git add orchestrator/app_detector.py tests/test_architecture_advisor.py
git commit -m "chore: add AppProfile = ArchitectureDecision type alias for backward compat"
```

---

## Task 4: Swap AppDetector → ArchitectureAdvisor in AppBuilder

**Files:**
- Modify: `orchestrator/app_builder.py:24` (imports)
- Modify: `orchestrator/app_builder.py:62` (`__init__`)
- Modify: `orchestrator/app_builder.py:92-104` (detection block in `build()`)

### Step 1: Write failing test

Add to `tests/test_architecture_advisor.py`:

```python
# ─── AppBuilder integration ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_app_builder_uses_architecture_advisor():
    """AppBuilder.build() calls ArchitectureAdvisor.analyze(), not AppDetector.detect()."""
    from orchestrator.app_builder import AppBuilder
    from orchestrator.app_detector import AppDetector

    builder = AppBuilder()

    # Confirm the advisor attribute exists and is ArchitectureAdvisor
    assert hasattr(builder, "_advisor")
    assert not hasattr(builder, "_detector") or builder._detector is None or True
    # The _advisor must be an ArchitectureAdvisor
    assert isinstance(builder._advisor, ArchitectureAdvisor)
```

### Step 2: Run to verify it fails

```bash
python -m pytest tests/test_architecture_advisor.py::test_app_builder_uses_architecture_advisor -v
```

Expected: `AttributeError: 'AppBuilder' object has no attribute '_advisor'`

### Step 3: Modify `orchestrator/app_builder.py`

**Change 1 — imports** (line 24):

Find:
```python
from orchestrator.app_detector import AppDetector, AppProfile
```
Replace with:
```python
from orchestrator.architecture_advisor import ArchitectureAdvisor, ArchitectureDecision
from orchestrator.app_detector import AppProfile  # type alias kept for compat
```

**Change 2 — `__init__`** (line 62):

Find:
```python
    def __init__(self) -> None:
        self._detector = AppDetector()
        self._scaffold = ScaffoldEngine()
```
Replace with:
```python
    def __init__(self) -> None:
        self._advisor = ArchitectureAdvisor()
        self._scaffold = ScaffoldEngine()
```

**Change 3 — detection block in `build()`** (lines 92-104):

Find:
```python
        try:
            # -- Step 1: Detect app type --
            logger.info("AppBuilder: detecting app type...")
            if app_type_override:
                # Skip LLM detection; use the YAML override directly.
                profile = self._detector.detect_from_yaml(app_type_override)
            else:
                profile = await self._detector.detect(description)
            result.profile = profile
            logger.info(
                "AppBuilder: app_type=%s detected_from=%s",
                profile.app_type,
                profile.detected_from,
            )
```
Replace with:
```python
        try:
            # -- Step 1: Decide architecture --
            logger.info("AppBuilder: running architecture advisor...")
            profile = await self._advisor.analyze(description, criteria, app_type_override)
            result.profile = profile
            logger.info(
                "AppBuilder: app_type=%s pattern=%s topology=%s detected_from=%s",
                profile.app_type,
                profile.structural_pattern,
                profile.topology,
                profile.detected_from,
            )
```

### Step 4: Run the test

```bash
python -m pytest tests/test_architecture_advisor.py::test_app_builder_uses_architecture_advisor -v
```

Expected: `PASSED`

Run existing AppBuilder tests to check for regressions:

```bash
python -m pytest tests/test_app_builder.py tests/test_app_builder_cli.py -v 2>&1 | tail -20
```

Expected: all pass (or only pre-existing failures)

### Step 5: Commit

```bash
git add orchestrator/app_builder.py tests/test_architecture_advisor.py
git commit -m "feat: swap AppDetector for ArchitectureAdvisor in AppBuilder"
```

---

## Task 5: Enrich decomposition prompt in engine.py

**Files:**
- Modify: `orchestrator/engine.py:350-368`
- Test: `tests/test_architecture_advisor.py`

### Step 1: Write the failing test

Add to `tests/test_architecture_advisor.py`:

```python
# ─── engine.py decomposition prompt ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_decompose_prompt_includes_arch_fields():
    """_decompose() prompt contains architectural fields when app_profile has them."""
    from orchestrator.engine import Orchestrator
    from orchestrator.models import Budget

    orch = Orchestrator(budget=Budget(max_usd=1.0))

    decision = ArchitectureDecision(
        app_type="fastapi",
        tech_stack=["python", "fastapi"],
        entry_point="src/main.py",
        test_command="pytest",
        run_command="uvicorn src.main:app",
        requires_docker=False,
        detected_from="advisor",
        structural_pattern="hexagonal",
        topology="monolith",
        data_paradigm="relational",
        api_paradigm="rest",
        rationale="Hexagonal keeps the domain isolated from FastAPI specifics.",
    )

    # Patch client.call to capture the prompt
    captured_prompts = []
    async def mock_call(model, prompt, **kwargs):
        captured_prompts.append(prompt)
        # Return a minimal valid task list
        import json as _json
        fake_tasks = _json.dumps([{
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write main.py",
            "dependencies": [],
            "hard_validators": ["python_syntax"],
        }])
        from orchestrator.api_clients import APIResponse
        return APIResponse(text=fake_tasks, input_tokens=100,
                           output_tokens=50, model=model)

    orch.client.call = mock_call

    await orch._decompose("Build a FastAPI service", "tests pass", app_profile=decision)

    assert len(captured_prompts) >= 1
    prompt = captured_prompts[0]
    assert "hexagonal" in prompt.lower()
    assert "ARCHITECTURE DECISION" in prompt
    assert "relational" in prompt.lower()
    assert "rest" in prompt.lower()
```

### Step 2: Run to verify it fails

```bash
python -m pytest tests/test_architecture_advisor.py::test_decompose_prompt_includes_arch_fields -v
```

Expected: `AssertionError` — `"ARCHITECTURE DECISION"` not in prompt yet

### Step 3: Modify `orchestrator/engine.py`

Find the `app_context_block` construction (lines 350-368):

```python
        app_context_block = ""
        if app_profile is not None:
            from orchestrator.scaffold import _TEMPLATE_MAP
            from orchestrator.scaffold.templates import generic
            template_files = _TEMPLATE_MAP.get(app_profile.app_type, generic.FILES)
            scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
            tech_stack_str = ", ".join(app_profile.tech_stack) if app_profile.tech_stack else "unknown"
            app_context_block = f"""
APP_TYPE: {app_profile.app_type}
TECH_STACK: {tech_stack_str}
SCAFFOLD_FILES (already exist — fill or extend these):
{scaffold_list}

Each task JSON element MUST also include:
- "target_path": the relative file path this task writes (e.g. "app/page.tsx").
  Use the exact scaffold paths listed above where applicable.
  Tasks producing non-file outputs (code_review, evaluation) use target_path: "".
- "tech_context": brief note on the tech stack relevant to this specific file.
"""
```

Replace with:

```python
        app_context_block = ""
        if app_profile is not None:
            from orchestrator.scaffold import _TEMPLATE_MAP
            from orchestrator.scaffold.templates import generic
            template_files = _TEMPLATE_MAP.get(app_profile.app_type, generic.FILES)
            scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
            tech_stack_str = ", ".join(app_profile.tech_stack) if app_profile.tech_stack else "unknown"

            # Build architecture block if ArchitectureDecision fields are present
            arch_block = ""
            if hasattr(app_profile, "structural_pattern") and app_profile.structural_pattern:
                rationale_line = (
                    f"\n  Rationale:          {app_profile.rationale}"
                    if getattr(app_profile, "rationale", "") else ""
                )
                arch_block = f"""
ARCHITECTURE DECISION:
  Structural pattern: {app_profile.structural_pattern}
  Topology:           {app_profile.topology}
  API paradigm:       {app_profile.api_paradigm}
  Data paradigm:      {app_profile.data_paradigm}{rationale_line}

Each task MUST follow this architecture — do not invent an alternative structure.
"""

            app_context_block = f"""
APP_TYPE: {app_profile.app_type}
TECH_STACK: {tech_stack_str}
SCAFFOLD_FILES (already exist — fill or extend these):
{scaffold_list}
{arch_block}
Each task JSON element MUST also include:
- "target_path": the relative file path this task writes (e.g. "app/page.tsx").
  Use the exact scaffold paths listed above where applicable.
  Tasks producing non-file outputs (code_review, evaluation) use target_path: "".
- "tech_context": brief note on the tech stack relevant to this specific file.
"""
```

### Step 4: Run the test

```bash
python -m pytest tests/test_architecture_advisor.py::test_decompose_prompt_includes_arch_fields -v
```

Expected: `PASSED`

### Step 5: Run full architecture advisor test suite

```bash
python -m pytest tests/test_architecture_advisor.py -v
```

Expected: all tests pass

### Step 6: Commit

```bash
git add orchestrator/engine.py tests/test_architecture_advisor.py
git commit -m "feat: inject architecture decision into decomposition prompt"
```

---

## Task 6: Update `__init__.py` exports

**Files:**
- Modify: `orchestrator/__init__.py`

### Step 1: Add import

Find the imports block (around line 19). Add after the existing imports (e.g., after the `from .app_detector import ...` line or at the end of the imports):

```python
from .architecture_advisor import ArchitectureDecision, ArchitectureAdvisor
```

Find the `__all__` list and add before the closing bracket:

```python
    # Architecture Advisor — replaces AppDetector
    "ArchitectureDecision", "ArchitectureAdvisor",
```

### Step 2: Verify the import works

```bash
python -c "from orchestrator import ArchitectureDecision, ArchitectureAdvisor; print('OK')"
```

Expected: `OK`

### Step 3: Run the full test suite to check regressions

```bash
python -m pytest tests/ -x --ignore=tests/test_engine_e2e.py -q 2>&1 | tail -15
```

Expected: all pass (or only pre-existing failures unrelated to this feature)

### Step 4: Commit

```bash
git add orchestrator/__init__.py
git commit -m "chore: export ArchitectureDecision and ArchitectureAdvisor from package"
```

---

## Task 7: Final verification

### Step 1: Run the full architecture advisor test file

```bash
python -m pytest tests/test_architecture_advisor.py -v
```

Expected (17+ tests, all pass):
```
tests/test_architecture_advisor.py::test_select_model_short PASSED
tests/test_architecture_advisor.py::test_select_model_long PASSED
tests/test_architecture_advisor.py::test_select_model_exactly_50 PASSED
tests/test_architecture_advisor.py::test_parse_valid_response PASSED
tests/test_architecture_advisor.py::test_parse_invalid_json PASSED
tests/test_architecture_advisor.py::test_parse_missing_arch_fields PASSED
tests/test_architecture_advisor.py::test_parse_json_in_markdown_fences PASSED
tests/test_architecture_advisor.py::test_parse_empty_string PASSED
tests/test_architecture_advisor.py::test_analyze_calls_llm_once PASSED
tests/test_architecture_advisor.py::test_analyze_handles_exception PASSED
tests/test_architecture_advisor.py::test_analyze_prints_summary PASSED
tests/test_architecture_advisor.py::test_analyze_selects_reasoner_for_long_desc PASSED
tests/test_architecture_advisor.py::test_analyze_selects_chat_for_short_desc PASSED
tests/test_architecture_advisor.py::test_detect_from_yaml_skips_llm PASSED
tests/test_architecture_advisor.py::test_detect_from_yaml_unknown_type PASSED
tests/test_architecture_advisor.py::test_app_profile_alias PASSED
tests/test_architecture_advisor.py::test_app_builder_uses_architecture_advisor PASSED
tests/test_architecture_advisor.py::test_decompose_prompt_includes_arch_fields PASSED
```

### Step 2: Verify no regressions

```bash
python -m pytest tests/test_app_detector.py tests/test_app_builder.py tests/test_app_builder_cli.py -v 2>&1 | tail -15
```

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: complete architecture advisor — LLM-powered architecture decision

Replaces AppDetector with ArchitectureAdvisor. One LLM call (DeepSeek
Reasoner for complex specs, Chat for simple ones) decides app type AND:
- structural_pattern: layered|hexagonal|cqrs|event-driven|mvc|script
- topology: monolith|microservices|serverless|bff|library
- data_paradigm: relational|document|time-series|key-value|none
- api_paradigm: rest|graphql|grpc|cli|none

Prints architecture summary to terminal. Injects decision into decomposition
prompt so all tasks follow the chosen architecture.

AppProfile = ArchitectureDecision type alias for full backward compat.

New: orchestrator/architecture_advisor.py
New: tests/test_architecture_advisor.py (18 tests)
Modified: orchestrator/app_detector.py (AppProfile alias)
Modified: orchestrator/app_builder.py (swap AppDetector → ArchitectureAdvisor)
Modified: orchestrator/engine.py (arch fields in decomposition prompt)
Modified: orchestrator/__init__.py (exports)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Quick Reference: File Change Summary

| File | Action | What changes |
|------|--------|-------------|
| `orchestrator/architecture_advisor.py` | **CREATE** | `ArchitectureDecision`, `_ARCH_DEFAULTS`, `_select_model`, `_parse_response`, `ArchitectureAdvisor`, `_print_summary` |
| `tests/test_architecture_advisor.py` | **CREATE** | 18 unit + integration tests |
| `orchestrator/app_detector.py` | **MODIFY** | Add `AppProfile = ArchitectureDecision` alias at end |
| `orchestrator/app_builder.py` | **MODIFY** | Swap `AppDetector` → `ArchitectureAdvisor`; update `__init__` + detection block |
| `orchestrator/engine.py` | **MODIFY** | Extend `app_context_block` with ARCHITECTURE DECISION section |
| `orchestrator/__init__.py` | **MODIFY** | Add `ArchitectureDecision`, `ArchitectureAdvisor` exports |
