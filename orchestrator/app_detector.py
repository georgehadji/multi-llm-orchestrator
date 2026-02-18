"""
AppDetector — analyses a project description and returns an AppProfile.
Author: Georgios-Chrysovalantis Chatzivantsidis

Detection logic:
- If app_type_override is provided (YAML field) → skip LLM, return profile directly
- Otherwise → single async LLM call → parse JSON → return AppProfile
- Fallback on any error → AppProfile(app_type="script", detected_from="auto")

FIX: _call_llm() is now wired to UnifiedClient using the cheapest available model
     (gemini-flash or gpt-4o-mini) for a single low-cost classification call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Supported app types
AppType = Literal["fastapi", "flask", "cli", "library", "script", "react-fastapi", "nextjs", "generic"]

# Per-type defaults used by detect_from_yaml and as fallback after LLM parse
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

_FALLBACK_DEFAULTS = _TYPE_DEFAULTS["script"]


@dataclass
class AppProfile:
    """Describes the type and runtime characteristics of the app to build."""

    app_type: str = "script"
    tech_stack: list[str] = field(default_factory=list)
    entry_point: str = "main.py"
    test_command: str = "pytest"
    run_command: str = "python main.py"
    requires_docker: bool = False
    detected_from: str = "auto"  # "auto" | "yaml_override"


class AppDetector:
    """
    Detects the type and characteristics of an app from its description.

    Usage:
        detector = AppDetector()
        profile = await detector.detect(description)          # LLM-based
        profile = await detector.detect(description, "cli")   # YAML override

    Optionally accepts a UnifiedClient for LLM-based detection. If not provided,
    a new client (with default DiskCache) is created on the first LLM call.
    """

    _DETECTION_PROMPT = (
        "You are an expert software architect. Analyse the following project description "
        "and return a JSON object (no markdown, no explanation) with these fields:\n"
        "  app_type: one of fastapi|flask|cli|library|script|react-fastapi|nextjs|generic\n"
        "  tech_stack: list of technology strings\n"
        "  entry_point: relative path to the main entry file\n"
        "  test_command: shell command to run tests\n"
        "  run_command: shell command to start the app\n"
        "  requires_docker: boolean\n\n"
        "Project description:\n{description}"
    )

    def __init__(self, client=None):
        """
        Parameters
        ----------
        client : UnifiedClient | None
            Pre-built API client. If None, a new one is lazily created on first
            LLM call. Accepting an existing client avoids re-initialising all
            provider SDKs (saves ~1s of startup time in AppBuilder).
        """
        self._client = client  # UnifiedClient or None; lazily initialized

    def _get_client(self):
        """Return the client, creating one lazily if needed."""
        if self._client is None:
            from .api_clients import UnifiedClient
            from .cache import DiskCache
            self._client = UnifiedClient(cache=DiskCache(), max_concurrency=1)
        return self._client

    def detect_from_yaml(self, app_type: str) -> AppProfile:
        """Return an AppProfile for a given app_type string without calling the LLM."""
        defaults = _TYPE_DEFAULTS.get(app_type, _FALLBACK_DEFAULTS)
        # If unknown type, fall back to script defaults but preserve the intent
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type '%s' in YAML; falling back to 'script'", app_type)
        resolved_type = app_type if app_type in _TYPE_DEFAULTS else "script"
        return AppProfile(
            app_type=resolved_type,
            tech_stack=list(defaults["tech_stack"]),
            entry_point=defaults["entry_point"],
            test_command=defaults["test_command"],
            run_command=defaults["run_command"],
            requires_docker=defaults["requires_docker"],
            detected_from="yaml_override",
        )

    async def detect(
        self,
        description: str,
        app_type_override: Optional[str] = None,
    ) -> AppProfile:
        """
        Detect the app type from a description.

        If app_type_override is given (from YAML), skip the LLM call.
        On any LLM or parse failure, return a safe 'script' fallback.
        """
        if app_type_override:
            return self.detect_from_yaml(app_type_override)

        try:
            raw = await self._call_llm(description)
            return self._parse_llm_response(raw)
        except Exception as exc:
            logger.warning("AppDetector LLM call failed (%s); falling back to 'script'", exc)
            return AppProfile(app_type="script", detected_from="auto")

    async def _call_llm(self, description: str) -> str:
        """
        Make a single LLM classification call using the cheapest available model.

        Preference order (cheapest first):
          gemini-2.5-flash → gpt-4o-mini → deepseek-chat → claude-haiku → any

        The response must be a JSON object matching the detection schema.
        Isolated into its own method so tests can patch it cleanly.
        """
        from .models import Model, ROUTING_TABLE, TaskType
        client = self._get_client()

        # Pick cheapest available model — detection is a simple classification task
        _PREFERENCE = [
            Model.GEMINI_FLASH,
            Model.GPT_4O_MINI,
            Model.DEEPSEEK_CHAT,
            Model.CLAUDE_HAIKU,
        ]
        model = next((m for m in _PREFERENCE if client.is_available(m)), None)
        if model is None:
            # Ultimate fallback: pick any available model
            model = next((m for m in Model if client.is_available(m)), None)
        if model is None:
            raise RuntimeError(
                "AppDetector: no LLM provider available. "
                "Check API keys in .env and provider SDK installation."
            )

        prompt = self._DETECTION_PROMPT.format(description=description)
        resp = await client.call(
            model=model,
            prompt=prompt,
            system=(
                "You are a software architect. Return only a JSON object "
                "with the exact fields requested — no markdown, no explanation."
            ),
            max_tokens=512,
            temperature=0.1,  # low temperature: classification task
            timeout=30,
            retries=1,
        )
        logger.debug(
            "AppDetector: used %s for detection (cost $%.6f)", model.value, resp.cost_usd
        )
        return resp.text

    def _parse_llm_response(self, raw: str) -> AppProfile:
        """
        Parse the LLM's JSON response into an AppProfile.

        Handles JSON embedded in markdown code fences.
        Raises ValueError or json.JSONDecodeError on bad input.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening fence (```json, ```, etc.) and closing fence (```)
            content_lines = [
                line for i, line in enumerate(lines)
                if i > 0 and not (line.strip() == "```")
            ]
            text = "\n".join(content_lines).strip()

        data = json.loads(text)

        app_type = str(data.get("app_type", "script"))
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type '%s' from LLM; using 'generic'", app_type)
            app_type = "generic"

        defaults = _TYPE_DEFAULTS.get(app_type, _FALLBACK_DEFAULTS)

        return AppProfile(
            app_type=app_type,
            tech_stack=list(data.get("tech_stack", defaults["tech_stack"])),
            entry_point=str(data.get("entry_point", defaults["entry_point"])),
            test_command=str(data.get("test_command", defaults["test_command"])),
            run_command=str(data.get("run_command", defaults["run_command"])),
            requires_docker=bool(data.get("requires_docker", defaults["requires_docker"])),
            detected_from="auto",
        )
