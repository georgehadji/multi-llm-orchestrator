from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from .models import Model

logger = logging.getLogger("orchestrator.architecture_advisor")


@dataclass
class ArchitectureDecision:
    app_type: str = "script"
    tech_stack: list[str] = field(default_factory=list)
    entry_point: str = "main.py"
    test_command: str = "pytest"
    run_command: str = "python main.py"
    requires_docker: bool = False
    detected_from: str = "advisor"
    structural_pattern: str = "script"
    topology: str = "monolith"
    data_paradigm: str = "none"
    api_paradigm: str = "none"
    rationale: str = ""


_TYPE_DEFAULTS = {
    "fastapi": {"tech_stack": ["python","fastapi","uvicorn"],"entry_point":"src/main.py","test_command":"pytest","run_command":"uvicorn src.main:app --reload","requires_docker":False},
    "flask":   {"tech_stack": ["python","flask"],"entry_point":"src/app.py","test_command":"pytest","run_command":"flask run","requires_docker":False},
    "cli":     {"tech_stack": ["python"],"entry_point":"cli.py","test_command":"pytest","run_command":"python cli.py","requires_docker":False},
    "library": {"tech_stack": ["python"],"entry_point":"src/__init__.py","test_command":"pytest","run_command":"","requires_docker":False},
    "script":  {"tech_stack": ["python"],"entry_point":"main.py","test_command":"pytest","run_command":"python main.py","requires_docker":False},
    "react-fastapi": {"tech_stack":["python","fastapi","react","typescript"],"entry_point":"backend/main.py","test_command":"pytest","run_command":"uvicorn backend.main:app","requires_docker":True},
    "nextjs":  {"tech_stack":["typescript","nextjs","react"],"entry_point":"pages/index.tsx","test_command":"npm test","run_command":"npm run dev","requires_docker":False},
    "generic": {"tech_stack": ["python"],"entry_point":"main.py","test_command":"pytest","run_command":"python main.py","requires_docker":False},
}

_ARCH_DEFAULTS = {
    "fastapi":       {"structural_pattern":"layered","topology":"monolith","data_paradigm":"relational","api_paradigm":"rest"},
    "flask":         {"structural_pattern":"mvc","topology":"monolith","data_paradigm":"relational","api_paradigm":"rest"},
    "nextjs":        {"structural_pattern":"mvc","topology":"monolith","data_paradigm":"relational","api_paradigm":"rest"},
    "react-fastapi": {"structural_pattern":"layered","topology":"monolith","data_paradigm":"relational","api_paradigm":"rest"},
    "cli":           {"structural_pattern":"script","topology":"library","data_paradigm":"none","api_paradigm":"cli"},
    "library":       {"structural_pattern":"script","topology":"library","data_paradigm":"none","api_paradigm":"none"},
    "script":        {"structural_pattern":"script","topology":"library","data_paradigm":"none","api_paradigm":"none"},
    "generic":       {"structural_pattern":"layered","topology":"monolith","data_paradigm":"relational","api_paradigm":"rest"},
}

_FALLBACK_TYPE = "script"
_FALLBACK_ARCH = _ARCH_DEFAULTS["script"]


def _select_model(description: str) -> Model:
    return Model.DEEPSEEK_REASONER if len(description.split()) > 50 else Model.DEEPSEEK_CHAT


def _parse_response(raw: str) -> ArchitectureDecision:
    fallback_defaults = _TYPE_DEFAULTS[_FALLBACK_TYPE]
    fallback_arch = _FALLBACK_ARCH
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = chr(10).join(line for i, line in enumerate(lines) if i > 0 and line.strip() != "```").strip()
        if not text:
            raise ValueError("empty response")
        data = json.loads(text)
        app_type = str(data.get("app_type", _FALLBACK_TYPE))
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type from LLM; using generic")
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
        logger.warning("ArchitectureAdvisor JSON parse failed: %s", exc)
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
