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
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop opening fence line (e.g. "```json") and closing fence line
            if lines and lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()
        if not text:
            raise ValueError("empty response")
        data = json.loads(text)
        app_type = str(data.get("app_type", _FALLBACK_TYPE))
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type %r from LLM; normalising to 'generic'", app_type)
            app_type = "generic"
        type_defaults = _TYPE_DEFAULTS[app_type]
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

from typing import Optional

_SYSTEM_PROMPT = (
    "You are a senior software architect.\n"
    "Given a project description and success criteria, select the optimal architecture.\n"
    "Return only a JSON object — no markdown, no explanation."
)

_USER_PROMPT_TEMPLATE = (
    "PROJECT: {description}\n"
    "SUCCESS CRITERIA: {criteria}\n\n"
    "Return a JSON object with these exact fields:\n"
    "{{\n"
    '  "app_type": "fastapi|flask|cli|library|script|react-fastapi|nextjs|generic",\n'
    '  "tech_stack": ["list", "of", "technologies"],\n'
    '  "entry_point": "relative/path/to/main.py",\n'
    '  "test_command": "pytest",\n'
    '  "run_command": "command to start app",\n'
    '  "requires_docker": false,\n'
    '  "structural_pattern": "layered|hexagonal|cqrs|event-driven|mvc|script",\n'
    '  "topology": "monolith|microservices|serverless|bff|library",\n'
    '  "data_paradigm": "relational|document|time-series|key-value|none",\n'
    '  "api_paradigm": "rest|graphql|grpc|cli|none",\n'
    '  "rationale": "2-3 sentences explaining all architectural choices"\n'
    "}}\n\n"
    "Rules:\n"
    "- Choose the architecture that best fits the project scale and requirements\n"
    "- layered: routes -> services -> repositories (standard API services)\n"
    "- hexagonal: ports & adapters (when testing or swappable infrastructure matters)\n"
    "- cqrs: separate read/write paths (high-read or event-sourced systems)\n"
    "- event-driven: async message-passing (real-time, decoupled components)\n"
    "- mvc: model-view-controller (web apps with server-side rendering)\n"
    "- Return ONLY the JSON object, no markdown fences"
)

_TIMEOUT_S = 30


class ArchitectureAdvisor:
    """
    Analyzes a project spec and decides optimal software architecture.

    Replaces AppDetector. One LLM call returns app type + 4 architecture
    dimensions. Prints a summary block to terminal (inform mode, no prompt).

    Usage:
        advisor = ArchitectureAdvisor()
        decision = await advisor.analyze(description, criteria)
        # Always returns ArchitectureDecision, never raises
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
        Returns fallback defaults on any error, never raises.
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
            logger.warning("ArchitectureAdvisor LLM call failed (%s), using fallback", exc)
            decision = _parse_response("")

        _print_summary(decision, model_label)
        return decision

    def detect_from_yaml(self, app_type: str) -> ArchitectureDecision:
        """Return an ArchitectureDecision for a YAML app_type override without LLM."""
        if app_type not in _TYPE_DEFAULTS:
            logger.warning("Unknown app_type %r in YAML; falling back to 'script'", app_type)
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
    """Print the architecture summary block to terminal."""
    pat = decision.structural_pattern.capitalize()
    top = decision.topology.capitalize()
    api = decision.api_paradigm.upper() if decision.api_paradigm != "none" else "None"
    data = decision.data_paradigm.capitalize() if decision.data_paradigm != "none" else "None"

    print(f"\n\U0001f3d7  Architecture Decision ({model_label}):")
    print(f"    Pattern: {pat}  |  Topology: {top}  |  API: {api}  |  Storage: {data}")
    if decision.rationale:
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
    print("-" * 78)
