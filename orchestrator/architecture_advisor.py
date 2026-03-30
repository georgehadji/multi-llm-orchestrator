from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .models import Model

logger = logging.getLogger("orchestrator.architecture_advisor")


# Nexus Search integration (lazy import to avoid circular dependencies)
def _get_nexus_search():
    """Lazy import of Nexus Search to avoid circular dependencies."""
    try:
        from orchestrator.nexus_search import SearchSource
        from orchestrator.nexus_search import search as nexus_search
        return nexus_search, SearchSource
    except ImportError:
        return None, None


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
    """
    Select optimal model for architecture decisions.
    Updated v3.0 with best models for architectural reasoning.
    
    Architecture decisions require:
    - Strong reasoning capabilities
    - Understanding of design patterns and trade-offs
    - Knowledge of scalability, maintainability, cost considerations
    
    Best models (in priority order):
    1. Qwen3.5-397B-A17B - 397B MoE, excellent reasoning, best value
    2. Xiaomi MiMo-V2-Pro - 1T+ params, 1M+ context for complex systems
    3. Grok 4.20 Beta - Lowest hallucination for critical decisions
    4. Claude Sonnet 4.6 - Premium quality for high-stakes architecture
    """
    word_count = len(description.split())
    
    # Simple projects: use cost-effective reasoning models
    if word_count <= 50:
        # Try budget-friendly options first (in priority order)
        priority_models = [
            Model.STEPFUN_STEP_3_5_FLASH,   # $0.10/$0.30, 196B MoE ⭐ BEST VALUE
            Model.ZHIPU_GLM_4_7_FLASH,      # $0.06/$0.40, ultra-cheap
            Model.XIAOMI_MIMO_V2_FLASH,     # $0.09/$0.29, #1 SWE-bench
        ]
    else:
        # Complex projects: use powerful reasoning models
        priority_models = [
            Model.QWEN_3_5_397B_A17B,       # $0.39/$2.34, 397B MoE ⭐ BEST OVERALL
            Model.XIAOMI_MIMO_V2_PRO,       # $1.00/$3.00, 1T+ params, 1M+ ctx
            Model.XAI_GROK_4_20_BETA,       # $2.00/$6.00, lowest hallucination
            Model.CLAUDE_SONNET_4_6,        # $3.00/$15.00, premium quality
        ]
    
    # Return first available model from priority list
    # Note: We can't check api_health here (no orchestrator instance)
    # so we just return the best model and let the client handle fallbacks
    return priority_models[0]


def _parse_response(raw: str) -> ArchitectureDecision:
    fallback_defaults = _TYPE_DEFAULTS[_FALLBACK_TYPE]
    fallback_arch = _FALLBACK_ARCH
    try:
        text = raw.strip()
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop opening fence line (e.g. "```json") and closing fence line
            lines = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
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
            **dict(fallback_arch.items()),
        )


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

    Nexus Search Integration:
    - Searches latest architecture patterns and best practices
    - Provides real-time data for architecture decisions

    Usage:
        advisor = ArchitectureAdvisor(nexus_enabled=True)
        decision = await advisor.analyze(description, criteria)
        # Always returns ArchitectureDecision, never raises
    """

    def __init__(self, client=None, nexus_enabled: bool = True):
        """
        client: UnifiedClient instance (created lazily if None).
        nexus_enabled: Enable Nexus Search for architecture research (default: True)
        """
        self._client = client
        self.nexus_enabled = nexus_enabled

    def _get_client(self):
        if self._client is None:
            from .api_clients import UnifiedClient
            self._client = UnifiedClient()
        return self._client

    async def _get_architecture_context(self, description: str) -> str:
        """Get architecture context using Nexus Search.

        Parameters
        ----------
        description : str
            Project description

        Returns
        -------
        str
            Architecture context from search results, or empty string
        """
        if not self.nexus_enabled:
            return ""

        nexus_search, SearchSource = _get_nexus_search()
        if nexus_search is None:
            logger.debug("Nexus Search not available")
            return ""

        try:
            # Extract key architecture-related terms
            query = f"{description[:100]} architecture patterns best practices 2026"

            # Search for architecture patterns
            results = await nexus_search(
                query=query,
                sources=[SearchSource.TECH, SearchSource.ACADEMIC],
                num_results=5,
            )

            # Build context from top results
            context_parts = []
            for result in results.top[:3]:
                if result.content:
                    context_parts.append(f"- {result.content}")

            if context_parts:
                logger.info(f"Nexus Search found {len(context_parts)} architecture results")
                return "\n".join(context_parts)
        except Exception as e:
            logger.debug(f"Nexus Search failed for architecture: {e}")

        return ""

    async def analyze(
        self,
        description: str,
        criteria: str,
        app_type_override: str | None = None,
        use_web_context: bool = True,
    ) -> ArchitectureDecision:
        """Decide architecture for this project.

        If app_type_override is provided (from YAML), skip LLM.
        Returns fallback defaults on any error, never raises.

        Parameters
        ----------
        description : str
            Project description
        criteria : str
            Success criteria
        app_type_override : Optional[str]
            Override app type from YAML
        use_web_context : bool
            Use Nexus Search for architecture context (default: True)
        """
        if app_type_override:
            return self.detect_from_yaml(app_type_override)

        # Get architecture context from Nexus Search
        arch_context = ""
        if use_web_context and self.nexus_enabled:
            arch_context = await self._get_architecture_context(description)

        model = _select_model(description)
        # Updated v3.0: Show actual model name instead of hardcoded labels
        model_label = model.value.split('/')[-1].replace('-', ' ').title()

        try:
            # Build prompt with optional architecture context
            if arch_context:
                prompt = _USER_PROMPT_TEMPLATE.format(
                    description=description,
                    criteria=criteria,
                )
                prompt = (
                    f"Latest Architecture Best Practices:\n{arch_context}\n\n"
                    f"Consider these patterns when making architecture decisions.\n\n{prompt}"
                )
            else:
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
