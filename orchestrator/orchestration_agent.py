"""
Orchestration Agent
===================
Converts natural-language intent into typed JobSpecV2 + PolicySpecV2 specs
and supports a human refine() loop before submission to ControlPlane.

Design principles:
  • Outputs ONLY specs, never calls task executors
  • Every chosen constraint is explained in the rationale field
  • Uses the cheapest capable model for spec generation
  • analyze_run() implements the B5 continuous-improvement loop

Usage:
    agent = OrchestrationAgent()
    draft = await agent.draft("θέλω pipeline code review, budget $2, EU data only")
    print(draft.rationale)
    # iterate:
    draft = await agent.refine(draft, "also add ruff linting")
    # submit:
    from orchestrator.control_plane import ControlPlane
    state = await ControlPlane().submit(draft.job, draft.policy)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .specs import (
    Constraints,
    InputSpec,
    JobSpecV2,
    PolicySpecV2,
    RoutingHint,
    SLAs,
    ValidationRule,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Draft
# ─────────────────────────────────────────────

@dataclass
class AgentDraft:
    """Output of OrchestrationAgent.draft() / refine()."""

    job: JobSpecV2
    policy: PolicySpecV2
    rationale: str = ""
    nl_intent: str = ""


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an Orchestration Agent. Your ONLY task is to produce JobSpecV2 and
PolicySpecV2 specifications in JSON. You NEVER invoke model APIs directly and
you NEVER execute tasks yourself.

Available hard constraints:
  - "no_training"     — never send data to models that use it for training
  - "eu_only"         — restrict all API calls to EU-hosted models
  - "no_pii_logging"  — disable PII capture in audit logs

Soft constraint keys (0.0–1.0 weight):
  - "prefer_low_cost"    — minimise API spend
  - "prefer_low_latency" — minimise wall-clock time

data_locality values: "eu" | "us" | "any"

For EVERY constraint you choose, you MUST explain WHY in the rationale field.

Return ONLY valid JSON with this structure (no markdown, no explanation):
{
  "job": {
    "goal": "<string>",
    "inputs": {
      "data_locality": "any",
      "contains_pii": false
    },
    "slas": {
      "max_cost_usd": null,
      "min_quality_tier": 0.85
    },
    "constraints": {
      "hard": [],
      "soft": {}
    },
    "metrics": []
  },
  "policy": {
    "allow_deny_rules": [],
    "routing_hints": [],
    "validation_rules": [],
    "escalation_rules": []
  },
  "rationale": "<explanation of every constraint chosen>"
}
"""

_CAPABILITY_LIBRARY = """\
Available validators: python_syntax, json_schema, pytest, ruff, latex, length
Available routing targets: eu_safe_only, high_quality, any
Available escalation actions: human_review, abort, fallback_model
"""


# ─────────────────────────────────────────────
# OrchestrationAgent
# ─────────────────────────────────────────────

class OrchestrationAgent:
    """
    Converts NL intent into draft specs and supports a human-in-the-loop
    refine() cycle.

    Parameters
    ----------
    client : UnifiedClient | None
        Optional pre-built API client.  If None, one is lazily created.
    """

    def __init__(self, client=None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            from .api_clients import UnifiedClient
            from .cache import DiskCache
            self._client = UnifiedClient(cache=DiskCache(), max_concurrency=1)
        return self._client

    async def draft(self, nl_intent: str) -> AgentDraft:
        """
        Convert a natural-language intent string into a draft AgentDraft.

        The LLM produces JSON; we parse it into typed dataclasses.
        On any parse failure, a safe default draft is returned.
        """
        prompt = (
            f"CAPABILITY LIBRARY:\n{_CAPABILITY_LIBRARY}\n\n"
            f"USER INTENT:\n{nl_intent}\n\n"
            "Produce the JSON spec described in your system prompt."
        )
        raw = await self._call_llm(prompt)
        return self._parse_draft(raw, nl_intent)

    async def refine(self, draft: AgentDraft, feedback: str) -> AgentDraft:
        """
        Refine an existing draft based on human feedback.
        """
        import dataclasses as _dc

        current_job_json = json.dumps(_dc.asdict(draft.job), indent=2, default=str)
        current_policy_json = json.dumps(_dc.asdict(draft.policy), indent=2, default=str)

        prompt = (
            f"CAPABILITY LIBRARY:\n{_CAPABILITY_LIBRARY}\n\n"
            f"CURRENT JOB SPEC:\n{current_job_json}\n\n"
            f"CURRENT POLICY SPEC:\n{current_policy_json}\n\n"
            f"CURRENT RATIONALE:\n{draft.rationale}\n\n"
            f"USER FEEDBACK:\n{feedback}\n\n"
            "Produce a revised JSON spec that incorporates the feedback."
        )
        raw = await self._call_llm(prompt)
        return self._parse_draft(raw, draft.nl_intent)

    async def analyze_run(
        self,
        state,  # ProjectState
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> str:
        """
        B5 — Continuous improvement loop.

        Sends a summary of the completed run to the LLM and returns
        suggestions for tightening constraints or adding validators.
        """
        import dataclasses as _dc

        summary = {
            "status": state.status.value,
            "total_cost_usd": state.budget.spent_usd,
            "task_count": len(state.results),
            "constraints_used": job.constraints.hard,
            "avg_score": (
                sum(r.score for r in state.results.values()) / len(state.results)
                if state.results else 0.0
            ),
        }

        prompt = (
            f"You just ran a job. Here is the run summary:\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            f"Original job goal: {job.goal}\n\n"
            "Suggest 2-3 specific improvements to the JobSpecV2 or PolicySpecV2 "
            "that would improve cost efficiency, quality, or compliance. "
            "Format each suggestion as a single line starting with 'Suggestion: '."
        )
        raw = await self._call_llm(prompt)
        return raw.strip()

    # ─────────────────────────────────────────
    # LLM call
    # ─────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        from .models import Model
        client = self._get_client()

        _PREFERENCE = [
            Model.GEMINI_FLASH,
            Model.GPT_4O_MINI,
            Model.DEEPSEEK_CHAT,
            Model.CLAUDE_HAIKU,
        ]
        model = next((m for m in _PREFERENCE if client.is_available(m)), None)
        if model is None:
            model = next((m for m in Model if client.is_available(m)), None)
        if model is None:
            raise RuntimeError(
                "OrchestrationAgent: no LLM provider available. "
                "Check API keys in .env."
            )

        resp = await client.call(
            model=model,
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.1,
            timeout=60,
        )
        logger.debug(
            "OrchestrationAgent: used %s (cost $%.6f)", model.value, resp.cost_usd
        )
        return resp.text

    # ─────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────

    def _parse_draft(self, raw: str, nl_intent: str) -> AgentDraft:
        """Parse LLM JSON output into an AgentDraft. Returns a safe default on failure."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                ln for i, ln in enumerate(lines)
                if i > 0 and ln.strip() != "```"
            ).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("OrchestrationAgent: failed to parse LLM output: %s", exc)
            return self._default_draft(nl_intent)

        try:
            return self._build_draft(data, nl_intent)
        except Exception as exc:
            logger.warning("OrchestrationAgent: failed to build draft: %s", exc)
            return self._default_draft(nl_intent)

    def _build_draft(self, data: dict, nl_intent: str) -> AgentDraft:
        job_data = data.get("job", {})
        pol_data = data.get("policy", {})

        inputs_data = job_data.get("inputs", {})
        inputs = InputSpec(
            data_locality=inputs_data.get("data_locality", "any"),
            contains_pii=bool(inputs_data.get("contains_pii", False)),
        )

        slas_data = job_data.get("slas", {})
        slas = SLAs(
            max_cost_usd=slas_data.get("max_cost_usd"),
            min_quality_tier=float(slas_data.get("min_quality_tier", 0.85)),
        )

        constraints_data = job_data.get("constraints", {})
        constraints = Constraints(
            hard=list(constraints_data.get("hard", [])),
            soft={k: float(v) for k, v in constraints_data.get("soft", {}).items()},
        )

        job = JobSpecV2(
            goal=str(job_data.get("goal", nl_intent)),
            inputs=inputs,
            slas=slas,
            constraints=constraints,
            metrics=list(job_data.get("metrics", [])),
        )

        routing_hints = [
            RoutingHint(condition=rh.get("condition", ""), target=rh.get("target", ""))
            for rh in pol_data.get("routing_hints", [])
        ]
        validation_rules = [
            ValidationRule(
                node_pattern=vr.get("node_pattern", "*"),
                mandatory_validators=list(vr.get("mandatory_validators", [])),
            )
            for vr in pol_data.get("validation_rules", [])
        ]

        policy = PolicySpecV2(
            allow_deny_rules=list(pol_data.get("allow_deny_rules", [])),
            routing_hints=routing_hints,
            validation_rules=validation_rules,
        )

        return AgentDraft(
            job=job,
            policy=policy,
            rationale=str(data.get("rationale", "")),
            nl_intent=nl_intent,
        )

    def _default_draft(self, nl_intent: str) -> AgentDraft:
        """Safe default when LLM output cannot be parsed."""
        return AgentDraft(
            job=JobSpecV2(goal=nl_intent),
            policy=PolicySpecV2(),
            rationale="Default draft (LLM output could not be parsed).",
            nl_intent=nl_intent,
        )


__all__ = ["AgentDraft", "OrchestrationAgent"]
