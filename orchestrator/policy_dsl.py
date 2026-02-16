"""
Policy DSL — YAML/JSON policy file loader and static analyzer.
==============================================================
Allows policy definitions to be externalised from Python code into
YAML or JSON files, enabling non-programmer configuration of compliance rules.

YAML file format (pyyaml required; JSON always works via stdlib):
    global:               # or "org" as an alias
      - name: gdpr
        allow_training_on_output: false
        enforcement_mode: hard
    team:
      eng:
        - name: eu_only
          allowed_regions: [eu, global]
    job:
      job_001:
        - name: cost_cap
          max_cost_per_task_usd: 0.50
    node: {}

Usage:
    from orchestrator.policy_dsl import load_policy_file, PolicyAnalyzer

    hierarchy = load_policy_file("policies.yml")
    policies  = hierarchy.policies_for(team="eng", job_id="job_001")

    report = PolicyAnalyzer.analyze(policies)
    if not report.is_clean():
        print("Policy errors:", report.errors)
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import Model
from .policy import EnforcementMode, Policy, PolicyHierarchy, PolicySet, RateLimit

logger = logging.getLogger("orchestrator.policy_dsl")


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisReport:
    """
    Result of a static analysis pass over a list of policies.

    Attributes
    ----------
    errors   : blocking issues (e.g. impossible constraints, hard contradictions)
    warnings : potential issues (e.g. conflicting policies, overlapping allow/block)
    info     : informational observations (e.g. no cost cap, no latency SLA)
    """
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info:     list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        """Return True if there are no errors and no warnings."""
        return len(self.errors) == 0 and len(self.warnings) == 0


# ─────────────────────────────────────────────────────────────────────────────
# PolicyAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class PolicyAnalyzer:
    """
    Static analysis of policy lists. Does not require a running orchestrator.

    Checks performed:
      1. Contradiction: allowed_providers ∩ blocked_providers is non-empty → error
      2. Impossible: allowed_regions == [] → error (blocks every model)
      3. Conflict: two policies both have allowed_providers whose intersection is empty → warning
      4. Coverage: no policy defines a cost cap (max_cost_per_task_usd) → info
      5. Coverage: no policy defines a latency SLA (max_latency_ms) → info
    """

    @staticmethod
    def analyze(policies: list[Policy]) -> AnalysisReport:
        errors:   list[str] = []
        warnings: list[str] = []
        info:     list[str] = []

        has_cost_cap    = False
        has_latency_sla = False

        # Collect per-policy allowed_providers sets for cross-policy conflict check
        ap_sets: list[tuple[str, set[str]]] = []

        for p in policies:

            # 1. Contradiction: overlap between allowed_providers and blocked_providers
            if p.allowed_providers is not None and p.blocked_providers is not None:
                overlap = set(p.allowed_providers) & set(p.blocked_providers)
                if overlap:
                    errors.append(
                        f"[{p.name}] allowed_providers and blocked_providers overlap: {sorted(overlap)}"
                    )

            # 2. Impossible: empty allowed_regions blocks every model
            if p.allowed_regions is not None and len(p.allowed_regions) == 0:
                errors.append(
                    f"[{p.name}] allowed_regions=[] is impossible — blocks all models"
                )

            # Coverage tracking
            if p.max_cost_per_task_usd is not None:
                has_cost_cap = True
            if p.max_latency_ms is not None:
                has_latency_sla = True

            # Collect allowed_providers for cross-policy conflict check
            if p.allowed_providers is not None:
                ap_sets.append((p.name, set(p.allowed_providers)))

        # 3. Cross-policy conflict: two allowed_providers sets with empty intersection
        for i in range(len(ap_sets)):
            for j in range(i + 1, len(ap_sets)):
                name_a, set_a = ap_sets[i]
                name_b, set_b = ap_sets[j]
                if set_a.isdisjoint(set_b):
                    warnings.append(
                        f"[{name_a}] and [{name_b}] have no common allowed_providers — "
                        f"no model can satisfy both simultaneously"
                    )

        # 4–5. Coverage info
        if not has_cost_cap:
            info.append("No policy defines a cost cap (max_cost_per_task_usd) — tasks are cost-unconstrained")
        if not has_latency_sla:
            info.append("No policy defines a latency SLA (max_latency_ms) — tasks are latency-unconstrained")

        return AnalysisReport(errors=errors, warnings=warnings, info=info)


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_rate_limit(d: dict) -> Optional[RateLimit]:
    """Parse a rate_limit sub-dict into a RateLimit dataclass."""
    if not d:
        return None
    return RateLimit(
        calls_per_minute  = d.get("calls_per_minute"),
        cost_usd_per_hour = d.get("cost_usd_per_hour"),
        tokens_per_day    = d.get("tokens_per_day"),
    )


def _parse_policy(d: dict) -> Policy:
    """
    Parse a single policy dict into a Policy dataclass.

    Handles:
    - enforcement_mode string → EnforcementMode enum (unknown strings → None + warning)
    - blocked_models list[str] → list[Model] (unknown model strings → skipped + warning)
    - rate_limit sub-dict → RateLimit dataclass
    """
    name = d.get("name", "<unnamed>")

    # enforcement_mode
    em_raw = d.get("enforcement_mode")
    enforcement_mode: Optional[EnforcementMode] = None
    if em_raw is not None:
        try:
            enforcement_mode = EnforcementMode(str(em_raw).lower())
        except ValueError:
            logger.warning(
                "Unknown enforcement_mode %r in policy %r — defaulting to None (HARD)",
                em_raw, name,
            )

    # blocked_models: list of model value strings → list of Model enums
    blocked_models_raw = d.get("blocked_models")
    blocked_models: Optional[list[Model]] = None
    if blocked_models_raw is not None:
        parsed_models: list[Model] = []
        for ms in blocked_models_raw:
            try:
                parsed_models.append(Model(str(ms)))
            except ValueError:
                logger.warning("Unknown model %r in blocked_models of policy %r — skipped", ms, name)
        blocked_models = parsed_models if parsed_models else None

    # rate_limit
    rl_raw = d.get("rate_limit")
    rate_limit = _parse_rate_limit(rl_raw) if rl_raw else None

    return Policy(
        name                   = name,
        allowed_providers      = d.get("allowed_providers"),
        blocked_providers      = d.get("blocked_providers"),
        allowed_regions        = d.get("allowed_regions"),
        blocked_models         = blocked_models,
        allow_training_on_output = d.get("allow_training_on_output", True),
        pii_allowed            = d.get("pii_allowed", True),
        max_cost_per_task_usd  = d.get("max_cost_per_task_usd"),
        max_latency_ms         = d.get("max_latency_ms"),
        enforcement_mode       = enforcement_mode,
        rate_limit             = rate_limit,
    )


def _parse_policy_list(items: list[dict]) -> list[Policy]:
    return [_parse_policy(item) for item in items if isinstance(item, dict)]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_policy_dict(d: dict) -> PolicyHierarchy:
    """
    Convert a policy dict (as loaded from YAML/JSON) into a PolicyHierarchy.

    Top-level keys recognised:
      global / org  — list of policy dicts (org-level)
      team          — dict mapping team_name → list of policy dicts
      job           — dict mapping job_id   → list of policy dicts
      node          — dict mapping task_id  → list of policy dicts
    """
    # "global" and "org" are accepted aliases for the top-level org policies
    org_raw  = d.get("global") or d.get("org") or []
    team_raw = d.get("team")  or {}
    job_raw  = d.get("job")   or {}
    node_raw = d.get("node")  or {}

    org_policies = _parse_policy_list(org_raw if isinstance(org_raw, list) else [])

    team_policies: dict[str, list[Policy]] = {}
    for team_name, team_items in team_raw.items():
        if isinstance(team_items, list):
            team_policies[str(team_name)] = _parse_policy_list(team_items)

    job_policies: dict[str, list[Policy]] = {}
    for job_id, job_items in job_raw.items():
        if isinstance(job_items, list):
            job_policies[str(job_id)] = _parse_policy_list(job_items)

    node_policies: dict[str, list[Policy]] = {}
    for task_id, node_items in node_raw.items():
        if isinstance(node_items, list):
            node_policies[str(task_id)] = _parse_policy_list(node_items)

    return PolicyHierarchy(
        org  = org_policies or None,
        team = team_policies or None,
        job  = job_policies or None,
        node = node_policies or None,
    )


def load_policy_file(path: str | Path) -> PolicyHierarchy:
    """
    Load a policy file and return a PolicyHierarchy.

    Supported formats:
      .json          — always available (stdlib)
      .yaml / .yml   — requires pyyaml (pip install pyyaml)

    Raises
    ------
    ImportError       if a .yaml/.yml file is passed but pyyaml is not installed
    ValueError        if the file extension is not recognised
    FileNotFoundError if the path does not exist
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"Cannot load YAML policy file '{p.name}': pyyaml is not installed. "
                "Install it with: pip install pyyaml"
            ) from exc
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        raise ValueError(
            f"Unsupported policy file extension '{suffix}'. "
            "Supported: .json, .yaml, .yml"
        )

    if not isinstance(data, dict):
        raise ValueError(f"Policy file '{p.name}' must contain a YAML/JSON object at the top level.")

    return load_policy_dict(data)
