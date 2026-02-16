"""
Project File Loader — load a JobSpec from a YAML file
======================================================
Supports a human-friendly YAML schema that maps onto the full
Policy / PolicySet / JobSpec / Budget API.

Schema reference (all fields except `project` and `criteria` are optional):

    project: "Build a FastAPI service"      # or block scalar with |
    criteria: "All tests pass"              # or block scalar with |
    budget_usd: 5.0                         # default 8.0
    time_seconds: 3600                      # default 5400
    concurrency: 3                          # default 3
    verbose: false                          # default false
    project_id: "my-project-v1"            # default: auto-generated
    output_dir: "./results"                # optional: write output files here
    quality_targets:
      code_generation: 0.90
      code_review:     0.88
      complex_reasoning: 0.92
      evaluation:      0.95
    policies:
      - name: no_training
        allow_training_on_output: false
      - name: eu_only
        allowed_regions: [eu]
      - name: no_openai
        blocked_providers: [openai]
      - name: no_specific_model
        blocked_models: [kimi-k2.5]
      - name: fast_sla
        max_latency_ms: 8000
      - name: pii_safe
        pii_allowed: false
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .models import Budget, Model, TaskType
from .policy import Policy, PolicySet, JobSpec


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectFileResult:
    """Everything extracted from a YAML project file."""
    spec: JobSpec
    concurrency: int
    verbose: bool
    project_id: str
    output_dir: Optional[str] = None   # write output files here (optional)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_project_file(path: str | Path) -> ProjectFileResult:
    """
    Parse a YAML project file and return a ProjectFileResult.

    Raises
    ------
    FileNotFoundError  — file doesn't exist
    ValueError         — required fields missing or values invalid
    ImportError        — PyYAML not installed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: PyYAML is required to load project files.\n"
            "Install it with:  pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # ── Required fields ───────────────────────────────────────────────────────
    project = raw.get("project", "").strip()
    criteria = raw.get("criteria", "").strip()
    if not project:
        raise ValueError(f"'{path}': 'project' field is required")
    if not criteria:
        raise ValueError(f"'{path}': 'criteria' field is required")

    # ── Budget ────────────────────────────────────────────────────────────────
    budget_usd = float(raw.get("budget_usd", 8.0))
    time_seconds = float(raw.get("time_seconds", 5400.0))
    budget = Budget(max_usd=budget_usd, max_time_seconds=time_seconds)

    # ── Misc CLI-level settings ───────────────────────────────────────────────
    concurrency = int(raw.get("concurrency", 3))
    verbose = bool(raw.get("verbose", False))
    project_id = str(raw.get("project_id", "")).strip()
    output_dir_raw = raw.get("output_dir", None)
    output_dir: Optional[str] = str(output_dir_raw).strip() if output_dir_raw else None

    # ── Quality targets ───────────────────────────────────────────────────────
    quality_targets: dict[TaskType, float] = {}
    for key, value in (raw.get("quality_targets") or {}).items():
        try:
            task_type = TaskType(key)
        except ValueError:
            valid = [t.value for t in TaskType]
            raise ValueError(
                f"'{path}': unknown task type '{key}' in quality_targets. "
                f"Valid values: {valid}"
            )
        quality_targets[task_type] = float(value)

    # ── Policies ──────────────────────────────────────────────────────────────
    policies: list[Policy] = []
    for p_raw in (raw.get("policies") or []):
        policies.append(_parse_policy(p_raw, path))
    policy_set = PolicySet(global_policies=policies)

    # ── Assemble JobSpec ──────────────────────────────────────────────────────
    spec = JobSpec(
        project_description=project,
        success_criteria=criteria,
        budget=budget,
        policy_set=policy_set,
        quality_targets=quality_targets,
    )

    return ProjectFileResult(
        spec=spec,
        concurrency=concurrency,
        verbose=verbose,
        project_id=project_id,
        output_dir=output_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_policy(raw: dict[str, Any], source_path: Path) -> Policy:
    """Convert a YAML policy dict → Policy dataclass."""
    name = str(raw.get("name", "unnamed"))

    # allowed_providers / blocked_providers: list[str]
    allowed_providers: Optional[list[str]] = raw.get("allowed_providers")
    blocked_providers: Optional[list[str]] = raw.get("blocked_providers")

    # allowed_regions: list[str]
    allowed_regions: Optional[list[str]] = raw.get("allowed_regions")

    # blocked_models: list[Model]
    blocked_models: Optional[list[Model]] = None
    if raw.get("blocked_models"):
        blocked_models = []
        for m_val in raw["blocked_models"]:
            try:
                blocked_models.append(Model(m_val))
            except ValueError:
                valid = [m.value for m in Model]
                raise ValueError(
                    f"'{source_path}': unknown model '{m_val}' in policy '{name}'. "
                    f"Valid values: {valid}"
                )

    allow_training: bool = bool(raw.get("allow_training_on_output", True))
    pii_allowed: bool = bool(raw.get("pii_allowed", True))
    max_cost: Optional[float] = (
        float(raw["max_cost_per_task_usd"]) if "max_cost_per_task_usd" in raw else None
    )
    max_latency: Optional[float] = (
        float(raw["max_latency_ms"]) if "max_latency_ms" in raw else None
    )

    return Policy(
        name=name,
        allowed_providers=allowed_providers,
        blocked_providers=blocked_providers,
        allowed_regions=allowed_regions,
        blocked_models=blocked_models,
        allow_training_on_output=allow_training,
        pii_allowed=pii_allowed,
        max_cost_per_task_usd=max_cost,
        max_latency_ms=max_latency,
    )
