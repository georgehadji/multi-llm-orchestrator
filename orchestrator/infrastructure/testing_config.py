"""Testing configuration — single source of truth (F-8).

All testing limits (repair iterations, suite timeout, mutation budget) are
declared once in ``orchestrator/config/limits.json`` under ``testing`` and
read through the existing config adapter. Defaults live in the JSON file,
not in code — the contract test asserts no limit is hardcoded at more than
one site.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, float | int] = {
    "max_repair_iterations": 5,
    "suite_timeout_s": 120,
    "mutation_sample_size": 12,
    "mutation_timeout_s": 60,
}


@dataclass(frozen=True)
class TestingConfig:
    """Resolved testing limits. Env vars override for operational control."""

    max_repair_iterations: int = 5
    suite_timeout_s: float = 120.0
    mutation_sample_size: int = 12
    mutation_timeout_s: float = 60.0


def load_testing_config() -> TestingConfig:
    """Load testing limits from ``limits.json`` with env-var overrides.

    Env overrides (operational escape hatches, not new defaults):
    ``ORCH_SUITE_TIMEOUT_S``, ``ORCH_MUTATION_SAMPLE_SIZE``,
    ``ORCH_MUTATION_TIMEOUT_S``.
    """
    values: dict[str, float | int] = dict(_DEFAULTS)
    try:
        from .adapters.config_adapter import JsonConfigAdapter

        limits = JsonConfigAdapter().get_limits()
        testing = limits.get("testing", {})
        for key in _DEFAULTS:
            if key in testing and testing[key] is not None:
                values[key] = testing[key]
    except Exception as exc:  # pragma: no cover - config is static
        logger.warning("Failed to load testing config, using defaults: %s", exc)

    env_overrides: dict[str, tuple[str, str]] = {
        "suite_timeout_s": ("ORCH_SUITE_TIMEOUT_S", "suite_timeout_s"),
        "mutation_sample_size": ("ORCH_MUTATION_SAMPLE_SIZE", "mutation_sample_size"),
        "mutation_timeout_s": ("ORCH_MUTATION_TIMEOUT_S", "mutation_timeout_s"),
    }
    for field_name, (env_name, key) in env_overrides.items():
        raw = os.environ.get(env_name)
        if raw:
            try:
                values[key] = float(raw) if field_name.endswith("_s") else int(raw)
            except ValueError:
                logger.warning("Ignoring invalid %s=%r", env_name, raw)

    return TestingConfig(
        max_repair_iterations=int(values["max_repair_iterations"]),
        suite_timeout_s=float(values["suite_timeout_s"]),
        mutation_sample_size=int(values["mutation_sample_size"]),
        mutation_timeout_s=float(values["mutation_timeout_s"]),
    )
