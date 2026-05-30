"""
ConfigAdapter — Infrastructure adapter for architectural configuration
======================================================================
Loads costs, routing, fallbacks, and thresholds from JSON files.
Satisfies orchestrator.domain.ports.ConfigPort.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.infrastructure.config")


class JsonConfigAdapter:
    """Loads configuration from the orchestrator/config directory."""

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            # Default to orchestrator/config relative to this file
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = config_dir
        self._cache: dict[str, Any] = {}

    def _load_json(self, filename: str) -> dict[str, Any]:
        if filename in self._cache:
            return self._cache[filename]

        file_path = self.config_dir / filename
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cache[filename] = data
                return data
        except Exception as e:
            logger.error(f"Failed to load config {file_path}: {e}")
            return {}

    def get_costs(self) -> dict[str, dict[str, float]]:
        return self._load_json("costs.json")

    def get_routing(self) -> dict[str, list[str]]:
        return self._load_json("routing.json")

    def get_fallbacks(self) -> dict[str, str]:
        return self._load_json("fallbacks.json")

    def get_thresholds(self) -> dict[str, float]:
        return self._load_json("thresholds.json")

    def get_limits(self) -> dict[str, int]:
        return self._load_json("limits.json")
