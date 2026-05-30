"""
ConfigSync - Push/pull config with backends (Supabase, custom).
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 10, Phase B4 (Base44-inspired).
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SyncTarget:
    name: str
    type: str  # "supabase", "custom_api", "env_file"
    url: str = ""
    api_key: str = ""
    headers: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, name, prefix):
        return cls(
            name=name,
            type=os.environ.get(f"{prefix}_TYPE", "custom_api"),
            url=os.environ.get(f"{prefix}_URL", ""),
            api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        )


class ConfigSync:
    """Syncs application config with external backends."""

    def __init__(self, config_path="orchestrator_config.json"):
        self.config_path = Path(config_path)
        self.targets: list[SyncTarget] = []
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.targets = [
                    SyncTarget(
                        name=t["name"],
                        type=t.get("type", "custom_api"),
                        url=t.get("url", ""),
                        api_key=t.get("api_key", ""),
                    )
                    for t in data.get("targets", [])
                ]
            except Exception:
                pass

    def save_config(self):
        self.config_path.write_text(
            json.dumps(
                {"targets": [{"name": t.name, "type": t.type, "url": t.url} for t in self.targets]},
                indent=2,
            ),
            encoding="utf-8",
        )

    def add_target(self, target):
        self.targets.append(target)
        self.save_config()

    async def push(self, target_name, data):
        """Push config data to a target backend."""
        target = next((t for t in self.targets if t.name == target_name), None)
        if not target:
            return False, f"Target '{target_name}' not found"

        import aiohttp

        try:
            headers = {"Content-Type": "application/json", **target.headers}
            if target.api_key:
                headers["Authorization"] = f"Bearer {target.api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.post(target.url, json=data, headers=headers, timeout=30) as resp:
                    result = await resp.json()
                    return resp.status == 200, result
        except Exception as e:
            logger.warning(f"Push to {target_name} failed: {e}")
            return False, str(e)

    async def pull(self, target_name):
        """Pull config data from a target backend."""
        target = next((t for t in self.targets if t.name == target_name), None)
        if not target:
            return None, f"Target '{target_name}' not found"

        import aiohttp

        try:
            headers = {**target.headers}
            if target.api_key:
                headers["Authorization"] = f"Bearer {target.api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(target.url, headers=headers, timeout=30) as resp:
                    return await resp.json(), None
        except Exception as e:
            logger.warning(f"Pull from {target_name} failed: {e}")
            return None, str(e)
