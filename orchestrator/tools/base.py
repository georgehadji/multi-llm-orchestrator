"""
Tool base — Standardized tool interface for agents
====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 4 of the Agentic System Implementation Plan.
Every agent accesses the world through tools. This defines the
standard interface and a simple registry.

Tools handle: file I/O, shell commands, git operations,
package management, test running, building, and web search.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.tools.base")


class ToolPermission(str, Enum):
    """Permissions that a tool may require."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL_EXECUTE = "shell_execute"
    NETWORK_REQUEST = "network_request"
    PACKAGE_INSTALL = "package_install"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"


@dataclass
class ToolResult:
    """Structured result from a tool execution."""

    success: bool
    output: str = ""
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Standardized tool interface for all agents."""

    name: str = ""
    description: str = ""
    required_permissions: list[ToolPermission] = []

    @abstractmethod
    async def execute(self, params: dict) -> ToolResult:
        """Execute the tool with the given parameters."""

    def validate_params(self, params: dict) -> bool:
        """Validate parameters before execution. Override for custom validation."""
        return True


class ToolRegistry:
    """Registry of available tools with permission enforcement."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.granted_permissions: set[str] = set()

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def grant(self, permission: ToolPermission) -> None:
        self.granted_permissions.add(permission.value)

    def can_execute(self, tool: Tool) -> bool:
        for perm in tool.required_permissions:
            if perm.value not in self.granted_permissions:
                return False
        return True
