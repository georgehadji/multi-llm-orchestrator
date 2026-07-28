"""Cargo test runner using --format json for machine-readable output."""

from __future__ import annotations

from .base import TestRunnerBase


class CargoRunner(TestRunnerBase):
    """Runs cargo test --format json and parses structured output."""

    @property
    def framework(self) -> str:
        return "cargo"

    def build_command(
        self,
        workspace: AnyWorkspace,
        selection: AnyTestSelection | None = None,
    ) -> list[str]:
        return ["cargo", "test", "--format", "json"]


AnyWorkspace = object
AnyTestSelection = object
