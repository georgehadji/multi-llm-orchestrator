"""Jest runner using --json for machine-readable output."""

from __future__ import annotations

from .base import TestRunnerBase


class JestRunner(TestRunnerBase):
    """Runs jest with --json and parses structured output."""

    @property
    def framework(self) -> str:
        return "jest"

    def build_command(
        self,
        workspace: AnyWorkspace,
        selection: AnyTestSelection | None = None,
    ) -> list[str]:
        return ["npx", "jest", "--json"]


AnyWorkspace = object
AnyTestSelection = object
