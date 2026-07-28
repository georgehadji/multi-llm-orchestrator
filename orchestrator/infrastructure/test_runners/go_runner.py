"""Go test runner using -json for machine-readable output."""

from __future__ import annotations

from .base import TestRunnerBase


class GoRunner(TestRunnerBase):
    """Runs go test -json and parses structured output."""

    @property
    def framework(self) -> str:
        return "go"

    def build_command(
        self,
        workspace: AnyWorkspace,
        selection: AnyTestSelection | None = None,
    ) -> list[str]:
        return ["go", "test", "./", "-json"]


AnyWorkspace = object
AnyTestSelection = object
