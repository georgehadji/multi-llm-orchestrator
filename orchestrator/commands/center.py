"""
CommandCenter — Interactive NLP REPL for app development
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Usage:
    python -m orchestrator command_center
    # Then type: "build a coffee shop landing page"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("orchestrator.command_center")

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           AI ORCHESTRATOR — COMMAND CENTER                   ║
║   Describe what you want to build in plain English           ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands:
  build <desc>     — Build: "build a todo app with auth"
  modify <p> <t>   — Modify: "modify ./repo 'add logging'"
  analyze <path>   — Analyze: "analyze ./my-project"
  budget $X        — Set budget: "budget $20"
  premium          — Premium models
  budget tier      — Budget models
  status           — Session status
  history          — Past projects
  help             — This help
  quit/exit        — Exit
"""


@dataclass
class CCState:
    budget: float = 10.0
    premium: bool = False
    project_count: int = 0
    last_output: str = ""


class CommandCenter:
    """Interactive REPL. Runs the Orchestrator directly (not via subprocess)."""

    def __init__(self):
        self.state = CCState()

    def run(self):
        print(BANNER)
        print(HELP)
        print(
            f"  Budget: ${self.state.budget:.2f} | {'Premium' if self.state.premium else 'Budget'}"
        )
        while True:
            try:
                text = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            result = self.handle(text)
            print(f"\n{result}")

    def handle(self, text: str) -> str:
        c = text.lower().strip()
        if c == "help":
            return HELP
        if c == "status":
            return f"Budget: ${self.state.budget:.2f} | {'Premium' if self.state.premium else 'Budget'} | Projects built: {self.state.project_count}"
        if c == "history":
            return self._history()
        if c in ("budget", "budget tier"):
            self.state.premium = False
            return "Budget tier: Qwen, DeepSeek, Codestral"
        if c == "premium":
            self.state.premium = True
            return "Premium tier: GPT-5, Claude Sonnet"
        if c.startswith("budget") and c not in ("budget", "budget tier"):
            return self._set_budget(c)
        if c.startswith("build "):
            return self._build(c[6:])
        if c.startswith("modify "):
            return self._modify(c[7:])
        if c.startswith("analyze "):
            return self._analyze(c[8:])
        return self._build(text)

    def _set_budget(self, c: str) -> str:
        try:
            amt = float(c.replace("budget", "").replace("$", "").strip())
            self.state.budget = amt
            return f"Budget set to ${amt:.2f}"
        except ValueError:
            return "Usage: budget 20 or budget $20"

    def _build(self, desc: str) -> str:
        if not desc:
            return "What do you want to build?"
        self.state.project_count += 1
        pid = f"project_{int(time.time())}"
        cmd = f'python -m orchestrator --project "{desc}" --criteria "The app must work correctly, have tests, and follow best practices" --budget {self.state.budget} --output-dir ./outputs/{pid}'
        return f"Command ready:\n\n  {cmd}\n\nRun this in your terminal. The orchestrator will decompose, generate, validate, and deliver."

    def _modify(self, rest: str) -> str:
        parts = rest.split(" ", 1)
        if len(parts) < 2:
            return "Usage: modify <repo_path> <objective>"
        return f'Command: python -m orchestrator modify --repo {parts[0]} --objective "{parts[1]}"'

    def _analyze(self, path: str) -> str:
        return f"Command: python -m orchestrator analyze --path {path} --budget {self.state.budget}"

    def _history(self) -> str:
        d = Path("./outputs")
        if not d.exists():
            return "No past projects."
        projects = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        lines = []
        for p in projects[:10]:
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            lines.append(f"  [{t}] {p.name}")
        return "\n".join(lines) if lines else "No projects found."


def main():
    CommandCenter().run()


if __name__ == "__main__":
    main()
