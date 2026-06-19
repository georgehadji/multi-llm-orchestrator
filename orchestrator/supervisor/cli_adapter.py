"""
Supervisor CLI adapter — persistent REPL and one-shot modes.

Each REPL line becomes a ``Directive`` handled by the Supervisor.  Sessions and
lessons survive process restarts in ``~/.orchestrator_cache/supervisor.db``.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.progress import ProgressRenderer

from .models import Directive
from .service import Supervisor
from .store import SupervisorStore

logger = logging.getLogger("orchestrator.supervisor.cli_adapter")


def _default_factory(default_budget: float) -> Callable[[float | None], Orchestrator]:
    """Build a fresh Orchestrator for each Supervisor run.

    The per-directive budget (from ``Directive.budget`` or text extraction) takes
    precedence; ``default_budget`` is the REPL/CLI fallback when none is given.
    """

    def factory(budget: float | None = None) -> Orchestrator:
        effective = budget if budget is not None else default_budget
        return Orchestrator(
            budget=Budget(max_usd=effective, max_time_seconds=5400),
        )

    return factory


async def run_repl(
    budget: float = 8.0,
    db_path: str | None = None,
) -> int:
    """Run the interactive supervisor REPL.

    Commands:
      /sessions  — list recent sessions
      /quit      — exit
      anything else — treated as a directive
    """
    store = SupervisorStore(db_path=Path(db_path) if db_path else None)
    await store.connect()

    supervisor = Supervisor(store, _default_factory(budget))

    print("Supervisor REPL. Type a directive, '/sessions' to list, '/quit' to exit.")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/sessions":
                sessions = await store.list_sessions(limit=20)
                for session in sessions:
                    print(f"  {session.id}  {session.status}  {session.summary[:60]}")
                continue

            directive = Directive(source="human", text=line, budget=budget)
            renderer = ProgressRenderer(quiet=False)
            try:
                result = await supervisor.handle(directive, on_event=renderer.handle)
                print(
                    f"[{result.project_status}] session={result.session_id} "
                    f"lessons={result.lessons_recorded}"
                )
            except Exception as exc:
                logger.exception("REPL directive failed")
                print(f"Error: {exc}")
    finally:
        await store.close()
    return 0
