"""
CLI chat mode — interactive spec-gathering session in the terminal.

Invoked via:  python -m orchestrator chat
              python -m orchestrator chat --budget 12.0 --dry-run
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

# ANSI colours (gracefully degraded on Windows without colorama)
try:
    import colorama

    colorama.init(autoreset=True)
    _C_AGENT = "\033[1;36m"  # bold cyan  — agent messages
    _C_HINT = "\033[0;33m"  # yellow     — suggestions
    _C_DIM = "\033[0;90m"  # dark grey  — meta info
    _C_RESET = "\033[0m"
    _C_BOLD = "\033[1m"
    _C_GREEN = "\033[1;32m"
    _C_RED = "\033[1;31m"
except ImportError:
    _C_AGENT = _C_HINT = _C_DIM = _C_RESET = _C_BOLD = _C_GREEN = _C_RED = ""


def _print_agent(text: str) -> None:
    prefix = f"{_C_AGENT}Orchestrator ▸{_C_RESET} "
    for line in text.splitlines():
        print(prefix + line)
        prefix = "               "  # indent continuation lines


def _print_suggestions(suggestions: list[str]) -> None:
    if not suggestions:
        return
    print()
    for s in suggestions:
        print(f"  {_C_HINT}✦ {s}{_C_RESET}")


def _print_confidence(conf: float) -> None:
    bar_len = 20
    filled = int(conf * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(conf * 100)
    print(f"\n  {_C_DIM}Spec completeness  [{bar}] {pct}%{_C_RESET}")


def _print_divider() -> None:
    print(f"\n{_C_DIM}{'─' * 60}{_C_RESET}\n")


async def run_chat(
    budget: float = 8.0,
    dry_run: bool = False,
    output_dir: str = "",
) -> None:
    """Full interactive chat loop — blocks until spec is ready, then builds."""
    from ..api_clients import UnifiedClient
    from ..cache import DiskCache
    from .conversation_agent import ConversationAgent

    # Build a minimal client for the conversation agent
    try:
        cache = DiskCache()
        client = UnifiedClient(cache=cache)
    except Exception as exc:
        print(f"{_C_RED}Could not initialise LLM client: {exc}{_C_RESET}")
        sys.exit(1)

    agent = ConversationAgent(client=client)

    print()
    print(f"{_C_BOLD}{'═' * 60}")
    print("  Multi-LLM Orchestrator — Interactive Mode")
    print(f"{'═' * 60}{_C_RESET}")
    print(f"\n  {_C_DIM}Type your idea and I'll help you build the perfect spec.")
    print(f"  Say 'ok' / 'go' / 'build it' when you're happy to proceed.{_C_RESET}\n")

    # Opening message
    opening = await agent.start()
    _print_agent(opening.content)

    # Conversation loop
    accepted_enhancements: list[str] = []

    while not agent.ready:
        print()
        try:
            user_input = input(f"{_C_BOLD}You ▸{_C_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_C_DIM}Session cancelled.{_C_RESET}")
            return

        if not user_input:
            continue

        # Handle enhancement acceptance inline
        # e.g. "yes to stripe" / "add stripe" / "yes 1" / "no"
        pending = getattr(agent, "_pending_suggestions", [])
        if pending and user_input.lower() in ("y", "yes", "all", "yes all", "add all"):
            for s in pending:
                agent.accept_enhancement(s)
                accepted_enhancements.append(s)
            user_input = "yes to all suggestions"

        _print_divider()
        turn = await agent.process_turn(user_input)

        _print_agent(turn.content)
        _print_suggestions(turn.suggestions)

        # Store pending suggestions so user can bulk-accept next turn
        agent._pending_suggestions = turn.suggestions  # type: ignore[attr-defined]

        if turn.suggestions:
            print(f"\n  {_C_DIM}Reply 'yes' to add all, or just keep going.{_C_RESET}")

        _print_confidence(turn.confidence)

        if turn.ready_to_build:
            break

    # ── Build phase ──────────────────────────────────────────────────────────
    _print_divider()
    spec = agent.spec

    if dry_run:
        print(f"{_C_GREEN}✓ Spec ready (dry-run — not building){_C_RESET}\n")
        _display_spec(spec)
        return

    print(f"{_C_GREEN}✓ Spec complete — starting build...{_C_RESET}\n")
    _display_spec(spec)
    print()

    # Delegate to the main orchestrator pipeline
    await _launch_build(spec, budget=budget, output_dir=output_dir)


def _display_spec(spec: Any) -> None:
    """Print a human-readable summary of the ProjectSpec."""
    print(f"{_C_BOLD}── Project Spec ───────────────────────────────────────{_C_RESET}")
    print(f"  {_C_BOLD}Description:{_C_RESET}  {spec.project_description}")
    print(f"  {_C_BOLD}Target users:{_C_RESET} {spec.target_users}")
    print(f"  {_C_BOLD}Platform:{_C_RESET}     {spec.platform or '(not specified)'}")
    print(f"  {_C_BOLD}Tech stack:{_C_RESET}   {spec.tech_stack or '(not specified)'}")
    print(f"  {_C_BOLD}Auth:{_C_RESET}         {spec.auth_requirements or '(not specified)'}")
    print(f"  {_C_BOLD}Data:{_C_RESET}         {spec.data_persistence or '(not specified)'}")
    if spec.core_features:
        print(f"  {_C_BOLD}Features:{_C_RESET}")
        for f in spec.core_features:
            print(f"    • {f}")
    if spec.enhancements_accepted:
        print(f"  {_C_BOLD}Enhancements:{_C_RESET}")
        for e in spec.enhancements_accepted:
            print(f"    ✦ {e}")
    if spec.integrations:
        print(f"  {_C_BOLD}Integrations:{_C_RESET} {', '.join(spec.integrations)}")
    print(f"  {_C_BOLD}Success criteria:{_C_RESET} {spec.success_criteria or '(defaults)'}")


async def _launch_build(spec: Any, budget: float, output_dir: str) -> None:
    """Hand off the completed spec to the Orchestrator pipeline."""
    from ..engine import Orchestrator
    from ..budget import Budget

    args = spec.to_orchestrator_args()
    args["budget"] = budget

    orch = Orchestrator(  # type: ignore[call-arg]
        budget=Budget(max_usd=budget),
        verbose=True,
    )

    try:
        async with orch:
            state = await orch.run_project(
                project_description=args["project"],
                success_criteria=args["criteria"],
                output_dir=output_dir or None,
            )
        print(f"\n{_C_GREEN}✓ Build complete — status: {state.status.value}{_C_RESET}")
        if output_dir or hasattr(state, "output_dir"):
            loc = output_dir or getattr(state, "output_dir", "")
            if loc:
                print(f"  Output: {loc}")
    except Exception as exc:
        print(f"\n{_C_RED}Build failed: {exc}{_C_RESET}")
        raise
