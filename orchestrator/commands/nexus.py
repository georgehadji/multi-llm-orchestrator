"""nexus command module."""

from __future__ import annotations


def register(subparsers) -> None:
    """Register the 'nexus' subcommand for Nexus Search."""
    parser = subparsers.add_parser(
        "nexus",
        help="Nexus Search - Web search for AI Orchestrator",
    )
    nexus_sub = parser.add_subparsers(dest="nexus_command", metavar="COMMAND")

    # Search command
    search_p = nexus_sub.add_parser("search", help="Perform web search")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("-s", "--sources", help="Sources (web,academic,tech,news,code)")
    search_p.add_argument(
        "-o", "--optimization", default="balanced", choices=["speed", "balanced", "quality"]
    )
    search_p.add_argument("-n", "--num-results", type=int, default=10)
    search_p.add_argument("--json", action="store_true")
    search_p.set_defaults(func=search)

    # Research command
    research_p = nexus_sub.add_parser("research", help="Deep research")
    research_p.add_argument("query", help="Research query")
    research_p.add_argument("-d", "--depth", type=int, default=3)
    research_p.add_argument("--json", action="store_true")
    research_p.set_defaults(func=research)

    # Status command
    status_p = nexus_sub.add_parser("status", help="Check Nexus Search status")
    status_p.add_argument("--json", action="store_true")
    status_p.set_defaults(func=status)

    # Classify command
    classify_p = nexus_sub.add_parser("classify", help="Classify a query")
    classify_p.add_argument("query", help="Query to classify")
    classify_p.add_argument("--json", action="store_true")
    classify_p.set_defaults(func=classify)


def search(args) -> int:
    """Execute Nexus search."""
    import asyncio

    from ..nexus_cli import cmd_search

    return asyncio.run(cmd_search(args))


def research(args) -> int:
    """Execute Nexus research."""
    import asyncio

    from ..nexus_cli import cmd_research

    return asyncio.run(cmd_research(args))


def status(args) -> int:
    """Execute Nexus status."""
    import asyncio

    from ..nexus_cli import cmd_status

    return asyncio.run(cmd_status(args))


def classify(args) -> int:
    """Execute Nexus classify."""
    import asyncio

    from ..nexus_cli import cmd_classify

    return asyncio.run(cmd_classify(args))
