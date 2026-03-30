"""
Nexus Search CLI Commands
==========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Command-line interface for Nexus Search operations.

Usage:
    python -m orchestrator.nexus_cli search "Python async"
    python -m orchestrator.nexus_cli research "Microservices patterns"
    python -m orchestrator.nexus_cli status
    python -m orchestrator.nexus_cli enable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .log_config import get_logger

logger = get_logger(__name__)


def print_search_results(results) -> None:
    """Print search results in a formatted way."""
    print(f"\n🔍 Search Results for: {results.query}")
    print(f"   Found {len(results)} results in {results.search_time:.0f}ms\n")

    for i, result in enumerate(results.top, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        if result.content:
            content = result.content[:150].replace('\n', ' ')
            print(f"   {content}...")
        print()

    if results.suggestions:
        print("Suggestions:")
        for suggestion in results.suggestions[:3]:
            print(f"  - {suggestion}")


def print_research_report(report) -> None:
    """Print research report in a formatted way."""
    print(f"\n📚 Research Report: {report.query}")
    print(f"   Iterations: {report.search_iterations}")
    print(f"   Sources: {report.source_count}")
    print(f"   Time: {report.total_time:.1f}s\n")

    print("Summary:")
    print("-" * 60)
    print(report.summary[:500] + "..." if len(report.summary) > 500 else report.summary)
    print("-" * 60)

    if report.findings:
        print(f"\nKey Findings ({len(report.findings)}):")
        for i, finding in enumerate(report.findings[:5], 1):
            print(f"{i}. [{finding.category}] {finding.content[:100]}...")


async def cmd_search(args) -> int:
    """Execute search command."""
    try:
        from .nexus_search import OptimizationMode, SearchSource, search

        # Parse sources
        sources = []
        if args.sources:
            for src in args.sources.split(','):
                src = src.strip().lower()
                if src in ['web', 'academic', 'tech', 'news', 'code']:
                    sources.append(SearchSource(src))

        # Parse optimization
        optimization = OptimizationMode(args.optimization.lower()) if args.optimization else OptimizationMode.BALANCED

        # Execute search
        results = await search(
            query=args.query,
            sources=sources if sources else None,
            optimization=optimization,
            num_results=args.num_results,
        )

        if args.json:
            print(json.dumps(results.to_dict(), indent=2))
        else:
            print_search_results(results)

        return 0

    except ImportError:
        print("❌ Nexus Search is not installed or configured")
        return 1
    except Exception as e:
        print(f"❌ Search failed: {e}")
        logger.debug("Search error", exc_info=True)
        return 1


async def cmd_research(args) -> int:
    """Execute research command."""
    try:
        from .nexus_search import research

        # Execute research
        report = await research(
            query=args.query,
            depth=args.depth,
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_research_report(report)

        return 0

    except ImportError:
        print("❌ Nexus Search is not installed or configured")
        return 1
    except Exception as e:
        print(f"❌ Research failed: {e}")
        logger.debug("Research error", exc_info=True)
        return 1


async def cmd_status(args) -> int:
    """Check Nexus Search status."""
    try:
        from .nexus_search import get_nexus_orchestrator

        orchestrator = get_nexus_orchestrator()
        status = await orchestrator.get_status()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n🔮 Nexus Search Status")
            print("=" * 40)
            print(f"Enabled:  {'✅ Yes' if status['enabled'] else '❌ No'}")
            print(f"Healthy:  {'✅ Yes' if status['healthy'] else '❌ No'}")
            print(f"API URL:  {status['api_url']}")

            if 'capabilities' in status:
                caps = status['capabilities']
                print("\nCapabilities:")
                print(f"  Sources: {', '.join(caps.get('sources', []))}")
                print(f"  Max Results: {caps.get('max_results', 'N/A')}")
                print(f"  Rate Limit: {caps.get('rate_limit', 'N/A')} queries/min")
                print(f"  Cache: {'Enabled' if caps.get('cache_enabled') else 'Disabled'}")

            print()

        return 0 if status['healthy'] else 1

    except ImportError:
        print("❌ Nexus Search is not installed")
        return 1
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1


async def cmd_classify(args) -> int:
    """Classify a query."""
    try:
        from .nexus_search import QueryType, classify

        query_type = await classify(args.query)

        if args.json:
            print(json.dumps({"query": args.query, "type": query_type.value}))
        else:
            type_emojis = {
                QueryType.FACTUAL: "📖",
                QueryType.RESEARCH: "🔬",
                QueryType.TECHNICAL: "💻",
                QueryType.ACADEMIC: "🎓",
                QueryType.CREATIVE: "🎨",
            }
            emoji = type_emojis.get(query_type, "❓")
            print(f"\n{emoji} Query Classification")
            print("=" * 40)
            print(f"Query: {args.query}")
            print(f"Type:  {query_type.value.upper()}")
            print()

        return 0

    except ImportError:
        print("❌ Nexus Search is not installed")
        return 1
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for Nexus CLI."""
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus Search - Web search for AI Orchestrator",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Perform a web search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-s", "--sources", help="Comma-separated sources (web,academic,tech,news,code)")
    search_parser.add_argument("-o", "--optimization", default="balanced",
                               choices=["speed", "balanced", "quality"],
                               help="Optimization mode")
    search_parser.add_argument("-n", "--num-results", type=int, default=10,
                               help="Maximum number of results")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.set_defaults(func=cmd_search)

    # Research command
    research_parser = subparsers.add_parser("research", help="Conduct deep research")
    research_parser.add_argument("query", help="Research query")
    research_parser.add_argument("-d", "--depth", type=int, default=3,
                                 help="Research depth (1-5)")
    research_parser.add_argument("--json", action="store_true", help="Output as JSON")
    research_parser.set_defaults(func=cmd_research)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check Nexus Search status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.set_defaults(func=cmd_status)

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a query")
    classify_parser.add_argument("query", help="Query to classify")
    classify_parser.add_argument("--json", action="store_true", help="Output as JSON")
    classify_parser.set_defaults(func=cmd_classify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for Nexus CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
