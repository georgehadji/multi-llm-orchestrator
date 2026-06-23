"""Cache stats command handler — extracted from cli.py."""
from __future__ import annotations

import argparse
import asyncio

def execute(args) -> None:
    """Show cache statistics."""
    from orchestrator.cache_optimizer import get_cache_optimizer

    optimizer = get_cache_optimizer()

    if args.clear:
        level = args.level if args.level else None
        asyncio.run(optimizer.clear(level))
        print(f"✓ Cache cleared (level: {level or 'all'})")
        return

    if args.cleanup:
        stats = asyncio.run(optimizer.cleanup())
        print(f"✓ Cleanup complete: {stats['l2_deleted']} expired entries removed")
        return

    # Print statistics
    stats = optimizer.get_stats()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    CACHE STATISTICS                              ║
╠══════════════════════════════════════════════════════════════════╣""")
    print(f"║ Total Requests:     {stats['total_requests']:>10,}                               ║")
    print(
        f"║ Total Hits:         {stats['total_hits']:>10,}  ({stats['overall_hit_rate']:.1%})                        ║"
    )
    print(f"║ Total Misses:       {stats['total_misses']:>10,}                               ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║ By Level:                                                        ║")
    print(f"║   L1 (Memory):      {stats['l1_hits']:>10,} hits                              ║")
    print(f"║   L2 (Disk):        {stats['l2_hits']:>10,} hits                              ║")
    print(f"║   L3 (Semantic):    {stats['l3_hits']:>10,} hits                              ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║ Savings:                                                         ║")
    print(f"║   Tokens Saved:     {stats['tokens_saved']:>10,}                               ║")
    print(f"║   Cost Saved:       ${stats['cost_saved']:>9.2f}                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # L1 detailed stats
    if stats.get("l1_stats"):
        l1 = stats["l1_stats"]
        print("\nL1 Memory Cache:")
        print(
            f"  Entries: {l1['entries']}/{l1['max_size']} ({100*l1['entries']/l1['max_size']:.1f}%)"
        )
        print(f"  Hit Rate: {l1['hit_rate']:.1%}")

def execute_stats(args: argparse.Namespace) -> int:
    """Handle cache-stats subcommand."""

    async def _run():
        from ..cache_optimizer import get_cache_optimizer

        optimizer = get_cache_optimizer()

        if args.clear:
            level = args.level
            if level:
                print(f"[CLEAR]  Clearing {level.upper()} cache...")
                if level == "l1":
                    optimizer.l1_cache.clear()
                elif level == "l2":
                    await optimizer.l2_cache.clear()
                elif level == "l3":
                    optimizer.l3_cache.clear()
                print(f"✅ {level.upper()} cache cleared")
            else:
                print("[CLEAR]  Clearing all cache levels...")
                optimizer.l1_cache.clear()
                await optimizer.l2_cache.clear()
                optimizer.l3_cache.clear()
                print("✅ All caches cleared")
            return 0

        if args.cleanup:
            print("[CLEAN] Cleaning up expired entries...")
            optimizer.l1_cache.cleanup()
            await optimizer.l2_cache.cleanup()
            optimizer.l3_cache.cleanup()
            print("✅ Cleanup complete")
            return 0

        # Show statistics
        optimizer.print_stats()
        return 0

    return asyncio.run(_run())



def register(subparsers) -> None:
    """Register the 'cache-stats' subcommand."""
    parser = subparsers.add_parser(
        "cache-stats",
        help="Show cache statistics and manage cache",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cache levels",
    )
    parser.add_argument(
        "--level",
        choices=["l1", "l2", "l3"],
        help="Specific cache level to clear (default: all)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove expired cache entries",
    )
    parser.set_defaults(func=execute)
