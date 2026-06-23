"""cmd_kanban command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> None:
    """Handle the 'kanban' subcommand: manage the work queue."""
    import asyncio
    from .kanban.board import KanbanBoard

    async def _run():
        board = KanbanBoard()
        if args.command == "enqueue":
            tid = await board.enqueue({"description": args.description or "auto"})
            print(f"Enqueued: {tid}")
        elif args.command == "list":
            tasks = await board.list_tasks(status=args.status)
            if not tasks:
                print("No tasks found.")
            else:
                for t in tasks:
                    print(f"  [{t.status:>8}] {t.task_id}: {t.project_spec[:50]}")
        elif args.command == "stats":
            stats = await board.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
        elif args.command == "start":
            from .kanban.dispatcher import KanbanDispatcher

            dispatcher = KanbanDispatcher(board)
            try:
                await dispatcher.start()
            except KeyboardInterrupt:
                await dispatcher.shutdown()
                print("\nDispatcher stopped.")

    asyncio.run(_run())


def register(subparsers) -> None:
    """Register the 'kanban' subcommand."""
    kp = subparsers.add_parser("kanban", help="Multi-project work queue")
    kp.add_argument("command", choices=["enqueue", "list", "stats", "start"], help="Kanban command")
    kp.add_argument("--description", "-d", default="", help="Project description")
    kp.add_argument("--status", "-s", default=None, help="Filter by status (list only)")
    kp.set_defaults(func=execute)
