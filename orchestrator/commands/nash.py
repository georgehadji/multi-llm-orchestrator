"""nash command module."""
from __future__ import annotations



def register(subparsers):
    """Add Nash stability subcommands."""
    nash_parser = subparsers.add_parser("nash", help="Nash stability management")
    nash_subparsers = nash_parser.add_subparsers(dest="nash_command", metavar="COMMAND")

    # nash status
    status_parser = nash_subparsers.add_parser("status", help="Show Nash stability status")
    status_parser.add_argument("--format", choices=["table", "json"], default="table")
    status_parser.add_argument("--watch", action="store_true", help="Watch mode")
    status_parser.set_defaults(func=_cmd_nash_status)

    # nash backup
    backup_parser = nash_subparsers.add_parser(
        "backup", help="Backup/restore accumulated knowledge"
    )
    backup_parser.add_argument("--list", action="store_true", help="List backups")
    backup_parser.add_argument("--restore", type=str, help="Restore from backup file")
    backup_parser.add_argument("--value", action="store_true", help="Show estimated value")
    backup_parser.set_defaults(func=_cmd_nash_backup)

    # nash tuning
    tuning_parser = nash_subparsers.add_parser("tuning", help="Auto-tuning control")
    tuning_parser.add_argument("--status", action="store_true", help="Show tuning status")
    tuning_parser.add_argument("--tune", type=str, help="Parameter to tune")
    tuning_parser.add_argument("--value", type=float, help="New value for parameter")
    tuning_parser.set_defaults(func=_cmd_nash_tuning)

    # nash compare
    compare_parser = nash_subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model_a", help="First model to compare")
    compare_parser.add_argument("model_b", help="Second model to compare")
    compare_parser.add_argument("--task-type", default="CODE_GEN", help="Task type")
    compare_parser.set_defaults(func=_cmd_nash_compare)
