"""nexusscope command module."""
from __future__ import annotations



def register(subparsers) -> None:
    """Register the 'nexusscope' profiling subcommand."""
    nsp = subparsers.add_parser("nexusscope", help="NexusScope statistical profiler")
    nsp_sub = nsp.add_subparsers(dest="nexusscope_command", metavar="COMMAND")

    sess_p = nsp_sub.add_parser("sessions", help="List recent profiling sessions")
    sess_p.add_argument("--name", "-n", default=None, help="Filter by session name")
    sess_p.add_argument("--last", "-l", type=int, default=20, help="Number to show")
    sess_p.set_defaults(func=_cmd_nexusscope_sessions)

    rep_p = nsp_sub.add_parser("report", help="Render profiling report")
    rep_p.add_argument("--name", "-n", default=None, help="Session name filter")
    rep_p.add_argument(
        "--format",
        "-f",
        choices=["text", "html", "json", "speedscope"],
        default="text",
        help="Output format",
    )
    rep_p.add_argument("--output", "-o", default=None, help="Write to file")
    rep_p.set_defaults(func=_cmd_nexusscope_report)
