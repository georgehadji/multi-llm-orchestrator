"""
CLI Dashboard Command
=====================
Run the Mission Control dashboard from command line.

Usage:
    python -m orchestrator.cli_dashboard [--host HOST] [--port PORT] [--no-browser]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dashboard", description="Launch Multi-LLM Orchestrator Mission Control Dashboard"
    )
    # Loopback by default: the dashboard has no auth and can start paid builds.
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")

    args = parser.parse_args()

    try:
        from orchestrator.dashboard import run_dashboard

        run_dashboard(host=args.host, port=args.port)
    except ImportError as e:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  ❌ Dashboard Dependencies Missing                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  To use the dashboard, install required dependencies:    ║
║                                                          ║
║      pip install fastapi uvicorn                        ║
║                                                          ║
║  Or install with dashboard extras:                       ║
║                                                          ║
║      pip install -e ".[dashboard]"                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Error: {e}
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()
