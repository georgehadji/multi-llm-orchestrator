#!/usr/bin/env python3
"""
Standalone dashboard runner - no installation needed.
Usage: python run_dashboard.py [--port PORT]
"""
import argparse
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.dashboard import run_dashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Mission Control Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()
    
    run_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)
