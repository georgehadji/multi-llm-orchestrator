#!/usr/bin/env python3
"""
Real-Time Dashboard Launcher
============================
Launches the dashboard with live data from orchestrator.

Usage:
    python run_dashboard_realtime.py [--port 8888]
"""
import argparse
import sys
from pathlib import Path

# Ensure orchestrator is importable
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Launch Mission Control Dashboard with Real-Time Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dashboard_realtime.py                    # Default: localhost:8888
    python run_dashboard_realtime.py --port 8080        # Custom port
    python run_dashboard_realtime.py --no-browser       # Don't open browser
    python run_dashboard_realtime.py --host 0.0.0.0     # Allow external access
        """
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to listen on (default: 8888)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    try:
        from orchestrator.dashboard_real import run_dashboard_realtime
        
        run_dashboard_realtime(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\n📦 Install required dependencies:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
