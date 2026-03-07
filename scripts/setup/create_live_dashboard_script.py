#!/usr/bin/env python3
"""Create live dashboard script."""
from pathlib import Path

script_content = '''#!/usr/bin/env python3
"""
Run Mission Control LIVE Dashboard v4.0
========================================
Gamified, real-time dashboard with WebSocket support.

Usage:
    python scripts/run_dashboard_live.py
    python scripts/run_dashboard_live.py --port 8888
"""
import argparse
from orchestrator.dashboard_live import run_live_dashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mission Control LIVE - Gamified Real-time Dashboard"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    run_live_dashboard(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )
'''

# Write to scripts folder
scripts_dir = Path("scripts")
if scripts_dir.exists():
    script_path = scripts_dir / "run_dashboard_live.py"
    script_path.write_text(script_content)
    print(f"✅ Created: {script_path}")
else:
    print("⚠️ scripts/ folder not found, creating in current directory...")
    script_path = Path("run_dashboard_live.py")
    script_path.write_text(script_content)
    print(f"✅ Created: {script_path}")
