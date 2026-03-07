#!/usr/bin/env python3
"""
LLM Orchestrator Dashboard Launcher
====================================
Entry point για εκκίνηση του dashboard.

Χρήση:
    python start_dashboard.py
    
Ή:
    python start_dashboard.py --no-browser  # Χωρίς auto-open browser
"""
import sys
import os
import io

# Force UTF-8 for stdout/stderr on Windows to prevent 'charmap' codec errors
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Change to the project directory so orchestrator can be imported as a package
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to path (not orchestrator directly)
if '' not in sys.path and os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LLM Orchestrator Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8888, help='Port to bind (default: 8888)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  LLM Orchestrator Dashboard - Startup")
    print("=" * 60)
    
    try:
        print("[DEBUG] Loading dashboard from:",
        __import__('orchestrator.dashboard_mission_control', fromlist=['']).__file__)

        # Import as a package module
        from orchestrator.dashboard_mission_control import run_mission_control
        
        print(f"\n[NET] Binding to: http://{args.host}:{args.port}")
        print(f"[WEB] Browser: {'Disabled' if args.no_browser else 'Enabled'}")
        print("\n" + "-" * 60)
        
        run_mission_control(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
        
    except KeyboardInterrupt:
        print("\n\n[BYE] Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
