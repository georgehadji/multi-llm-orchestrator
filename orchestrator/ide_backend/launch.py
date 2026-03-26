"""
AI Orchestrator IDE - Launcher Script
======================================
Run this script to start the IDE server with the React frontend.

Usage:
    python -m orchestrator.ide_backend.launch
    python -m orchestrator.ide_backend.launch --port 9000
    python -m orchestrator.ide_backend.launch --no-frontend
"""
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    parser = argparse.ArgumentParser(description="AI Orchestrator IDE Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--frontend", action="store_true", default=True, help="Serve frontend")
    parser.add_argument("--no-frontend", action="store_false", dest="frontend", help="Don't serve frontend")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Find frontend build directory
    frontend_path = None
    if args.frontend:
        possible_paths = [
            Path(__file__).parent.parent.parent / "ide_frontend" / "dist",
            Path(__file__).parent / ".." / ".." / "ide_frontend" / "dist",
            Path.cwd() / "ide_frontend" / "dist",
        ]
        for path in possible_paths:
            if path.exists() and (path / "index.html").exists():
                frontend_path = path
                break

        if not frontend_path:
            print("⚠️  Frontend build not found. Run 'npm run build' in ide_frontend/")
            print("   Starting with API-only mode...")

    # Print startup info
    print("\n" + "=" * 60)
    print("  AI Orchestrator IDE Server")
    print("=" * 60)
    print(f"  🌐 Server: http://{args.host}:{args.port}")
    if frontend_path:
        print(f"  📁 Frontend: {frontend_path}")
    print(f"  🔄 Reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 60 + "\n")

    # Import server module directly (avoid orchestrator package init)
    import importlib.util
    server_path = Path(__file__).parent / "server.py"
    spec = importlib.util.spec_from_file_location("ide_server", server_path)
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    try:
        server_module.run_ide_server(
            host=args.host,
            port=args.port,
            frontend_path=frontend_path,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
