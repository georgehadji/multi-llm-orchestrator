#!/usr/bin/env python3
"""
Mission Control v5.0 - Performance Optimized Launcher
=====================================================
Features:
- Redis caching layer
- Gzip compression
- ETag support
- Debounced updates
- Performance monitoring

Usage:
    python run_optimized_dashboard.py [--port 8888] [--redis]
"""
import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

# Ensure orchestrator is importable
sys.path.insert(0, str(Path(__file__).parent))


def print_banner(host: str, port: int):
    """Print startup banner."""
    url = f"http://{host}:{port}"
    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           ◈ MISSION CONTROL v5.0 ◈                               ║
║           PERFORMANCE OPTIMIZED                                  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Dashboard URL: {url:<43} ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  PERFORMANCE FEATURES:                                           ║
║                                                                  ║
║    ✓ Gzip compression (Level 6)                                  ║
║    ✓ Redis + In-memory dual-layer caching                        ║
║    ✓ ETag support for 304 Not Modified                           ║
║    ✓ External CSS (24h browser cache)                            ║
║    ✓ Debounced real-time updates (2s interval)                   ║
║    ✓ Connection pooling ready                                    ║
║    ✓ Performance monitoring & KPIs                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  TARGET PERFORMANCE:                                             ║
║                                                                  ║
║    ⚡ First Contentful Paint: <100ms                             ║
║    ⚡ Time to First Byte: <50ms                                  ║
║    ⚡ Cache Hit Rate: >85%                                       ║
║    ⚡ P95 Response Time: <300ms                                  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  MONITORING ENDPOINTS:                                           ║
║                                                                  ║
║    📊 {url + '/api/metrics':<48} ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

📖 Keyboard Shortcuts:
   1-5    Switch views
   Q      Toggle Quick-Action FAB
   ?      Show shortcuts
   Esc    Dismiss modals
   Ctrl+K Command palette
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Mission Control v5.0 - Performance Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_optimized_dashboard.py                    # Default: localhost:8080
    python run_optimized_dashboard.py --port 8888        # Custom port
    python run_optimized_dashboard.py --no-browser       # Don't open browser
    python run_optimized_dashboard.py --host 0.0.0.0     # Allow external access
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
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)"
    )
    
    args = parser.parse_args()
    
    print_banner(args.host, args.port)
    
    try:
        # Import optimized dashboard
        from orchestrator.dashboard_optimized import OptimizedDashboardServer, PerformanceConfig
        
        # Configure Redis if specified
        if args.redis_host:
            from orchestrator.dashboard_optimized import cache
            cache._host = args.redis_host
            cache._port = args.redis_port
            print(f"🔧 Redis configured: {args.redis_host}:{args.redis_port}")
        
        # Create server
        server = OptimizedDashboardServer(host=args.host, port=args.port)
        
        # Open browser
        if not args.no_browser:
            url = f"http://{args.host}:{args.port}"
            webbrowser.open(url)
        
        # Run server
        print("🚀 Starting server...")
        asyncio.run(server.run())
        
    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies - {e}")
        print("\n📦 Install with:")
        print("   pip install fastapi uvicorn websockets httpx")
        if sys.platform == "win32":
            print("\n   Or on Windows:")
            print("   pip install fastapi uvicorn[standard] websockets httpx")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
