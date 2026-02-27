#!/usr/bin/env python3
"""Create dashboard module files."""
import os
from pathlib import Path

# Create dashboard directory
dashboard_dir = Path("orchestrator/dashboard")
dashboard_dir.mkdir(parents=True, exist_ok=True)

# __init__.py
init_content = '''"""
Multi-LLM Orchestrator Dashboard
=================================
Mission Control interface for real-time monitoring and control.

Features:
- Real-time model status monitoring
- Live project execution tracking
- Budget and cost visualization
- Prompt management and templates
- Configuration management
- WebSocket-based live updates

Usage:
    from orchestrator.dashboard import run_dashboard
    run_dashboard(port=8080)
"""

__all__ = ["run_dashboard", "DashboardServer", "ModelMonitor", "ProjectMonitor"]

try:
    from .server import DashboardServer, run_dashboard
    from .monitor import ModelMonitor, ProjectMonitor
except ImportError as e:
    # Graceful degradation if dependencies not installed
    import warnings
    warnings.warn(f"Dashboard dependencies not installed: {e}")
    
    def run_dashboard(*args, **kwargs):
        raise ImportError(
            "Dashboard requires additional dependencies. "
            "Install with: pip install fastapi uvicorn"
        )
'''

(dashboard_dir / "__init__.py").write_text(init_content, encoding='utf-8')
print(f"✓ Created: {dashboard_dir / '__init__.py'}")

# Cleanup
Path(__file__).unlink()
print("✓ Dashboard module created!")
