#!/usr/bin/env python3
"""
Mission Control Launcher
========================
🚀 Mission Control v6.0 - Full Project Management Dashboard

Χρήση:
    python start_mission_control.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🚀 MISSION CONTROL v6.0                                         ║
║                                                                  ║
║  Το απόλυτο dashboard για διαχείριση projects!                  ║
║                                                                  ║
║  ✨ Features:                                                    ║
║     • 📝 Εισαγωγή prompt & criteria                              ║
║     • 🎯 Επιλογή τύπου project                                   ║
║     • 📊 Real-time progress tracking                             ║
║     • 🏗️ Architecture visualization                             ║
║     • 🤖 Model usage monitoring                                  ║
║     • ⚡ Live task execution                                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        from orchestrator import run_mission_control
        run_mission_control()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Εγκατάσταση dependencies: pip install fastapi uvicorn websockets")
        sys.exit(1)
