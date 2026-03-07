#!/usr/bin/env python3
"""
Restart Dashboard Server
=========================
Kills any running dashboard server and starts a fresh one.
"""
import subprocess
import sys
import os
import time
import signal

def find_and_kill_dashboard():
    """Find and kill any running dashboard processes."""
    print("🔍 Looking for dashboard processes...")
    
    try:
        import psutil
        killed = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('dashboard' in str(c).lower() or 'mission_control' in str(c).lower() for c in cmdline):
                    if 'python' in proc.info.get('name', '').lower() or any('python' in str(c).lower() for c in cmdline):
                        print(f"   Found: PID {proc.info['pid']} - {' '.join(cmdline[:3])}...")
                        proc.terminate()
                        killed.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if killed:
            print(f"✅ Terminated {len(killed)} process(es)")
            time.sleep(2)
        else:
            print("ℹ️ No dashboard processes found")
            
    except ImportError:
        print("⚠️ psutil not installed, trying taskkill...")
        # Windows fallback
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq *dashboard*'], 
                      capture_output=True)


def main():
    print("=" * 60)
    print(" DASHBOARD SERVER RESTART")
    print("=" * 60)
    
    # Kill existing
    find_and_kill_dashboard()
    
    # Clear cache
    print("\n🧹 Clearing Python cache...")
    import shutil
    for root, dirs, files in os.walk('orchestrator'):
        for d in dirs:
            if d == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(root, d))
                    print(f"   Removed {os.path.join(root, d)}")
                except:
                    pass
    
    # Start fresh
    print("\n🚀 Starting fresh dashboard server...")
    print("=" * 60)
    
    os.environ['PYTHONPATH'] = os.getcwd()
    
    try:
        subprocess.run([sys.executable, "start_dashboard.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
