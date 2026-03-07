#!/usr/bin/env python3
"""Launch the dashboard server with timeout"""
import subprocess
import sys
import os
import time

# Kill any existing python processes on port 8888
try:
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('dashboard' in str(cmd) for cmd in proc.info['cmdline']):
                print(f"Killing process {proc.info['pid']}")
                psutil.Process(proc.info['pid']).terminate()
        except:
            pass
except ImportError:
    pass

# Start server
print("Starting server...")
proc = subprocess.Popen(
    [sys.executable, "-c", """
import sys
sys.path.insert(0, 'orchestrator')
from dashboard_mission_control import run_mission_control
run_mission_control()
"""],
    stdout=open("dashboard_output.txt", "w"),
    stderr=subprocess.STDOUT,
    cwd=os.getcwd()
)

# Wait for output
print(f"Server PID: {proc.pid}")
print("Waiting 15 seconds for startup...")
time.sleep(15)

# Check output
print("\n--- Server Output ---")
try:
    with open("dashboard_output.txt", "r") as f:
        print(f.read())
except:
    print("No output yet")

# Check if still running
if proc.poll() is None:
    print(f"\nServer is still running (PID: {proc.pid})")
else:
    print(f"\nServer exited with code: {proc.returncode}")

# Terminate
proc.terminate()
print("\nServer terminated")
