#!/usr/bin/env python3
"""Kill any process using port 8888"""
import subprocess
import sys
import os

print("Finding process on port 8888...")

# Find process using port 8888
try:
    result = subprocess.run(
        ['netstat', '-ano', '|', 'findstr', ':8888'],
        capture_output=True,
        text=True,
        shell=True
    )
    
    if result.returncode == 0 and result.stdout:
        lines = result.stdout.strip().split('\n')
        pids = set()
        for line in lines:
            if 'LISTENING' in line or 'ESTABLISHED' in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.add(pid)
        
        if pids:
            print(f"Found PIDs: {pids}")
            for pid in pids:
                print(f"Killing PID {pid}...")
                subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
            print("Done!")
        else:
            print("No PIDs found in netstat output")
    else:
        print("No process found on port 8888")
        
except Exception as e:
    print(f"Error: {e}")

# Also try to kill python processes with dashboard
print("\nChecking for Python dashboard processes...")
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'processid,commandline'],
        capture_output=True,
        text=True
    )
    
    if 'dashboard' in result.stdout.lower() or 'mission_control' in result.stdout.lower():
        for line in result.stdout.split('\n'):
            if 'dashboard' in line.lower() or 'mission_control' in line.lower():
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit():
                        print(f"Killing Python dashboard PID {pid}...")
                        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
except Exception as e:
    print(f"Error: {e}")

print("\nPort 8888 should be free now.")
print("You can start the server with: python start_dashboard.py")
