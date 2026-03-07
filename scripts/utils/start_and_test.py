#!/usr/bin/env python3
"""
Start server and test
=====================
Starts the dashboard server and runs basic tests.
"""
import subprocess
import sys
import os
import time
import signal
import urllib.request
import json

def check_server(url="http://127.0.0.1:8888/api/ping", timeout=1):
    """Check if server is running."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except:
        return False

def start_server():
    """Start the server."""
    print("🚀 Starting dashboard server...")
    
    # Kill any existing
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('dashboard' in str(c).lower() for c in cmdline):
                    proc.terminate()
            except:
                pass
    except:
        pass
    
    time.sleep(1)
    
    # Start new server
    proc = subprocess.Popen(
        [sys.executable, "start_dashboard.py"],
        stdout=open("dashboard.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=os.getcwd()
    )
    
    print(f"   PID: {proc.pid}")
    print("   Waiting for server to start...")
    
    # Wait for server
    for i in range(30):
        time.sleep(1)
        if check_server():
            print("   ✅ Server is running!")
            return proc
        print(f"   ... {i+1}s")
    
    print("   ❌ Server failed to start")
    print("\n   Logs:")
    try:
        with open("dashboard.log", "r", encoding="utf-8") as f:
            print(f.read()[-2000:])  # Last 2000 chars
    except:
        print("   (no logs)")
    
    return None

def test_endpoints():
    """Test API endpoints."""
    BASE = "http://127.0.0.1:8888"
    
    print("\n📡 Testing endpoints...")
    
    # Ping
    try:
        resp = urllib.request.urlopen(f"{BASE}/api/ping", timeout=5)
        data = json.loads(resp.read())
        print(f"   ✅ /api/ping: {data}")
    except Exception as e:
        print(f"   ❌ /api/ping: {e}")
        return False
    
    # Debug
    try:
        resp = urllib.request.urlopen(f"{BASE}/api/debug", timeout=5)
        data = json.loads(resp.read())
        print(f"   ✅ /api/debug: {len(data.get('api_status', []))} APIs")
    except Exception as e:
        print(f"   ❌ /api/debug: {e}")
    
    return True

def main():
    print("=" * 60)
    print(" DASHBOARD SERVER START & TEST")
    print("=" * 60)
    
    # Check if already running
    if check_server():
        print("\nℹ️ Server already running")
    else:
        proc = start_server()
        if not proc:
            print("\n❌ Failed to start server")
            sys.exit(1)
    
    # Test
    if test_endpoints():
        print("\n✅ Server is ready!")
        print("   URL: http://127.0.0.1:8888")
        print("\n   Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping...")
    else:
        print("\n❌ Tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
