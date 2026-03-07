#!/usr/bin/env python3
"""Quick test of project creation"""
import urllib.request
import json
import time
import sys

BASE = "http://127.0.0.1:8888"

def post(url, data, timeout=30):
    """POST request with JSON."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def get(url, timeout=5):
    """GET request."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

print("=" * 60)
print(" QUICK PROJECT TEST")
print("=" * 60)

# 1. Check server
print("\n1. Checking server...")
try:
    result = get(f"{BASE}/api/ping")
    print(f"   ✅ Server OK - {result.get('projects', 0)} projects")
except Exception as e:
    print(f"   ❌ Server error: {e}")
    sys.exit(1)

# 2. Start project
print("\n2. Starting test project...")
try:
    result = post(f"{BASE}/api/project/start", {
        "name": "Quick Test",
        "prompt": "Create a Python function that adds two numbers",
        "project_type": "python",
        "criteria": "Function works correctly",
        "budget": 0.5,
        "time_seconds": 120,
        "concurrency": 2
    }, timeout=10)
    
    if result.get('status') == 'error':
        print(f"   ❌ Error: {result.get('message')}")
        sys.exit(1)
    
    project_id = result.get('project_id')
    print(f"   ✅ Started: {project_id}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Monitor project
print("\n3. Monitoring project (30 seconds)...")
for i in range(6):
    time.sleep(5)
    try:
        result = get(f"{BASE}/api/debug")
        for p in result.get('active_projects', []):
            if p['id'] == project_id:
                print(f"   [{i*5}s] Status: {p['status']}, Progress: {p['progress']}%, Task: {p.get('current_task', 'N/A')[:40]}...")
                break
    except Exception as e:
        print(f"   [{i*5}s] Error checking: {e}")

# 4. Final status
print("\n4. Final status...")
try:
    result = get(f"{BASE}/api/debug")
    for p in result.get('active_projects', []):
        if p['id'] == project_id:
            print(f"   Status: {p['status']}")
            print(f"   Progress: {p['progress']}%")
            print(f"   Cost: ${p.get('cost', 0):.4f}")
            print(f"   Tasks: {p.get('tasks_completed', 0)} done")
            
            # Get logs
            try:
                logs = get(f"{BASE}/api/project/{project_id}/logs")
                if logs.get('logs'):
                    print(f"\n   Last 5 logs:")
                    for log in logs['logs'][-5:]:
                        print(f"      [{log.get('time', '?')}] {log.get('message', '')[:60]}")
            except:
                pass
            break
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print(" Test complete!")
print("=" * 60)
