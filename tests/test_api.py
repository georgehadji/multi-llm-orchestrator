#!/usr/bin/env python3
"""Quick API test"""

import urllib.request
import json
import sys


def test_endpoint(url, timeout=5):
    """Test a single endpoint."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        return {"error": str(e)}


BASE = "http://127.0.0.1:8888"

print("Testing API endpoints...")
print("=" * 50)

# Test ping
print("\n1. GET /api/ping")
result = test_endpoint(f"{BASE}/api/ping")
print(f"   Result: {result}")

# Test state
print("\n2. GET /api/state")
result = test_endpoint(f"{BASE}/api/state")
if "error" in result:
    print(f"   Error: {result['error']}")
else:
    print(f"   Status: OK")
    print(f"   Projects: {result.get('total_projects', 0)}")

# Test debug
print("\n3. GET /api/debug")
result = test_endpoint(f"{BASE}/api/debug")
if "error" in result:
    print(f"   Error: {result['error']}")
else:
    print(f"   Status: OK")
    print(f"   APIs: {len(result.get('api_status', []))}")

# Test POST with small timeout
print("\n4. POST /api/project/start (timeout 3s)")
try:
    data = json.dumps({"name": "Test", "prompt": "Test prompt", "budget": 1.0}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/project/start",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        result = json.loads(resp.read().decode())
        print(f"   Result: {result}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
