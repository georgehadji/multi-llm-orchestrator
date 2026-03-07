"""
Check if server is running and diagnose issues
"""
import urllib.request
import urllib.error
import json
import sys

def check_endpoint(url, name):
    """Check if an endpoint is accessible."""
    try:
        req = urllib.request.Request(url, method='GET')
        req.add_header('Accept', 'application/json')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read().decode('utf-8')
            try:
                json_data = json.loads(data)
                return True, json_data
            except:
                return True, data[:200]
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, str(e)

BASE_URL = "http://127.0.0.1:8888"

print("=" * 60)
print("🌐 SERVER CONNECTION CHECK")
print("=" * 60)

# Check main page
print(f"\n1. Checking {BASE_URL}/...")
ok, result = check_endpoint(BASE_URL + "/", "main page")
if ok:
    print("   ✅ Server is responding")
    if isinstance(result, dict):
        print(f"   Version: {result.get('version', 'N/A')}")
else:
    print(f"   ❌ {result}")

# Check API state
print(f"\n2. Checking {BASE_URL}/api/state...")
ok, result = check_endpoint(BASE_URL + "/api/state", "state API")
if ok and isinstance(result, dict):
    print("   ✅ API is responding")
    print(f"   Version: {result.get('version')}")
    print(f"   Server status: {result.get('server_status')}")
    print(f"   Projects: {result.get('total_projects', 0)} (running: {result.get('projects_running', 0)})")
    apis = result.get('api_status', [])
    if apis:
        print(f"   APIs: {len(apis)} configured")
        for api in apis:
            status = api.get('status', 'unknown')
            icon = "✅" if status == "connected" else "⚠️" if status == "no_key" else "❌"
            print(f"     {icon} {api.get('provider', 'unknown')}: {status}")
else:
    print(f"   ❌ {result}")

# Check WebSocket
print(f"\n3. Checking WebSocket (ws://127.0.0.1:8888/ws)...")
try:
    import websocket
    ws = websocket.create_connection("ws://127.0.0.1:8888/ws", timeout=5)
    ws.settimeout(2)
    try:
        msg = ws.recv()
        data = json.loads(msg)
        print("   ✅ WebSocket connected")
        if data.get('type') == 'init':
            print("   ✅ Received init message")
    except:
        print("   ⚠️ WebSocket connected but no message received")
    ws.close()
except ImportError:
    print("   ⚠️ websocket-client not installed, skipping WebSocket check")
    print("      Install with: pip install websocket-client")
except Exception as e:
    print(f"   ❌ WebSocket failed: {e}")

print("\n" + "=" * 60)
