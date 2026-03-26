#!/usr/bin/env python3
"""
Test Project - Simple diagnostic project
========================================
Creates a minimal project to test if the orchestrator works.
"""
import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8888"

def wait_for_project(project_id, timeout=300):
    """Wait for project to complete."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/api/state")
        data = resp.json()
        
        for p in data.get('active_projects', []):
            if p['id'] == project_id:
                print(f"  Status: {p['status']}, Progress: {p['progress']}%, Task: {p.get('current_task', 'N/A')}")
                
                if p['status'] in ('completed', 'failed', 'stopped'):
                    return p
        
        time.sleep(5)
    
    return None

def main():
    print("=" * 60)
    print(" TEST PROJECT - Simple Python Script")
    print("=" * 60)
    
    # Check server
    try:
        resp = requests.get(f"{BASE_URL}/api/debug", timeout=5)
        debug = resp.json()
        print(f"\n✅ Server OK - {len(debug.get('api_status', []))} APIs configured")
    except Exception as e:
        print(f"\n❌ Server not running: {e}")
        print("   Start with: python start_dashboard.py")
        sys.exit(1)
    
    # Start simple project
    print("\n🚀 Starting test project...")
    print("   Prompt: Create a simple Python calculator")
    
    project_data = {
        "name": "Test Calculator",
        "prompt": "Create a simple Python calculator with add, subtract, multiply, divide functions. Include basic error handling.",
        "project_type": "python",
        "criteria": "Functions work correctly with proper error handling",
        "budget": 1.0,
        "time_seconds": 300,
        "concurrency": 2,
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/project/start", json=project_data, timeout=10)
        result = resp.json()
        
        if result.get('status') == 'error':
            print(f"❌ Failed to start: {result.get('message')}")
            return
        
        project_id = result.get('project_id')
        print(f"✅ Project started: {project_id}")
        
        # Wait for completion
        print("\n⏳ Waiting for completion (timeout: 5 minutes)...")
        final = wait_for_project(project_id)
        
        if final:
            print(f"\n{'=' * 60}")
            print(f" RESULT: {final['status'].upper()}")
            print(f"{'=' * 60}")
            print(f"Progress: {final['progress']}%")
            print(f"Cost: ${final.get('cost', 0):.4f}")
            print(f"Tasks: {final.get('tasks_completed', 0)} / {final.get('tasks_total', 0)}")
            
            # Get logs
            resp = requests.get(f"{BASE_URL}/api/project/{project_id}/logs")
            logs_data = resp.json()
            if logs_data.get('logs'):
                print(f"\n📋 Last 10 log entries:")
                for log in logs_data['logs'][-10:]:
                    print(f"   [{log.get('time', '?')}] {log.get('level', '?').upper()}: {log.get('message', '')}")
        else:
            print("\n⏱️ Timeout - project still running")
            print("   Check dashboard at http://localhost:8888")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
