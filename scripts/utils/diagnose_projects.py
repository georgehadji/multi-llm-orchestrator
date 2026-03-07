#!/usr/bin/env python3
"""
Project Diagnostics Tool
========================
Diagnose why projects are not completing.
"""
import asyncio
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

async def diagnose_projects():
    """Fetch and analyze project state from the dashboard."""
    import aiohttp
    
    print("=" * 70)
    print(" PROJECT DIAGNOSTICS")
    print("=" * 70)
    
    base_url = "http://127.0.0.1:8888"
    
    async with aiohttp.ClientSession() as session:
        # Check if server is running
        try:
            async with session.get(f"{base_url}/api/state", timeout=5) as resp:
                if resp.status != 200:
                    print(f"❌ Server returned status {resp.status}")
                    return
                data = await resp.json()
        except Exception as e:
            print(f"❌ Cannot connect to dashboard server: {e}")
            print("   Make sure the server is running on port 8888")
            return
        
        # Get debug info
        try:
            async with session.get(f"{base_url}/api/debug", timeout=5) as resp:
                debug_data = await resp.json()
        except Exception as e:
            print(f"⚠️ Could not fetch debug info: {e}")
            debug_data = {}
        
        print(f"\n📊 Server Status: OK")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Active Projects: {len(data.get('active_projects', []))}")
        print(f"   API Connections: {len([a for a in data.get('api_status', []) if a['status'] in ('connected', 'available')])}")
        
        projects = data.get('active_projects', [])
        
        if not projects:
            print("\n✅ No active projects found")
            return
        
        print(f"\n{'─' * 70}")
        print(" PROJECT DETAILS")
        print("─" * 70)
        
        for i, p in enumerate(projects, 1):
            print(f"\n{i}. {p.get('name', 'Unnamed')} ({p.get('id', 'no-id')[:8]}...)")
            print(f"   Status: {p.get('status', 'unknown')}")
            print(f"   Progress: {p.get('progress', 0)}%")
            print(f"   Current Task: {p.get('current_task', 'N/A')}")
            print(f"   Mode: {p.get('mode', 'unknown')}")
            print(f"   Cost: ${p.get('cost', 0):.4f} / ${p.get('budget', 5):.2f}")
            print(f"   Tasks: {p.get('tasks_completed', 0)} / {p.get('tasks_total', 0)} completed")
            print(f"   Elapsed: {p.get('elapsed', 0):.0f}s")
            
            # Get debug info for this project
            debug_project = None
            for dp in debug_data.get('active_projects', []):
                if dp.get('id') == p.get('id'):
                    debug_project = dp
                    break
            
            if debug_project:
                print(f"   Has Orchestrator: {debug_project.get('has_orchestrator', False)}")
                task_handle = debug_project.get('task_handle', {})
                if task_handle:
                    print(f"   Task Handle Done: {task_handle.get('done', 'N/A')}")
                    print(f"   Task Handle Cancelled: {task_handle.get('cancelled', 'N/A')}")
            
            # Analyze issues
            issues = []
            
            if p.get('status') == 'running':
                elapsed = p.get('elapsed', 0)
                if elapsed > 600:  # 10 minutes
                    issues.append(f"Running for {elapsed/60:.1f} minutes - may be stuck")
                
                if p.get('progress', 0) == 0 and elapsed > 120:
                    issues.append("No progress after 2 minutes - check API connections")
                
                if debug_project and not debug_project.get('has_orchestrator'):
                    issues.append("No orchestrator instance - project may have crashed")
                
                task_handle = debug_project.get('task_handle', {}) if debug_project else {}
                if task_handle and task_handle.get('done') and not task_handle.get('cancelled'):
                    issues.append("Task handle is done but project still shows running - sync issue")
            
            if issues:
                print(f"\n   ⚠️  POTENTIAL ISSUES:")
                for issue in issues:
                    print(f"      • {issue}")
            else:
                print(f"   ✅ No obvious issues detected")
        
        # Check API connections
        print(f"\n{'─' * 70}")
        print(" API CONNECTIONS")
        print("─" * 70)
        
        api_status = data.get('api_status', [])
        connected = [a for a in api_status if a['status'] in ('connected', 'available')]
        no_key = [a for a in api_status if a['status'] == 'no_key']
        error = [a for a in api_status if a['status'] == 'error']
        
        print(f"   Connected/Available: {len(connected)}/{len(api_status)}")
        if connected:
            print(f"   ✅ {', '.join([a['provider'] for a in connected])}")
        if no_key:
            print(f"   ⚠️  No API Key: {', '.join([a['provider'] for a in no_key])}")
        if error:
            print(f"   ❌ Error: {', '.join([a['provider'] for a in error])}")
        
        if len(connected) == 0:
            print("\n   ❌ CRITICAL: No API providers connected!")
            print("      Projects cannot run without at least one API key.")
            print("      Set OPENAI_API_KEY, DEEPSEEK_API_KEY, etc. in your .env file")
        
        # Recommendations
        print(f"\n{'─' * 70}")
        print(" RECOMMENDATIONS")
        print("─" * 70)
        
        if len(connected) == 0:
            print("   1. Add API keys to your .env file")
            print("   2. Restart the dashboard server")
        elif any(p.get('status') == 'running' for p in projects):
            print("   1. Check the browser console for errors")
            print("   2. Look at the project logs in the dashboard")
            print("   3. Check if API providers are responding")
            print("   4. Try stopping and restarting the project")
            print("   5. Check logs/ directory for detailed error logs")
        else:
            print("   1. Start a new project with a simple prompt")
            print("   2. Monitor the console for real-time updates")
        
        print(f"\n{'=' * 70}")

if __name__ == "__main__":
    try:
        asyncio.run(diagnose_projects())
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
