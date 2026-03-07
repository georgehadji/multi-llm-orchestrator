#!/usr/bin/env python3
"""Quick self-test that writes to file"""
import sys
import os
import time

# Redirect output to file
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "self_test_log.txt")

with open(log_file, "w") as log:
    def log_print(msg):
        print(msg, file=log, flush=True)
        # Also print to stderr for visibility
        print(msg, file=sys.stderr)
    
    log_print("="*50)
    log_print("SELF TEST STARTING")
    log_print("="*50)
    
    try:
        log_print("Step 1: Importing...")
        from dashboard_mission_control import MissionControlServer
        log_print("✓ Import successful")
        
        log_print("\nStep 2: Creating server...")
        server = MissionControlServer(host="127.0.0.1", port=8888)
        log_print("✓ Server created")
        
        log_print("\nStep 3: Checking API connections...")
        import asyncio
        
        async def test_connections():
            await server._check_api_connections()
        
        asyncio.run(test_connections())
        log_print("✓ API connections checked")
        
        log_print("\nStep 4: Checking state...")
        log_print(f"  Active projects: {len(server.state.active_projects)}")
        
        log_print("\n" + "="*50)
        log_print("ALL TESTS PASSED!")
        log_print("="*50)
        
    except Exception as e:
        log_print(f"\n✗ ERROR: {e}")
        import traceback
        log_print(traceback.format_exc())
        sys.exit(1)

print(f"\nLog written to: {log_file}")
