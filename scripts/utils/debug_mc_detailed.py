#!/usr/bin/env python3
"""Detailed debug for Mission Control import issues"""
import sys
import ast

sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

filepath = r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\dashboard_mission_control.py"

# First check syntax
print("="*60)
print("1️⃣ Checking Python syntax...")
print("="*60)
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    ast.parse(source)
    print("✅ Syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax Error:")
    print(f"   Line {e.lineno}: {e.text}")
    print(f"   {e.msg}")
    sys.exit(1)

# Try importing step by step
print("\n" + "="*60)
print("2️⃣ Testing imports step by step...")
print("="*60)

try:
    print("   Importing logging...")
    from orchestrator.logging import get_logger
    print("   ✅ logging")
    
    print("   Importing models...")
    from orchestrator.models import Model, TaskType, TaskStatus, COST_TABLE, ROUTING_TABLE, get_provider, Task, TaskResult
    print("   ✅ models")
    
    print("   Importing api_clients...")
    from orchestrator.api_clients import UnifiedClient
    print("   ✅ api_clients")
    
    print("   Importing engine...")
    from orchestrator.engine import Orchestrator
    print("   ✅ engine")
    
    print("   Importing budget...")
    from orchestrator.budget import Budget
    print("   ✅ budget")
    
    print("   Importing hooks...")
    from orchestrator.hooks import HookRegistry, EventType
    print("   ✅ hooks")
    
    print("   Importing state...")
    from orchestrator.state import StateManager
    print("   ✅ state")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try importing the actual module
print("\n" + "="*60)
print("3️⃣ Importing dashboard_mission_control module...")
print("="*60)
try:
    import orchestrator.dashboard_mission_control as dmc
    print(f"✅ Module imported successfully")
    print(f"   Module file: {dmc.__file__}")
    
    # Check for MissionControlServer
    if hasattr(dmc, 'MissionControlServer'):
        print(f"✅ MissionControlServer found: {dmc.MissionControlServer}")
    else:
        print("❌ MissionControlServer NOT FOUND in module!")
        print(f"   Available attributes: {dir(dmc)}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try creating an instance
print("\n" + "="*60)
print("4️⃣ Creating MissionControlServer instance...")
print("="*60)
try:
    server = dmc.MissionControlServer(host="127.0.0.1", port=8888)
    print(f"✅ Instance created: {server}")
    print(f"✅ State version: {server.state.version}")
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
