#!/usr/bin/env python3
"""
Diagnose Mission Control Import Issues
======================================
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("🔍 DIAGNOSING MISSION CONTROL IMPORT")
print("="*70)

# Test 1: Direct file access
print("\n1️⃣ Checking file exists and is readable...")
filepath = os.path.join(os.path.dirname(__file__), 'orchestrator', 'dashboard_mission_control.py')
if os.path.exists(filepath):
    size = os.path.getsize(filepath)
    print(f"   ✅ File exists: {filepath}")
    print(f"   📁 Size: {size:,} bytes")
else:
    print(f"   ❌ File NOT FOUND: {filepath}")
    sys.exit(1)

# Test 2: Syntax check
print("\n2️⃣ Checking Python syntax...")
import ast
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    ast.parse(source)
    print("   ✅ Syntax is valid")
except SyntaxError as e:
    print(f"   ❌ Syntax error at line {e.lineno}: {e.msg}")
    sys.exit(1)

# Test 3: Module import
print("\n3️⃣ Importing module...")
try:
    import orchestrator.dashboard_mission_control as dmc
    print(f"   ✅ Module imported: {dmc}")
    
    # Check attributes
    attrs = [attr for attr in dir(dmc) if not attr.startswith('_')]
    print(f"   📋 Available: {', '.join(attrs)}")
    
    if hasattr(dmc, 'run_mission_control'):
        print(f"   ✅ run_mission_control: {dmc.run_mission_control}")
    else:
        print("   ❌ run_mission_control NOT FOUND!")
        
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create instance
print("\n4️⃣ Creating server instance...")
try:
    server = dmc.MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server created: {server}")
    print(f"   📊 State: {server.state.to_dict()}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Via orchestrator package
print("\n5️⃣ Testing import via orchestrator package...")
try:
    import orchestrator
    print(f"   ✅ orchestrator imported")
    print(f"   📋 run_mission_control: {orchestrator.run_mission_control}")
except Exception as e:
    print(f"   ⚠️ Import via package failed: {e}")
    print("   💡 This is expected if __init__.py has issues")

print("\n" + "="*70)
print("✅ DIAGNOSTIC COMPLETE")
print("="*70)
print("\n🚀 To start Mission Control:")
print("   python run_mission_control_standalone.py")
