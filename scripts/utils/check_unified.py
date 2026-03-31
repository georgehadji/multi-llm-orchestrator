"""Check unified dashboard syntax"""
import ast
import sys

filepath = r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\unified_dashboard.py"

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    ast.parse(source)
    print("✅ unified_dashboard.py - Syntax OK")
    print(f"   Size: {len(source):,} characters ({len(source)/1024:.1f} KB)")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)

# Test imports
print("\n🧪 Testing imports...")
sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

try:
    from orchestrator.unified_dashboard import (
        UnifiedDashboardServer,
        ApiConnectionManager,
        run_unified_dashboard,
    )
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Unified Dashboard ready to use!")
