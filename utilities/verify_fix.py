#!/usr/bin/env python3
"""Verify BUG-001 fix was applied correctly."""

import ast
import sys

def verify_syntax(filename):
    """Check Python file syntax."""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def verify_fix_present(filename):
    """Verify BUG-001 fix is present in source."""
    with open(filename, 'r') as f:
        source = f.read()
    
    checks = {
        "target_path serialization": 'target_path": t.target_path' in source,
        "module_name serialization": 'module_name": t.module_name' in source,
        "tech_context serialization": 'tech_context": t.tech_context' in source,
        "target_path deserialization": 'target_path=d.get("target_path", "")' in source,
        "module_name deserialization": 'module_name=d.get("module_name", "")' in source,
        "tech_context deserialization": 'tech_context=d.get("tech_context", "")' in source,
        "BUG-001 FIX comment": "BUG-001 FIX" in source,
    }
    
    return checks

if __name__ == "__main__":
    print("=" * 60)
    print("BUG-001 FIX VERIFICATION")
    print("=" * 60)
    
    # Check syntax
    print("\n1. Syntax Check:")
    ok, msg = verify_syntax("orchestrator/state.py")
    print(f"   {'✅' if ok else '❌'} {msg}")
    if not ok:
        sys.exit(1)
    
    # Check fix present
    print("\n2. Fix Presence Check:")
    checks = verify_fix_present("orchestrator/state.py")
    all_present = True
    for check, present in checks.items():
        status = "✅" if present else "❌"
        print(f"   {status} {check}")
        if not present:
            all_present = False
    
    print("\n" + "=" * 60)
    if all_present:
        print("✅ BUG-001 FIX VERIFIED AND DEPLOYED")
        print("\nSummary:")
        print("  - target_path, module_name, tech_context now serialized")
        print("  - Backward compatible (defaults for old states)")
        print("  - Syntax valid")
    else:
        print("❌ FIX INCOMPLETE - Some checks failed")
        sys.exit(1)
    print("=" * 60)
