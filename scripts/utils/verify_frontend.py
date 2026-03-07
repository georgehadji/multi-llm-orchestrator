"""Verify frontend_rules.py syntax."""
import ast
import sys

# Read the file
with open(r'E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\frontend_rules.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Try to parse it
try:
    ast.parse(source)
    print("✅ frontend_rules.py - Syntax OK")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)

# Read __init__.py
with open(r'E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\__init__.py', 'r', encoding='utf-8') as f:
    source = f.read()

try:
    ast.parse(source)
    print("✅ __init__.py - Syntax OK")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    sys.exit(1)

print("\n✅ All syntax checks passed!")
