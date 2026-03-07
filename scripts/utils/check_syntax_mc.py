"""Check syntax of dashboard_mission_control.py"""
import ast

filepath = r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\dashboard_mission_control.py"

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Try to parse
    ast.parse(source)
    print("✅ Syntax OK")
    print(f"   File size: {len(source):,} bytes")
    print(f"   Lines: {source.count(chr(10)) + 1}")
except SyntaxError as e:
    print(f"❌ Syntax Error at line {e.lineno}:")
    print(f"   {e.text}")
    print(f"   {e.msg}")
