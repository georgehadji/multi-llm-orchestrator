"""Check all orchestrator files for syntax errors."""
import ast
from pathlib import Path

root = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator")
errors = []
fixed = []

for py_file in root.glob("*.py"):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        fixed.append(py_file.name)
    except SyntaxError as e:
        errors.append((py_file.name, str(e)))

print("=" * 60)
print("Syntax Check Results")
print("=" * 60)

if fixed:
    print(f"\n✅ OK ({len(fixed)} files):")
    for name in fixed:
        print(f"   ✓ {name}")

if errors:
    print(f"\n❌ Errors ({len(errors)} files):")
    for name, err in errors:
        print(f"   ✗ {name}: {err}")
else:
    print("\n🎉 All files have valid syntax!")

print("=" * 60)
