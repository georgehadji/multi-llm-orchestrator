"""Check Python syntax of key files."""
import ast
import sys

files = [
    r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\output_organizer.py",
    r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\__init__.py",
]

all_ok = True
for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print(f"✅ {filepath.split('\\')[-1]} - OK")
    except SyntaxError as e:
        print(f"❌ {filepath.split('\\')[-1]} - {e}")
        all_ok = False

if all_ok:
    print("\n✅ All files OK! Ready to start dashboard.")
else:
    sys.exit(1)
