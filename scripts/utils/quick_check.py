"""Quick syntax check for dashboard files."""
import ast

files = [
    "orchestrator/dashboard_live.py",
    "orchestrator/dashboard_antd.py",
    "orchestrator/dashboard_enhanced.py",
    "orchestrator/output_organizer.py",
]

all_ok = True
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            ast.parse(file.read())
        print(f"✅ {f}")
    except SyntaxError as e:
        print(f"❌ {f}: {e}")
        all_ok = False

print()
if all_ok:
    print("🎉 All files OK! Run: python start_dashboard.py live")
else:
    print("⚠️  Fix errors above before running dashboard")
