"""Delete old dashboard files that are now replaced by unified_dashboard.py"""

from pathlib import Path
import shutil

root = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator")

# Files to delete (replaced by unified_dashboard.py)
files_to_delete = [
    "dashboard.py",  # Old v1.0
    "dashboard_real.py",  # Replaced by unified
    "dashboard_optimized.py",  # Replaced by unified
    # Keep: unified_dashboard.py (NEW)
    # Keep: dashboard_live.py (for compatibility)
    # Keep: dashboard_enhanced.py (for compatibility)
    # Keep: dashboard_antd.py (for compatibility)
]

print("=" * 60)
print("🧹 Cleaning up old dashboard files")
print("=" * 60)

for filename in files_to_delete:
    filepath = root / filename
    if filepath.exists():
        try:
            filepath.unlink()
            print(f"✅ Deleted: {filename}")
        except Exception as e:
            print(f"❌ Error deleting {filename}: {e}")
    else:
        print(f"⏭️  Already gone: {filename}")

print("\n" + "=" * 60)
print("Remaining dashboards:")
print("=" * 60)

remaining = list(root.glob("dashboard*.py"))
for f in sorted(remaining):
    size = f.stat().st_size / 1024
    print(f"  • {f.name:30} ({size:6.1f} KB)")

print("\n✅ Cleanup complete!")
print(
    '\n🚀 Use: python -c "from orchestrator import run_unified_dashboard; run_unified_dashboard()"'
)
