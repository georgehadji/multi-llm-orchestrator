"""
Reset Dashboard - Clear all state and cached data
"""
import shutil
import os

print("🧹 Resetting Dashboard...")

# Remove uploads folder
if os.path.exists('uploads'):
    shutil.rmtree('uploads')
    print("✅ Removed uploads/ folder")

# Remove __pycache__ to force reimport
for root, dirs, files in os.walk('.'):
    for d in dirs:
        if d == '__pycache__':
            path = os.path.join(root, d)
            shutil.rmtree(path)
            print(f"✅ Removed {path}")

print("\n✨ Dashboard reset complete!")
print("\nNow run: Start_Mission_Control.bat")
