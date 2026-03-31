#!/usr/bin/env python3
"""Delete temporary and backup files."""

from pathlib import Path

files_to_delete = [
    # Backup files
    "CAPABILITIES.md.bak",
    "USAGE_GUIDE.md.bak",
    # Temporary files
    "temp_fix.py",
    "temp_setup_slack.py",
    "create_slack_dir.py",
    "setup_slack_module.py",
    # Old session files
    ".env_old.txt",
    "COMMIT_SUMMARY.txt",
    "CONTINUE_SESSION.txt",
]

root = Path(".")
deleted = 0

print("Deleting temporary files...\n")

for filename in files_to_delete:
    filepath = root / filename
    if filepath.exists():
        try:
            filepath.unlink()
            print(f"✅ Deleted: {filename}")
            deleted += 1
        except Exception as e:
            print(f"❌ Error: {filename} - {e}")
    else:
        print(f"⚠️  Not found: {filename}")

print(f"\n{'='*40}")
print(f"Total: {deleted}/{len(files_to_delete)} files deleted")
