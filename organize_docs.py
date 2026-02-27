#!/usr/bin/env python3
"""
Organize documentation files into docs/ folder
"""
import shutil
from pathlib import Path

# Define file categories
doc_files = {
    "debugging": [
        "DEBUGGING_GUIDE.md",
        "DEBUGGING_OVERVIEW.md",
        "PROJECT_DEBUGGING.md",
        "TROUBLESHOOTING_CHEATSHEET.md",
    ],
    "performance": [
        "PERFORMANCE_OPTIMIZATION.md",
        "PERFORMANCE_SUMMARY.md",
        "MANAGEMENT_SYSTEMS.md",
    ],
    "guides": [
        "PUSH_TO_GITHUB.md",
        "GIT_COMMANDS.txt",
        "COMMIT_SUMMARY.txt",
        "UPDATE_SUMMARY.md",
        "DOCUMENTATION_UPDATE_v5.1.md",
    ],
}

# Create subdirectories
docs_path = Path("docs")
for category in doc_files.keys():
    (docs_path / category).mkdir(exist_ok=True)
    print(f"✓ Created docs/{category}/")

# Move files
moved = []
for category, files in doc_files.items():
    for filename in files:
        src = Path(filename)
        dst = docs_path / category / filename
        
        if src.exists():
            shutil.move(str(src), str(dst))
            moved.append(f"docs/{category}/{filename}")
            print(f"✓ Moved {filename} → docs/{category}/")
        else:
            print(f"⚠ {filename} not found")

# Move helper scripts to scripts/ folder
scripts_dir = Path("scripts")
scripts_dir.mkdir(exist_ok=True)

script_files = [
    "git_commit_push.py",
    "git_auto_commit.py",
    "execute_git.py",
    "COMMIT_COMMANDS.bat",
    "commit_and_push.sh",
    "test_performance_import.py",
]

for script in script_files:
    src = Path(script)
    if src.exists():
        shutil.move(str(src), str(scripts_dir / src.name))
        print(f"✓ Moved {script} → scripts/")

print("\n" + "="*60)
print("📁 REORGANIZATION COMPLETE")
print("="*60)
print("\nNew structure:")
print("""
docs/
├── debugging/
│   ├── DEBUGGING_GUIDE.md
│   ├── DEBUGGING_OVERVIEW.md
│   ├── PROJECT_DEBUGGING.md
│   └── TROUBLESHOOTING_CHEATSHEET.md
├── performance/
│   ├── PERFORMANCE_OPTIMIZATION.md
│   ├── PERFORMANCE_SUMMARY.md
│   └── MANAGEMENT_SYSTEMS.md
├── guides/
│   ├── PUSH_TO_GITHUB.md
│   ├── GIT_COMMANDS.txt
│   ├── COMMIT_SUMMARY.txt
│   ├── UPDATE_SUMMARY.md
│   └── DOCUMENTATION_UPDATE_v5.1.md
├── ADVERSARIAL_STRESS_TEST.md
├── DASHBOARD.md
└── plans/

scripts/
├── git_commit_push.py
├── git_auto_commit.py
├── execute_git.py
├── COMMIT_COMMANDS.bat
├── commit_and_push.sh
└── test_performance_import.py
""")

# Create README for docs folder
docs_readme = """# Documentation

Complete documentation for Multi-LLM Orchestrator.

## Quick Navigation

| Folder | Contents |
|--------|----------|
| `debugging/` | Debugging guides and troubleshooting |
| `performance/` | Performance optimization and management systems |
| `guides/` | Setup guides and helper documentation |

## Main Documentation

- [../README.md](../README.md) - Project overview
- [../CAPABILITIES.md](../CAPABILITIES.md) - Feature reference
- [../USAGE_GUIDE.md](../USAGE_GUIDE.md) - Usage examples

## Debugging

- [debugging/DEBUGGING_GUIDE.md](debugging/DEBUGGING_GUIDE.md) - Comprehensive debugging
- [debugging/TROUBLESHOOTING_CHEATSHEET.md](debugging/TROUBLESHOOTING_CHEATSHEET.md) - Quick fixes

## Performance & Management

- [performance/PERFORMANCE_OPTIMIZATION.md](performance/PERFORMANCE_OPTIMIZATION.md)
- [performance/MANAGEMENT_SYSTEMS.md](performance/MANAGEMENT_SYSTEMS.md)
"""

(docs_path / "README.md").write_text(docs_readme, encoding="utf-8")
print("✓ Created docs/README.md")
