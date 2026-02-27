#!/usr/bin/env python3
"""
Complete documentation reorganization
Moves files to docs/ and updates all links
"""
import shutil
from pathlib import Path

print("=" * 70)
print("📁 DOCUMENTATION REORGANIZATION")
print("=" * 70)

# Create directory structure
docs = Path("docs")
(docs / "debugging").mkdir(exist_ok=True)
(docs / "performance").mkdir(exist_ok=True)
(docs / "guides").mkdir(exist_ok=True)

scripts = Path("scripts")
scripts.mkdir(exist_ok=True)

print("\n✓ Created directory structure")

# Define moves
def move_file(src, dst):
    """Move file if it exists."""
    src_path = Path(src)
    if src_path.exists():
        shutil.move(str(src_path), str(dst))
        print(f"✓ Moved {src} → {dst}")
        return True
    return False

# Move documentation files
doc_moves = [
    # Debugging
    ("DEBUGGING_GUIDE.md", "docs/debugging/DEBUGGING_GUIDE.md"),
    ("DEBUGGING_OVERVIEW.md", "docs/debugging/DEBUGGING_OVERVIEW.md"),
    ("PROJECT_DEBUGGING.md", "docs/debugging/PROJECT_DEBUGGING.md"),
    ("TROUBLESHOOTING_CHEATSHEET.md", "docs/debugging/TROUBLESHOOTING_CHEATSHEET.md"),
    
    # Performance
    ("PERFORMANCE_OPTIMIZATION.md", "docs/performance/PERFORMANCE_OPTIMIZATION.md"),
    ("PERFORMANCE_SUMMARY.md", "docs/performance/PERFORMANCE_SUMMARY.md"),
    ("MANAGEMENT_SYSTEMS.md", "docs/performance/MANAGEMENT_SYSTEMS.md"),
    
    # Guides
    ("PUSH_TO_GITHUB.md", "docs/guides/PUSH_TO_GITHUB.md"),
    ("GIT_COMMANDS.txt", "docs/guides/GIT_COMMANDS.txt"),
    ("COMMIT_SUMMARY.txt", "docs/guides/COMMIT_SUMMARY.txt"),
    ("UPDATE_SUMMARY.md", "docs/guides/UPDATE_SUMMARY.md"),
    ("DOCUMENTATION_UPDATE_v5.1.md", "docs/guides/DOCUMENTATION_UPDATE_v5.1.md"),
]

for src, dst in doc_moves:
    move_file(src, dst)

# Move scripts
script_moves = [
    ("git_commit_push.py", "scripts/git_commit_push.py"),
    ("git_auto_commit.py", "scripts/git_auto_commit.py"),
    ("execute_git.py", "scripts/execute_git.py"),
    ("COMMIT_COMMANDS.bat", "scripts/COMMIT_COMMANDS.bat"),
    ("commit_and_push.sh", "scripts/commit_and_push.sh"),
    ("test_performance_import.py", "scripts/test_performance_import.py"),
    ("organize_docs.py", "scripts/organize_docs.py"),
    ("update_doc_links.py", "scripts/update_doc_links.py"),
    ("finalize_organization.py", "scripts/finalize_organization.py"),
]

for src, dst in script_moves:
    move_file(src, dst)

# Create docs/README.md
docs_readme = """# Documentation

Complete documentation for Multi-LLM Orchestrator v5.1

## 📚 Folder Structure

```
docs/
├── debugging/          Debugging & troubleshooting guides
├── performance/        Performance optimization & management systems
├── guides/            Setup guides and helper documentation
├── ADVERSARIAL_STRESS_TEST.md
├── DASHBOARD.md
└── plans/
```

## 🚀 Quick Start

- [Main README](../README.md) - Project overview
- [Usage Guide](../USAGE_GUIDE.md) - Detailed usage examples
- [Capabilities](../CAPABILITIES.md) - Feature reference

## 🐛 Debugging

- [Debugging Guide](debugging/DEBUGGING_GUIDE.md) - Comprehensive debugging manual
- [Troubleshooting Cheatsheet](debugging/TROUBLESHOOTING_CHEATSHEET.md) - Quick fixes
- [Project Debugging](debugging/PROJECT_DEBUGGING.md) - Debug generated projects

## ⚡ Performance

- [Performance Optimization](performance/PERFORMANCE_OPTIMIZATION.md)
- [Management Systems](performance/MANAGEMENT_SYSTEMS.md)

## 🛠️ Development

- [Push to GitHub Guide](guides/PUSH_TO_GITHUB.md)
- [Git Commands](guides/GIT_COMMANDS.txt)
"""

(docs / "README.md").write_text(docs_readme, encoding="utf-8")
print("✓ Created docs/README.md")

# Create scripts/README.md
scripts_readme = """# Scripts

Helper scripts for Multi-LLM Orchestrator

## Available Scripts

### Git & Release
- `COMMIT_COMMANDS.bat` - Windows batch for git commit
- `commit_and_push.sh` - Unix shell script for git commit
- `git_commit_push.py` - Python git automation
- `git_auto_commit.py` - Automated commit script

### Testing
- `test_performance_import.py` - Verify module imports

### Maintenance
- `organize_docs.py` - Documentation organization
- `update_doc_links.py` - Update documentation links
- `finalize_organization.py` - This script

## Usage

```bash
# Windows
scripts\\COMMIT_COMMANDS.bat

# Unix/Mac
./scripts/commit_and_push.sh

# Python
python scripts/git_auto_commit.py
```
"""

(scripts / "README.md").write_text(scripts_readme, encoding="utf-8")
print("✓ Created scripts/README.md")

print("\n" + "=" * 70)
print("✅ REORGANIZATION COMPLETE")
print("=" * 70)
print("""
📁 New Structure:

Root/
├── README.md                    (updated)
├── USAGE_GUIDE.md              (updated)
├── CAPABILITIES.md             (updated)
├── docs/
│   ├── README.md               (new)
│   ├── debugging/              (4 files)
│   ├── performance/            (3 files)
│   ├── guides/                 (5 files)
│   ├── ADVERSARIAL_STRESS_TEST.md
│   ├── DASHBOARD.md
│   └── plans/
├── scripts/                    (9 files)
└── orchestrator/               (code)
""")
