#!/usr/bin/env python3
"""
Complete Documentation Reorganization Script
Moves files to proper locations and updates all internal links
"""
import shutil
import re
from pathlib import Path

print("=" * 70)
print("📁 DOCUMENTATION REORGANIZATION - v5.1")
print("=" * 70)

# Step 1: Create directory structure
docs = Path("docs")
(docs / "debugging").mkdir(exist_ok=True)
(docs / "performance").mkdir(exist_ok=True)
(docs / "guides").mkdir(exist_ok=True)

scripts = Path("scripts")
scripts.mkdir(exist_ok=True)

print("\n✓ Step 1: Created directory structure")

# Step 2: Define all file moves
moves = [
    # Documentation files
    ("DEBUGGING_GUIDE.md", "docs/debugging/DEBUGGING_GUIDE.md"),
    ("DEBUGGING_OVERVIEW.md", "docs/debugging/DEBUGGING_OVERVIEW.md"),
    ("PROJECT_DEBUGGING.md", "docs/debugging/PROJECT_DEBUGGING.md"),
    ("TROUBLESHOOTING_CHEATSHEET.md", "docs/debugging/TROUBLESHOOTING_CHEATSHEET.md"),
    
    ("PERFORMANCE_OPTIMIZATION.md", "docs/performance/PERFORMANCE_OPTIMIZATION.md"),
    ("PERFORMANCE_SUMMARY.md", "docs/performance/PERFORMANCE_SUMMARY.md"),
    ("MANAGEMENT_SYSTEMS.md", "docs/performance/MANAGEMENT_SYSTEMS.md"),
    
    ("PUSH_TO_GITHUB.md", "docs/guides/PUSH_TO_GITHUB.md"),
    ("GIT_COMMANDS.txt", "docs/guides/GIT_COMMANDS.txt"),
    ("COMMIT_SUMMARY.txt", "docs/guides/COMMIT_SUMMARY.txt"),
    ("UPDATE_SUMMARY.md", "docs/guides/UPDATE_SUMMARY.md"),
    ("DOCUMENTATION_UPDATE_v5.1.md", "docs/guides/DOCUMENTATION_UPDATE_v5.1.md"),
    
    # Scripts
    ("git_commit_push.py", "scripts/git_commit_push.py"),
    ("git_auto_commit.py", "scripts/git_auto_commit.py"),
    ("execute_git.py", "scripts/execute_git.py"),
    ("COMMIT_COMMANDS.bat", "scripts/COMMIT_COMMANDS.bat"),
    ("commit_and_push.sh", "scripts/commit_and_push.sh"),
    ("test_performance_import.py", "scripts/test_performance_import.py"),
    ("organize_docs.py", "scripts/organize_docs.py"),
    ("update_doc_links.py", "scripts/update_doc_links.py"),
    ("finalize_organization.py", "scripts/finalize_organization.py"),
    ("COMPLETE_REORGANIZATION.py", "scripts/COMPLETE_REORGANIZATION.py"),
]

# Step 3: Move files
moved_count = 0
for src, dst in moves:
    src_path = Path(src)
    if src_path.exists():
        shutil.move(str(src_path), str(dst))
        moved_count += 1
        print(f"✓ Moved: {src} → {dst}")

print(f"\n✓ Step 2: Moved {moved_count} files")

# Step 4: Update links in moved documentation files
link_patterns = [
    # From debugging folder
    (r"\./USAGE_GUIDE\.md", "../../USAGE_GUIDE.md"),
    (r"\./CAPABILITIES\.md", "../../CAPABILITIES.md"),
    (r"\./README\.md", "../../README.md"),
    (r"\./MANAGEMENT_SYSTEMS\.md", "../performance/MANAGEMENT_SYSTEMS.md"),
    (r"\./PERFORMANCE_OPTIMIZATION\.md", "../performance/PERFORMANCE_OPTIMIZATION.md"),
    
    # From performance folder
    (r"\./DEBUGGING_OVERVIEW\.md", "../debugging/DEBUGGING_OVERVIEW.md"),
    (r"\./DEBUGGING_GUIDE\.md", "../debugging/DEBUGGING_GUIDE.md"),
    (r"\./TROUBLESHOOTING_CHEATSHEET\.md", "../debugging/TROUBLESHOOTING_CHEATSHEET.md"),
    (r"\./PROJECT_DEBUGGING\.md", "../debugging/PROJECT_DEBUGGING.md"),
]

def update_file_links(filepath):
    """Update links in a file."""
    if not filepath.exists():
        return False
    
    content = filepath.read_text(encoding='utf-8')
    original = content
    
    for pattern, replacement in link_patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False

# Update all markdown files in docs/
updated = 0
for md_file in docs.rglob("*.md"):
    if update_file_links(md_file):
        updated += 1
        print(f"✓ Updated links in: {md_file}")

print(f"\n✓ Step 3: Updated links in {updated} files")

# Step 5: Create README files
# docs/README.md
docs_readme = """# Documentation

Complete documentation for Multi-LLM Orchestrator v5.1

## 📂 Folder Structure

| Folder | Contents |
|--------|----------|
| `debugging/` | Debugging guides and troubleshooting |
| `performance/` | Performance optimization and management systems |
| `guides/` | Setup guides and helper documentation |

## 🚀 Quick Navigation

### Main Documentation
- [Main README](../README.md) - Project overview
- [Usage Guide](../USAGE_GUIDE.md) - Detailed usage examples
- [Capabilities](../CAPABILITIES.md) - Feature reference

### Debugging & Troubleshooting
- [Debugging Guide](debugging/DEBUGGING_GUIDE.md) - Comprehensive debugging manual
- [Troubleshooting Cheatsheet](debugging/TROUBLESHOOTING_CHEATSHEET.md) - Quick fixes
- [Project Debugging](debugging/PROJECT_DEBUGGING.md) - Debug generated projects
- [Debugging Overview](debugging/DEBUGGING_OVERVIEW.md) - Navigation hub

### Performance & Management Systems
- [Performance Optimization](performance/PERFORMANCE_OPTIMIZATION.md) - Performance tuning
- [Management Systems](performance/MANAGEMENT_SYSTEMS.md) - Knowledge, Project, Product, Quality

### Development Guides
- [Push to GitHub](guides/PUSH_TO_GITHUB.md) - Git workflow
- [Git Commands](guides/GIT_COMMANDS.txt) - Quick reference
- [Documentation Update](guides/DOCUMENTATION_UPDATE_v5.1.md) - v5.1 changes

## 📄 Other Files

- [Adversarial Stress Test](ADVERSARIAL_STRESS_TEST.md)
- [Dashboard](DASHBOARD.md)
- [plans/](plans/) - Planning documents

---

Last Updated: 2026-02-26 | Version: v5.1
"""

(docs / "README.md").write_text(docs_readme, encoding="utf-8")
print("✓ Created docs/README.md")

# scripts/README.md
scripts_readme = """# Scripts

Helper scripts for Multi-LLM Orchestrator

## 🛠️ Available Scripts

### Git & Release
| Script | Purpose | Platform |
|--------|---------|----------|
| `COMMIT_COMMANDS.bat` | Git commit automation | Windows |
| `commit_and_push.sh` | Git commit automation | Unix/Mac |
| `git_commit_push.py` | Git automation | Python |
| `git_auto_commit.py` | Automated commit | Python |

### Testing
| Script | Purpose |
|--------|---------|
| `test_performance_import.py` | Verify module imports |

### Maintenance
| Script | Purpose |
|--------|---------|
| `organize_docs.py` | Documentation organization |
| `update_doc_links.py` | Update doc links |
| `finalize_organization.py` | Reorganization script |
| `COMPLETE_REORGANIZATION.py` | This script |

## 🚀 Usage

### Windows
```batch
scripts\\COMMIT_COMMANDS.bat
```

### Unix/Mac
```bash
./scripts/commit_and_push.sh
```

### Python
```bash
python scripts/git_auto_commit.py
```

---

Last Updated: 2026-02-26
"""

(scripts / "README.md").write_text(scripts_readme, encoding="utf-8")
print("✓ Created scripts/README.md")

print("\n" + "=" * 70)
print("✅ REORGANIZATION COMPLETE!")
print("=" * 70)
print(f"""
📊 Summary:
- Moved {moved_count} files
- Updated links in {updated} files
- Created 2 README files

📁 New Structure:

Root/
├── README.md
├── USAGE_GUIDE.md
├── CAPABILITIES.md
├── docs/
│   ├── README.md
│   ├── debugging/          (4 files)
│   ├── performance/        (3 files)
│   ├── guides/            (5 files)
│   ├── ADVERSARIAL_STRESS_TEST.md
│   ├── DASHBOARD.md
│   └── plans/
├── scripts/               (10 files)
└── orchestrator/          (code modules)
""")
