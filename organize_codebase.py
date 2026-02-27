#!/usr/bin/env python3
"""
Codebase Organization Script
============================
Organizes files into proper directories:
- docs/ - Documentation
- scripts/ - Helper scripts  
- archive/ - Old/temp files
- tools/ - Development tools
"""
import shutil
import os
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("📁 CODEBASE ORGANIZATION")
print("=" * 70)

# Create directory structure
dirs = [
    "docs",
    "docs/debugging",
    "docs/performance", 
    "docs/guides",
    "scripts",
    "tools",
    "archive/temp_files",
    "archive/old_docs",
    "archive/backup",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {d}")

# File categories
file_moves = {
    # Documentation files -> docs/
    "docs/": [
        ("DEBUGGING_GUIDE.md", "debugging/DEBUGGING_GUIDE.md"),
        ("DEBUGGING_OVERVIEW.md", "debugging/DEBUGGING_OVERVIEW.md"),
        ("PROJECT_DEBUGGING.md", "debugging/PROJECT_DEBUGGING.md"),
        ("TROUBLESHOOTING_CHEATSHEET.md", "debugging/TROUBLESHOOTING_CHEATSHEET.md"),
        ("PERFORMANCE_OPTIMIZATION.md", "performance/PERFORMANCE_OPTIMIZATION.md"),
        ("PERFORMANCE_SUMMARY.md", "performance/PERFORMANCE_SUMMARY.md"),
        ("MANAGEMENT_SYSTEMS.md", "performance/MANAGEMENT_SYSTEMS.md"),
        ("docs_performance_PROJECT_ANALYZER.md", "performance/PROJECT_ANALYZER.md"),
        ("PUSH_TO_GITHUB.md", "guides/PUSH_TO_GITHUB.md"),
        ("GIT_COMMANDS.txt", "guides/GIT_COMMANDS.txt"),
        ("COMMIT_SUMMARY.txt", "guides/COMMIT_SUMMARY.txt"),
        ("UPDATE_SUMMARY.md", "guides/UPDATE_SUMMARY.md"),
        ("DOCUMENTATION_UPDATE_v5.1.md", "guides/DOCUMENTATION_UPDATE_v5.1.md"),
        ("FINAL_COMMIT_GUIDE.md", "guides/FINAL_COMMIT_GUIDE.md"),
        ("FEATURE_PROJECT_ANALYZER.md", "guides/FEATURE_PROJECT_ANALYZER.md"),
    ],
    
    # Scripts -> scripts/
    "scripts/": [
        ("git_commit_push.py", "git_commit_push.py"),
        ("git_auto_commit.py", "git_auto_commit.py"),
        ("execute_git.py", "execute_git.py"),
        ("COMMIT_COMMANDS.bat", "COMMIT_COMMANDS.bat"),
        ("commit_and_push.sh", "commit_and_push.sh"),
        ("test_performance_import.py", "test_performance_import.py"),
        ("organize_docs.py", "organize_docs.py"),
        ("update_doc_links.py", "update_doc_links.py"),
        ("finalize_organization.py", "finalize_organization.py"),
        ("COMPLETE_REORGANIZATION.py", "COMPLETE_REORGANIZATION.py"),
        ("create_docs_dir.py", "create_docs_dir.py"),
        ("organize_codebase.py", "organize_codebase.py"),
        ("fix_git_errors.py", "fix_git_errors.py"),
        ("clean_and_push.py", "clean_and_push.py"),
    ],
    
    # Tools -> tools/
    "tools/": [
        ("check_outputs.py", "check_outputs.py"),
        ("check_state.py", "check_state.py"),
    ],
    
    # Archive old/temp files
    "archive/temp_files/": [
        ("cleanup_all.py", "cleanup_all.py"),
        ("cleanup_final.py", "cleanup_final.py"),
        ("cleanup_temp.py", "cleanup_temp.py"),
        ("cleanup_temp_files.py", "cleanup_temp_files.py"),
        ("final_cleanup.py", "final_cleanup.py"),
        ("final_cleanup_all.py", "final_cleanup_all.py"),
    ],
    
    "archive/old_docs/": [
        ("CAPABILITIES.md.bak", "CAPABILITIES.md.bak"),
        ("USAGE_GUIDE.md.bak", "USAGE_GUIDE.md.bak"),
        ("LLM_ANALYSIS_REPORT.md", "LLM_ANALYSIS_REPORT.md"),
        ("LLM_ROUTING_UPDATES_APPLIED.md", "LLM_ROUTING_UPDATES_APPLIED.md"),
        ("OPTIMIZATIONS_APPLIED.md", "OPTIMIZATIONS_APPLIED.md"),
        ("OPTIMIZATIONS_BATCH_2.md", "OPTIMIZATIONS_BATCH_2.md"),
        ("OPTIMIZATION_REPORT.md", "OPTIMIZATION_REPORT.md"),
    ],
    
    "archive/backup/": [
        (".env_old.txt", ".env_old.txt"),
    ],
}

# Move files
moved_count = 0
for base_dir, moves in file_moves.items():
    for src, dst in moves:
        src_path = Path(src)
        if src_path.exists():
            dst_path = Path(base_dir) / dst
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"✓ Moved: {src} → {base_dir}{dst}")
                moved_count += 1
            except Exception as e:
                print(f"⚠ Could not move {src}: {e}")

print(f"\n✓ Moved {moved_count} files")

# Create/update docs/README.md
docs_readme = """# Documentation

Multi-LLM Orchestrator v5.1 Documentation

## 📂 Structure

| Folder | Contents |
|--------|----------|
| `debugging/` | Debugging guides and troubleshooting |
| `performance/` | Performance, management systems, project analyzer |
| `guides/` | Setup guides, Git workflow, update notes |

## 🚀 Quick Links

### Main Documentation
- [Main README](../README.md)
- [Usage Guide](../USAGE_GUIDE.md)
- [Capabilities](../CAPABILITIES.md)

### Debugging
- [Debugging Guide](debugging/DEBUGGING_GUIDE.md)
- [Troubleshooting Cheatsheet](debugging/TROUBLESHOOTING_CHEATSHEET.md)
- [Project Debugging](debugging/PROJECT_DEBUGGING.md)

### Performance & Management
- [Performance Optimization](performance/PERFORMANCE_OPTIMIZATION.md)
- [Management Systems](performance/MANAGEMENT_SYSTEMS.md)
- [Project Analyzer](performance/PROJECT_ANALYZER.md)

### Guides
- [Push to GitHub](guides/PUSH_TO_GITHUB.md)
- [Git Commands](guides/GIT_COMMANDS.txt)

---
Last Updated: """ + datetime.now().strftime("%Y-%m-%d") + "\n"

Path("docs/README.md").write_text(docs_readme, encoding='utf-8')
print("✓ Updated docs/README.md")

# Create scripts/README.md
scripts_readme = """# Scripts

Helper scripts for Multi-LLM Orchestrator

## Available Scripts

### Git & Release
- `COMMIT_COMMANDS.bat` - Windows git automation
- `commit_and_push.sh` - Unix/Mac git automation
- `git_commit_push.py` - Python git helper

### Testing
- `test_performance_import.py` - Verify imports

### Organization
- `organize_docs.py` - Documentation organization
- `organize_codebase.py` - This script

---
Last Updated: """ + datetime.now().strftime("%Y-%m-%d") + "\n"

Path("scripts/README.md").write_text(scripts_readme, encoding='utf-8')
print("✓ Updated scripts/README.md")

print("\n" + "=" * 70)
print("✅ ORGANIZATION COMPLETE!")
print("=" * 70)
print(f"""
📊 Summary:
- Created {len(dirs)} directories
- Moved {moved_count} files
- Updated 2 README files

📁 Structure:
Root/
├── docs/           # Documentation (organized)
├── scripts/        # Helper scripts
├── tools/          # Development tools
├── archive/        # Old/temp files
└── orchestrator/   # Main code
""")
