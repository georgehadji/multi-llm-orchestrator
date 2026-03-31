#!/usr/bin/env python3
"""
Sync FROM D: drive TO E: drive
Ensures E: has all the latest files from D:
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def copy_file(src: Path, dst: Path) -> bool:
    """Copy a file from src to dst."""
    if not src.exists():
        return False
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if dst.exists():
            shutil.copy2(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        log(f"❌ Error: {e}")
        return False

def main():
    log("=" * 60)
    log("Syncing D: -> E: (AI Orchestrator)")
    log("=" * 60)
    
    d_root = Path("D:/Vibe-Coding/Ai Orchestrator")
    e_root = Path("E:/Documents/Vibe-Coding/Ai Orchestrator")
    
    if not d_root.exists():
        log("❌ D: source not found!")
        return 1
    
    e_root.mkdir(parents=True, exist_ok=True)
    
    # Sync orchestrator package
    log("\n📦 Syncing orchestrator package...")
    d_orch = d_root / "orchestrator"
    e_orch = e_root / "orchestrator"
    
    copied = 0
    for py_file in d_orch.rglob("*.py"):
        rel_path = py_file.relative_to(d_orch)
        dst_file = e_orch / rel_path
        if copy_file(py_file, dst_file):
            copied += 1
    
    log(f"  Copied {copied} Python files")
    
    # Sync docs
    log("\n📚 Syncing docs...")
    d_docs = d_root / "docs"
    e_docs = e_root / "docs"
    
    if d_docs.exists():
        copied = 0
        for md_file in d_docs.rglob("*.md"):
            rel_path = md_file.relative_to(d_docs)
            dst_file = e_docs / rel_path
            if copy_file(md_file, dst_file):
                copied += 1
        log(f"  Copied {copied} documentation files")
    
    # Sync config files
    log("\n⚙️  Syncing config files...")
    config_files = ["pyproject.toml", ".env.example", ".gitignore", "README.md", "CONTRIBUTING.md"]
    for filename in config_files:
        src = d_root / filename
        dst = e_root / filename
        if src.exists() and copy_file(src, dst):
            log(f"  ✅ {filename}")
    
    log("\n" + "=" * 60)
    log("✅ Sync complete!")
    log("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
