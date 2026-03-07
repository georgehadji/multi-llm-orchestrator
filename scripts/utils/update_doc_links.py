#!/usr/bin/env python3
"""
Update documentation links after reorganization
"""
from pathlib import Path

# Define link mappings
link_updates = {
    "../DEBUGGING_GUIDE.md": "./DEBUGGING_GUIDE.md",
    "../TROUBLESHOOTING_CHEATSHEET.md": "./TROUBLESHOOTING_CHEATSHEET.md",
    "../PROJECT_DEBUGGING.md": "./PROJECT_DEBUGGING.md",
    "../PERFORMANCE_OPTIMIZATION.md": "../performance/PERFORMANCE_OPTIMIZATION.md",
    "../MANAGEMENT_SYSTEMS.md": "../performance/MANAGEMENT_SYSTEMS.md",
    "../USAGE_GUIDE.md": "../../USAGE_GUIDE.md",
    "../README.md": "../../README.md",
    "../CAPABILITIES.md": "../../CAPABILITIES.md",
}

def update_links_in_file(filepath):
    """Update links in a single file."""
    content = filepath.read_text(encoding='utf-8')
    original = content
    
    for old_link, new_link in link_updates.items():
        content = content.replace(old_link, new_link)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        print(f"✓ Updated {filepath}")
        return True
    return False

def main():
    docs_path = Path("docs")
    
    # Update all markdown files in docs/
    updated = 0
    for md_file in docs_path.rglob("*.md"):
        if update_links_in_file(md_file):
            updated += 1
    
    print(f"\n✓ Updated {updated} files")

if __name__ == "__main__":
    main()
