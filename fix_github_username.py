"""
Update GitHub username in all documentation files.
gchatz22/georrgehadji → georgehadji
"""
import os
import sys
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OLD_USERNAMES = ["gchatz22", "georrgehadji"]
NEW_USERNAME = "georgehadji"

def update_file(filepath: Path) -> int:
    """Update username in file. Returns number of replacements."""
    try:
        content = filepath.read_text(encoding='utf-8')
        count = 0
        for old in OLD_USERNAMES:
            count += content.count(old)
        
        if count > 0:
            new_content = content
            for old in OLD_USERNAMES:
                new_content = new_content.replace(old, NEW_USERNAME)
            filepath.write_text(new_content, encoding='utf-8')
            return count
        return 0
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return 0

def main():
    root = Path(__file__).parent
    total_replacements = 0
    files_updated = []
    
    # Find all markdown files
    for md_file in root.rglob("*.md"):
        # Skip .git directory and Useful Github Projects
        if ".git" in str(md_file) or "Useful Github Projects" in str(md_file):
            continue
        
        count = update_file(md_file)
        if count > 0:
            files_updated.append((md_file.relative_to(root), count))
            total_replacements += count
    
    # Update batch files
    for bat_file in root.rglob("*.bat"):
        if ".git" in str(bat_file):
            continue
        count = update_file(bat_file)
        if count > 0:
            files_updated.append((bat_file.relative_to(root), count))
            total_replacements += count
    
    # Print results
    print("=" * 70)
    print("GITHUB USERNAME UPDATE COMPLETE")
    print("=" * 70)
    print(f"\nUpdated: {OLD_USERNAMES} -> {NEW_USERNAME}")
    print(f"\nFiles updated ({len(files_updated)}):")
    for filepath, count in files_updated:
        print(f"  [OK] {filepath}: {count} replacements")
    print(f"\nTotal replacements: {total_replacements}")
    print("=" * 70)

if __name__ == "__main__":
    main()
