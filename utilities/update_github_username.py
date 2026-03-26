"""
Update GitHub username in all documentation files.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

OLD_USERNAME = "gchatz22"
NEW_USERNAME = "georrgehadji"

def update_file(filepath: Path) -> int:
    """Update username in file. Returns number of replacements."""
    try:
        content = filepath.read_text(encoding='utf-8')
        count = content.count(OLD_USERNAME)
        if count > 0:
            new_content = content.replace(OLD_USERNAME, NEW_USERNAME)
            filepath.write_text(new_content, encoding='utf-8')
            return count
        return 0
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return 0

def main():
    root = Path(__file__).parent
    total_replacements = 0
    
    # Find all markdown files
    for md_file in root.rglob("*.md"):
        # Skip .git directory
        if ".git" in str(md_file):
            continue
        
        count = update_file(md_file)
        if count > 0:
            print(f"[OK] {md_file.relative_to(root)}: {count} replacements")
            total_replacements += count
    
    print(f"\n{'='*60}")
    print(f"Total replacements: {total_replacements}")
    print(f"Username updated: {OLD_USERNAME} -> {NEW_USERNAME}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
