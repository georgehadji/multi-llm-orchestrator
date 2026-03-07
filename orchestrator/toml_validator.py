"""
TOML Validator — Validates and fixes TOML files
=================================================

Usage:
    python -m orchestrator.toml_validator fix outputs/cinematic_webgl_framework/pyproject.toml
"""

import sys
import tomllib
from pathlib import Path
import re


def validate_toml(file_path: Path) -> tuple[bool, str]:
    """
    Validate a TOML file.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            tomllib.load(f)
        return True, ""
    except tomllib.TOMLDecodeError as e:
        return False, str(e)


def fix_toml_newlines(file_path: Path) -> bool:
    """
    Fix newline characters in TOML strings.
    
    This is a common issue when LLMs generate TOML with multi-line strings.
    """
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Find and fix multi-line strings in TOML values
    # Pattern: key = "value\ncontinued"
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if line has unclosed string
        if '=' in line and '"' in line:
            # Count quotes
            quote_count = line.count('"') - line.count('\\"')
            
            if quote_count % 2 == 1:  # Odd = unclosed string
                # Merge with next lines until closed
                merged = line
                while quote_count % 2 == 1 and i + 1 < len(lines):
                    i += 1
                    next_line = lines[i].strip()
                    # Remove leading/trailing whitespace but keep content
                    merged = merged + ' ' + next_line
                    quote_count += next_line.count('"') - next_line.count('\\"')
                
                # Now merge into single line
                # Replace the multi-line string with single line
                line = merged
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    # Additional fix: Replace literal \n in strings with space
    # This handles cases like: description = "text\ncontinued"
    content = re.sub(r'"([^"]*)\\n([^"]*)"', r'"\1 \2"', content)
    
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ Fixed TOML file: {file_path}")
        return True
    
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m orchestrator.toml_validator <command> <file>")
        print("Commands:")
        print("  validate - Check if TOML is valid")
        print("  fix      - Fix common TOML issues")
        sys.exit(1)
    
    command = sys.argv[1]
    file_path = Path(sys.argv[2])
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    if command == "validate":
        is_valid, error = validate_toml(file_path)
        if is_valid:
            print(f"✅ Valid TOML: {file_path}")
        else:
            print(f"❌ Invalid TOML: {file_path}")
            print(f"   Error: {error}")
            sys.exit(1)
    
    elif command == "fix":
        # First validate
        is_valid, error = validate_toml(file_path)
        if is_valid:
            print(f"✅ Already valid: {file_path}")
            return
        
        print(f"🔧 Fixing TOML file: {file_path}")
        print(f"   Error: {error}")
        
        if fix_toml_newlines(file_path):
            # Re-validate
            is_valid, error = validate_toml(file_path)
            if is_valid:
                print(f"✅ Successfully fixed!")
            else:
                print(f"⚠️  Still has errors: {error}")
                sys.exit(1)
        else:
            print(f"⚠️  No automatic fixes available")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
