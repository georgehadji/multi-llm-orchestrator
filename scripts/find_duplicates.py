#!/usr/bin/env python3
"""
Find duplicate modules between orchestrator/ root and subpackages.

A duplicate is defined as a root-level .py file that has ≥ 90% similarity
with a subpackage file of the same name.
"""
import ast
import difflib
import sys
from pathlib import Path


def get_file_content(filepath: Path) -> str:
    """Read file content, normalizing whitespace."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Normalize: remove comments, docstrings for comparison
        return content
    except Exception:
        return ""


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_duplicate_modules(orchestrator_dir: Path) -> list[dict]:
    """Find root/package duplicate module pairs."""
    duplicates = []
    root_dir = orchestrator_dir

    # Get all .py files in root
    root_files = list(root_dir.glob("*.py"))

    for root_file in root_files:
        if root_file.name.startswith("test_"):
            continue

        root_content = get_file_content(root_file)
        if not root_content.strip():
            continue

        # Check for subpackage versions
        module_name = root_file.stem

        # Possible locations: module.py, module/module.py, or subpackage/module.py
        candidates = [
            root_dir / f"{module_name}.py",  # self (skip)
            root_dir / module_name / "__init__.py",
            root_dir / module_name / f"{module_name}.py",
        ]

        # Also check common subpackage patterns
        for subpkg in [
            "infrastructure",
            "application",
            "domain",
            "agents",
            "analysis",
            "events",
            "cost_optimization",
            "generators",
        ]:
            candidates.extend([
                root_dir / subpkg / f"{module_name}.py",
                root_dir / subpkg / module_name / "__init__.py",
            ])

        for candidate in candidates:
            if candidate == root_file:
                continue
            if not candidate.exists():
                continue

            pkg_content = get_file_content(candidate)
            if not pkg_content.strip():
                continue

            sim = similarity(root_content, pkg_content)

            if sim >= 0.90:
                duplicates.append({
                    "root_file": str(root_file.relative_to(root_dir.parent)),
                    "pkg_file": str(candidate.relative_to(root_dir.parent)),
                    "similarity": sim,
                    "root_lines": len(root_content.splitlines()),
                    "pkg_lines": len(pkg_content.splitlines()),
                })
                break  # Found a match, move to next root file

    return duplicates


def main():
    orchestrator_dir = Path(__file__).parent.parent / "orchestrator"

    if not orchestrator_dir.exists():
        print(f"Error: {orchestrator_dir} does not exist")
        sys.exit(1)

    print("Scanning for duplicate modules...")
    print(f"Root directory: {orchestrator_dir}")
    print()

    duplicates = find_duplicate_modules(orchestrator_dir)

    if not duplicates:
        print("No duplicate modules found (≥ 90% similarity)")
        sys.exit(0)

    print(f"Found {len(duplicates)} duplicate module pairs:\n")

    # Sort by similarity
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)

    for dup in duplicates:
        print(f"  {dup['root_file']}")
        print(f"    = {dup['similarity']:.1%} similar")
        print(f"    -> {dup['pkg_file']}")
        print(f"    ({dup['root_lines']} lines vs {dup['pkg_lines']} lines)")
        print()

    # Output as JSON for automation
    import json

    output_file = Path(".tmp/duplicate_modules_report.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(duplicates, f, indent=2)

    print(f"Full report saved to: {output_file}")

    # Summary stats
    total_root_lines = sum(d["root_lines"] for d in duplicates)
    print(f"\nSummary: {len(duplicates)} duplicates, ~{total_root_lines} lines of dead code")


if __name__ == "__main__":
    main()
