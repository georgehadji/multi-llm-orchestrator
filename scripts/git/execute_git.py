#!/usr/bin/env python3
"""Execute git commands using Python subprocess."""
import subprocess
import sys

def run_git(args):
    """Run git command."""
    cmd = ["git"] + args
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python execute_git.py <git-args...>")
        sys.exit(1)
    
    sys.exit(run_git(sys.argv[1:]))
