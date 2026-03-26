#!/usr/bin/env python3
"""
Push AI Orchestrator Updates to GitHub
========================================

Commits and pushes all recent changes:
- Issue tracking integration
- Slack integration  
- Git service integration
- Sync scripts
- Documentation updates

Usage:
    python push_to_github.py --dry-run   # Preview
    python push_to_github.py --apply     # Execute
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run(cmd: str, check: bool = False) -> tuple[int, str, str]:
    """Run a shell command."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr

def log(msg: str, level: str = "info"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": "\033[94m", "success": "\033[92m",
        "warning": "\033[93m", "error": "\033[91m"
    }
    color = colors.get(level, "")
    reset = "\033[0m"
    print(f"{color}[{ts}] {msg}{reset}")

def main():
    dry_run = "--apply" not in sys.argv
    
    print("\n" + "="*60)
    print("  Push to GitHub")
    print("="*60)
    print(f"\n  Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")
    
    # Check git status
    log("Checking git status...")
    code, stdout, stderr = run("git status --short")
    
    if not stdout.strip():
        log("No changes to commit", "warning")
        return 0
    
    print("\nChanged files:")
    print(stdout)
    
    # Check which branch
    code, branch, _ = run("git branch --show-current")
    branch = branch.strip()
    log(f"Current branch: {branch}")
    
    # Files to commit
    new_files = [
        "orchestrator/issue_tracking.py",
        "orchestrator/slack_integration.py", 
        "orchestrator/git_service.py",
        "orchestrator/git_hooks.py",
        "orchestrator/git_integration_example.py",
        "docs/ISSUE_TRACKING.md",
        "full_sync_and_cleanup.py",
        "sync_d_to_e.py",
        "SYNC_README.md",
    ]
    
    # Generate commit message
    commit_msg = f"""feat: Add issue tracking, Slack & Git integrations

New features:
- Issue tracking integration (Jira/Linear)
- Slack notifications & slash commands
- Git service (GitHub/GitLab) for CI/CD
- Sync scripts for D:/E: drive management
- Comprehensive documentation

Author: Georgios-Chrysovalantis Chatzivantsidis
Date: {datetime.now().strftime('%Y-%m-%d')}"""
    
    print("\n" + "="*60)
    print("Proposed commit message:")
    print("="*60)
    print(commit_msg)
    print("="*60)
    
    if dry_run:
        print("\n⚠️  DRY RUN - No changes pushed.")
        print("Run with --apply to execute.\n")
        return 0
    
    # Execute git commands
    log("Adding files...")
    run("git add -A")
    
    log("Committing...")
    code, _, stderr = run(f'git commit -m "{commit_msg}"')
    if code != 0:
        log(f"Commit error: {stderr}", "error")
        return 1
    
    log("Pushing to GitHub...")
    code, _, stderr = run(f"git push origin {branch}")
    if code != 0:
        log(f"Push error: {stderr}", "error")
        return 1
    
    log("✅ Successfully pushed to GitHub!", "success")
    return 0

if __name__ == "__main__":
    sys.exit(main())
