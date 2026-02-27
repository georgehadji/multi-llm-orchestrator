#!/usr/bin/env python3
"""
Automated Git Commit for v5.1 Release
"""
import subprocess
import sys
from pathlib import Path

def run(cmd, check=True):
    """Run shell command."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout if result.stdout else "(no output)")
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     🚀 GIT AUTO COMMIT - Multi-LLM Orchestrator v5.1             ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Check if we're in a git repo
    if not Path(".git").exists():
        print("❌ Not a git repository!")
        return 1
    
    # Git status
    print("📋 Checking git status...")
    if not run("git status --short", check=False):
        print("⚠️ No changes to commit or git error")
        return 1
    
    # Add all changes
    print("\n📦 Staging changes...")
    if not run("git add -A"):
        return 1
    
    # Create commit message
    commit_msg = """feat: v5.1 Management Systems + v5.0 Performance Optimization

🆕 NEW: Management Systems (v5.1)
- Knowledge Management: Semantic search, pattern recognition
- Project Management: Critical path analysis, resource scheduling  
- Product Management: RICE prioritization, feature flags
- Quality Control: Multi-level testing, compliance gates

⚡ NEW: Performance Optimization (v5.0)
- Dashboard v5.0: 5x faster load (<100ms FCP)
- Dual-layer caching: Redis + LRU fallback
- Connection pooling: Bounded resource management
- KPI monitoring: Real-time performance tracking

📚 Updated: CAPABILITIES.md, README.md, USAGE_GUIDE.md
📁 New files: 7 orchestrator modules, tests, docs

Performance targets: FCP<100ms, Cache>85%, P95<300ms"""
    
    # Write commit message to temp file
    msg_file = Path(".git/COMMIT_MSG")
    msg_file.write_text(commit_msg, encoding="utf-8")
    
    # Commit
    print("\n📝 Creating commit...")
    if not run(f"git commit -F {msg_file}"):
        print("⚠️ Commit may have failed or nothing to commit")
        return 1
    
    # Show commit
    print("\n✅ Commit created successfully!")
    run("git log -1 --oneline")
    run("git log -1 --stat")
    
    # Push instructions
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🎉 COMMIT COMPLETE!                                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  To push to GitHub:                                              ║
║                                                                  ║
║     git push origin main                                         ║
║                                                                  ║
║  Or create a PR branch:                                          ║
║                                                                  ║
║     git checkout -b release/v5.1                                 ║
║     git push -u origin release/v5.1                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
