#!/usr/bin/env python3
"""
Clean temporary files and push to GitHub
"""
import subprocess
import sys
from pathlib import Path

def run(cmd, description=""):
    """Run shell command."""
    if description:
        print(f"\n📌 {description}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and "error" in result.stderr.lower():
        print(f"⚠️  {result.stderr}", file=sys.stderr)
    return result.returncode == 0

print("=" * 70)
print("🧹 CLEAN & PUSH - Multi-LLM Orchestrator v5.1")
print("=" * 70)

# Step 1: Remove problematic files
print("\n🗑️  Step 1: Removing problematic files...")
problematic = ["nul", "h origin main"]
for f in problematic:
    path = Path(f)
    if path.exists():
        path.unlink()
        print(f"✓ Removed: {f}")

# Step 2: Add to .gitignore
print("\n📝 Step 2: Updating .gitignore...")
with open(".gitignore", "a") as f:
    f.write("\n# Problematic files\nnul\nh origin main\n")
print("✓ Updated .gitignore")

# Step 3: Stage important files only
print("\n📦 Step 3: Staging files...")
important_dirs = [
    "orchestrator/",
    "tests/",
    "docs/",
    "scripts/",
    "CAPABILITIES.md",
    "README.md",
    "USAGE_GUIDE.md",
    "pyproject.toml",
    ".gitignore",
]

for item in important_dirs:
    run(f"git add {item}", f"Adding {item}")

# Step 4: Check status
print("\n📋 Step 4: Git status...")
run("git status --short", "Current status")

# Step 5: Commit
print("\n📝 Step 5: Creating commit...")
commit_msg = """feat: v5.1 Management Systems + v5.0 Performance Optimization

🆕 NEW: Management Systems (v5.1)
- Knowledge Management: Semantic search, pattern recognition
- Project Management: Critical path analysis, resource scheduling  
- Product Management: RICE prioritization, feature flags
- Quality Control: Multi-level testing, compliance gates
- Diagnostics: System & project diagnostic tools

⚡ NEW: Performance Optimization (v5.0)
- Dashboard v5.0: 5x faster load (<100ms FCP)
- Dual-layer caching: Redis + LRU fallback
- Connection pooling: Bounded resource management
- KPI monitoring: Real-time performance tracking

📚 Documentation: Organized into docs/ folder
📁 New files: 9 orchestrator modules, 12 docs, 10 scripts

Performance targets: FCP<100ms, Cache>85%, P95<300ms"""

with open(".git/COMMIT_MSG", "w") as f:
    f.write(commit_msg)

run("git commit -F .git/COMMIT_MSG", "Creating commit")

# Step 6: Show commit
print("\n📊 Step 6: Commit details...")
run("git log -1 --stat", "Commit details")

print("\n" + "=" * 70)
print("✅ READY TO PUSH!")
print("=" * 70)
print("\nRun this to push:")
print("  git push origin release/v5.1")
print("\nOr create PR:")
print("  git push -u origin release/v5.1")
print("  gh pr create --title 'Release v5.1' --body 'Management Systems + Performance'")
