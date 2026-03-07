#!/bin/bash
# Git Commit & Push Script for Multi-LLM Orchestrator v5.1

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     🚀 GIT COMMIT & PUSH - Multi-LLM Orchestrator v5.1           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Step 1: Git Status${NC}"
git status --short

echo ""
echo -e "${BLUE}📦 Step 2: Stage All Changes${NC}"
git add -A
git status --short

echo ""
echo -e "${BLUE}📝 Step 3: Create Commit${NC}"

COMMIT_MSG='feat: v5.1 Management Systems + v5.0 Performance Optimization

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

Performance targets: FCP<100ms, Cache>85%, P95<300ms'

git commit -m "$COMMIT_MSG"

echo ""
echo -e "${GREEN}✅ Step 4: Commit Created${NC}"
git log -1 --oneline
git log -1 --stat --name-only

echo ""
echo -e "${YELLOW}🚀 Step 5: Push to GitHub${NC}"
echo ""
echo "Choose push option:"
echo "  1) Push to main branch directly"
echo "  2) Create and push to release/v5.1 branch (recommended)"
echo "  3) Skip push for now"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
  1)
    echo "Pushing to main..."
    git push origin main
    echo -e "${GREEN}✅ Pushed to main!${NC}"
    ;;
  2)
    echo "Creating release/v5.1 branch..."
    git checkout -b release/v5.1
    git push -u origin release/v5.1
    echo -e "${GREEN}✅ Created and pushed release/v5.1 branch!${NC}"
    echo "Create PR at: https://github.com/gchatz22/multi-llm-orchestrator/pull/new/release/v5.1"
    ;;
  3)
    echo "Skipped push. To push later run:"
    echo "  git push origin main"
    ;;
  *)
    echo "Invalid choice. To push later run:"
    echo "  git push origin main"
    ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🎉 ALL DONE!                                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
