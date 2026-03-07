"""Create scripts directory and all script files."""
import os
from pathlib import Path

# Create scripts directory
scripts_dir = Path("scripts")
scripts_dir.mkdir(exist_ok=True)

scripts = {
    "setup.sh": '''#!/usr/bin/env bash
# Development Environment Setup Script
set -euo pipefail

echo "Setting up development environment..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION"

# Create venv
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -e ".[dev,security]"
echo "✓ Dependencies installed"

# Setup pre-commit
pre-commit install 2>/dev/null || echo "pre-commit not installed"

# Create .env if not exists
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✓ .env created from .env.example"
fi

# Create directories
mkdir -p logs outputs data .cache

echo ""
echo "✓ Setup complete!"
echo "Next: source .venv/bin/activate && vim .env"
''',
    
    "lint.sh": '''#!/usr/bin/env bash
# Code Quality Check Script
set -euo pipefail

echo "Running code quality checks..."
echo ""

ERRORS=0

# Black
echo "→ Black formatting check..."
if black --check orchestrator/ tests/ 2>/dev/null; then
    echo "  ✓ Black"
else
    echo "  ✗ Black: Run 'black orchestrator/ tests/' to fix"
    ERRORS=$((ERRORS + 1))
fi

# Ruff
echo "→ Ruff linting..."
if ruff check orchestrator/ tests/ 2>/dev/null; then
    echo "  ✓ Ruff"
else
    echo "  ✗ Ruff errors found"
    ERRORS=$((ERRORS + 1))
fi

# MyPy
echo "→ MyPy type check..."
if mypy orchestrator/ --ignore-missing-imports 2>/dev/null; then
    echo "  ✓ MyPy"
else
    echo "  ✗ MyPy errors found"
    ERRORS=$((ERRORS + 1))
fi

# Bandit
echo "→ Bandit security scan..."
bandit -r orchestrator/ -q 2>/dev/null && echo "  ✓ Bandit" || echo "  ⚠ Bandit issues found"

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
else
    echo "✗ $ERRORS check(s) failed"
    exit 1
fi
''',
    
    "test.sh": '''#!/usr/bin/env bash
# Test Runner Script
set -euo pipefail

echo "Running tests..."

# Activate venv if exists
[ -d ".venv" ] && source .venv/bin/activate

# Parse args
TEST_PATH="tests/"
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit) TEST_PATH="tests/unit"; shift ;;
        -i|--integration) TEST_PATH="tests/integration"; shift ;;
        -c|--coverage) COVERAGE="--cov=orchestrator"; shift ;;
        *) shift ;;
    esac
done

pytest -v $COVERAGE "$TEST_PATH"
''',
    
    "release.sh": '''#!/usr/bin/env bash
# Release Script
set -euo pipefail

VERSION_TYPE="${1:-patch}"

echo "Preparing $VERSION_TYPE release..."

# Check clean git
if ! git diff-index --quiet HEAD --; then
    echo "ERROR: Uncommitted changes"
    exit 1
fi

# Run tests
echo "→ Running tests..."
make ci

# Get current version
CURRENT=$(grep '__version__' orchestrator/__init__.py | head -1 | grep -o '".*"' | tr -d '"')
echo "Current version: $CURRENT"

# Bump version (simple logic)
IFS='.' read -r major minor patch <<< "$CURRENT"
case $VERSION_TYPE in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
esac
NEW="$major.$minor.$patch"
echo "New version: $NEW"

# Update version
sed -i.bak "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" orchestrator/__init__.py
rm orchestrator/__init__.py.bak

# Commit and tag
git add -A
git commit -m "Release version $NEW"
git tag -a "v$NEW" -m "Release v$NEW"

# Push
git push origin HEAD
git push origin "v$NEW"

echo "✓ Release v$NEW complete!"
''',
}

# Write scripts
for filename, content in scripts.items():
    filepath = scripts_dir / filename
    filepath.write_text(content, encoding='utf-8')
    # Make executable (works on Unix-like systems)
    os.chmod(filepath, 0o755)
    print(f"✓ Created: {filepath}")

print(f"\n✓ All scripts created in {scripts_dir}/")

# Self cleanup
Path(__file__).unlink()
