#!/usr/bin/env python3
"""Initialize complete project structure."""
import os
import stat
from pathlib import Path

def create_file(path: Path, content: str, executable: bool = False) -> None:
    """Create a file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    if executable:
        os.chmod(path, 0o755)
    print(f"✓ {path}")

# Base directory
base = Path(".")

# ═══════════════════════════════════════════════════════════════════════════════
# SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPTS = {
    "scripts/setup.sh": '''#!/usr/bin/env bash
set -euo pipefail
echo "Setting up development environment..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,security]"
echo "✓ Dependencies installed"

pre-commit install 2>/dev/null || echo "pre-commit not installed"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✓ .env created from .env.example"
fi

mkdir -p logs outputs data .cache
echo ""
echo "✓ Setup complete!"
''',
    "scripts/lint.sh": '''#!/usr/bin/env bash
set -euo pipefail
echo "Running code quality checks..."
ERRORS=0

if black --check orchestrator/ tests/ 2>/dev/null; then
    echo "✓ Black"
else
    echo "✗ Black: Run 'black orchestrator/ tests/' to fix"
    ERRORS=$((ERRORS + 1))
fi

if ruff check orchestrator/ tests/ 2>/dev/null; then
    echo "✓ Ruff"
else
    echo "✗ Ruff errors found"
    ERRORS=$((ERRORS + 1))
fi

if mypy orchestrator/ --ignore-missing-imports 2>/dev/null; then
    echo "✓ MyPy"
else
    echo "✗ MyPy errors found"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
else
    exit 1
fi
''',
    "scripts/test.sh": '''#!/usr/bin/env bash
set -euo pipefail
echo "Running tests..."
[ -d ".venv" ] && source .venv/bin/activate
pytest -v "$@"
''',
    "scripts/release.sh": '''#!/usr/bin/env bash
set -euo pipefail
VERSION_TYPE="${1:-patch}"

if ! git diff-index --quiet HEAD --; then
    echo "ERROR: Uncommitted changes"
    exit 1
fi

echo "→ Running tests..."
make ci

CURRENT=$(grep '__version__' orchestrator/__init__.py | head -1 | grep -o '".*"' | tr -d '"')
IFS='.' read -r major minor patch <<< "$CURRENT"
case $VERSION_TYPE in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
esac
NEW="$major.$minor.$patch"

echo "Bumping: $CURRENT → $NEW"
sed -i.bak "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" orchestrator/__init__.py
rm orchestrator/__init__.py.bak

git add -A
git commit -m "Release version $NEW"
git tag -a "v$NEW" -m "Release v$NEW"
git push origin HEAD
git push origin "v$NEW"
echo "✓ Release v$NEW complete!"
''',
}

for filepath, content in SCRIPTS.items():
    create_file(base / filepath, content, executable=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FILES
# ═══════════════════════════════════════════════════════════════════════════════

create_file(base / ".editorconfig", '''# EditorConfig
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4
max_line_length = 100

[*.{yml,yaml}]
indent_size = 2

[Makefile]
indent_style = tab
''')

create_file(base / "LICENSE", '''MIT License

Copyright (c) 2024 Georgios-Chrysovalantis Chatzivantsidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
''')

create_file(base / "CHANGELOG.md", '''# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready project structure
- Exception hierarchy for proper error handling
- Structured logging with correlation IDs
- Docker multi-stage build
- GitHub Actions CI/CD pipeline
- Pre-commit hooks
- Development scripts
- Comprehensive Makefile

## [1.1.0] - 2024-01-15

### Added
- Multi-provider LLM support
- Cost-optimized routing
- Policy enforcement system
- Resume capability

## [1.0.0] - 2024-01-01

### Added
- Initial release
''')

create_file(base / "docker-compose.yml", '''version: "3.8"

services:
  orchestrator:
    build:
      context: .
      target: development
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
''')

# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

create_file(base / ".github/ISSUE_TEMPLATE/bug_report.yml", '''name: Bug Report
description: Create a report to help us improve
title: "[BUG] "
labels: ["bug", "triage"]
body:
  - type: textarea
    id: description
    attributes:
      label: Describe the bug
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: To Reproduce
    validations:
      required: true
  - type: input
    id: version
    attributes:
      label: Version
    validations:
      required: true
''')

create_file(base / ".github/ISSUE_TEMPLATE/feature_request.yml", '''name: Feature Request
description: Suggest an idea
title: "[FEATURE] "
labels: ["enhancement", "triage"]
body:
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution\'d like
    validations:
      required: true
''')

create_file(base / ".github/PULL_REQUEST_TEMPLATE.md", '''## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass (`make ci`)
''')

create_file(base / ".github/workflows/ci.yml", '''name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install black ruff mypy
      - run: black --check orchestrator/ tests/
      - run: ruff check orchestrator/ tests/
      - run: mypy orchestrator/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest --cov=orchestrator --cov-report=xml
      - uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.11'

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install bandit[toml] safety
      - run: bandit -r orchestrator/
      - run: safety check || true
''')

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

create_file(base / "docs/ARCHITECTURE.md", '''# Architecture Overview

## System Diagram

```
CLI → Orchestrator → API Clients → LLM Providers
         ↓
    Policy Engine
         ↓
    State Manager
```

## Components

### Engine
Core orchestration managing project lifecycle.

### Router
Intelligent model selection by task type, cost, quality.

### API Clients
Unified LLM interface with retries and tracking.

### Policy Engine
Enforces budgets, model restrictions, content rules.
''')

create_file(base / "docs/ROUTING.md", '''# Model Routing

| Task Type | Priority | Rationale |
|-----------|----------|-----------|
| CODE_GEN | DeepSeek → Minimax | Best cost/performance |
| REASONING | DeepSeek-R → GPT-4o | Complex logic |
| WRITING | GPT-4o → Gemini | Quality |

## Fallback Chains
```
DeepSeek → GPT-4o → Gemini
Kimi → GLM-4 → GPT-4o-mini
```
''')

create_file(base / "docs/POLICIES.md", '''# Policy System

## Types

### BudgetPolicy
```python
Policy(max_usd=100.0, enforcement=HARD)
```

### ModelPolicy
```python
Policy(allowed=[GPT_4O], blocked=[GPT_4O_MINI])
```

## Enforcement Modes
| Mode | Behavior |
|------|----------|
| HARD | Block on violation |
| SOFT | Warn but allow |
''')

print("\n" + "="*60)
print("✓ ALL PRODUCTION-READY FILES CREATED!")
print("="*60)
print("\nCreated:")
print("  • scripts/ (setup.sh, lint.sh, test.sh, release.sh)")
print("  • .editorconfig, LICENSE, CHANGELOG.md, docker-compose.yml")
print("  • .github/ (workflows, templates)")
print("  • docs/ (ARCHITECTURE.md, ROUTING.md, POLICIES.md)")

# Cleanup
Path(__file__).unlink()
print("\n✓ Cleanup complete")
