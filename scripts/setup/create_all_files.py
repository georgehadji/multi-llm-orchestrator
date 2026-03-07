"""Create all remaining project files."""
import os
import stat
from pathlib import Path

print("Creating production-ready project files...\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════
scripts_dir = Path("scripts")
scripts_dir.mkdir(exist_ok=True)

SCRIPTS = {
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

pytest -v "$@"
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

# Get and bump version
CURRENT=$(grep '__version__' orchestrator/__init__.py | head -1 | grep -o '".*"' | tr -d '"')
IFS='.' read -r major minor patch <<< "$CURRENT"
case $VERSION_TYPE in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
esac
NEW="$major.$minor.$patch"

echo "Bumping version: $CURRENT → $NEW"

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

for filename, content in SCRIPTS.items():
    filepath = scripts_dir / filename
    filepath.write_text(content, encoding='utf-8')
    # Make executable
    os.chmod(filepath, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    print(f"✓ scripts/{filename}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIG FILES
# ═══════════════════════════════════════════════════════════════════════════════

# .editorconfig
EDITORCONFIG = '''# EditorConfig
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
'''

Path(".editorconfig").write_text(EDITORCONFIG, encoding='utf-8')
print("✓ .editorconfig")

# LICENSE
LICENSE = '''MIT License

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
'''

Path("LICENSE").write_text(LICENSE, encoding='utf-8')
print("✓ LICENSE")

# CHANGELOG.md
CHANGELOG = '''# Changelog

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
- Development scripts (setup.sh, lint.sh, test.sh, release.sh)
- Comprehensive Makefile with all development tasks

## [1.1.0] - 2024-01-15

### Added
- Multi-provider LLM support (DeepSeek, OpenAI, Google, Kimi, Minimax, Zhipu)
- Cost-optimized routing with budget hierarchy
- Policy enforcement system
- Resume capability for interrupted projects
- Deterministic validation
- Real-time telemetry and OpenTracing support

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic orchestration engine
'''

Path("CHANGELOG.md").write_text(CHANGELOG, encoding='utf-8')
print("✓ CHANGELOG.md")

# docker-compose.yml
DOCKER_COMPOSE = '''version: "3.8"

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
    command: redis-server --appendonly yes

volumes:
  redis_data:
'''

Path("docker-compose.yml").write_text(DOCKER_COMPOSE, encoding='utf-8')
print("✓ docker-compose.yml")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GITHUB TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# Create directories
github_dir = Path(".github")
issue_dir = github_dir / "ISSUE_TEMPLATE"
issue_dir.mkdir(parents=True, exist_ok=True)

# Bug report
BUG_REPORT = '''name: Bug Report
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
'''

(issue_dir / "bug_report.yml").write_text(BUG_REPORT, encoding='utf-8')
print("✓ .github/ISSUE_TEMPLATE/bug_report.yml")

# Feature request
FEATURE_REQUEST = '''name: Feature Request
description: Suggest an idea
title: "[FEATURE] "
labels: ["enhancement", "triage"]
body:
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
    validations:
      required: true
'''

(issue_dir / "feature_request.yml").write_text(FEATURE_REQUEST, encoding='utf-8')
print("✓ .github/ISSUE_TEMPLATE/feature_request.yml")

# PR template
PR_TEMPLATE = '''## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] All tests pass (`make ci`)
'''

(github_dir / "PULL_REQUEST_TEMPLATE.md").write_text(PR_TEMPLATE, encoding='utf-8')
print("✓ .github/PULL_REQUEST_TEMPLATE.md")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)

# ARCHITECTURE.md
ARCHITECTURE = '''# Architecture Overview

## System Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│     CLI     │────▶│  Orchestrator│────▶│  API Clients │
└─────────────┘     └──────────────┘     └──────────────┘
                             │                     │
                             ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │Policy Engine │     │LLM Providers │
                    └──────────────┘     └──────────────┘
```

## Components

### Engine (`orchestrator.engine`)
Core orchestration logic managing project lifecycle.

### Router (`orchestrator.models`)
Intelligent model selection based on task type, cost, quality.

### API Clients (`orchestrator.api_clients`)
Unified interface to all LLM providers with retries and tracking.

### Policy Engine (`orchestrator.policy_engine`)
Enforces governance: budgets, model restrictions, content filtering.

## Data Flow

1. CLI receives project spec
2. Engine decomposes into tasks
3. Router selects optimal model
4. API Client executes call
5. Response validated and stored
6. Output generated
'''

(docs_dir / "ARCHITECTURE.md").write_text(ARCHITECTURE, encoding='utf-8')
print("✓ docs/ARCHITECTURE.md")

# ROUTING.md
ROUTING = '''# Model Routing

## Routing Table

| Task Type | Priority Order | Rationale |
|-----------|---------------|-----------|
| CODE_GEN | DeepSeek → Minimax → Kimi | Best cost/performance |
| REASONING | DeepSeek-R → GPT-4o → Gemini | Complex logic |
| WRITING | GPT-4o → Gemini → Kimi | Quality |
| DATA_EXTRACT | GPT-4o-mini → GPT-4o | Structured output |

## Fallback Chains

```
DeepSeek → GPT-4o → Gemini
Kimi → GLM-4 → GPT-4o-mini
```

## Cost Points (per 1M tokens)

| Provider | Input | Output |
|----------|-------|--------|
| DeepSeek | $0.27 | $1.10 |
| Kimi | $0.14 | $0.56 |
| GPT-4o | $2.50 | $10.00 |
'''

(docs_dir / "ROUTING.md").write_text(ROUTING, encoding='utf-8')
print("✓ docs/ROUTING.md")

# POLICIES.md
POLICIES = '''# Policy System

## Policy Types

### BudgetPolicy
```python
Policy(
    name="monthly_budget",
    max_usd=100.0,
    enforcement=EnforcementMode.HARD
)
```

### ModelPolicy
```python
Policy(
    name="approved_models",
    allowed_models=[Model.GPT_4O],
    blocked_models=[Model.GPT_4O_MINI]
)
```

## Enforcement Modes

| Mode | Behavior |
|------|----------|
| HARD | Block on violation |
| SOFT | Warn but allow |
| MONITOR | Log only |
'''

(docs_dir / "POLICIES.md").write_text(POLICIES, encoding='utf-8')
print("✓ docs/POLICIES.md")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GITHUB WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

workflow_dir = github_dir / "workflows"
workflow_dir.mkdir(exist_ok=True)

WORKFLOW = '''name: CI/CD

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
          cache: 'pip'
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
          cache: 'pip'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=orchestrator --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

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

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build
'''

(workflow_dir / "ci.yml").write_text(WORKFLOW, encoding='utf-8')
print("✓ .github/workflows/ci.yml")

print("\n" + "="*60)
print("✓ ALL PRODUCTION-READY FILES CREATED!")
print("="*60)
print("\nNew files:")
print("  • scripts/ (setup.sh, lint.sh, test.sh, release.sh)")
print("  • .editorconfig")
print("  • LICENSE")
print("  • CHANGELOG.md")
print("  • docker-compose.yml")
print("  • .github/ (workflows, issue templates, PR template)")
print("  • docs/ (ARCHITECTURE.md, ROUTING.md, POLICIES.md)")

# Self cleanup
Path(__file__).unlink()
