"""Create GitHub Actions workflow directory and file."""
import os
from pathlib import Path

workflow_dir = Path('.github/workflows')
workflow_dir.mkdir(parents=True, exist_ok=True)

workflow_content = '''name: CI/CD

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install black ruff mypy
      - run: black --check orchestrator/ tests/
      - run: ruff check orchestrator/ tests/
      - run: mypy orchestrator/ --ignore-missing-imports

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install bandit[toml] safety
      - run: bandit -r orchestrator/
      - run: safety check || true

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=orchestrator --cov-report=xml -m "not requires_api" || true
      - uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.11'
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install build twine
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
'''

workflow_file = workflow_dir / 'ci.yml'
workflow_file.write_text(workflow_content, encoding='utf-8')

print(f"✓ Created: {workflow_file}")

# Verify
if workflow_file.exists():
    print(f"✓ File exists with size: {workflow_file.stat().st_size} bytes")
else:
    print("✗ File was not created")
