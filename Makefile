# Multi-LLM Orchestrator - Development Makefile
# =============================================

.PHONY: help install install-dev test test-unit test-integration lint format type-check security-check clean docker-build docker-run ci

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Multi-LLM Orchestrator - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install package with dependencies"
	@echo "  install-dev       Install with development dependencies"
	@echo "  install-all       Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run               Run the CLI"
	@echo "  test              Run all tests with coverage"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-fast         Run tests in parallel (faster)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run linters (ruff)"
	@echo "  lint-fix          Run linters with auto-fix"
	@echo "  format            Format code (black)"
	@echo "  format-check      Check code formatting"
	@echo "  type-check        Run type checker (mypy)"
	@echo "  security-check    Run security scans (bandit, safety)"
	@echo "  precommit         Install and run pre-commit hooks"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean             Remove build artifacts"
	@echo "  clean-all         Remove all generated files"
	@echo "  update-deps       Update dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-run        Run Docker container"
	@echo "  docker-test       Run tests in Docker"
	@echo ""
	@echo "CI:"
	@echo "  ci                Run all CI checks locally"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════════

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,security]"
	pre-commit install

install-all:
	pip install -e ".[dev,security,tracing,docs]"
	pre-commit install

update-deps:
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,security,tracing,docs]"

# ═══════════════════════════════════════════════════════════════════════════════
# Development
# ═══════════════════════════════════════════════════════════════════════════════

run:
	python -m orchestrator --help

test:
	pytest -xvs

test-unit:
	pytest tests/unit -xvs -m unit

test-integration:
	pytest tests/integration -xvs -m integration --ignore=tests/integration/test_api_clients.py

test-fast:
	pytest -x --forked -n auto

test-cov:
	pytest --cov=orchestrator --cov-report=html --cov-report=term

test-ci:
	pytest --cov=orchestrator --cov-report=xml --cov-fail-under=70

# ═══════════════════════════════════════════════════════════════════════════════
# Code Quality
# ═══════════════════════════════════════════════════════════════════════════════

lint:
	ruff check orchestrator/ tests/

lint-fix:
	ruff check --fix orchestrator/ tests/

format:
	black orchestrator/ tests/

format-check:
	black --check orchestrator/ tests/

type-check:
	mypy orchestrator/

security-check:
	@echo "Running Bandit security scan..."
	bandit -r orchestrator/ -f json -o bandit-report.json || true
	bandit -r orchestrator/
	@echo ""
	@echo "Running Safety check..."
	safety check || true

precommit:
	pre-commit install
	pre-commit run --all-files

# ═══════════════════════════════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf coverage_html/ bandit-report.json .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf outputs/ results/ logs/
	find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════════════════════

docker-build:
	docker build -t multi-llm-orchestrator:latest .

docker-run:
	docker run --rm -it --env-file .env multi-llm-orchestrator:latest

docker-test:
	docker build --target test -t multi-llm-orchestrator:test .
	docker run --rm multi-llm-orchestrator:test

# ═══════════════════════════════════════════════════════════════════════════════
# CI - Run all checks
# ═══════════════════════════════════════════════════════════════════════════════

ci: format-check lint type-check test-ci security-check
	@echo ""
	@echo "============================================"
	@echo "✓ All CI checks passed!"
	@echo "============================================"

# Convenience aliases
fmt: format
check: ci
