# Contributing to Multi-LLM Orchestrator

Thank you for your interest in contributing! This document provides guidelines for development.

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/multi-llm-orchestrator.git
cd multi-llm-orchestrator

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install with dev dependencies
pip install -e ".[dev,security]"
pre-commit install

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Code Quality Standards

### Type Hints
- All functions must have type annotations
- Use `from __future__ import annotations` for Python 3.10+ features
- Run `make type-check` to verify with mypy

### Code Style
- **Black**: Line length 100, Python 3.10+ target
- **Ruff**: Fast linting with selected rules
- Run `make format` before committing

### Testing
- Write tests for new features
- Maintain >70% coverage
- Run `make test` before submitting PR

```python
# Example test
def test_feature():
    """Test description following Google style."""
    result = my_feature()
    assert result == expected
```

### Error Handling

Use the exception hierarchy:

```python
from orchestrator.exceptions import ModelUnavailableError, TaskTimeoutError

try:
    result = await execute_task(task)
except ModelUnavailableError as e:
    logger.warning(f"Model unavailable: {e.model}")
    # Retry with fallback
except TaskTimeoutError as e:
    logger.error(f"Task timed out: {e.task_id}")
    # Handle timeout
```

### Logging

Use structured logging:

```python
from orchestrator.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

# Set correlation ID at entry point
set_correlation_id("req-123")

# Log with extra fields
logger.info("Processing", extra={"task_id": task.id})
```

## Submitting Changes

1. **Branch**: Create a feature branch (`git checkout -b feature/amazing`)
2. **Commit**: Use clear commit messages
3. **Test**: Ensure all tests pass (`make ci`)
4. **Push**: Push to your fork
5. **PR**: Open a pull request with description

## CI Checks

All PRs must pass:
- ✅ Black formatting
- ✅ Ruff linting
- ✅ MyPy type checking
- ✅ pytest test suite
- ✅ Bandit security scan

## Code Review Process

- All changes require review
- Address feedback promptly
- Maintain constructive discussion

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
