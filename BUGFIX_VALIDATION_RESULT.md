# Bug Fix: ValidationResult.error_message

## Issue
`ValidationResult` class in `code_validator.py` uses `errors` (list) instead of `error_message` (string).

## Fix Applied

**File**: `orchestrator/output_writer.py`

**Before**:
```python
logger.warning(
    f"⚠️ Code validation failed for {filename}: {validation_result.error_message}"
)
```

**After**:
```python
error_msg = "; ".join(validation_result.errors) if validation_result.errors else "Unknown error"
logger.warning(
    f"⚠️ Code validation failed for {filename}: {error_msg}"
)
```

## ValidationResult Class Definition

```python
@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    syntax_valid: bool = True
    security_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dangerous_patterns_found: list[str] = field(default_factory=list)
    ast_nodes: int = 0
```

Note: Uses `errors: list[str]` not `error_message: str`.

## Related Classes (Not Affected)

These classes correctly use `error_message`:
- `TestValidationResult` (test_validator.py) - for test validation
- `ActionRecord` (accountability.py) - for action tracking
- `IntegrationHealth` (integration_circuit_breaker.py) - for health checks
- `TaskRecord` (meta_orchestrator.py) - for meta-orchestration
- `TestFailure` (test_fixer.py) - for test failures

---

**Date**: 2026-03-30  
**Fix**: Convert `errors` list to semicolon-separated string for logging
