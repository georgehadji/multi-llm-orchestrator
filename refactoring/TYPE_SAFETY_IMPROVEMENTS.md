# Type Safety Improvements - Phase 4

## Status: IN PROGRESS

### Completed
- ✅ Fixed `git_integration_example.py` syntax error (unclosed string)
- ✅ Ran `ruff check --fix --unsafe-fixes`: Fixed 21,291/21,745 errors (98% fixed)
- ✅ mypy now runs without syntax errors

### Remaining Type Issues (454 ruff errors + mypy type errors)

#### High Priority (Blocking Type Safety)

1. **Missing Return Type Annotations** (~50 functions)
   - Add `-> None` to void functions
   - Add return types to functions with returns
   
2. **Missing Type Annotations for Arguments** (~30 functions)
   - Add types to function parameters
   - Especially in `exceptions.py`, `design_system.py`

3. **Untyped dict/list Generics** (~20 occurrences)
   - Change `dict` → `dict[str, Any]`
   - Change `list` → `list[str]` etc.

#### Medium Priority (Type Refinements)

4. **TYPE_CHECKING Blocks** (~60 imports)
   - Move typing-only imports to `TYPE_CHECKING` blocks
   - Use string annotations for forward references

5. **Optional/Union Types** (~40 occurrences)
   - Use `X | None` instead of `Optional[X]` (PEP 604)
   - Use `X | Y` instead of `Union[X, Y]`

### Action Plan

#### Step 1: Fix Critical Files (1-2 hours)
```bash
# Fix exceptions.py
# Fix design_system.py
# Fix token_optimizer.py
# Fix website_validator.py
```

#### Step 2: Add TYPE_CHECKING Blocks (1 hour)
```python
# Before
from typing import TYPE_CHECKING, Optional
from .models import Task, TaskResult  # Only used in type hints

# After
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Task, TaskResult
```

#### Step 3: Update Generic Types (30 mins)
```python
# Before
def process(data: dict) -> dict:
    return {}

# After
def process(data: dict[str, Any]) -> dict[str, Any]:
    return {}
```

#### Step 4: Configure mypy Strictness
Update `pyproject.toml`:
```toml
[tool.mypy]
# Current: strict = true (but many ignores)
# Target: Gradually enable strict checks

# Temporary relaxations
disallow_untyped_defs = false  # Enable gradually
check_untyped_defs = true
warn_return_any = true

# Keep strict
warn_redundant_casts = true
warn_unused_ignores = true
```

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Ruff errors | 454 | < 50 |
| mypy errors | ~200 | < 50 |
| Untyped functions | ~100 | < 20 |
| TYPE_CHECKING usage | Low | High |

### Files Needing Attention

Critical (most errors):
1. `orchestrator/design_system.py` - 7 errors
2. `orchestrator/exceptions.py` - 9 errors  
3. `orchestrator/token_optimizer.py` - 7 errors
4. `orchestrator/website_validator.py` - 4 errors
5. `orchestrator/indesign_plugin_rules.py` - 6 errors
6. `orchestrator/frontend_rules.py` - 5 errors
7. `orchestrator/native_features.py` - 7 errors

### Commands

```bash
# Check current status
ruff check orchestrator --statistics
mypy orchestrator --ignore-missing-imports

# After fixes, verify
ruff check orchestrator
mypy orchestrator --ignore-missing-imports --no-error-summary
```

### Notes

- Priority is on **core engine modules** first
- Plugin rules files can have relaxed typing (generated code)
- Test files don't need strict typing (already configured in mypy)
