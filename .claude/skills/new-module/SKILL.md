---
name: new-module
description: Scaffold a new orchestrator module following project conventions. Use when creating any new file in orchestrator/. Provides header template, logger setup, test file template, and __init__.py export instructions.
---

# New Module — AI Orchestrator

Δημιουργεί νέο module στο `orchestrator/` ακολουθώντας τα project conventions.

## Χρήση

```
/new-module <module_name> <pattern> <description>
```

Παράδειγμα: `/new-module autonomy "Configuration Object" "AutonomyLevel enum and presets"`

---

## Step 1: Δημιούργησε το αρχείο

**Mandatory header** για κάθε νέο `orchestrator/<module_name>.py`:

```python
"""
<ClassName> — <One-line description>
======================================
<Context: τι κάνει αυτό το module, γιατί υπάρχει, τι pattern ακολουθεί>

Pattern: <Strategy | Decorator | Repository | State Machine | κτλ.>
Async: <Yes — για I/O-bound | No — για pure computation>
Layer: <L1 Infrastructure | L2 Verification | L3 Agents | L4 Supervisor |
        L5 Events | L6 Observability | L7 Human Interface>

Usage:
    from orchestrator.<module_name> import <ClassName>

    obj = <ClassName>(...)
    result = obj.method(...)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("orchestrator.<module_name>")


# ── Main class ────────────────────────────────────────────────────────────────

@dataclass
class <ClassName>:
    """<One-line docstring>."""

    # fields here

    def method(self) -> ...:
        """<What it does>."""
        ...
```

---

## Step 2: Σύνδεσε με τα υπάρχοντα

**Έλεγξε dependencies:**
```python
# Typical imports (add only what's needed):
from .models import Model, TaskType, Task, TaskResult
from .policy import ModelProfile, Policy
from .cost import BudgetHierarchy, CostPredictor
```

**Circular import rule:** Μόνο `models.py` → `policy.py` → `planner.py` → `engine.py`
Το νέο module ΔΕΝ import-άρει `engine.py`.

---

## Step 3: Export από `__init__.py`

Άνοιξε `orchestrator/__init__.py` και πρόσθεσε:

```python
# Στην αντίστοιχη tier section:
try:
    from .<module_name> import <ClassName>
    HAS_<MODULE_NAME> = True
except ImportError:
    HAS_<MODULE_NAME> = False
```

Πρόσθεσε `"<ClassName>"` στο `__all__` list.

---

## Step 4: Δημιούργησε test file

**`tests/test_<module_name>.py`:**

```python
"""
Tests for <ClassName>.

Pattern: <pattern>
Async: <Yes|No>
"""
from __future__ import annotations

import pytest
# from unittest.mock import AsyncMock, MagicMock  # αν χρειάζεται


# ── Fixtures ──────────────────────────────────────────────────────────────────

# @pytest.fixture
# def instance():
#     return <ClassName>(...)


# ── Tests ─────────────────────────────────────────────────────────────────────

class Test<ClassName>:

    def test_basic_creation(self):
        """Module imports and basic instantiation works."""
        from orchestrator.<module_name> import <ClassName>
        obj = <ClassName>()
        assert obj is not None

    # def test_<method>_<scenario>(self):
    #     """<What it tests>."""
    #     ...

    # async def test_<async_method>(self):  # για async modules
    #     """<What it tests>."""
    #     ...
```

**Test conventions:**
- `tmp_path` fixture για file I/O
- `async def test_...` (no decorator — asyncio_mode=auto)
- `AsyncMock` για external API calls
- ΔΕΝ κάνουν real network calls
- `@pytest.mark.slow` για tests > 5s

---

## Step 5: Verify

```bash
# RED: Τρέξε πριν γράψεις implementation
python -m pytest tests/test_<module_name>.py -v --no-cov
# Αναμένεται: ImportError ή AssertionError

# GREEN: Τρέξε μετά την implementation
python -m pytest tests/test_<module_name>.py -v --no-cov
# Αναμένεται: PASSED

# REGRESSION CHECK:
python -m pytest tests/ -v --no-cov -m "not slow" -x

# IMPORT CHECK:
python -c "from orchestrator.<module_name> import <ClassName>; print('OK')"

# LINT:
python -m ruff check orchestrator/<module_name>.py
```

---

## Quick Reference: Pattern Templates

### Pure Dataclass (sync, domain layer)
```python
@dataclass
class Config:
    level: SomeEnum
    threshold: float = 0.7

    @classmethod
    def from_preset(cls, preset_name: str) -> "Config":
        return PRESETS[preset_name]
```

### Strategy (sync, routing/selection)
```python
class Router:
    def select(self, candidates: list[Model], context: dict) -> Model:
        ...

class CostRouter(Router):
    def select(self, candidates, context):
        return min(candidates, key=lambda m: context["costs"][m])
```

### Async Service (I/O-bound)
```python
class Service:
    def __init__(self, client: "UnifiedClient"):
        self._client = client

    async def process(self, input: str) -> str:
        try:
            result = await self._client.complete(...)
            return result
        except Exception as e:
            logger.warning("Service failed: %s — falling back", e)
            return input  # fail-open
```

### Repository (persistence)
```python
class Store:
    def __init__(self, base_dir: Path):
        self._dir = base_dir / ".orchestrator" / "store"
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: dict) -> None:
        (self._dir / f"{key}.json").write_text(json.dumps(data, indent=2))

    def load(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None
```
