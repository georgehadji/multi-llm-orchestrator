"""tomllib for every supported Python: stdlib on 3.11+, tomli on 3.10.

pyproject.toml declares ``requires-python = ">=3.10"`` and a ``tomli`` fallback
dependency for python_version < 3.11. Import ``tomllib`` from here, never from
the stdlib or tomli directly, so the version gate lives in one place
(tests/unit/test_hunt_t0_census.py enforces this).
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["tomllib"]
