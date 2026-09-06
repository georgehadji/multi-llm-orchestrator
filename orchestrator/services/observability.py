"""ObservabilityService — re-export shim from application/ (canonical
location per services/__init__.py; see services/executor.py for why this
submodule-level shim matters, not just the package's own re-export)."""

from ..application.observability import *  # noqa: F401, F403
