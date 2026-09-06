"""SwiftStack integration — re-export shim from root.

This copy was byte-identical to ``orchestrator/swiftstack_integration.py`` but
sat one package deeper, so its root-relative imports (``from .api_builder
import ...``) resolved against ``orchestrator.integrations.*`` and raised
ModuleNotFoundError on every import (hunt T17).
"""

from ..swiftstack_integration import *  # noqa: F401, F403
