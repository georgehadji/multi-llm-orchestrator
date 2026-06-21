"""CLI command handler modules — extracted from orchestrator/cli.py.

Each module exposes an ``execute(args)`` function that the argparse
dispatcher in cli.py delegates to.
"""

# Backward-compat re-exports — pre-existing from before Phase 2 extraction
from .center import *  # noqa: F401, F403
from .integration import *  # noqa: F401, F403
from .server import *  # noqa: F401, F403
from .registry import *  # noqa: F401, F403
