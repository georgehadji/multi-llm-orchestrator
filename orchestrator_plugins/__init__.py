"""
Orchestrator Official Plugins — Package Init
============================================
Save this as orchestrator_plugins/__init__.py
"""

__version__ = "1.0.0"

# Try to import optional subpackages
try:
    from . import validators
    HAS_VALIDATORS = True
except ImportError:
    HAS_VALIDATORS = False

try:
    from . import integrations
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False

try:
    from . import dashboards
    HAS_DASHBOARDS = True
except ImportError:
    HAS_DASHBOARDS = False

try:
    from . import feedback
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False


def get_available_plugins() -> dict:
    """List all available plugin categories."""
    return {
        "validators": HAS_VALIDATORS,
        "integrations": HAS_INTEGRATIONS,
        "dashboards": HAS_DASHBOARDS,
        "feedback": HAS_FEEDBACK,
    }
