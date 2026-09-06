"""ComponentLibrary — re-export shim from root.

Was a byte-identical 879-line duplicate of ``orchestrator/component_library.py``
exposed through ``design/__init__.py``'s wildcard, so a fix landing on either
copy would silently not reach the other (hunt T17).
"""

from ..component_library import *  # noqa: F401, F403
