"""
CodebaseWriter — Backward-compatibility shim
==============================================
The canonical implementation lives in orchestrator/codebase/writer.py, which
also has SEARCH/REPLACE block patching (Aider-style) and a pre-destructive-
operation snapshot safety net (CodeWhale Phase 2) that this module used to
lack — the two had silently diverged. See
docs/hunts/t2-credentials/inventory.md.
"""

from .codebase.writer import *  # noqa: F401, F403
