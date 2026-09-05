"""
AppVerifier — Backward-compatibility shim
===========================================
The canonical implementation is in orchestrator/app_verifier.py — the copy
that used to live here had silently diverged from it, missing a fix for a
subprocess pip-install call: verify_local() runs subprocess.run(...,
cwd=output_dir), so a requirements.txt path built relative to repo-root
would be resolved against output_dir a second time and fail to open. The
root module fixes this via str(req_file.resolve()); this copy still used
str(req_file). Because orchestrator/appbuilder/__init__.py does
`from .verifier import *` *after* `from .builder import *`, that divergence
also meant orchestrator.appbuilder.AppVerifier (the package's own public
name) resolved to this buggy class rather than the fixed one that
AppBuilder itself actually constructs — see
docs/hunts/t7-execution/inventory.md C1.
"""

from ..app_verifier import *  # noqa: F401, F403
