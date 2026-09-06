"""Re-export shim — canonical source: orchestrator.secrets_generator

Previously self-referential (`from ..generators.secrets_generator import *`,
which — from within orchestrator.generators — resolves to this very module),
so `import *` picked up nothing from an empty partially-initialized module and
this file was silently empty. See docs/hunts/t2-credentials/inventory.md.
"""

from ..secrets_generator import *  # noqa: F401, F403
