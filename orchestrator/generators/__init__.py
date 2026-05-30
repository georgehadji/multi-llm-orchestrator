"""Generators package.

Individual generators are importable directly:
    from orchestrator.generators.database_generator import DatabaseGenerator
    from orchestrator.generators.secrets_manager import SecretsManager

The root-level shims (e.g. orchestrator/secrets_manager.py) re-export from here.
Wildcard imports are intentionally NOT done in this __init__ because several
generator modules have optional or broken transitive dependencies that would
cause ImportError on any import of the generators package.
"""
