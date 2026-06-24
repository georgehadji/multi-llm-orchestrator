"""
Model Registry — Backward-compatibility shim
===============================================
The canonical ModelRegistry now lives in orchestrator/domain/model_registry.py.
"""

from .domain.model_registry import ModelConfig  # noqa: F401
from .domain.model_registry import ModelRegistry  # noqa: F401
from .domain.model_registry import get_timeout  # noqa: F401
from .domain.model_registry import get_cost  # noqa: F401
from .domain.model_registry import get_max_tokens  # noqa: F401
from .domain.model_registry import is_valid_model  # noqa: F401
from .domain.model_registry import get_replacement  # noqa: F401
from .domain.model_registry import migrate_deprecated_models  # noqa: F401

__all__ = [
    "ModelConfig",
    "ModelRegistry",
    "get_timeout",
    "get_cost",
    "get_max_tokens",
    "is_valid_model",
    "get_replacement",
    "migrate_deprecated_models",
]
