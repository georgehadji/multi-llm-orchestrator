"""
product_manager — Backward-compatibility shim
The canonical implementation lives in orchestrator/product/product_manager.py.
New code should import from `orchestrator.product.product_manager` directly.
"""

from .product.product_manager import *  # noqa: F401, F403
