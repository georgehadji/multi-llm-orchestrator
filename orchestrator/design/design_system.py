"""DesignSystem — re-export shim from root.

This copy was a stale 155-line fork of the 310-line canonical
``orchestrator/design_system.py``, missing the ``tone``/``font_heading``/
``font_body``/``accessibility`` fields and the ``__post_init__`` that
materialises ``spacing``/``shadow``/``animation``/``border_radius`` — all of
which ``website_generator.py`` formats into its output. Every real consumer
imports the root module, but this fork was exposed through
``design/__init__.py``'s wildcard, so ``from orchestrator.design import
DesignSystem`` handed out a class that raises AttributeError on those
attributes (hunt T17).
"""

from ..design_system import *  # noqa: F401, F403
