"""Pytest collection guard for the VS regression suite.

``test_vs_regression.py`` is a *standalone* unittest script: at import time it
replaces ``sys.modules["orchestrator"]`` (and several submodules) with stubs so
it can exercise the Verbalized Sampling code in isolation, bypassing the real
package. That global mutation runs during pytest's collection phase and poisons
``sys.modules`` for every test collected afterwards, causing spurious
"cannot import name ... (unknown location)" / "is not a package" errors across
the suite.

It is designed to run on its own (``python tests/regression/test_vs_regression.py``),
so we exclude it from normal pytest collection rather than letting it corrupt
the shared interpreter state.
"""

collect_ignore = ["test_vs_regression.py"]
