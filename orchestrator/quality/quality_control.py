"""
quality_control — Backward-compatibility shim
The canonical implementation lives in orchestrator/quality_control.py.
New code should import from `orchestrator.quality_control` directly.

This copy was dead (zero live callers — hunt T13; both real callers,
project_analyzer.py and project_mgmt/analyzer.py, import the root module)
and had fallen out of sync with a fix already applied to the root copy:
`_run_security_checks` silently skipped unreadable files with a bare
`except Exception: pass`, reporting a false-clean "No security issues
found" even when every file failed to read. The root copy now counts and
logs skipped files and reports them as a scan issue.

F-6: this module also hosts the deprecated ``TestRunner`` shim. The root
module's old ``TestRunner`` was renamed ``QualityCheckRunner`` (its unit-test
execution now delegates to ``infrastructure.test_runners``); ``TestRunner``
remains importable from ``orchestrator.quality_control`` (lazily re-exported
from here) so existing importers keep working.
"""

import warnings

from ..quality_control import *  # noqa: F401, F403
from ..quality_control import QualityCheckRunner


class TestRunner(QualityCheckRunner):
    """DEPRECATED alias of :class:`QualityCheckRunner` (F-6 shim)."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "TestRunner is deprecated; use orchestrator.quality_control.QualityCheckRunner "
            "(unit-test execution lives in orchestrator.infrastructure.test_runners).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
