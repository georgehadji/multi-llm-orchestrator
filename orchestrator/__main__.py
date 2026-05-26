#!/usr/bin/env python3
"""Entry point for python -m orchestrator"""

import sys
import warnings

# Windows consoles default to a locale-specific encoding (e.g. cp1253) that
# cannot represent Unicode box-drawing characters and emoji used in dashboard
# and plan rendering.  Reconfigure stdout/stderr to UTF-8 with replacement
# so any non-encodable character becomes "?" rather than crashing.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress FutureWarning from instructor's legacy google-generativeai import.
# (?s) flag required: the warning message starts with \n so . must match newlines.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*google\.generativeai.*",
    category=FutureWarning,
)

from orchestrator.cli import main

if __name__ == "__main__":
    main()
