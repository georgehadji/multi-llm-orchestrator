#!/usr/bin/env python3
"""Entry point for python -m orchestrator"""

import warnings

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
