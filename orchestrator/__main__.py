#!/usr/bin/env python3
"""Entry point for python -m orchestrator"""

import warnings

# Suppress FutureWarning from instructor's legacy google-generativeai import.
# instructor 1.14.x still references google.generativeai internally.
# The warning message starts with \n so we need (?s) (DOTALL) for . to match it.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*google\.generativeai.*",
    category=FutureWarning,
)

from orchestrator.cli import main

if __name__ == "__main__":
    main()
