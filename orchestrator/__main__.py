#!/usr/bin/env python3
"""Entry point for python -m orchestrator"""

import warnings

# Suppress FutureWarning from instructor's legacy google-generativeai import.
# instructor 1.14.x still references google.generativeai internally; the warning
# is noise until instructor ships a release that targets google.genai instead.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"instructor\..*",
)
# Also suppress the google package's own emission of this warning
warnings.filterwarnings(
    "ignore",
    message=r".*google\.generativeai.*",
    category=FutureWarning,
)

from orchestrator.cli import main

if __name__ == "__main__":
    main()
