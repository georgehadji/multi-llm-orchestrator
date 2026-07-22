#!/usr/bin/env python3
"""Check that every test file has at least one pytest marker."""

from __future__ import annotations

import os
import re
import sys

TEST_DIR = "tests"


def main() -> int:
    missing: list[str] = []
    for dirpath, _, files in os.walk(TEST_DIR):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if not fn.startswith("test_") or not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if not re.search(r"@pytest\.mark\.\w+|pytestmark\s*=", content):
                missing.append(os.path.relpath(fpath).replace(os.sep, "/"))

    if missing:
        print(f"FAIL: {len(missing)} test files missing pytest markers:")
        for m in missing:
            print(f"  {m}")
        return 1

    print("OK: all test files have pytest markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
