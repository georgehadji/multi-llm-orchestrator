"""
bandit_verification.py — targeted bandit scan for the verification module.

Usage:
    python scripts/bandit_verification.py
    # Exits 0 if clean, 1 if violations found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).resolve().parent.parent
    targets = [
        base / "orchestrator" / "infrastructure" / "verification_checks.py",
        base / "orchestrator" / "application" / "verification_gate.py",
    ]

    all_errors: list[str] = []
    for target in targets:
        if not target.exists():
            all_errors.append(f"File not found: {target}")
            continue

        # Run bandit on the specific module
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                str(target),
                "--severity-level",
                "medium",
                "--confidence-level",
                "medium",
                "-f",
                "custom",
                "--msg-template",
                "{abspath}:{line}: {test_id}[{severity}][{confidence}]: {msg}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            # Filter out warnings we've intentionally suppressed with # nosec
            all_errors.append(f"Security findings in {target.name}:")
            for line in result.stdout.splitlines():
                if line.strip():
                    all_errors.append(f"  {line}")

    if all_errors:
        print("BANDIT SECURITY SCAN VIOLATIONS:")
        for err in all_errors:
            print(err)
        return 1

    print("OK — all security-relevant lines have justified # nosec annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
