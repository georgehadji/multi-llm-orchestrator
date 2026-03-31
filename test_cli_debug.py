#!/usr/bin/env python3
"""Debug script to test CLI execution."""

import sys
import time

print("=" * 60, file=sys.stderr)
print("DEBUG: Script started", file=sys.stderr)
print("=" * 60, file=sys.stderr)
sys.stderr.flush()

# Mock sys.argv
sys.argv = ["orchestrator", "--file", "projects/frontend_react_realtime.yaml"]

print("DEBUG: About to import main", file=sys.stderr)
sys.stderr.flush()

try:
    from orchestrator.cli import main
    print("DEBUG: main imported successfully", file=sys.stderr)
    sys.stderr.flush()

    print("DEBUG: Calling main()", file=sys.stderr)
    sys.stderr.flush()

    start = time.time()
    main()
    elapsed = time.time() - start

    print(f"DEBUG: main() completed in {elapsed:.1f}s", file=sys.stderr)
    sys.stderr.flush()
except Exception as e:
    print(f"DEBUG: Exception in main(): {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

print("DEBUG: Script finished", file=sys.stderr)
sys.stderr.flush()
