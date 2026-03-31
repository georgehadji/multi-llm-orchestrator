#!/usr/bin/env python3
"""Debug script to test which import hangs."""

import sys

print("1. Testing load_dotenv", file=sys.stderr, flush=True)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("   ✓ load_dotenv works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ load_dotenv failed: {e}", file=sys.stderr, flush=True)

print("2. Testing assembler import", file=sys.stderr, flush=True)
try:
    from orchestrator.assembler import assemble_project
    print("   ✓ assembler import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ assembler import failed: {e}", file=sys.stderr, flush=True)

print("3. Testing engine import", file=sys.stderr, flush=True)
try:
    from orchestrator.engine import Orchestrator
    print("   ✓ engine import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ engine import failed: {e}", file=sys.stderr, flush=True)

print("4. Testing models import", file=sys.stderr, flush=True)
try:
    from orchestrator.models import Budget
    print("   ✓ models import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ models import failed: {e}", file=sys.stderr, flush=True)

print("5. Testing output_organizer import", file=sys.stderr, flush=True)
try:
    from orchestrator.output_organizer import (
        organize_project_output,
        suppress_cache_messages,
    )
    print("   ✓ output_organizer import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ output_organizer import failed: {e}", file=sys.stderr, flush=True)

print("6. Testing output_writer import", file=sys.stderr, flush=True)
try:
    from orchestrator.output_writer import write_output_dir
    print("   ✓ output_writer import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ output_writer import failed: {e}", file=sys.stderr, flush=True)

print("7. Testing progress import", file=sys.stderr, flush=True)
try:
    from orchestrator.progress import ProgressRenderer
    print("   ✓ progress import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ progress import failed: {e}", file=sys.stderr, flush=True)

print("8. Testing project_file import", file=sys.stderr, flush=True)
try:
    from orchestrator.project_file import load_project_file
    print("   ✓ project_file import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ project_file import failed: {e}", file=sys.stderr, flush=True)

print("9. Testing run_tests import", file=sys.stderr, flush=True)
try:
    from orchestrator.run_tests import run_project_tests
    print("   ✓ run_tests import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ run_tests import failed: {e}", file=sys.stderr, flush=True)

print("10. Testing state import", file=sys.stderr, flush=True)
try:
    from orchestrator.state import StateManager
    print("   ✓ state import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ state import failed: {e}", file=sys.stderr, flush=True)

print("11. Testing tracing import", file=sys.stderr, flush=True)
try:
    from orchestrator.tracing import TracingConfig
    print("   ✓ tracing import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ tracing import failed: {e}", file=sys.stderr, flush=True)

print("12. Testing unified_events import", file=sys.stderr, flush=True)
try:
    from orchestrator.unified_events import ProjectCompletedEvent as _ProjectCompleted
    print("   ✓ unified_events import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ unified_events import failed: {e}", file=sys.stderr, flush=True)

print("13. Testing resume_detector import", file=sys.stderr, flush=True)
try:
    from orchestrator.resume_detector import (
        ResumeCandidate,
        _extract_keywords,
        _is_exact_match,
        _recency_factor,
        _score_candidates,
    )
    print("   ✓ resume_detector import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ resume_detector import failed: {e}", file=sys.stderr, flush=True)

print("14. Testing visualization import", file=sys.stderr, flush=True)
try:
    from orchestrator.visualization import DagRenderer
    print("   ✓ visualization import works", file=sys.stderr, flush=True)
except Exception as e:
    print(f"   ✗ visualization import failed: {e}", file=sys.stderr, flush=True)

print("\n✅ All imports completed", file=sys.stderr, flush=True)
