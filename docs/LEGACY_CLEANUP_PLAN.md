"""
Legacy Module Cleanup Plan
==========================
Author: Georgios-Chrysovalantis Chatzivantsidis

This document identifies deprecated, duplicate, and unused modules
that should be removed or consolidated to improve maintainability.

Priority Levels:
- P0: Critical - Remove immediately (broken/unsafe)
- P1: High - Remove in next sprint (duplicate/deprecated)
- P2: Medium - Consolidate in next release
- P3: Low - Review and decide
"""

# ═══════════════════════════════════════════════════════════════════════════════
# P0: CRITICAL - Remove Immediately
# ═══════════════════════════════════════════════════════════════════════════════

P0_CRITICAL = [
    # Broken modules that could cause issues
    {
        "file": "orchestrator/output_writer_trimmed.py",
        "reason": "Trimmed/partial version of output_writer.py - confusing and unused",
        "lines": 201,
        "action": "DELETE",
    },
    {
        "file": "orchestrator/progress.py",
        "reason": "Replaced by progress_writer.py - zero test coverage",
        "lines": 79,
        "action": "DELETE",
    },
    {
        "file": "orchestrator/run_tests.py",
        "reason": "3 lines, unused wrapper",
        "lines": 3,
        "action": "DELETE",
    },
    {
        "file": "orchestrator/pricing_cache.py",
        "reason": "Unused - 0% coverage, no imports",
        "lines": 138,
        "action": "DELETE",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# P1: HIGH - Remove (Duplicates/Deprecated)
# ═══════════════════════════════════════════════════════════════════════════════

P1_HIGH = [
    # Dashboard duplicates (7 implementations - should be 1)
    {
        "file": "orchestrator/dashboard.py",
        "reason": "Original dashboard - replaced by dashboard_core/",
        "lines": "~1500",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_enhanced.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~1400",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_optimized.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~1400",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_antd.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~800",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_live.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~1100",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_mc_simple.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~200",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_mission_control.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~1400",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_mission_control_fix.py",
        "reason": "Temporary fix file - should be merged",
        "lines": "~1400",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/dashboard_real.py",
        "reason": "Duplicate dashboard implementation",
        "lines": "~1100",
        "action": "DELETE after migration",
    },
    
    # Event system duplicates (4 implementations - should be 1)
    {
        "file": "orchestrator/events.py",
        "reason": "Original events - replaced by unified_events/",
        "lines": "~500",
        "action": "DELETE after migration",
    },
    {
        "file": "orchestrator/events_proposed.py",
        "reason": "Proposed events - never implemented",
        "lines": "~300",
        "action": "DELETE",
    },
    {
        "file": "orchestrator/events_resilient.py",
        "reason": "Resilient events - merged into unified_events/",
        "lines": "~400",
        "action": "DELETE after migration",
    },
    
    # Engine duplicates
    {
        "file": "orchestrator/engine_with_events.py",
        "reason": "Transitional file - features merged into engine.py",
        "lines": "~500",
        "action": "DELETE",
    },
    
    # State management duplicates
    {
        "file": "orchestrator/state_fix_bug001.py",
        "reason": "Bug fix script - fix already applied to state.py",
        "lines": 106,
        "action": "DELETE",
    },
    
    # Cache duplicates
    {
        "file": "orchestrator/caching.py",
        "reason": "Duplicate of cache.py",
        "lines": "~300",
        "action": "DELETE after review",
    },
    
    # Other duplicates
    {
        "file": "orchestrator/progressive_output.py",
        "reason": "Replaced by progress_writer.py",
        "lines": 203,
        "action": "DELETE",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# P2: MEDIUM - Consolidate
# ═══════════════════════════════════════════════════════════════════════════════

P2_MEDIUM = [
    # Multiple small utilities that could be consolidated
    {
        "file": "orchestrator/diagnostics.py",
        "reason": "Could be merged into telemetry.py",
        "lines": "~500",
        "action": "CONSOLIDATE",
    },
    {
        "file": "orchestrator/audit.py",
        "reason": "Could be merged into telemetry.py",
        "lines": "~200",
        "action": "CONSOLIDATE",
    },
    {
        "file": "orchestrator/metrics.py",
        "reason": "Could be merged into telemetry.py",
        "lines": "~150",
        "action": "CONSOLIDATE",
    },
    
    # Plugin system duplicates
    {
        "file": "orchestrator/plugin_isolation.py",
        "reason": "Duplicate plugin isolation",
        "lines": 234,
        "action": "CONSOLIDATE",
    },
    {
        "file": "orchestrator/plugin_isolation_secure.py",
        "reason": "Duplicate plugin isolation",
        "lines": 206,
        "action": "CONSOLIDATE",
    },
    
    # Search duplicates
    {
        "file": "orchestrator/bm25_search.py",
        "reason": "Could be consolidated with nexus_search/",
        "lines": "~200",
        "action": "CONSOLIDATE",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# P3: LOW - Review and Decide
# ═══════════════════════════════════════════════════════════════════════════════

P3_LOW = [
    # External integrations (keep if used, remove if not)
    {
        "file": "orchestrator/slack_integration.py",
        "reason": "430 lines - verify if used",
        "lines": 430,
        "action": "REVIEW",
    },
    {
        "file": "orchestrator/issue_tracking.py",
        "reason": "1200+ lines - verify if used",
        "lines": 1200,
        "action": "REVIEW",
    },
    {
        "file": "orchestrator/connectors.py",
        "reason": "500+ lines - verify if used",
        "lines": 523,
        "action": "REVIEW",
    },
    
    # Experimental features
    {
        "file": "orchestrator/federated_learning.py",
        "reason": "Experimental - verify if production-ready",
        "lines": "~400",
        "action": "REVIEW",
    },
    {
        "file": "orchestrator/a2a_protocol.py",
        "reason": "A2A protocol - verify adoption",
        "lines": "~700",
        "action": "REVIEW",
    },
    
    # Legacy scaffolding
    {
        "file": "orchestrator/scaffold/",
        "reason": "Template scaffolding - verify usage",
        "lines": "~50",
        "action": "REVIEW",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Search Folder (External Libraries - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_FOLDER = {
    "path": "Search/",
    "reason": "External library code - should be moved to vendor/ or deleted",
    "action": "ARCHIVE or DELETE",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup Script
# ═══════════════════════════════════════════════════════════════════════════════

CLEANUP_SCRIPT = """
#!/bin/bash
# Legacy Module Cleanup Script
# Run with: bash cleanup_legacy.sh

set -e

echo "=== Legacy Module Cleanup ==="
echo ""

# P0: Critical - Safe to delete immediately
echo "P0: Deleting critical (broken/unused) files..."
rm -f orchestrator/output_writer_trimmed.py
rm -f orchestrator/progress.py
rm -f orchestrator/run_tests.py
rm -f orchestrator/pricing_cache.py
rm -f orchestrator/state_fix_bug001.py
rm -f orchestrator/progressive_output.py
rm -f orchestrator/events_proposed.py
rm -f orchestrator/engine_with_events.py

echo "P0 cleanup complete."
echo ""

# Note: P1 requires migration to dashboard_core/ first
echo "P1: Dashboard consolidation required first."
echo "  - Ensure all features migrated to dashboard_core/"
echo "  - Then delete duplicate dashboards"
echo ""

# Note: P2 requires careful consolidation
echo "P2: Consolidation requires manual review."
echo "  - Merge diagnostics/audit/metrics into telemetry.py"
echo "  - Consolidate plugin isolation modules"
echo ""

echo "=== Cleanup Plan Complete ==="
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARY = """
Legacy Module Cleanup Summary
=============================

Total Files Identified: 35+
Total Lines to Remove: ~15,000

By Priority:
- P0 Critical: 6 files (~600 lines) - DELETE IMMEDIATELY
- P1 High: 18 files (~10,000 lines) - DELETE after migration
- P2 Medium: 6 files (~1,500 lines) - CONSOLIDATE
- P3 Low: 5 files (~2,500 lines) - REVIEW

Expected Benefits:
1. Reduced cognitive load (fewer files to navigate)
2. Clearer architecture (single source of truth)
3. Faster CI/CD (less code to test)
4. Improved coverage % (denominator decreases)
5. Easier maintenance (less dead code)

Risks:
1. Breaking changes if deprecated APIs still used
2. Migration effort for dashboard consolidation
3. Potential loss of experimental features

Mitigation:
1. Check git history for last usage
2. Create migration guide for dashboard users
3. Archive experimental features before deletion

Timeline:
- Week 1: P0 cleanup (safe deletes)
- Week 2-3: Dashboard migration to dashboard_core/
- Week 4: P1 cleanup (after migration)
- Week 5: P2 consolidation
- Week 6: P3 review and decision
"""

if __name__ == "__main__":
    print(SUMMARY)
    print("\nSee CLEANUP_SCRIPT for bash commands.")
