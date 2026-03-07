#!/usr/bin/env python3
"""
Setup script for Orchestrator v6.0 Paradigm Optimizations
===========================================================

Run this to set up the optimized structure:
    python setup_v6_optimizations.py

What it does:
1. Creates directory structure
2. Moves files to proper locations
3. Updates __init__.py
4. Creates missing __init__.py files
5. Runs integration tests
"""

import os
import shutil
import sys

BASE_DIR = r"D:\Vibe-Coding\Ai Orchestrator"

def banner(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def step1_create_dirs():
    """Step 1: Create directory structure"""
    banner("STEP 1: Creating Directory Structure")
    
    dirs = [
        "orchestrator/dashboard_core",
        "orchestrator/unified_events",
        "orchestrator_plugins",
        "orchestrator_plugins/validators",
        "orchestrator_plugins/integrations",
        "orchestrator_plugins/dashboards",
        "orchestrator_plugins/feedback",
    ]
    
    for d in dirs:
        path = os.path.join(BASE_DIR, d)
        os.makedirs(path, exist_ok=True)
        # Create minimal __init__.py
        init_file = os.path.join(path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f'"""{d.replace("/", ".")} module."""\n')
        print(f"[OK] {d}/")

def step2_move_files():
    """Step 2: Move files to proper locations"""
    banner("STEP 2: Moving Files")
    
    moves = [
        ("orchestrator/dashboard_core_core.py", "orchestrator/dashboard_core/core.py"),
        ("orchestrator/dashboard_core_mission_control.py", "orchestrator/dashboard_core/mission_control.py"),
        ("orchestrator/unified_events_core.py", "orchestrator/unified_events/core.py"),
        ("orchestrator/orchestrator_compat_layer.py", "orchestrator/compat.py"),
        ("orchestrator_plugins_init.py", "orchestrator_plugins/__init__.py"),
        ("orchestrator_plugins_validators.py", "orchestrator_plugins/validators/__init__.py"),
        ("orchestrator_plugins_integrations.py", "orchestrator_plugins/integrations/__init__.py"),
        ("orchestrator_plugins_dashboards.py", "orchestrator_plugins/dashboards/__init__.py"),
    ]
    
    for src, dst in moves:
        src_path = os.path.join(BASE_DIR, src)
        dst_path = os.path.join(BASE_DIR, dst)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"[OK] {src} -> {dst}")
        else:
            print(f"[WARN] {src} not found (may already be moved)")

def step3_create_module_inits():
    """Step 3: Create proper __init__.py files for modules"""
    banner("STEP 3: Creating Module __init__.py Files")
    
    # dashboard_core/__init__.py
    dashboard_core_init = '''"""
Unified Dashboard Core
======================
Single dashboard core with pluggable views.
"""
from .core import (
    DashboardCore,
    get_dashboard_core,
    DashboardView,
    ViewContext,
    ViewRegistry,
    run_dashboard,
)
from .mission_control import MissionControlView

__all__ = [
    "DashboardCore",
    "get_dashboard_core",
    "DashboardView",
    "ViewContext",
    "ViewRegistry",
    "run_dashboard",
    "MissionControlView",
]
'''
    with open(os.path.join(BASE_DIR, "orchestrator/dashboard_core/__init__.py"), "w") as f:
        f.write(dashboard_core_init)
    print("[OK] orchestrator/dashboard_core/__init__.py")
    
    # unified_events/__init__.py
    unified_events_init = '''"""
Unified Events System
=====================
Single event bus for all orchestrator events.
"""
from .core import (
    UnifiedEventBus,
    get_event_bus,
    DomainEvent,
    EventType,
    ProjectStartedEvent,
    ProjectCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskProgressEvent,
    CapabilityUsedEvent,
    CapabilityCompletedEvent,
    BudgetWarningEvent,
    ModelSelectedEvent,
    FallbackTriggeredEvent,
    log_capability_use,
    set_current_project,
    get_current_project,
)

__all__ = [
    "UnifiedEventBus",
    "get_event_bus",
    "DomainEvent",
    "EventType",
    "ProjectStartedEvent",
    "ProjectCompletedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskProgressEvent",
    "CapabilityUsedEvent",
    "CapabilityCompletedEvent",
    "BudgetWarningEvent",
    "ModelSelectedEvent",
    "FallbackTriggeredEvent",
    "log_capability_use",
    "set_current_project",
    "get_current_project",
]
'''
    with open(os.path.join(BASE_DIR, "orchestrator/unified_events/__init__.py"), "w") as f:
        f.write(unified_events_init)
    print("[OK] orchestrator/unified_events/__init__.py")

def step4_update_main_init():
    """Step 4: Update main orchestrator/__init__.py"""
    banner("STEP 4: Updating Main __init__.py")
    
    # Backup original
    init_path = os.path.join(BASE_DIR, "orchestrator/__init__.py")
    backup_path = os.path.join(BASE_DIR, "orchestrator/__init__.py.v5.backup")
    
    if os.path.exists(init_path) and not os.path.exists(backup_path):
        shutil.copy(init_path, backup_path)
        print(f"[OK] Backed up original to __init__.py.v5.backup")
    
    # Read new init
    new_init_path = os.path.join(BASE_DIR, "orchestrator/__init__v2.py")
    if os.path.exists(new_init_path):
        shutil.move(new_init_path, init_path)
        print(f"[OK] Updated orchestrator/__init__.py with v6.0 exports")
    else:
        print(f"[WARN] {new_init_path} not found, skipping")

def step5_cleanup():
    """Step 5: Clean up temporary files"""
    banner("STEP 5: Cleanup")
    
    files_to_remove = [
        "create_optimization_dirs.py",
        "execute_optimization.py",
    ]
    
    for f in files_to_remove:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"[OK] Removed {f}")

def step6_test():
    """Step 6: Run integration tests"""
    banner("STEP 6: Running Integration Tests")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "test_optimization_integration.py"],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

def main():
    """Main setup flow"""
    banner("ORCHESTRATOR v6.0 PARADIGM OPTIMIZATION SETUP")
    print("\nThis will set up the optimized architecture:")
    print("  1. Unified Dashboard Core (replaces 7 dashboards)")
    print("  2. Unified Events System (replaces 4 event systems)")
    print("  3. Plugin Architecture (core + optional plugins)")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    try:
        step1_create_dirs()
        step2_move_files()
        step3_create_module_inits()
        step4_update_main_init()
        step5_cleanup()
        
        banner("SETUP COMPLETE")
        print("\n[SUCCESS] v6.0 optimizations have been set up!")
        print("\nNext steps:")
        print("  1. Review the changes")
        print("  2. Run: python test_optimization_integration.py")
        print("  3. Update your code to use new APIs")
        print("  4. See OPTIMIZATION_SETUP_GUIDE.md for details")
        
        # Ask if user wants to run tests
        print("\nRun integration tests now? (y/n)")
        if input().lower() == 'y':
            exit_code = step6_test()
            sys.exit(exit_code)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
