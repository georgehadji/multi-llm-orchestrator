#!/usr/bin/env python3
"""Fix logging.py name conflict by renaming to log_config.py"""
import os
import shutil

orch_dir = os.path.join(os.path.dirname(__file__), 'orchestrator')

# 1. Rename logging.py to log_config.py
old_path = os.path.join(orch_dir, 'logging.py')
new_path = os.path.join(orch_dir, 'log_config.py')

if os.path.exists(old_path):
    shutil.move(old_path, new_path)
    print(f"✅ Renamed: logging.py -> log_config.py")
else:
    print(f"⚠️ logging.py not found at {old_path}")

# 2. Update all imports in all Python files
files_to_update = [
    'architecture_rules.py',
    'dashboard_antd.py',
    'dashboard_enhanced.py',
    'dashboard_live.py',
    'dashboard_mc_simple.py',
    'dashboard_mission_control.py',
    'diagnostics.py',
    'knowledge_base.py',
    'monitoring.py',
    'output_organizer.py',
    'performance.py',
    'product_manager.py',
    'project_analyzer.py',
    'project_manager.py',
    'quality_control.py',
    'unified_dashboard_simple.py',
    'unified_dashboard.py',
    '__init__.py',
    'log_config.py',  # Update its own docstring
]

for filename in files_to_update:
    filepath = os.path.join(orch_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace imports
        new_content = content.replace('from .logging import', 'from .log_config import')
        new_content = new_content.replace('from orchestrator.logging import', 'from orchestrator.log_config import')
        
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Updated: {filename}")
        else:
            print(f"⏩ No changes: {filename}")
    else:
        print(f"⚠️ Not found: {filename}")

print("\n🎉 Done! Logging conflict fixed.")
