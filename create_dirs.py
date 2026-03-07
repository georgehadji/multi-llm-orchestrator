#!/usr/bin/env python3
"""Create directory structure for orchestrator."""

import os

# Define the directories to create
DIRECTORIES = [
    'orchestrator/dashboard_core/',
    'orchestrator/unified_events/',
    'orchestrator_plugins/validators/',
    'orchestrator_plugins/integrations/',
    'orchestrator_plugins/dashboards/',
    'orchestrator_plugins/feedback/'
]

def create_directories():
    """Create all directories and confirm creation."""
    base_path = r'D:\Vibe-Coding\Ai Orchestrator'
    
    print('=' * 60)
    print('Creating directories...')
    print('=' * 60)
    
    all_success = True
    for dir_path in DIRECTORIES:
        full_path = os.path.join(base_path, dir_path)
        try:
            os.makedirs(full_path, exist_ok=True)
            if os.path.isdir(full_path):
                print(f'✓ Created: {dir_path}')
            else:
                print(f'✗ Failed: {dir_path}')
                all_success = False
        except Exception as e:
            print(f'✗ Error creating {dir_path}: {e}')
            all_success = False
    
    print()
    print('=' * 60)
    print('Final Verification:')
    print('=' * 60)
    
    for dir_path in DIRECTORIES:
        full_path = os.path.join(base_path, dir_path)
        exists = os.path.isdir(full_path)
        status = '✓ EXISTS' if exists else '✗ MISSING'
        print(f'{status}: {dir_path}')
    
    print()
    if all_success:
        print('All directories created successfully!')
    else:
        print('Some directories could not be created.')
    
    return all_success

if __name__ == '__main__':
    create_directories()
