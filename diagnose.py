#!/usr/bin/env python3
"""Diagnose the orchestrator setup."""
import sys
import os

print("=== Python Path ===")
for p in sys.path:
    print(f"  {p}")

print("\n=== Checking orchestrator module location ===")
try:
    import orchestrator
    print(f"Module location: {orchestrator.__file__}")
    
    # Check api_clients
    from orchestrator import api_clients
    print(f"api_clients location: {api_clients.__file__}")
    
    # Check for Zhipu references
    import inspect
    source = inspect.getsource(api_clients)
    if 'zhipu' in source.lower():
        print("\n⚠️  WARNING: Zhipu code FOUND in api_clients.py!")
        # Show line numbers
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            if 'zhipu' in line.lower():
                print(f"  Line {i}: {line.strip()}")
    else:
        print("\n✅ Zhipu code NOT found in api_clients.py")
        
    # Check models
    from orchestrator import models
    print(f"\nmodels location: {models.__file__}")
    
    # List all models
    from orchestrator.models import Model
    print("\n=== Available Models ===")
    for m in Model:
        print(f"  {m.value}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Environment Variables ===")
for key in ['ZHIPUAI_API_KEY', 'ZHIPU_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']:
    val = os.environ.get(key)
    if val:
        print(f"  {key}: {'*' * len(val)}")
    else:
        print(f"  {key}: NOT SET")
