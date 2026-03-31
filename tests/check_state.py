#!/usr/bin/env python3
"""Check state files."""

from pathlib import Path
import json

state_dir = Path(".orchestrator/state")
print("State files:")
for f in sorted(state_dir.glob("*.json")):
    print(f"  {f.name}")
