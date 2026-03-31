#!/usr/bin/env python3
"""Fix httpx compatibility issue by installing compatible version"""
import subprocess
import sys

print("Installing compatible httpx version...")
result = subprocess.run([
    sys.executable, "-m", "pip", "install", "httpx>=0.24.0,<0.28.0", "--quiet"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ httpx installed successfully")
    print("Now run: python -m orchestrator --project \"Your project\"")
else:
    print(f"❌ Installation failed: {result.stderr}")
