#!/usr/bin/env python3
"""Test assembler import."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

try:
    from orchestrator.project_assembler import ProjectAssembler
    print("ProjectAssembler imported successfully")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
