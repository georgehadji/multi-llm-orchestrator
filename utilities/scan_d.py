#!/usr/bin/env python3
"""Scan D: drive directory."""
import os
import json
from pathlib import Path
from datetime import datetime

def scan_directory(path):
    files = {}
    path_obj = Path(path)
    
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            full_path = Path(root) / filename
            try:
                rel_path = full_path.relative_to(path_obj)
                rel_path_str = str(rel_path).replace('\\', '/')
                stat = full_path.stat()
                files[rel_path_str] = {
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                print(f"Error: {full_path} - {e}")
    
    return files

if __name__ == '__main__':
    d_path = r'D:\Vibe-Coding\Ai Orchestrator'
    print(f"Scanning {d_path}...")
    d_files = scan_directory(d_path)
    print(f"Found {len(d_files)} files")
    
    with open('d_files.json', 'w', encoding='utf-8') as f:
        json.dump(d_files, f, indent=2)
    print(f"Saved to d_files.json")
