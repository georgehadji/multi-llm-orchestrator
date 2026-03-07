#!/usr/bin/env python3
"""Compare two AI Orchestrator project directories."""

import os
import json
from datetime import datetime
from pathlib import Path

def scan_directory(path, base_name):
    """Scan a directory and return file information."""
    files = {}
    path_obj = Path(path)
    
    if not path_obj.exists():
        print(f"WARNING: {path} does not exist!")
        return files
    
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
                print(f"Error processing {full_path}: {e}")
    
    print(f"Scanned {len(files)} files in {base_name}")
    return files

def compare_directories(d_files, e_files):
    """Compare two directory scans and return differences."""
    
    d_paths = set(d_files.keys())
    e_paths = set(e_files.keys())
    
    only_in_d = sorted(d_paths - e_paths)
    only_in_e = sorted(e_paths - d_paths)
    in_both = sorted(d_paths & e_paths)
    
    # Find files with differences
    different_files = []
    for path in in_both:
        d_info = d_files[path]
        e_info = e_files[path]
        
        if d_info['size'] != e_info['size'] or d_info['mtime'] != e_info['mtime']:
            different_files.append({
                'path': path,
                'd_size': d_info['size'],
                'e_size': e_info['size'],
                'd_mtime': d_info['mtime_str'],
                'e_mtime': e_info['mtime_str'],
                'size_diff': d_info['size'] - e_info['size']
            })
    
    return only_in_d, only_in_e, different_files

def main():
    # Paths to compare
    d_path = r'D:\Vibe-Coding\Ai Orchestrator'
    e_path = r'E:\Documents\Vibe-Coding\Ai Orchestrator'
    
    print("=" * 80)
    print("AI ORCHESTRATOR PROJECT - DIRECTORY COMPARISON")
    print("=" * 80)
    print(f"D: {d_path}")
    print(f"E: {e_path}")
    print()
    
    # Scan both directories
    print("Scanning D: drive...")
    d_files = scan_directory(d_path, "D:")
    
    print("Scanning E: drive...")
    e_files = scan_directory(e_path, "E:")
    
    # Compare
    only_in_d, only_in_e, different_files = compare_directories(d_files, e_files)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Files only in D:
    report.append("-" * 80)
    report.append(f"FILES ONLY IN D: ({len(only_in_d)} files)")
    report.append("-" * 80)
    for f in only_in_d:
        info = d_files[f]
        report.append(f"  {f}")
        report.append(f"    Size: {info['size']:,} bytes | Modified: {info['mtime_str']}")
    report.append("")
    
    # Files only in E:
    report.append("-" * 80)
    report.append(f"FILES ONLY IN E: ({len(only_in_e)} files)")
    report.append("-" * 80)
    for f in only_in_e:
        info = e_files[f]
        report.append(f"  {f}")
        report.append(f"    Size: {info['size']:,} bytes | Modified: {info['mtime_str']}")
    report.append("")
    
    # Files with differences
    report.append("-" * 80)
    report.append(f"FILES WITH DIFFERENCES (size/modification time) - {len(different_files)} files")
    report.append("-" * 80)
    for diff in different_files:
        report.append(f"  {diff['path']}")
        report.append(f"    D: Size={diff['d_size']:,} bytes, Modified={diff['d_mtime']}")
        report.append(f"    E: Size={diff['e_size']:,} bytes, Modified={diff['e_mtime']}")
        if diff['size_diff'] != 0:
            sign = "+" if diff['size_diff'] > 0 else ""
            report.append(f"    Size difference: {sign}{diff['size_diff']:,} bytes")
        report.append("")
    
    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append(f"Total files in D: {len(d_files)}")
    report.append(f"Total files in E: {len(e_files)}")
    report.append(f"Files only in D: {len(only_in_d)}")
    report.append(f"Files only in E: {len(only_in_e)}")
    report.append(f"Files with differences: {len(different_files)}")
    report.append(f"Identical files: {len(d_paths & e_paths) - len(different_files)}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    output_path = Path("directory_comparison_report.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {output_path.absolute()}")
    
    # Save JSON for further analysis
    comparison_data = {
        'd_path': d_path,
        'e_path': e_path,
        'd_files': d_files,
        'e_files': e_files,
        'only_in_d': only_in_d,
        'only_in_e': only_in_e,
        'different_files': different_files
    }
    
    with open('directory_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Detailed data saved to: directory_comparison.json")

if __name__ == '__main__':
    main()
