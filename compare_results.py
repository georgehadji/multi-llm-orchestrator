#!/usr/bin/env python3
"""Compare the scanned results."""
import json

def main():
    # Load scanned data
    try:
        with open('d_files.json', 'r', encoding='utf-8') as f:
            d_files = json.load(f)
    except FileNotFoundError:
        print("ERROR: d_files.json not found. Run scan_d.py first.")
        return
    
    try:
        with open('e_files.json', 'r', encoding='utf-8') as f:
            e_files = json.load(f)
    except FileNotFoundError:
        print("ERROR: e_files.json not found. Run scan_e.py first.")
        return
    
    d_paths = set(d_files.keys())
    e_paths = set(e_files.keys())
    
    only_in_d = sorted(d_paths - e_paths)
    only_in_e = sorted(e_paths - d_paths)
    in_both = sorted(d_paths & e_paths)
    
    # Find differences
    different = []
    for path in in_both:
        d_info = d_files[path]
        e_info = e_files[path]
        if d_info['size'] != e_info['size'] or d_info['mtime'] != e_info['mtime']:
            different.append({
                'path': path,
                'd_size': d_info['size'],
                'e_size': e_info['size'],
                'd_mtime': d_info['mtime_str'],
                'e_mtime': e_info['mtime_str'],
                'size_diff': d_info['size'] - e_info['size']
            })
    
    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("AI ORCHESTRATOR - DIRECTORY COMPARISON REPORT")
    lines.append("D: \\Vibe-Coding\Ai Orchestrator")
    lines.append("E: \\Documents\Vibe-Coding\Ai Orchestrator")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("-" * 80)
    lines.append(f"FILES ONLY IN D: ({len(only_in_d)} files)")
    lines.append("-" * 80)
    for f in only_in_d:
        info = d_files[f]
        lines.append(f"  {f}")
        lines.append(f"    Size: {info['size']:,} bytes | Modified: {info['mtime_str']}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append(f"FILES ONLY IN E: ({len(only_in_e)} files)")
    lines.append("-" * 80)
    for f in only_in_e:
        info = e_files[f]
        lines.append(f"  {f}")
        lines.append(f"    Size: {info['size']:,} bytes | Modified: {info['mtime_str']}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append(f"FILES WITH DIFFERENCES ({len(different)} files)")
    lines.append("-" * 80)
    for diff in different:
        lines.append(f"  {diff['path']}")
        lines.append(f"    D: Size={diff['d_size']:,}, Modified={diff['d_mtime']}")
        lines.append(f"    E: Size={diff['e_size']:,}, Modified={diff['e_mtime']}")
        if diff['size_diff']:
            sign = "+" if diff['size_diff'] > 0 else ""
            lines.append(f"    Diff: {sign}{diff['size_diff']:,} bytes")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total in D: {len(d_files)}")
    lines.append(f"Total in E: {len(e_files)}")
    lines.append(f"Only in D: {len(only_in_d)}")
    lines.append(f"Only in E: {len(only_in_e)}")
    lines.append(f"Different: {len(different)}")
    lines.append(f"Identical: {len(in_both) - len(different)}")
    
    report = "\n".join(lines)
    print(report)
    
    with open('directory_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\nReport saved to: directory_comparison_report.txt")

if __name__ == '__main__':
    main()
