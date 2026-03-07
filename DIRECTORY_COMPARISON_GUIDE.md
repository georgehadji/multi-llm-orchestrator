# AI Orchestrator - Directory Comparison Guide

## Overview

This document provides a guide for comparing the AI Orchestrator project between two locations:
- **D:** `D:\Vibe-Coding\Ai Orchestrator`
- **E:** `E:\Documents\Vibe-Coding\Ai Orchestrator`

## D: Drive File Inventory

### Summary Statistics

| Directory | File Count |
|-----------|------------|
| Root (non-recursive) | 220 entries |
| orchestrator/ | 190 files |
| docs/ | 31 files |
| examples/ | 0 files (doesn't exist) |
| scripts/ | 0 files (doesn't exist) |
| tests/ | 147 files |
| outputs/ | 1000+ files (includes venv) |
| projects/ | 32 files |

### Key Directories in D: Drive

#### 1. orchestrator/ (Main Package)
The main Python package with 190 files including:
- Core modules: `engine.py`, `agents.py`, `models.py`, `api_clients.py`
- Dashboard modules: `dashboard.py`, `dashboard_live.py`, `dashboard_mission_control.py`, etc.
- Analyzer modules: `analyzer.py`, `codebase_analyzer.py`, `project_analyzer.py`
- Rules: `frontend_rules.py`, `wordpress_plugin_rules.py`, `indesign_plugin_rules.py`
- Scaffold templates: `scaffold/templates/` with 8 template files
- Cache, telemetry, monitoring modules
- And many more...

#### 2. docs/ (Documentation)
31 files including:
- Architecture and capability documentation
- Dashboard guides (Ant Design, Enhanced, Live)
- Plugin rules documentation (WordPress, InDesign)
- Plans folder with design/implementation docs dated 2025-02 to 2026-02

#### 3. tests/ (Test Suite)
147 files including:
- Unit tests for various modules
- Pytest cache files
- Test utilities and verification scripts

#### 4. outputs/ (Generated Output)
1000+ files including:
- Project analysis outputs (food_delivery, greek_airbnb, etc.)
- Virtual environments (venv folders with full pip packages)
- Generated code and documentation

#### 5. projects/ (Project Specifications)
32 YAML project specification files including:
- Analysis projects (food_delivery, greek_airbnb, elections, real_estate)
- Backend projects (graphql_api, microservices, rest_api, websocket_chat)
- Frontend projects (nextjs_ecommerce, react_dashboard, vue_kanban)
- Portfolio projects
- Engine projects (polytonic_ocr, symplectic_engine)

### Root Directory Files

220 entries including:
- Configuration files: `.env`, `.env.example`, `pyproject.toml`, `Dockerfile`
- Documentation: `README.md`, `CLAUDE.md`, `USAGE_GUIDE.md`, `CAPABILITIES.md`
- Batch files: `*.bat` files for starting dashboards, mission control
- Python scripts: Various utilities for setup, cleanup, debugging

## How to Perform the Comparison

### Method 1: Using Provided Scripts

1. Open Command Prompt or PowerShell
2. Navigate to the D: drive project:
   ```cmd
   cd "D:\Vibe-Coding\Ai Orchestrator"
   ```

3. Run the comparison batch file:
   ```cmd
   run_comparison.bat
   ```

This will:
- Scan D: drive and save to `d_files.json`
- Scan E: drive and save to `e_files.json`
- Compare and generate `directory_comparison_report.txt`

### Method 2: Manual PowerShell Comparison

```powershell
# Get files from D:
$dFiles = Get-ChildItem -Path "D:\Vibe-Coding\Ai Orchestrator\orchestrator" -Recurse -File | 
    Select-Object FullName, Length, LastWriteTime | 
    Sort-Object FullName

# Get files from E:
$eFiles = Get-ChildItem -Path "E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator" -Recurse -File | 
    Select-Object FullName, Length, LastWriteTime | 
    Sort-Object FullName

# Find files only in D:
Compare-Object $dFiles $eFiles -Property FullName | Where-Object {$_.SideIndicator -eq "<="}

# Find files only in E:
Compare-Object $dFiles $eFiles -Property FullName | Where-Object {$_.SideIndicator -eq "=>"}
```

### Method 3: Using robocopy (List-only mode)

```cmd
robocopy "D:\Vibe-Coding\Ai Orchestrator" "E:\Documents\Vibe-Coding\Ai Orchestrator" /E /L /NS /NC /NFL /NDL /LOG:d_to_e_comparison.log
```

## Comparison Scripts Created

The following scripts have been created to help with the comparison:

1. **compare_directories.py** - Full comparison script
2. **scan_d.py** - Scan D: drive directory
3. **scan_e.py** - Scan E: drive directory
4. **compare_results.py** - Compare JSON results
5. **run_comparison.bat** - Batch file to run all steps

## Expected Comparison Results

### Files Likely Only in D: (Newer files)
- Comparison scripts created during this session
- Recently modified configuration files
- New test files

### Files Likely Only in E: (If E: is an older backup)
- Older versions of scripts
- Files that were deleted from D: but preserved in E:

### Files Likely Different
- `README.md` and documentation files (frequently updated)
- `CLAUDE.md` (project context file)
- Dashboard and main engine files (actively developed)
- Configuration files (`.env`, etc.)

## Notes

- The `outputs/` directory contains generated files and virtual environments that may differ significantly
- `__pycache__/` folders are auto-generated and will likely differ
- `.git/` folders track history and may differ
- The `examples/` and `scripts/` directories mentioned in the original request don't exist in the D: drive location

## Next Steps

1. Run the comparison scripts to generate actual diff data
2. Review files that exist in both locations with differences
3. Decide on sync strategy:
   - Copy newer files from D: to E:
   - Preserve specific files from E: that may have been lost
   - Merge specific changes

---
*Generated: 2026-03-01*
