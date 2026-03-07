# Project Cleanup Guide

## 📁 Files to Organize

### Root Directory Cleanup

#### Keep in Root (Essential Files)
```
.env                          # Environment variables
.env.example                  # Environment template
.gitignore                    # Git ignore rules
LICENSE                       # License file
README.md                     # Main readme
CLAUDE.md                     # Claude context
USAGE_GUIDE.md                # Usage documentation
CAPABILITIES.md               # Capabilities list
pyproject.toml               # Package configuration
setup.py                      # Setup script (optional)
Makefile                      # Build automation
setup_project.py              # Project setup script ⭐
```

#### Move to `scripts/`
```
run_dashboard.py              # Start dashboard
run_dashboard_realtime.py     # Start real-time dashboard
run_optimized_dashboard.py    # Start optimized dashboard
```

#### Move to `examples/`
```
example_deepseek_coder.py
example_deepseek_coder_v2.py
example_enhanced_dashboard.py
```

#### Temporary/Setup Files (Delete after setup)
```
check_outputs.py              # Debugging
check_state.py                # Debugging
clean_and_push.py             # Git helper
cleanup_all.py                # Cleanup
cleanup_final.py              # Cleanup
cleanup_temp.py               # Cleanup
cleanup_temp_files.py         # Cleanup
COMPLETE_REORGANIZATION.py    # Setup
create_all_files.py           # Setup
create_configs.py             # Setup
create_dashboard.py           # Setup
create_docs_dir.py            # Setup
create_github_templates.py    # Setup
create_github_workflow.py     # Setup
create_scripts.py             # Setup
create_scripts_folder.py      # Setup
create_workflow.py            # Setup
execute_git.py                # Git helper
final_cleanup.py              # Cleanup
final_cleanup_all.py          # Cleanup
finalize_organization.py      # Setup
fix_git_errors.py             # Git helper
git_auto_commit.py            # Git helper
git_commit_push.py            # Git helper
init_project_structure.py     # Setup
organize_codebase.py          # Setup
organize_docs.py              # Setup
remove_all_temp.py            # Cleanup
setup_github_workflow.py      # Setup
setup_scripts.py              # Setup
update_doc_links.py           # Setup
```

#### Test Files (Keep or Move to `tests/`)
```
test_ant_design_dashboard.py      # ⭐ Keep in root (new)
test_assembler_import.py          # Can delete
test_deepseek.py                  # Keep in root
test_enhanced_dashboard.py        # ⭐ Keep in root (new)
test_output_organizer.py          # ⭐ Keep in root (new)
test_performance_import.py        # Can delete
```

## 🧹 Cleanup Commands

### Step 1: Run Setup Script

```bash
# Create scripts folder and organize
python setup_project.py
```

### Step 2: Create Examples Directory

```bash
mkdir -p examples
mv example_*.py examples/
```

### Step 3: Move Dashboard Scripts

```bash
# These will be created by setup_project.py in scripts/
# Original files can be removed after:
rm run_dashboard.py
rm run_dashboard_realtime.py
rm run_optimized_dashboard.py
```

### Step 4: Remove Temporary Files

```bash
# Remove all temporary setup files
rm check_outputs.py
rm check_state.py
rm clean_and_push.py
rm cleanup_*.py
rm COMPLETE_REORGANIZATION.py
rm create_*.py
rm execute_git.py
rm final_cleanup*.py
rm fix_git_errors.py
rm git_*.py
rm init_project_structure.py
rm organize_*.py
rm remove_all_temp.py
rm setup_github_workflow.py
rm setup_scripts.py
rm update_doc_links.py
```

### Step 5: Clean Test Files

```bash
# Remove old test files
rm test_assembler_import.py
rm test_performance_import.py

# Keep these test files:
# - test_ant_design_dashboard.py
# - test_deepseek.py
# - test_enhanced_dashboard.py
# - test_output_organizer.py
```

## 📂 Final Project Structure

```
multi-llm-orchestrator/
├── orchestrator/               # Main package
│   ├── __init__.py
│   ├── cli.py
│   ├── engine.py
│   ├── ...
│   └── dashboard_antd.py      # ⭐ NEW
│
├── scripts/                    # ⭐ NEW
│   ├── __init__.py
│   ├── run_dashboard.py        # ⭐ NEW
│   ├── run_tests.py            # ⭐ NEW
│   ├── organize_output.py      # ⭐ NEW
│   ├── check_models.py         # ⭐ NEW
│   ├── cleanup_cache.py        # ⭐ NEW
│   └── create_project.py       # ⭐ NEW
│
├── examples/                   # ⭐ NEW
│   ├── example_deepseek_coder.py
│   ├── example_deepseek_coder_v2.py
│   └── example_enhanced_dashboard.py
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE_RULES.md
│   ├── ANT_DESIGN_DASHBOARD.md   # ⭐ NEW
│   ├── ENHANCED_DASHBOARD.md
│   ├── OUTPUT_ORGANIZER.md
│   └── PROJECT_STRUCTURE.md      # ⭐ NEW
│
├── tests/                      # Test suite
│   └── ...
│
├── test_*.py                   # Root test files
│
├── outputs/                    # Default output
├── projects/                   # Project files
│
├── setup_project.py            # ⭐ Setup script
├── PROJECT_CLEANUP_GUIDE.md    # ⭐ This file
├── README.md
├── USAGE_GUIDE.md
├── CLAUDE.md
├── CAPABILITIES.md
├── pyproject.toml
├── Makefile
├── .env
├── .env.example
├── .gitignore
└── LICENSE
```

## ✅ Post-Cleanup Checklist

- [ ] `scripts/` folder exists with utility scripts
- [ ] `examples/` folder exists with example files
- [ ] `docs/` folder has all documentation
- [ ] Temporary setup files removed
- [ ] Root directory has only essential files
- [ ] Test files organized
- [ ] Dashboard can be started with `python scripts/run_dashboard.py`

## 🚀 Quick Test After Cleanup

```bash
# 1. Test scripts
python scripts/check_models.py

# 2. Test dashboard
python scripts/run_dashboard.py --no-browser

# 3. Test organizer
python scripts/organize_output.py ./outputs/<project_id>

# 4. Run tests
python scripts/run_tests.py
```
