# Output Organizer & Test Automation

Αυτόματη οργάνωση του output και test execution μετά την ολοκλήρωση ενός project.

## 🎯 Features

### 1. Cache Message Suppression
Καταστολή των ενοχλητικών "Cache hit" μηνυμάτων για καθαρότερο output.

### 2. Task Files Organization
Αυτόματη μετακίνηση των task files σε `tasks/` folder:
```
output/
├── tasks/                    # ← Task files moved here
│   ├── task_001_code_generation.py
│   ├── task_002_code_review.md
│   └── task_003_data_extract.json
├── src/                      # Source code
├── tests/                    # Tests
└── summary.json
```

### 3. Auto-Test Generation
Ανάλυση source files και αυτόματη δημιουργία tests για functions/classes που δεν έχουν.

### 4. Test Execution
Αυτόματο τρέξιμο όλων των tests με coverage reporting.

### 5. Test Files Organization
Μετακίνηση test files σε `tests/` folder.

## 📖 Usage

### Αυτόματη Χρήση (μέσω CLI)

Όταν τρέχετε ένα project μέσω CLI, ο organizer καλείται αυτόματα:

```bash
python -m orchestrator \
  --project "Build a FastAPI REST API" \
  --criteria "All endpoints tested" \
  --budget 5.0

# Output:
# 📁 Organizing project output...
#   ✅ Tasks moved: 8
#   ✅ Tests: 5/6 passed
```

### Manual Χρήση

```python
from orchestrator.output_organizer import OutputOrganizer

# Create organizer
organizer = OutputOrganizer(
    output_dir=Path("./output/my_project"),
    auto_generate_tests=True,  # Generate missing tests
    run_tests=True,            # Run tests after generation
    min_coverage=80.0,         # Minimum coverage target
)

# Run organization
report = await organizer.organize_project()

# Access results
print(f"Tasks moved: {len(report.tasks_moved)}")
print(f"Tests created: {len(report.tests_created)}")
print(f"Tests run: {len(report.tests_run)}")
```

### Cache Suppression

```python
from orchestrator.output_organizer import suppress_cache_messages

# Global suppression (call once at startup)
suppress_cache_messages()

# Or use context manager
from orchestrator.output_organizer import CacheMessageSuppressor

with CacheMessageSuppressor():
    # Cache messages suppressed here
    await orch.run_project(...)
```

## 📁 Directory Structure

Μετά την οργάνωση:

```
output/
├── tasks/                      # All task files
│   ├── task_001_code_generation.py
│   ├── task_002_code_review.md
│   └── ...
├── tests/                      # All test files
│   ├── test_main.py
│   ├── test_utils.py
│   └── ...
├── src/                        # Source code (if exists)
│   ├── main.py
│   └── utils.py
├── app/                        # App code (if exists)
├── organization_report.json    # Organization report
└── summary.json                # Project summary
```

## 🧪 Test Generation

### Auto-Generated Test Structure

```python
# test_example.py (auto-generated)
"""Auto-generated tests for example.py"""
import pytest
from example import MyClass, my_function


class TestMyClass:
    """Tests for MyClass class."""
    
    def test_myclass_initialization(self):
        """Test MyClass can be instantiated."""
        # TODO: Add proper initialization parameters
        # instance = MyClass()
        # assert instance is not None
        pass


def test_my_function():
    """Test my_function function."""
    # TODO: Add proper test parameters and assertions
    # result = my_function()
    # assert result is not None
    pass
```

### Coverage Reporting

Το organizer τρέχει tests με coverage:
```
📊 Organization Summary
============================================================
  📁 Task files moved: 8
  ✨ Tests generated: 2
  🧪 Tests run: 6
     ✅ Passed: 5
     ❌ Failed: 1
     📈 Coverage: 73.5%
  📂 Test files organized: 3
============================================================
```

## 📊 Organization Report

Το report αποθηκεύεται σε `organization_report.json`:

```json
{
  "tasks_moved": ["task_001.py", "task_002.md", ...],
  "tests_created": ["test_main.py", "test_utils.py"],
  "tests_run": [
    {
      "test_file": "test_main.py",
      "passed": true,
      "duration_ms": 125.5,
      "coverage_percent": 85.0
    }
  ],
  "summary": {
    "total_tests": 6,
    "passed_tests": 5,
    "failed_tests": 1,
    "coverage_avg": 73.5
  }
}
```

## 🔧 Configuration

### Organizer Options

| Option | Default | Description |
|--------|---------|-------------|
| `auto_generate_tests` | `True` | Generate tests for missing coverage |
| `run_tests` | `True` | Run tests after organization |
| `min_coverage` | `80.0` | Minimum coverage target (%) |

### Cache Suppression Options

```python
# Suppress specific loggers
with CacheMessageSuppressor([
    "orchestrator.cache",
    "orchestrator.api_clients",
    "custom.logger",
]):
    ...
```

## 🚀 Integration with Engine

### Automatic Integration

Ο organizer ενσωματώνεται αυτόματα στο CLI workflow:

1. Project execution
2. Output writing
3. **Organization** (tasks → tasks/, tests generation)
4. **Test execution**
5. Report generation

### Manual Integration

```python
from orchestrator import Orchestrator
from orchestrator.output_organizer import organize_project_output

async def run_with_organization():
    orch = Orchestrator(...)
    state = await orch.run_project(...)
    
    # Write output
    from orchestrator.output_writer import write_output_dir
    path = write_output_dir(state, "./output")
    
    # Organize
    report = await organize_project_output(path)
    
    return state, report
```

## 📝 Examples

### Example 1: Basic Organization

```python
import asyncio
from pathlib import Path
from orchestrator.output_organizer import OutputOrganizer

async def main():
    organizer = OutputOrganizer(
        output_dir=Path("./output/project_123"),
        auto_generate_tests=True,
        run_tests=True,
    )
    
    report = await organizer.organize_project()
    
    print(f"Moved {len(report.tasks_moved)} tasks")
    print(f"Generated {len(report.tests_created)} tests")
    print(f"Tests: {sum(1 for r in report.tests_run if r.passed)}/{len(report.tests_run)} passed")

asyncio.run(main())
```

### Example 2: Cache Suppression Only

```python
from orchestrator import Orchestrator
from orchestrator.output_organizer import suppress_cache_messages

# Suppress cache messages globally
suppress_cache_messages()

# Now run without cache spam
orch = Orchestrator()
```

### Example 3: Custom Test Generation

```python
from orchestrator.output_organizer import OutputOrganizer

organizer = OutputOrganizer(
    output_dir=Path("./output"),
    auto_generate_tests=True,
    min_coverage=90.0,  # Higher threshold
)

report = await organizer.organize_project()

# Check coverage
avg_coverage = sum(r.coverage_percent for r in report.tests_run) / len(report.tests_run)
if avg_coverage < 90.0:
    print("⚠️ Coverage below target!")
```

## 🔍 Troubleshooting

### Tests Not Generated
- Βεβαιωθείτε ότι υπάρχουν source files (`.py`)
- Check ότι τα files δεν είναι ήδη test files
- Βεβαιωθείτε ότι υπάρχουν public functions/classes

### Tests Fail to Run
- Ελέγξτε ότι το `pytest` είναι installed
- Βεβαιωθείτε ότι υπάρχουν test files
- Ελέγξτε τα error messages στο report

### Cache Messages Still Appear
- Καλέστε `suppress_cache_messages()` πριν από οποιαδήποτε άλλη operation
- Βεβαιωθείτε ότι δεν υπάρχει άλλο code που επαναφέρει τα log levels

## 📈 Future Improvements

- [ ] Support for more languages (JavaScript, TypeScript, etc.)
- [ ] Intelligent test case generation with AI
- [ ] Parallel test execution
- [ ] Integration with CI/CD pipelines
- [ ] Custom test templates
