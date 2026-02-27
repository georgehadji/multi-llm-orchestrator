# 🔍 Debugging Generated Projects
## How to debug and fix errors in projects created by the orchestrator

---

## 🎯 Overview

When the orchestrator creates a project, you may need to debug it. This guide covers:

1. **Runtime Errors** - The project crashes when run
2. **Logic Errors** - The project runs but produces wrong results
3. **Build Errors** - The project fails to compile/build
4. **Test Failures** - Tests don't pass

---

## 🐛 Runtime Errors

### Python Projects

#### Syntax Errors

```python
# Error: SyntaxError: invalid syntax
# Location: File "app.py", line 15

# Fix:
# 1. Check for missing colons
def hello()  # ❌ Missing colon
    pass

def hello():  # ✅ Fixed
    pass

# 2. Check indentation (mixing tabs and spaces)
# Use: autopep8 --in-place app.py
```

#### Import Errors

```python
# Error: ModuleNotFoundError: No module named 'xyz'

# Fix:
# 1. Install missing dependency
pip install xyz

# 2. Check for typos
from fastapi import FastAPI  # ✅ Correct
from fastapi import fastapi  # ❌ Wrong case

# 3. Check if module is in requirements.txt
pip install -r requirements.txt
```

#### Attribute Errors

```python
# Error: AttributeError: 'NoneType' object has no attribute 'x'

# Debug:
print(f"Variable type: {type(variable)}")
print(f"Variable value: {variable}")

# Fix with null check:
if variable is not None:
    variable.x()
else:
    # Handle missing value
    pass
```

#### Type Errors

```python
# Error: TypeError: unsupported operand type(s)

# Debug:
print(f"a type: {type(a)}, value: {a}")
print(f"b type: {type(b)}, value: {b}")

# Fix with type conversion:
result = int(a) + int(b)  # ✅ Explicit conversion
```

### JavaScript/TypeScript Projects

```bash
# Error: Cannot find module
npm install missing-package

# Error: Type error
npx tsc --noEmit  # Check types

# Error: Syntax error
npx eslint . --fix  # Auto-fix issues
```

---

## 🔧 Build Errors

### Python Build

```bash
# Error: Module not found during build
pip install build
python -m build

# Check dist/ for errors
ls -la dist/
```

### Node.js Build

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Check for peer dependency issues
npm ls

# Build with verbose output
npm run build --verbose
```

---

## 🧪 Test Failures

### Running Tests

```bash
# Python
pytest -v --tb=short

# JavaScript
npm test -- --verbose

# Specific test
pytest tests/test_specific.py::test_function -v
```

### Debugging Failed Tests

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use pytest debugger
pytest --pdb  # Drop into debugger on failure

# Print test output
pytest -s  # Show print statements
```

### Common Test Issues

| Issue | Solution |
|-------|----------|
| Test timeouts | Increase timeout: `pytest --timeout=60` |
| Missing fixtures | Check `conftest.py` exists |
| Database locks | Use `@pytest.fixture(scope="function")` |
| Async issues | Use `pytest-asyncio` plugin |

---

## 🎛️ Logic Errors

### Adding Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"Input: {data}")
    result = data.process()
    logger.debug(f"Output: {result}")
    return result
```

### Using Debugger

```python
# Built-in debugger
import pdb

def calculate(x, y):
    pdb.set_trace()  # Execution stops here
    result = x / y   # Check variable values
    return result

# IDE debugger (VSCode)
# Add to launch.json:
{
    "name": "Python: Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
}
```

### Assertions

```python
# Add sanity checks
def divide(a, b):
    assert b != 0, "Division by zero!"
    assert isinstance(a, (int, float)), "a must be numeric"
    assert isinstance(b, (int, float)), "b must be numeric"
    return a / b
```

---

## 🔍 Debugging Tools by Language

### Python

```bash
# Interactive debugging
python -i script.py  # Run then drop to REPL

# Post-mortem debugging
python -m pdb script.py

# Profiling
python -m cProfile -s cumulative script.py

# Memory debugging
pip install memory_profiler
python -m memory_profiler script.py
```

### JavaScript/TypeScript

```bash
# Node debugger
node --inspect script.js
# Then open chrome://inspect

# Console debugging
node script.js  # Add console.log() statements

# Type checking
npx tsc --noEmit --watch
```

---

## 🛠️ Fixing Common Generated Code Issues

### Issue 1: Missing Error Handling

**Generated:**
```python
def get_user(user_id):
    return db.query(User).get(user_id)
```

**Fixed:**
```python
def get_user(user_id):
    try:
        user = db.query(User).get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        return user
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise
```

### Issue 2: Hardcoded Values

**Generated:**
```python
API_KEY = "sk-12345"  # ❌ Security risk
```

**Fixed:**
```python
import os

API_KEY = os.getenv("API_KEY")  # ✅ Environment variable
if not API_KEY:
    raise ValueError("API_KEY not set")
```

### Issue 3: Missing Input Validation

**Generated:**
```python
def process(data):
    return data * 2
```

**Fixed:**
```python
def process(data):
    if not isinstance(data, (int, float)):
        raise TypeError("data must be numeric")
    if data < 0:
        raise ValueError("data must be non-negative")
    return data * 2
```

### Issue 4: Resource Leaks

**Generated:**
```python
def read_file(path):
    f = open(path, 'r')  # ❌ Never closed
    return f.read()
```

**Fixed:**
```python
def read_file(path):
    with open(path, 'r') as f:  # ✅ Auto-closes
        return f.read()
```

---

## 📊 Performance Debugging

### Slow Queries

```python
# Add timing
import time

start = time.time()
result = db.query(Model).all()
print(f"Query took: {time.time() - start:.2f}s")

# Enable SQL logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Memory Issues

```python
# Check memory usage
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Profile memory
from memory_profiler import profile

@profile
def heavy_function():
    large_list = [0] * 1000000
    return sum(large_list)
```

---

## 🔄 Iterative Improvement

### Workflow

```
1. Run → Find Error
   ↓
2. Add Logging/Debug
   ↓
3. Fix Issue
   ↓
4. Test Fix
   ↓
5. Repeat until clean
```

### Example Session

```bash
# 1. Run tests
pytest
# FAIL: test_calculator.py::test_add

# 2. Run specific test with details
pytest test_calculator.py::test_add -v --tb=long
# Error: AssertionError: add(2, 2) returned 5, expected 4

# 3. Check the code
cat calculator.py
# Bug: return a + b + 1  # Extra +1!

# 4. Fix
sed -i 's/return a + b + 1/return a + b/' calculator.py

# 5. Verify
pytest test_calculator.py::test_add -v
# PASS
```

---

## 🎯 Using Orchestrator to Fix Itself

You can use the orchestrator to help debug:

```python
from orchestrator import Orchestrator, Budget

async def fix_project():
    orch = Orchestrator(budget=Budget(max_usd=2.0))
    
    state = await orch.run_project(
        project_description="""
        Fix the bug in app.py where the divide function 
        crashes on zero division. Add proper error handling.
        """,
        success_criteria="All tests pass, no crashes on edge cases",
        output_dir="./fixed_project"
    )
    
    return state

# Apply fixes
cp fixed_project/app.py ./app.py
```

---

## 📚 Related

- [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md) - Orchestrator debugging
- [TROUBLESHOOTING_CHEATSHEET.md](./TROUBLESHOOTING_CHEATSHEET.md) - Quick fixes
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) - Usage examples

---

**Last Updated:** 2026-02-26 | **Version:** v5.1
