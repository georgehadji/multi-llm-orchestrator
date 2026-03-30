# TDD (Test-First Generation) Implementation Guide

**Version:** v3.0  
**Date:** 2026-03-30  
**Author:** Georgios-Chrysovalantis Chatzivantsidis

---

## 🎯 Overview

Test-First Generation (TDD Inversion) implements a paradigm shift in code generation:

**Traditional Approach:**
```
Generate Code → Verify → Fix (score-based, heuristic)
```

**TDD Approach:**
```
Generate Tests → Generate Code → Run Tests → Fix to Pass (deterministic)
```

### **Benefits:**
- ✅ **Machine-verifiable success** - "17/17 tests passed" vs "score: 0.85"
- ✅ **Executable specifications** - Tests document expected behavior
- ✅ **Higher code quality** - Tests enforce good design
- ✅ **Early bug detection** - Bugs caught during generation

---

## 🚀 Quick Start

### **Enable TDD via CLI:**

```bash
# Enable TDD with balanced quality tier
python -m orchestrator \
  --project "Build a Python calculator" \
  --criteria "All operations tested" \
  --tdd-first \
  --tdd-quality balanced

# Enable TDD with budget tier
python -m orchestrator \
  --project "Build a REST API" \
  --tdd-first \
  --tdd-quality budget

# Enable TDD with premium tier (maximum quality)
python -m orchestrator \
  --project "Build a payment processor" \
  --tdd-first \
  --tdd-quality premium
```

### **Enable TDD via Python API:**

```python
from orchestrator import Orchestrator
from orchestrator.cost_optimization import update_config, get_optimization_config

# Get current config
config = get_optimization_config()

# Enable TDD
config.enable_tdd_first = True
config.tdd_quality_tier = "balanced"  # or "budget", "premium"
config.tdd_max_iterations = 3
config.tdd_min_test_coverage = 0.8

update_config(config)

# Run orchestrator
orch = Orchestrator()
state = await orch.run_project(
    project_description="Build a Python calculator",
    success_criteria="All operations tested with pytest",
)
```

---

## 📊 Quality Tiers

### **Budget Tier** (`--tdd-quality budget`)

**Best for:** Simple tasks, prototypes, cost-conscious projects

| Phase | Model | Cost (per 1M tokens) |
|-------|-------|---------------------|
| Test Generation | Qwen3 Coder Next | $0.12/$0.75 |
| Implementation | Qwen3 Coder Next | $0.12/$0.75 |
| Test Review | DeepSeek V3.2 | $0.27/$1.10 |

**Estimated Cost per Task:** ~$0.006

---

### **Balanced Tier** (`--tdd-quality balanced`) ⭐ **RECOMMENDED**

**Best for:** Production code, most use cases

| Phase | Model | Cost (per 1M tokens) |
|-------|-------|---------------------|
| Test Generation | Claude Sonnet 4.6 | $3.00/$15.00 |
| Implementation | Qwen3 Coder Next | $0.12/$0.75 |
| Test Review | Claude Sonnet 4.6 | $3.00/$15.00 |

**Estimated Cost per Task:** ~$0.039

---

### **Premium Tier** (`--tdd-quality premium`)

**Best for:** Critical systems, enterprise code, complex requirements

| Phase | Model | Cost (per 1M tokens) |
|-------|-------|---------------------|
| Test Generation | GPT-5.4 Pro | $30.00/$180.00 |
| Implementation | GPT-5.4 Pro | $30.00/$180.00 |
| Test Review | GPT-5.4 Pro | $30.00/$180.00 |

**Estimated Cost per Task:** ~$0.449

---

## 🎯 Model Selection Logic

### **How Models Are Selected:**

```python
from orchestrator.cost_optimization import get_tdd_profile

# Get TDD profile
config = get_tdd_profile(
    tier="balanced",    # budget, balanced, premium
    language="python"   # Optional: python, javascript, typescript, go, rust
)

# Get model for specific phase
test_model = config.get_model("test_generation", "balanced")
impl_model = config.get_model("implementation", "balanced")
review_model = config.get_model("test_review", "balanced")
```

### **Language-Specific Profiles:**

```python
# Python + pytest
python_config = get_tdd_profile("balanced", "python")

# JavaScript + Jest
js_config = get_tdd_profile("balanced", "javascript")

# TypeScript + Vitest
ts_config = get_tdd_profile("balanced", "typescript")
```

---

## 📋 TDD Workflow

### **Step 1: Test Specification Generation**

```python
# Model: test_generation (e.g., Claude Sonnet 4.6)
# Prompt: "Generate comprehensive pytest tests for a calculator"
# Output: test_calculator.py with 15+ test cases
```

**What's Generated:**
- ✅ Test file structure
- ✅ Test functions for all requirements
- ✅ Edge cases (boundary values, error conditions)
- ✅ Happy path tests

---

### **Step 2: Implementation Generation**

```python
# Model: implementation (e.g., Qwen3 Coder Next)
# Prompt: "Implement calculator to pass these tests"
# Input: Test file from Step 1
# Output: calculator.py that passes all tests
```

**What's Generated:**
- ✅ Implementation code
- ✅ Type annotations
- ✅ Docstrings
- ✅ Error handling

---

### **Step 3: Test Execution**

```python
# Run tests in sandbox
result = sandbox.run_tests(
    test_file="test_calculator.py",
    implementation_file="calculator.py",
)

# Result: TestExecutionResult
# - tests_run: 15
# - tests_passed: 15
# - tests_failed: 0
# - passed: True
```

---

### **Step 4: Review & Refactoring (if needed)**

```python
# If tests failed:
if not result.passed:
    # Model: test_review (e.g., Claude Sonnet 4.6)
    # Prompt: "Fix implementation to pass failing tests"
    # Input: Failing test output
    # Output: Fixed implementation
    
    # Re-run tests
    result = sandbox.run_tests(...)
```

---

## 💰 Cost Tracking

### **Per-Task Cost Breakdown:**

```python
from orchestrator.test_first_generator import TDDResult

result: TDDResult = await tdd.generate_with_tests(task)

print(f"Test Generation Cost: ${result.test_generation_cost:.4f}")
print(f"Implementation Cost:  ${result.implementation_cost:.4f}")
print(f"Review Cost:          ${result.review_cost:.4f}")
print(f"Total Cost:           ${result.cost_usd:.4f}")
```

### **Cost Comparison:**

| Approach | Cost per Task | Quality |
|----------|---------------|---------|
| **Standard Generation** | $0.015 | Good (score: 0.85) |
| **TDD Budget** | $0.006 | Good (tests: 15/15 passed) |
| **TDD Balanced** | $0.039 | Excellent (tests: 15/15 passed) |
| **TDD Premium** | $0.449 | Outstanding (tests: 15/15 passed) |

---

## 🧪 Testing Framework Support

### **Automatically Detected:**

| Language | Framework | Detection |
|----------|-----------|-----------|
| **Python** | pytest | `.py` files, "pytest" keyword |
| **Python** | unittest | `.py` files, "unittest" keyword |
| **JavaScript** | Jest | `.js` files, "jest" keyword |
| **TypeScript** | Vitest | `.ts` files, "vitest" keyword |
| **Go** | go test | `.go` files, "go test" keyword |
| **Rust** | cargo test | `.rs` files, "cargo test" keyword |

### **Manual Override:**

```python
task = Task(
    id="task_001",
    type=TaskType.CODE_GEN,
    prompt="Build a calculator (use pytest for tests)",  # Explicit framework
    acceptance_threshold=0.85,
)
```

---

## ⚙️ Configuration Options

### **CLI Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--tdd-first` | Enable TDD generation | False |
| `--tdd-quality` | Quality tier (budget/balanced/premium) | balanced |
| `--tdd-max-iterations` | Max TDD iterations | 3 |
| `--tdd-min-coverage` | Minimum test coverage (0.0-1.0) | 0.8 |

### **Python API:**

```python
from orchestrator.cost_optimization import (
    OptimizationConfig,
    update_config,
    get_optimization_config,
)

config = get_optimization_config()
config.enable_tdd_first = True
config.tdd_quality_tier = "balanced"
config.tdd_max_iterations = 3
config.tdd_min_test_coverage = 0.8

update_config(config)
```

---

## 📊 Performance Metrics

### **Expected Outcomes:**

| Metric | Standard | TDD Budget | TDD Balanced | TDD Premium |
|--------|----------|------------|--------------|-------------|
| **Test Coverage** | 0% | 80%+ | 85%+ | 90%+ |
| **Bug Detection** | Post-generation | During generation | During generation | During generation |
| **Code Quality** | Good | Good | Excellent | Outstanding |
| **Generation Time** | 1x | 2-3x | 2-3x | 2-3x |
| **Success Rate** | 85% | 80% | 90% | 95% |

---

## 🐛 Troubleshooting

### **Issue: TDD not enabled**

**Solution:**
```bash
# Check if TDD is enabled
python -m orchestrator --help | grep tdd

# Enable explicitly
python -m orchestrator --tdd-first --tdd-quality balanced ...
```

---

### **Issue: Tests failing repeatedly**

**Solution:**
```python
# Increase max iterations
config.tdd_max_iterations = 5  # Default: 3

# Lower minimum coverage requirement
config.tdd_min_test_coverage = 0.6  # Default: 0.8

# Upgrade quality tier
config.tdd_quality_tier = "premium"  # Default: balanced
```

---

### **Issue: High TDD costs**

**Solution:**
```python
# Switch to budget tier
config.tdd_quality_tier = "budget"

# Or disable TDD for simple tasks
config.enable_tdd_first = False
```

---

## 📝 Best Practices

### **When to Use TDD:**

✅ **Use TDD for:**
- Production code
- API implementations
- Business logic
- Mathematical functions
- Data transformations

❌ **Skip TDD for:**
- Simple scripts
- Configuration files
- Documentation generation
- UI/CSS styling
- One-off utilities

---

### **Quality Tier Selection:**

```python
# Budget: Simple utilities, prototypes
if task_complexity < 3:
    tier = "budget"

# Balanced: Most production code (RECOMMENDED)
elif task_complexity < 7:
    tier = "balanced"

# Premium: Critical systems, complex logic
else:
    tier = "premium"
```

---

## 📚 Examples

### **Example 1: Python Calculator (Budget)**

```bash
python -m orchestrator \
  --project "Build a Python calculator with add, subtract, multiply, divide" \
  --criteria "All operations tested with pytest, 80%+ coverage" \
  --tdd-first \
  --tdd-quality budget
```

**Expected Output:**
```
✅ TDD enabled: quality=budget
✅ Task task_001: Using TDD-first generation
✅ Tests generated: 16 test cases
✅ Implementation generated
✅ Tests: 16/16 passed
✅ Cost: $0.0058
```

---

### **Example 2: REST API (Balanced)**

```bash
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT authentication" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --tdd-first \
  --tdd-quality balanced \
  --budget 5.0
```

**Expected Output:**
```
✅ TDD enabled: quality=balanced
✅ Task task_001: Using TDD-first generation
✅ Tests generated: 24 test cases (pytest)
✅ Implementation generated
✅ Tests: 24/24 passed
✅ Cost: $0.042
```

---

### **Example 3: Payment Processor (Premium)**

```bash
python -m orchestrator \
  --project "Build a payment processor with Stripe integration" \
  --criteria "All payment flows tested, PCI compliance" \
  --tdd-first \
  --tdd-quality premium \
  --budget 20.0
```

**Expected Output:**
```
✅ TDD enabled: quality=premium
✅ Task task_001: Using TDD-first generation
✅ Tests generated: 42 test cases (pytest)
✅ Implementation generated
✅ Tests: 42/42 passed
✅ Cost: $0.485
```

---

## 📖 Additional Resources

- **TDD_CONFIG.md** - TDD model configuration details
- **TDD_MODEL_SELECTION.md** - Model recommendations per phase
- **TDD_CLI_USAGE.md** - CLI usage examples
- **CAPABILITIES.md** - Full orchestrator capabilities

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** TDD Implementation Guide v1.0  
**Status:** ✅ Production Ready
