# 🐛 Debugging Overview
## Complete Debugging Resources for Multi-LLM Orchestrator

---

## 📚 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md) | Comprehensive debugging manual | Deep troubleshooting |
| [TROUBLESHOOTING_CHEATSHEET.md](./TROUBLESHOOTING_CHEATSHEET.md) | Quick fixes for common errors | Fast problem resolution |
| [PROJECT_DEBUGGING.md](./PROJECT_DEBUGGING.md) | Debug generated projects | Fixing created code |
| This file | Overview and navigation | Finding the right resource |

---

## 🎯 Quick Navigation

### I have an error...

#### 1. During Installation
→ [TROUBLESHOOTING_CHEATSHEET.md](#installation-errors)

#### 2. Running the Orchestrator
→ [DEBUGGING_GUIDE.md](#2-common-errors--solutions)

#### 3. In a Generated Project
→ [PROJECT_DEBUGGING.md](#-debugging-generated-projects)

#### 4. Performance Issues
→ [DEBUGGING_GUIDE.md](#6-diagnostic-tools)

---

## 🚀 Quick Diagnostic

```bash
# Run full system check
python -c "
import asyncio
from orchestrator import SystemDiagnostic, print_diagnostic_report

async def main():
    diag = SystemDiagnostic()
    report = await diag.run_full_check()
    print_diagnostic_report(report)

asyncio.run(main())
"
```

**Output:**
```
🟢 Overall Status: HEALTHY
   Checks: 12 passed, 0 failed

✅ No issues found!
```

---

## 🛠️ Common Scenarios

### Scenario 1: Project Keeps Failing

```python
# 1. Diagnose specific project
from orchestrator import ProjectDiagnostic

diag = ProjectDiagnostic("project_id")
report = await diag.diagnose()

print(f"Issues: {len(report['issues'])}")
for issue in report['issues']:
    print(f"- {issue['type']}: {issue['suggestion']}")

# 2. Resume with fixes
python -m orchestrator --resume project_id
```

### Scenario 2: High Costs

```python
# Check cost breakdown
from orchestrator.cost import CostAnalyzer

analyzer = CostAnalyzer()
report = analyzer.analyze_project("project_id")

print(f"Total: ${report.total:.2f}")
print(f"By model: {report.by_model}")

# Fix: Use cheaper models
from orchestrator import WeightedSumBackend
orch = Orchestrator(optimization_backend=WeightedSumBackend(w_cost=0.7))
```

### Scenario 3: Slow Performance

```python
# Check cache performance
from orchestrator import get_cache

cache = get_cache()
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']}")

# Fix: Enable caching
from orchestrator import cached

@cached(ttl=300)
async def expensive_operation():
    return await fetch_data()
```

---

## 🔧 Built-in Tools

### 1. System Diagnostic
```python
from orchestrator import SystemDiagnostic

diag = SystemDiagnostic()
report = await diag.run_full_check()
```

### 2. Project Diagnostic
```python
from orchestrator import ProjectDiagnostic

diag = ProjectDiagnostic("project_id")
report = await diag.diagnose()
```

### 3. Health Checker
```python
from orchestrator import health_checker

status = await health_checker.check()
print(f"System: {status['overall']}")
```

### 4. Log Analysis
```bash
# View recent errors
tail -50 logs/orchestrator.log | grep ERROR

# Monitor in real-time
tail -f logs/orchestrator.log | grep -E "(ERROR|WARN)"
```

---

## 📝 Debug Checklist

Before asking for help, check:

- [ ] **Environment**: `python -c "from orchestrator import Orchestrator"`
- [ ] **API Keys**: `echo $OPENAI_API_KEY` (at least one set)
- [ ] **Dependencies**: `pip list | grep orchestrator`
- [ ] **Disk Space**: `df -h .`
- [ ] **Network**: `ping api.openai.com`
- [ ] **Logs**: `cat logs/orchestrator.log | tail -20`
- [ ] **Diagnostics**: Run `SystemDiagnostic()`

---

## 🆘 Emergency Commands

```bash
# 1. Complete reset
rm -rf ~/.orchestrator_cache/*
pip uninstall multi-llm-orchestrator -y
pip install -e ".[dev]"

# 2. Test installation
python -c "from orchestrator import Orchestrator; print('✓ OK')"

# 3. Run diagnostic
python -c "
import asyncio
from orchestrator.diagnostics import SystemDiagnostic, print_diagnostic_report
asyncio.run(print_diagnostic_report(await SystemDiagnostic().run_full_check()))
"

# 4. Test API
python -c "
import asyncio
from orchestrator.api_clients import test_all_providers
results = asyncio.run(test_all_providers())
for p, s in results.items():
    print(f'{p}: {s}')
"
```

---

## 📖 Learning Path

### Beginner
1. Read [TROUBLESHOOTING_CHEATSHEET.md](./TROUBLESHOOTING_CHEATSHEET.md)
2. Learn common error codes
3. Practice with `SystemDiagnostic`

### Intermediate
1. Read [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md)
2. Understand log analysis
3. Master recovery procedures

### Advanced
1. Read [PROJECT_DEBUGGING.md](./PROJECT_DEBUGGING.md)
2. Learn to debug generated code
3. Create custom diagnostic tools

---

## 🔗 Related Resources

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) - How to use the orchestrator
- [CAPABILITIES.md](./CAPABILITIES.md) - Feature reference
- [MANAGEMENT_SYSTEMS.md](./MANAGEMENT_SYSTEMS.md) - Management systems
- [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md) - Performance tuning

---

## 💡 Pro Tips

1. **Always check logs first** - Most issues are logged
2. **Use dry-run mode** - Test before full execution
3. **Enable debug mode** - Set `ORCHESTRATOR_LOG_LEVEL=DEBUG`
4. **Run diagnostics** - Use `SystemDiagnostic` regularly
5. **Keep backups** - Archive old projects before major changes

---

**Last Updated:** 2026-02-26 | **Version:** v5.1
