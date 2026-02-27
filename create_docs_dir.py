#!/usr/bin/env python3
"""Create docs/performance directory and PROJECT_ANALYZER.md"""
import os
from pathlib import Path

# Create directory
docs_perf = Path("docs/performance")
docs_perf.mkdir(parents=True, exist_ok=True)
print(f"✓ Created {docs_perf}")

# Create PROJECT_ANALYZER.md
content = """# 🔍 Project Analyzer
## Automatic Post-Project Codebase Analysis

---

## 📋 Overview

The **Project Analyzer** automatically analyzes completed projects and suggests improvements.

---

## 🚀 Quick Start

```python
from orchestrator import Orchestrator, Budget

orch = Orchestrator(budget=Budget(max_usd=5.0))

state = await orch.run_project(
    project_description="Build a REST API",
    success_criteria="All endpoints tested",
    analyze_on_complete=True,
    output_dir=Path("./results/my_project")
)
```

---

## 🛠️ Manual Analysis

```python
from orchestrator import ProjectAnalyzer

analyzer = ProjectAnalyzer()
report = await analyzer.analyze_project(
    project_path=Path("./results/my_project"),
    project_id="my_project_123"
)

print(analyzer.generate_summary(report))
```

---

## 📊 Analysis Categories

- **Code Quality** - Complexity, duplication, style
- **Architecture** - Pattern detection, structure
- **Security** - Vulnerabilities, error handling
- **Performance** - Bottlenecks, optimizations
- **Testing** - Coverage, test structure
- **Documentation** - Completeness, examples

---

**Version:** v5.1
"""

(docs_perf / "PROJECT_ANALYZER.md").write_text(content, encoding='utf-8')
print(f"✓ Created {docs_perf / 'PROJECT_ANALYZER.md'}")
