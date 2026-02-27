# 🔍 Project Analyzer
## Automatic Post-Project Codebase Analysis

---

## 📋 Overview

The **Project Analyzer** automatically analyzes completed projects and suggests improvements. It runs after project completion to:

- ✅ Analyze code quality and complexity
- ✅ Detect architecture patterns
- ✅ Identify security issues
- ✅ Suggest performance optimizations
- ✅ Recommend missing tests/documentation
- ✅ Store findings in Knowledge Base

---

## 🚀 Quick Start

### Enable Automatic Analysis

```python
from orchestrator import Orchestrator, Budget
from pathlib import Path

orch = Orchestrator(budget=Budget(max_usd=5.0))

state = await orch.run_project(
    project_description="Build a REST API",
    success_criteria="All endpoints tested",
    analyze_on_complete=True,  # Enable analysis
    output_dir=Path("./results/my_project")
)
```

**Output Example:**
```
📊 Project Analysis: proj_abc123
============================================================
Overall Quality Score: 78.5/100
Test Coverage: 65.0%
Total Files: 12
Total Lines: 1,247

🎯 Top Suggestions:

🟠 [HIGH] Add comprehensive test suite
   Category: testing
   Effort: 4h
   Impact: Reduces bugs, enables refactoring
   
🟡 [MEDIUM] Improve API documentation
   Category: documentation
   Effort: 2h
   Impact: Easier onboarding

💡 Architecture: MVC/MVT
   Quality: 75/100

💾 5 suggestions stored in Knowledge Base
```

---

## 🛠️ Manual Analysis

```python
from orchestrator import ProjectAnalyzer, analyze_project
from pathlib import Path

# Method 1: Full control
analyzer = ProjectAnalyzer()
report = await analyzer.analyze_project(
    project_path=Path("./results/my_project"),
    project_id="my_project_123",
    run_quality_gate=True
)

# Get summary
print(analyzer.generate_summary(report))

# Access specific suggestions
critical = report.get_critical_suggestions()
for suggestion in report.suggestions[:5]:
    print(f"[{suggestion.priority.value}] {suggestion.title}")

# Method 2: Quick analysis
report = await analyze_project(
    Path("./results/my_project"),
    "my_project_123",
    print_summary=True
)
```

---

## 📊 Analysis Categories

| Category | Analyzes | Outputs |
|----------|----------|---------|
| **Code Quality** | Complexity, duplication, style issues | Quality score, issues list |
| **Architecture** | Pattern detection, structure | Architecture insights |
| **Security** | Bare excepts, hardcoded secrets, validation | Security warnings |
| **Performance** | Algorithmic complexity, bottlenecks | Optimization tips |
| **Testing** | Coverage, test structure | Testing recommendations |
| **Documentation** | Docstrings, README completeness | Documentation gaps |

---

## 📈 Report Structure

```python
@dataclass
class ProjectAnalysisReport:
    project_id: str
    analyzed_at: str
    
    # Metrics
    total_files: int
    total_lines: int
    languages: Dict[str, int]
    quality_score: float  # 0-100
    test_coverage: Optional[float]
    
    # Analysis
    issues: List[CodeIssue]
    suggestions: List[ImprovementSuggestion]
    architecture_insights: List[ArchitectureInsight]
```

### Suggestion Priority Levels

| Priority | Icon | Action Needed |
|----------|------|---------------|
| **CRITICAL** | 🔴 | Fix immediately (crashes, security) |
| **HIGH** | 🟠 | Address soon (performance, bugs) |
| **MEDIUM** | 🟡 | Plan to implement (optimizations) |
| **LOW** | 🔵 | Consider when convenient (style) |

---

## 🧠 Knowledge Base Integration

Findings are automatically stored:

```python
from orchestrator import get_knowledge_base

kb = get_knowledge_base()

# Patterns found
patterns = await kb.find_similar("MVC pattern")

# Lessons learned
lessons = await kb.find_similar(
    "authentication best practices",
    type_filter=KnowledgeType.LESSON
)
```

---

## 📁 Output Files

Analysis generates:
- Console summary
- `analysis_report.json` in project directory
- Knowledge Base entries

```json
{
  "project_id": "proj_abc123",
  "quality_score": 78.5,
  "test_coverage": 65.0,
  "suggestions": [
    {
      "id": "test_coverage",
      "title": "Add comprehensive test suite",
      "priority": "high",
      "estimated_effort": "4h"
    }
  ]
}
```

---

## 🎨 Use Cases

### 1. CI/CD Integration
```python
# Fail build if quality < 70
report = await analyzer.analyze_project(path, id)
if report.quality_score < 70:
    raise Exception(f"Quality {report.quality_score} below threshold")
```

### 2. Code Review
```python
# Generate review checklist
for suggestion in report.get_critical_suggestions():
    print(f"❗ Must fix: {suggestion.title}")
```

### 3. Learning & Improvement
```python
# Track quality over time
scores = [r.quality_score for r in historical_reports]
print(f"Quality trend: {sum(scores)/len(scores):.1f}")
```

---

## 🔧 Configuration

### Disable Analysis
```python
state = await orch.run_project(
    project_description="...",
    analyze_on_complete=False  # Skip analysis
)
```

### Custom Directory
```python
state = await orch.run_project(
    project_description="...",
    analyze_on_complete=True,
    output_dir=Path("./custom/path")
)
```

---

## 📚 Related

- [Knowledge Management](./MANAGEMENT_SYSTEMS.md)
- [Quality Control](./MANAGEMENT_SYSTEMS.md)
- [Debugging Guide](../debugging/DEBUGGING_GUIDE.md)

---

**Version:** v5.1 | **Last Updated:** 2026-02-26
