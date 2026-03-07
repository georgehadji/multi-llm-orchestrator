# 🏗️ Architecture Rules Engine
## Automatic Architecture Decision & Rules Generation with LLM Optimization

---

## 📋 Overview

Το **Architecture Rules Engine** αυτόματα:
1. **Αναλύει** τις απαιτήσεις του project
2. **Επιλέγει** βέλτιστη αρχιτεκτονική (rule-based)
3. **Βελτιστοποιεί** με LLM review και suggestions
4. **Προτείνει** technology stack
5. **Δημιουργεί** rules file (`.orchestrator-rules.yml`)
6. **Εφαρμόζει** constraints κατά την ανάπτυξη

---

## 🚀 Quick Start

### Αυτόματη Χρήση (στο run_project)

```python
from orchestrator import Orchestrator, Budget

orch = Orchestrator(budget=Budget(max_usd=5.0))

state = await orch.run_project(
    project_description="Build a scalable e-commerce API",
    success_criteria="Handle 10k requests/sec",
    output_dir=Path("./output")
)

# Αυτόματα δημιουργεί:
# - .orchestrator-rules.yml (machine-readable)
# - ARCHITECTURE.md (human-readable)
```

**Output Example:**
```
🏗️ ARCHITECTURE DECISION
============================================================

Decided by: LLM (Rule-based → Optimized)

Style: Event Driven
Paradigm: Object Oriented
API: GraphQL
Database: Document

Technology Stack:
  Primary: typescript
  Frameworks: react, next.js
  Libraries: tailwindcss, zustand

Key Constraints:
  • All code must be type-annotated
  • Maximum cyclomatic complexity of 10 per function
  • Minimum 80% test coverage

Recommended Patterns:
  • Repository Pattern
  • Dependency Injection
  • Factory Pattern

Rules file: .orchestrator-rules.yml
============================================================
```

---

## 🛠️ Manual Usage

### Basic Usage (Rule-based only)

```python
from orchestrator import ArchitectureRulesEngine, create_architecture_rules
from pathlib import Path

# Method 1: Full control
engine = ArchitectureRulesEngine()

rules = await engine.generate_rules(
    description="Build a real-time chat application",
    criteria="Support 10k concurrent users",
    project_type="web_api"
)

# Save rules
engine.save_rules(rules, Path("./output"))

# Print summary
print(engine.generate_summary(rules))
```

### With LLM Optimization

```python
from orchestrator import ArchitectureRulesEngine
from orchestrator.api_clients import UnifiedClient

# Create client for LLM optimization
client = UnifiedClient()
engine = ArchitectureRulesEngine(client=client)

# Generate with optimization
rules = await engine.generate_rules(
    description="Build event-driven analytics dashboard",
    criteria="Real-time updates, scalable"
)

# Check if optimized
print(f"LLM Generated: {rules._llm_generated}")   # False
print(f"LLM Optimized: {rules._llm_optimized}")   # True

print(engine.generate_summary(rules))
# Output: "Decided by: LLM (Rule-based → Optimized)"
```

### Method 2: Quick create

```python
rules = await create_architecture_rules(
    description="Build CLI tool",
    criteria="Cross-platform",
    output_dir=Path("./output")
)
```

---

## 🧠 How It Works

### Two-Phase Architecture Decision

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Rule-Based Detection                               │
│  ─────────────────────────────                               │
│  • Keyword analysis (microservice, event, graphql, etc.)     │
│  • Pattern matching for project type                         │
│  • Default stack selection                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: LLM Optimization (if client available)             │
│  ───────────────────────────────────────────────             │
│  • LLM reviews rule-based proposal                           │
│  • Suggests improvements if beneficial                       │
│  • Conservative approach: only clear wins                    │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌────────────────────┐
│  can_optimize   │    │  !can_optimize     │
│  = true         │    │  (no clear wins)   │
└────────┬────────┘    └────────┬───────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌────────────────────┐
│ Use Optimized   │    │ Use Rule-based     │
│ Architecture    │    │ (original)         │
└─────────────────┘    └────────────────────┘
```

### Decision Labels

| Label | Περιγραφή |
|-------|-----------|
| `Rule-based` | Μόνο keyword detection, χωρίς LLM |
| `LLM (Generated from scratch)` | LLM δημιούργησε όλη την αρχιτεκτονική |
| `LLM (Rule-based → Optimized)` | Rule-based + LLM βελτιστοποίηση |

---

## 📁 Generated Files

### 1. `.orchestrator-rules.yml`

Machine-readable rules file:

```yaml
version: "1.0"
created_at: "2026-02-26T10:30:00"
project_type: "web_api"

architecture:
  style: "event_driven"
  paradigm: "object_oriented"
  api_style: "graphql"
  database_type: "document"
  
  stack:
    primary_language: "typescript"
    frameworks:
      - "react"
      - "next.js"
    libraries:
      - "tailwindcss"
      - "zustand"
    databases:
      - "mongodb"
  
  constraints:
    - "All code must be type-annotated"
    - "Maximum cyclomatic complexity of 10"
    - "Minimum 80% test coverage"
  
  patterns:
    - "Repository Pattern"
    - "Event Sourcing"
    - "Pub/Sub"

coding_standards:
  naming_conventions:
    classes: "PascalCase"
    functions: "snake_case"
  max_line_length: 100
  test_coverage_min: 80.0

quality_gates:
  syntax_check: true
  type_check: true
  lint_check: true
  test_coverage_min: 80.0
```

### 2. `ARCHITECTURE.md`

Human-readable documentation:

```markdown
# Architecture Decision

## Project Overview
- **Type**: web_frontend
- **Generated**: 2026-02-26T10:30:00
- **Decision Method**: LLM (Rule-based → Optimized)

## Decisions

### Architecture Style
**Event Driven**

Selected for real-time updates and loose coupling...

### Technology Stack

**Primary Language**: typescript

**Frameworks**:
- react
- next.js
...
```

---

## 🎯 Architecture Decisions

### Supported Styles

| Style | When to Use | Keywords |
|-------|-------------|----------|
| **Microservices** | Distributed systems, independent scaling | microservice, kubernetes, scale |
| **Serverless** | Event-triggered, pay-per-use | lambda, serverless, function |
| **Event-Driven** | Real-time, streaming | kafka, event, streaming |
| **Hexagonal** | Testability, ports/adapters | ports, adapters, clean architecture |
| **Layered** | Traditional, simple | default |
| **CQRS** | Complex queries, event sourcing | read model, command, query |

### Supported Paradigms

- **Object Oriented** (default)
- **Functional** - immutable, pure functions
- **Reactive** - streams, observables
- **Declarative** - configuration over code

### API Styles

- **REST** (default)
- **GraphQL** - flexible queries
- **gRPC** - high performance
- **WebSocket** - real-time

### Database Types

- **Relational** (default) - PostgreSQL, MySQL
- **Document** - MongoDB
- **Key-Value** - Redis
- **Graph** - Neo4j
- **Time-Series** - InfluxDB

---

## 🧠 Technology Stack Selection

### Automatic Detection

Το σύστημα επιλέγει stack βάσει:

```python
# Keywords → Project Type → Stack

"frontend, react, ui" → web_frontend → {
    "primary": "typescript",
    "frameworks": ["react", "next.js"],
    "libraries": ["tailwindcss", "zustand"]
}

"cli, command line" → cli_tool → {
    "primary": "python",
    "frameworks": ["typer", "click"],
    "libraries": ["rich"]
}

"data pipeline, etl" → data_pipeline → {
    "primary": "python",
    "frameworks": ["apache spark", "pandas"],
    "libraries": ["numpy", "polars"]
}

"machine learning, ml" → machine_learning → {
    "primary": "python",
    "frameworks": ["pytorch", "scikit-learn"],
    "libraries": ["numpy", "pandas"]
}
```

---

## 🔮 LLM Optimization

### How It Works

Το LLM λαμβάνει:
```
PROJECT DESCRIPTION: {description}
SUCCESS CRITERIA: {criteria}

CURRENT ARCHITECTURE PROPOSAL:
- Style: {detected_style}
- Paradigm: {detected_paradigm}
- API: {detected_api}
- Database: {detected_db}
- Primary Language: {detected_language}
- Frameworks: {detected_frameworks}
...
```

Το LLM απαντά:
```json
{
    "can_optimize": true,
    "reasoning": "Event-driven better fits real-time requirements",
    "changes": [
        {
            "field": "style",
            "from": "layered",
            "to": "event_driven",
            "reason": "Better for real-time analytics"
        }
    ],
    "optimized_architecture": {
        "style": "event_driven",
        "paradigm": "object_oriented",
        ...
    }
}
```

### Conservative Approach

Το LLM προτείνει αλλαγές **μόνο αν**:
- Υπάρχει ξεκάθαρο architectural benefit
- Το project description υποστηρίζει την αλλαγή
- Το tradeoff είναι θετικό

---

## 📊 Constraints & Patterns

### Auto-Generated Constraints

Βάσει architecture style:

**Microservices:**
- Each service must have its own database
- Services communicate via events or HTTP
- No shared databases between services
- Services must be independently deployable

**Hexagonal:**
- Business logic must not depend on frameworks
- All external dependencies through ports/adapters
- Domain layer has no external dependencies

**Functional:**
- Prefer pure functions
- Minimize mutable state
- Use immutable data structures

### Recommended Patterns

- Repository Pattern
- Dependency Injection
- Factory Pattern
- Circuit Breaker (microservices)
- API Gateway (microservices)
- Event Sourcing (event-driven)

---

## 🔍 Loading & Using Rules

```python
from orchestrator import ArchitectureRulesEngine

engine = ArchitectureRulesEngine()

# Load existing rules
rules = engine.load_rules(Path("./output/.orchestrator-rules.yml"))

if rules:
    print(f"Architecture: {rules.architecture.style}")
    print(f"Stack: {rules.architecture.stack.primary_language}")
    
    # Check how decision was made
    if rules._llm_optimized:
        print("✨ This architecture was LLM-optimized")
    
    # Check constraints
    for constraint in rules.architecture.constraints:
        print(f"Constraint: {constraint}")
    
    # Access quality gates
    min_coverage = rules.quality_gates.get("test_coverage_min", 80.0)
    print(f"Min coverage: {min_coverage}%")
```

---

## 🎨 Integration με Orchestrator

Το Architecture Rules Engine είναι ενσωματωμένο στον Orchestrator:

```python
# Phase 0: Architecture Decision (automatic with optimization)
architecture_rules = await self._generate_architecture_rules(
    project_description,
    success_criteria,
    output_dir
)

# Phase 1: Decomposition (uses rules for constraints)
tasks = await self._decompose(
    project_description,
    success_criteria,
    architecture_rules=architecture_rules  # Pass rules to decomposition
)
```

---

## 📈 Benefits

1. **Consistency** - Κάθε project ακολουθεί standards
2. **Best Practices** - Αυτόματη εφαρμογή patterns
3. **Documentation** - Architecture decisions documented
4. **Quality Gates** - Enforced constraints
5. **Team Alignment** - Clear rules for all developers
6. **LLM Optimization** - Expert review των αποφάσεων

---

## 📚 Related

- [Architecture Advisor](./CAPABILITIES.md) - Original architecture decision
- [Project Analyzer](./PROJECT_ANALYZER.md) - Post-project analysis
- [Quality Control](./MANAGEMENT_SYSTEMS.md) - Constraint enforcement
- [Feature Documentation](./FEATURE_ARCHITECTURE_RULES.md) - Implementation details

---

**Version:** v5.2 | **Last Updated:** 2026-03-04

**🎉 Το Architecture Rules Engine με LLM Optimization είναι έτοιμο!**
