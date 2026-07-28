# Implementation Plan: Codebase-Aware Token Minimization & Optimizations
**Target System**: Multi-LLM Orchestrator  
**Author**: Gemini CLI (Specialist AI Agent)  
**Date**: Tuesday, July 28, 2026  

---

## 1. Executive Summary

Codebase-aware AI operations (`analyze` and `modify`) inherently struggle with high token consumption. Large file systems, structural indexes, and recursive critique loops can consume millions of tokens in a single execution. 

This implementation plan outlines the architecture for applying **four synergistic token minimization techniques** to the Orchestrator's codebase-aware pipelines:
1. **Dynamic Context Slicing (Semantic Slicing)**: Reduces base input tokens by up to 70% by pruning non-essential files based on task-types and dependency paths.
2. **Ephemeral Prompt Caching (Warm Prefixes)**: Cuts sequential task input costs by 80–90% by caching static system instructions and codebase context.
3. **Unified Diff & Block Patching (Diff-Only Synthesis)**: Slashes output token costs by up to 95% by generating only git-style diffs rather than full file rewrites.
4. **Decomposer Task Consolidation (Graph Optimization)**: Minimizes orchestration overhead and critique cycles by grouping micro-tasks at the target-file level.

These optimizations are aligned with the existing **Domain-Driven Design (DDD)** structure of the project, using clean abstractions, dependency inversion, and highly modular design patterns.

---

## 2. Software Architecture & Optimization Mapping

The optimizations span across three main layers of the AI Orchestrator architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Entrypoints                               │
│                (cli.py, commands/codebase.py, main.py)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Calls
┌───────────────────────────────────▼────────────────────────────────────┐
│                           Orchestrator Engine                          │
│               (orchestrator/engine.py, ProjectRunner)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Coordinates
┌───────────────────────────────────▼────────────────────────────────────┐
│                       Domain Services / Packages                       │
│                                                                        │
│ ┌──────────────────────────┐ ┌──────────────────────────┐ ┌──────────┐ │
│ │   orchestrator/codebase/ │ │orchestrator/cost_optim/  │ │models.py │ │
│ │  (Decomposer, Context,   │ │ (PromptCacher,           │ │(Task,    │ │
│ │   Reader, Writer)        │ │  SpeculativeGenerator)   │ │TaskType) │ │
│ └──────────────────────────┘ └──────────────────────────┘ └──────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Mapping Matrix

| Optimization | Targeted Component | Files to Edit/Create | Design Pattern Applied |
| :--- | :--- | :--- | :--- |
| **Dynamic Slicing** | `CodebaseContext`, `CodebaseReader` | `orchestrator/codebase/context.py` | **Strategy Pattern** |
| **Prompt Caching** | `Orchestrator`, `PromptCacher` | `orchestrator/engine.py` | **Proxy / Decorator Pattern** |
| **Diff-Only Output** | `CodebaseWriter`, `DiffEngine` | `orchestrator/codebase/writer.py` | **Command / Builder Pattern** |
| **Consolidation** | `CodebaseDecomposer` | `orchestrator/codebase/decomposer.py` | **Composite / Pipe-and-Filter** |

---

## 3. Detailed Modules Specification

### 3.1 Dynamic Context Slicing (Semantic Slicing)

#### Architectural Intent
Currently, `CodebaseContext` builds a general project summary of up to 60,000 tokens containing every file it can fit. Instead, we must dynamically adapt the codebase profile to the specific task:
1. **Dependency Installation (`install_dependency`)**: Slice context exclusively to lockfiles (`pyproject.toml`, `requirements.txt`, etc.).
2. **File Deletion (`delete_file`)**: Slice context to the path itself, its direct imports, and its immediate dependents in the Dependency Graph.
3. **File Editing (`modify_file`)**: Prioritize the target file, its AST-extracted class/function signatures, and sibling files within its direct import path.

#### Implementation Design
We will introduce `ContextSlicingStrategy` inside `orchestrator/codebase/context.py` using the **Strategy Pattern**:

```python
class ContextSlicingStrategy(abc.ABC):
    @abc.abstractmethod
    def slice_files(self, reader: CodebaseReader, target_path: str) -> list[FileNode]:
        pass

class ModifyFileSlicingStrategy(ContextSlicingStrategy):
    """Slices context around a target file, prioritizing AST imports and dependents."""
    def slice_files(self, reader: CodebaseReader, target_path: str) -> list[FileNode]:
        # 1. Add target file
        # 2. Extract AST imports from reader.symbols[target_path]
        # 3. Pull immediate neighbors from reader.graph
        # 4. Return localized file subset (typically 3-5 files instead of 100)
```

---

### 3.2 Ephemeral Prompt Caching (Warm Prefixes)

#### Architectural Intent
During the execution of a multi-task modification plan, the local Orchestrator executes tasks sequentially or in parallel levels. We must inject a cache-warming step to ensure the heavy, sliced `CodebaseContext` is persisted as a cached prompt block on the LLM provider's side.

#### Implementation Design
Using the **Proxy/Decorator Pattern**, wrap task execution inside `orchestrator/engine.py` to transparently pre-warm prompt contexts:

```python
# Inside orchestrator/engine.py -> modify_codebase()
from orchestrator.cost_optimization.prompt_cache import PromptCacher

cacher = PromptCacher(client=self.client)

# Build unified, cacheable system prefix
system_prefix = (
    "You are a senior software engineer modifying an existing codebase.\n"
    f"## Codebase Context\n{context.to_llm_prompt()}"
)

# Pre-warm cache for providers supporting it (Anthropic, OpenAI)
await cacher.warm_cache(system_prompt=system_prefix, project_context=objective)
```

By ensuring that the cache is pre-warmed before looping through the execution levels, subsequent tasks mapped to `anthropic/` or `openai/` will register cache hits for the heavy contextual blocks.

---

### 3.3 Unified Diff & Search/Replace Block Generation

#### Architectural Intent
Generating complete files during `modify_file` tasks is highly expensive and highly error-prone. By instructing the model to output *only* unified git-style diffs or specific `SEARCH / REPLACE` blocks (such as those popularized by tools like `Aider`), we compress output tokens from thousands to double digits.

#### Implementation Design
1. **Update Prompts**: Enhance `decomposer.py` or the generation prompt templates to request concise, standard diff blocks:
   ```
   <<<<<<< SEARCH
   original lines
   =======
   modified lines
   >>>>>>> REPLACE
   ```
2. **Enhance `DiffEngine` / `CodebaseWriter`**: Adapt `orchestrator/codebase/writer.py` to parse search/replace patterns and patch the target files in-memory before validation:
   ```python
   # Inside CodebaseWriter.apply() for TaskType.MODIFY_FILE
   # 1. Extract SEARCH/REPLACE block
   # 2. Match original lines in the target path
   # 3. Safely replace with modified lines
   # 4. Write patched file and pass it to the verification gate
   ```

This represents the single biggest **speed and output cost improvement** achievable in codebase editing.

---

### 3.4 Decomposer Task Consolidation

#### Architectural Intent
If a modification plan creates separate, dependency-chained tasks to modify the exact same file (e.g., adding imports, then modifying functions, then adding comments), the orchestrator must pay high pricing overheads for separate critique, revision, and testing gates.

#### Implementation Design
In `orchestrator/codebase/decomposer.py`, introduce a post-decomposition **Pipe-and-Filter consolidation stage** that groups tasks targeting the same file:

```python
# Inside CodebaseDecomposer._parse_response
# Group all MODIFY_FILE tasks by task.target_path
grouped_tasks: dict[str, list[Task]] = {}
for task in parsed_tasks:
    if task.type == TaskType.MODIFY_FILE and task.target_path:
        grouped_tasks.setdefault(task.target_path, []).append(task)

# Consolidate grouped tasks
for path, sub_tasks in grouped_tasks.items():
    if len(sub_tasks) > 1:
        # Merge sub_tasks into a single aggregate Task
        # Combine prompts: "1. {task1.prompt}\n2. {task2.prompt}"
        # Keep dependencies of all original tasks
```

---

## 4. Programming Paradigms & Design Patterns

We will strictly enforce software engineering standards to maintain the platform's architectural integrity:

1. **Async Concurrency (`asyncio`)**: Slicing and caching steps must be fully async-native to prevent UI freeze and maintain compatibility with CLI/API dashboard execution.
2. **Strategy Pattern**: Prompts and slicing policies must be encapsulated in interchangeable strategies (`ContextSlicingStrategy`) rather than large, complex `if/elif` blocks inside context structures.
3. **Decorator/Proxy Pattern**: Caching should wrap standard client execution streams transparently, keeping engine code dry and highly maintainable.
4. **Defensive Programming**: Diffs must fail-safe. If search/replace blocks do not find exact matches, the `CodebaseWriter` must abort cleanly, raise a detailed error, and trigger self-correcting agents to retry with looser constraints, preventing data loss.

---

## 5. Verification & Testing Plan

Our verification process covers syntax validation, unit testing, and empirical evaluation:

### Phase 1: Unit Testing (Mocked LLM)
Create a new test file `tests/test_codebase_optimizations.py` to verify:
*   `ModifyFileSlicingStrategy` selects correct AST neighbors and prunes unrelated files.
*   `CodebaseDecomposer` successfully consolidates multiple sub-tasks affecting `app.py` into a single task with unified requirements.
*   `SEARCH/REPLACE` parsing logic in `DiffEngine` handles edge-cases (whitespace mismatches, line-endings) and rolls back cleanly on mismatch.

### Phase 2: Static Verification & Linters
Run project validation checks to ensure zero regressions across other layers:
```bash
ruff check .
mypy orchestrator/
pytest tests/test_phase6_10_comprehensive.py
```

### Phase 3: Empirical Benchmarking
Run a dry-run modification operation on a local sample repository and log token cost savings:
```bash
python -m orchestrator modify --repo "./my-test-app" --objective "Add JWT auth" --dry-run
# Compare generated token counts against un-optimized baselines.
```

---
*End of Design Document.*
