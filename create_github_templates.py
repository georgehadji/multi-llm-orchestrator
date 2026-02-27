"""Create GitHub templates and documentation."""
from pathlib import Path

# Create directories
github_dir = Path(".github")
issue_template_dir = github_dir / "ISSUE_TEMPLATE"
issue_template_dir.mkdir(parents=True, exist_ok=True)

# Bug report template
bug_report = '''name: Bug Report
description: Create a report to help us improve
title: "[BUG] "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!

  - type: textarea
    id: description
    attributes:
      label: Describe the bug
      description: A clear and concise description of what the bug is.
      placeholder: Tell us what happened...
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: To Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Run command '...'
        2. With configuration '...'
        3. See error
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
      description: A clear description of what you expected to happen.
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Logs/Error Messages
      description: Paste any relevant logs or error messages
      render: shell

  - type: input
    id: version
    attributes:
      label: Version
      description: What version are you running?
      placeholder: e.g., 1.1.0
    validations:
      required: true

  - type: dropdown
    id: python
    attributes:
      label: Python Version
      options:
        - "3.10"
        - "3.11"
        - "3.12"
        - "3.13"
    validations:
      required: true

  - type: checkboxes
    id: terms
    attributes:
      label: Checklist
      options:
        - label: I have searched existing issues
          required: true
        - label: I have provided all necessary information
          required: true
'''

# Feature request template
feature_request = '''name: Feature Request
description: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!

  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear description of what the problem is.
      placeholder: I'm always frustrated when...

  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear description of what you want to happen.
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: Any alternative solutions or features you've considered.

  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Add any other context about the feature request here.
'''

# PR template
pr_template = '''## Description
<!-- Provide a brief description of the changes -->

## Type of Change
<!-- Mark relevant items with [x] -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement
- [ ] Other (please describe):

## Checklist
<!-- Mark completed items with [x] -->
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Testing
<!-- Describe the tests you ran -->
- [ ] Ran `make ci` locally
- [ ] Tested with multiple providers
- [ ] Verified error handling

## Screenshots (if applicable)
<!-- Add screenshots to help explain your changes -->

## Related Issues
<!-- Link related issues using "Fixes #123" or "Relates to #123" -->
Fixes #

## Additional Notes
<!-- Any other information relevant to this PR -->
'''

# Write templates
(issue_template_dir / "bug_report.yml").write_text(bug_report, encoding='utf-8')
print("✓ Created: .github/ISSUE_TEMPLATE/bug_report.yml")

(issue_template_dir / "feature_request.yml").write_text(feature_request, encoding='utf-8')
print("✓ Created: .github/ISSUE_TEMPLATE/feature_request.yml")

(github_dir / "PULL_REQUEST_TEMPLATE.md").write_text(pr_template, encoding='utf-8')
print("✓ Created: .github/PULL_REQUEST_TEMPLATE.md")

# Create documentation files
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)

# ARCHITECTURE.md
architecture_content = '''# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / API                                │
│                    (orchestrator.cli)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Engine                           │
│                   (orchestrator.engine)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Planner   │  │   Router    │  │    Budget Manager       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Policy Engine  │ │  API Clients    │ │  State Manager  │
│ (orchestrator.  │ │ (orchestrator.  │ │ (orchestrator.  │
│  policy_engine) │ │  api_clients)   │ │  state)         │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  LLM Providers  │ │  Cache Layer    │ │  Telemetry      │
│ • DeepSeek      │ │ (aiosqlite)     │ │ (OTEL)          │
│ • OpenAI        │ └─────────────────┘ └─────────────────┘
│ • Google        │
│ • Kimi          │
│ • Minimax       │
│ • Zhipu         │
└─────────────────┘
```

## Key Components

### 1. Engine (`orchestrator.engine`)
The core orchestration logic. Manages the full project lifecycle:
- Decomposes projects into tasks
- Routes tasks to optimal providers
- Manages execution flow
- Handles retries and fallbacks

### 2. Router (`orchestrator.models`)
Intelligent model selection based on:
- Task type (CODE_GEN, REASONING, etc.)
- Cost optimization
- Quality requirements
- Provider availability

### 3. API Clients (`orchestrator.api_clients`)
Unified interface to all LLM providers:
- Consistent error handling
- Automatic retries with backoff
- Token usage tracking
- Rate limiting

### 4. Policy Engine (`orchestrator.policy_engine`)
Enforces governance rules:
- Budget constraints
- Model restrictions
- Content filtering
- Audit logging

### 5. State Manager (`orchestrator.state`)
Persistence layer for:
- Project state
- Task results
- Budget tracking
- Resume capability

## Data Flow

```
1. User Input → CLI
        ↓
2. Project Planning → Task Decomposition
        ↓
3. Task Routing → Model Selection
        ↓
4. API Call → LLM Provider
        ↓
5. Response Processing → Validation
        ↓
6. State Update → Persistence
        ↓
7. Output Generation
```

## Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Testability**: Dependency injection for easy mocking
3. **Observability**: Structured logging and distributed tracing
4. **Resilience**: Circuit breakers and graceful degradation
5. **Cost Awareness**: Budget tracking at all levels

## Extension Points

- **Custom Validators**: Add to `orchestrator.validators`
- **Custom Policies**: Implement `Policy` interface
- **New Providers**: Extend `api_clients.py`
- **Custom Routers**: Implement routing logic
'''

# ROUTING.md
routing_content = '''# Model Routing System

## Overview

The routing system intelligently selects the best LLM provider for each task based on:
- Task type and requirements
- Cost optimization
- Model capabilities
- Current provider health
- User preferences

## Routing Table

### CODE_GEN (Code Generation)
Priority: DeepSeek Coder → Minimax → Kimi → GLM-4 → GPT-4o

**Rationale**: DeepSeek Coder offers best cost/performance for code.

### REASONING (Complex Reasoning)
Priority: DeepSeek Reasoner → GPT-4o → Gemini Pro

**Rationale**: Reasoning models excel at step-by-step logic.

### WRITING (Content Creation)
Priority: GPT-4o → Gemini Pro → Kimi → GLM-4

**Rationale**: GPT-4o has best general writing quality.

### DATA_EXTRACT (Structured Data)
Priority: GPT-4o Mini → GPT-4o → Gemini Flash

**Rationale**: Smaller models sufficient for structured extraction.

## Fallback Chains

When a provider fails, the system automatically falls back:

```
DeepSeek Coder → GPT-4o → Gemini Pro
GPT-4o → Gemini Pro → GLM-4
Kimi → GLM-4 → GPT-4o Mini
```

## Cost Optimization

### Price Points (per 1M tokens)

| Provider | Input | Output | Cached |
|----------|-------|--------|--------|
| DeepSeek | $0.27 | $1.10 | $0.07 |
| Kimi | $0.14 | $0.56 | - |
| GPT-4o | $2.50 | $10.00 | $1.25 |
| Gemini Pro | $1.25 | $5.00 | - |

### Smart Routing

The router considers:
1. **Historical Performance**: EMA-tracked quality scores
2. **Budget Remaining**: Adjusts for available budget
3. **Task Complexity**: Simpler tasks → cheaper models
4. **Latency Requirements**: Faster models for time-sensitive tasks

## Configuration

```python
from orchestrator.models import Model, ROUTING_TABLE

# Default routing
routing = ROUTING_TABLE[TaskType.CODE_GEN]

# Custom routing
orchestrator = Orchestrator(
    custom_routing={
        TaskType.CODE_GEN: [Model.GPT_4O, Model.DEEPSEEK_CODER]
    }
)
```

## Monitoring

Track routing decisions via:
- Logs: `grep "routing_decision" logs/app.log`
- Metrics: `orchestrator.metrics`
- Telemetry: OpenTelemetry spans
'''

# POLICIES.md
policies_content = '''# Policy System

## Overview

The policy system enforces governance rules across the orchestrator. Policies can control:
- Budget limits
- Model access
- Content restrictions
- Rate limiting
- Audit requirements

## Policy Types

### BudgetPolicy
Controls spending at organization, team, and job levels.

```python
from orchestrator.policy import Policy, EnforcementMode

budget_policy = Policy(
    name="monthly_budget",
    scope="team",
    max_usd=100.0,
    enforcement=EnforcementMode.HARD,  # BLOCK on violation
)
```

### ModelPolicy
Restricts which models can be used.

```python
model_policy = Policy(
    name="approved_models",
    allowed_models=[Model.GPT_4O, Model.DEEPSEEK_CODER],
    blocked_models=[Model.GPT_4O_MINI],  # Too low quality
)
```

### ContentPolicy
Filters input/output content.

```python
content_policy = Policy(
    name="content_filter",
    blocked_patterns=[r"password", r"secret_key"],
    require_audit=True,
)
```

## Enforcement Modes

| Mode | Behavior |
|------|----------|
| HARD | Block execution on violation |
| SOFT | Warn but allow execution |
| MONITOR | Log only, no action |

## Policy Hierarchy

```
Organization Policies (broadest)
    ↓
Team Policies
    ↓
Job Policies (most specific)
```

Lower-level policies can override higher-level ones if not marked `inheritable=False`.

## Policy DSL

Define policies in YAML:

```yaml
# policies.yaml
policies:
  - name: budget_protection
    type: budget
    max_usd: 50.0
    enforcement: hard
    
  - name: model_restrictions
    type: model
    allowed:
      - gpt-4o
      - deepseek-coder
    blocked:
      - gpt-4o-mini
```

Load and apply:

```python
from orchestrator.policy_dsl import load_policy_file

policies = load_policy_file("policies.yaml")
orchestrator = Orchestrator(policies=policies)
```

## Audit Trail

All policy decisions are logged:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "policy": "budget_protection",
  "decision": "ALLOW",
  "reason": "Budget remaining: $23.45",
  "task_id": "task_001"
}
```

## Best Practices

1. **Start with MONITOR mode** to understand impact
2. **Use SOFT mode** for non-critical policies
3. **Set HARD mode** for compliance requirements
4. **Regular review** of policy effectiveness
'''

# Write documentation
(docs_dir / "ARCHITECTURE.md").write_text(architecture_content, encoding='utf-8')
print("✓ Created: docs/ARCHITECTURE.md")

(docs_dir / "ROUTING.md").write_text(routing_content, encoding='utf-8')
print("✓ Created: docs/ROUTING.md")

(docs_dir / "POLICIES.md").write_text(policies_content, encoding='utf-8')
print("✓ Created: docs/POLICIES.md")

print("\n✓ All GitHub templates and documentation created!")

# Self cleanup
Path(__file__).unlink()
