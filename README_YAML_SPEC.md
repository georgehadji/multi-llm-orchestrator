# 📄 Project Specification YAML Format

The **YAML Spec** tab allows you to upload a complete project specification in YAML format.

## Supported Fields

```yaml
name: "Project Name"                    # Required - Project title
description: "What to build"             # Required - Detailed description
tech_stack:                              # Optional - Technologies to use
  - "Python/FastAPI"
  - "PostgreSQL"
  - "React"

requirements:                              # Optional - List of requirements
  - "User authentication"
  - "CRUD operations"
  - "API documentation"

features:                                  # Optional - Key features
  - "Login system"
  - "Dashboard"
  - "Reports"

architecture:                              # Optional - Architecture notes
  - "Clean architecture"
  - "Repository pattern"

success_criteria: "Tests pass 80%"        # Optional - Completion criteria
budget_usd: 5.0                           # Optional - Budget (can override in UI)
```

## Example Files

See `example_project_spec.yaml` for a complete example.

## Alternative Formats

You can also upload:
- **.json** - JSON format with same fields
- **.md** - Markdown file with project description
- **.txt** - Plain text specification

## How It Works

1. Create your spec file
2. Upload via the "YAML Spec" tab
3. Add optional additional instructions
4. Set budget
5. Click "Start Project from Spec"

The orchestrator will use the specification to generate the complete project.
