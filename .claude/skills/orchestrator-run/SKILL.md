# orchestrator-run

Quick reference for running the AI Orchestrator CLI. Use this skill when the user wants to run a project, resume one, analyze a codebase, or invoke any CLI subcommand.

## Entry points

```bash
python -m orchestrator <flags>   # module invocation (always works)
orchestrator <flags>             # installed script
mllm <flags>                     # alias
```

## Common invocations

| Goal | Command |
|------|---------|
| New project | `python -m orchestrator --project "<desc>" --criteria "<criteria>" --budget 2.0` |
| Resume crashed run | `python -m orchestrator --resume <project_id>` |
| List all projects | `python -m orchestrator --list-projects` |
| Analyze codebase | `python -m orchestrator analyze <path>` |
| Build app from desc | `python -m orchestrator build "<desc>"` |
| Agent mode | `python -m orchestrator agent "<intent>"` |
| Open dashboard | `python -m orchestrator dashboard` |
| Cache statistics | `python -m orchestrator cache-stats` |
| NexusScope sessions | `python -m orchestrator nexusscope sessions` |
| NexusScope report | `python -m orchestrator nexusscope report --format html --output profile.html` |
| Nash status | `python -m orchestrator nash status` |
| Kanban queue | `python -m orchestrator kanban enqueue "<desc>"` |
| Web search | `python -m orchestrator nexus search "<query>"` |

## Required flags for `--project`

- `--project "<desc>"` — plain-language description of what to build
- `--budget <float>` — USD ceiling (default 1.0); **always set explicitly**
- `--criteria "<text>"` — acceptance criteria (optional but recommended)

## Optional flags

```bash
--time <seconds>          # wall-clock time ceiling
--output-dir <path>       # where to write generated files
--tdd-first               # enforce TDD pipeline
--tracing                 # enable OpenTelemetry tracing
--profile                 # enable NexusScope profiling (requires .[profiling])
--profile-output <path>   # write profile report here on exit
--profile-format text|html|json|speedscope
--visualize               # generate dependency graph SVG
--critical-path           # highlight critical path in graph
```

## Environment variables

```bash
OPENROUTER_API_KEY=sk-or-...     # required (primary)
OPENAI_API_KEY=sk-...            # alternative
GOOGLE_API_KEY=...               # for Gemini models
ANTHROPIC_API_KEY=sk-ant-...     # for Claude models
ORCHESTRATOR_PROFILING=1         # enable NexusScope (same as --profile)
ORCHESTRATOR_LOG_LEVEL=DEBUG     # verbose output
```

## Install with all extras

```bash
pip install -e ".[dev,security,tracing,dashboard,profiling]"
```
