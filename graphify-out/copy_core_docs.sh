python -c "
import shutil, os
repo = r'C:\Users\tesse\.graphify\repos\nousresearch\hermes-agent'
dst = r'E:\Documents\Vibe-Coding\Ai Orchestrator\graphify-out'
core_files = [
    'AGENTS.md', 'pyproject.toml', 'agent/__init__.py',
    'agent/agent_runtime_helpers.py', 'agent/conversation_loop.py',
    'agent/tool_executor.py', 'agent/skill_utils.py',
    'agent/memory_manager.py', 'agent/system_prompt.py',
    'agent/context_engine.py', 'agent/prompt_builder.py',
    'agent/trajectory.py', 'agent/async_utils.py',
    'hermes_cli/__init__.py', 'hermes_cli/main.py', 'hermes_cli/commands.py',
    'hermes_constants.py', 'hermes_logging.py', 'hermes_state.py',
    'run_agent.py', 'cli.py'
]
for f in core_files:
    src = os.path.join(repo, f)
    if os.path.exists(src):
        name = f.replace('/', '_').replace('\\', '_')
        shutil.copy2(src, os.path.join(dst, f'hermes_{name}'))
        print(f'Copied {f}')
    else:
        print(f'Missing {f}')
"