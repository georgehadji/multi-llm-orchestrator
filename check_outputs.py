import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from dotenv import load_dotenv; load_dotenv()
from orchestrator.state import StateManager

sm = StateManager()
state = sm.load_project('60175c0a5948')

task_names = {
    'task_001': 'FastAPI backend skeleton',
    'task_002': 'Typed node types (HTTP, LLM, Email, Sheets, Transform)',
    'task_003': 'Execution trace viewer',
    'task_004': 'React Flow frontend scaffold',
    'task_005': 'React node components (3+)',
    'task_006': 'Workflow JSON templates (5x)',
    'task_007': 'Python export + Docker microservice',
    'task_008': 'Pytest unit tests [FAILED]',
    'task_009': 'Code review - backend [FAILED]',
    'task_010': 'Code review - nodes [FAILED]',
    'task_011': 'Final evaluation [FAILED]',
}

for tid, result in state.results.items():
    name = task_names.get(tid, tid)
    print(f'\n{"="*60}')
    print(f'{tid}: {name}')
    print(f'Score: {result.score} | Model: {result.model_used.value}')
    print(f'{"="*60}')
    if result.output:
        print(result.output[:800])
        if len(result.output) > 800:
            print(f'... ({len(result.output)} chars total)')

sm.close()
