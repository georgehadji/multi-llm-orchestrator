import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from dotenv import load_dotenv; load_dotenv()
from orchestrator.state import StateManager

sm = StateManager()
state = sm.load_project('60175c0a5948')

print('Status:', state.status.value)
print(f'Budget spent: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}')
print(f'Tasks total: {len(state.tasks)} | Results: {len(state.results)}')
print()
print('=== TASK RESULTS ===')
for tid, result in state.results.items():
    mark = 'OK' if result.status.value == 'completed' else 'FAIL' if result.status.value == 'failed' else 'DEG'
    print(f'[{mark}] {tid}: score={result.score:.3f} model={result.model_used.value} iters={result.iterations} cost=${result.cost_usd:.4f}')
    if result.status.value == 'failed':
        print(f'       critique: {result.critique[:100]}')

print()
print('=== ALL TASKS (with type & deps) ===')
for tid, task in state.tasks.items():
    done = tid in state.results
    status_str = state.results[tid].status.value if done else 'pending'
    print(f'  {tid}: [{task.type.value}] status={status_str} deps={task.dependencies}')

sm.close()
