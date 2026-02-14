import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
from dotenv import load_dotenv; load_dotenv()
from orchestrator.state import StateManager

def main():
    if len(sys.argv) < 2:
        sm = StateManager()
        projects = sm.list_projects()
        if not projects:
            print("No saved projects. Usage: python check_state.py <project_id>")
        else:
            print("Available projects (pass one as argument):")
            print(f"{'ID':<15} {'Status':<20} {'Updated'}")
            print("-" * 55)
            from datetime import datetime
            for p in projects:
                updated = datetime.fromtimestamp(p["updated_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{p['project_id']:<15} {p['status']:<20} {updated}")
        sm.close()
        return

    project_id = sys.argv[1]
    sm = StateManager()
    state = sm.load_project(project_id)

    if state is None:
        print(f"Project '{project_id}' not found.")
        sm.close()
        sys.exit(1)

    print('Status:', state.status.value)
    print(f'Budget spent: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}')
    print(f'Tasks total: {len(state.tasks)} | Results: {len(state.results)}')
    print()
    print('=== TASK RESULTS ===')
    for tid, result in state.results.items():
        mark = 'OK' if result.status.value == 'completed' else 'FAIL' if result.status.value == 'failed' else 'DEG'
        print(f'[{mark}] {tid}: score={result.score:.3f} model={result.model_used.value} iters={result.iterations} cost=${result.cost_usd:.4f}')
        if result.status.value == 'failed' and result.critique:
            print(f'       critique: {result.critique[:100]}')

    print()
    print('=== ALL TASKS (with type & deps) ===')
    for tid, task in state.tasks.items():
        done = tid in state.results
        status_str = state.results[tid].status.value if done else 'pending'
        print(f'  {tid}: [{task.type.value}] status={status_str} deps={task.dependencies}')

    sm.close()

if __name__ == '__main__':
    main()
