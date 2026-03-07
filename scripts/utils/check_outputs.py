# Author: Georgios-Chrysovalantis Chatzivantsidis
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
            print("No saved projects. Usage: python check_outputs.py <project_id>")
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

    for tid, result in state.results.items():
        print(f'\n{"="*60}')
        print(f'{tid}')
        print(f'Score: {result.score} | Model: {result.model_used.value} | Status: {result.status.value}')
        print(f'{"="*60}')
        if result.output:
            print(result.output[:800])
            if len(result.output) > 800:
                print(f'... ({len(result.output)} chars total)')
        else:
            print('(no output)')

    sm.close()

if __name__ == '__main__':
    main()
