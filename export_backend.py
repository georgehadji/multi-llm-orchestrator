import asyncio
import sys
from pathlib import Path
from orchestrator.infrastructure.state import StateManager
from orchestrator.output_writer import write_output_dir

async def main():
    sm = StateManager()
    await sm._get_conn()
    project_id = "backend-rest-api-v1"
    state = await sm.load_project(project_id)
    if state:
        print(f"Loaded project: {project_id}")
        print(f"Tasks: {len(state.tasks)}, Results: {len(state.results)}")
        out = Path("./outputs/backend_rest_api")
        path = write_output_dir(state, out, project_id=project_id)
        print(f"Successfully wrote output to: {path}")
    else:
        print("Project state not found.")

if __name__ == "__main__":
    asyncio.run(main())
