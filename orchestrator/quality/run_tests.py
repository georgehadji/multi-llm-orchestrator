"""
Minimal stub for running project tests.
"""


def run_project_tests(
    project_dir: str, *, fixtures: list[str] | None = None
) -> list[dict[str, object]]:
    """Run project tests if possible.

    This is a lightweight placeholder so higher-level CLI code can import the helper.
    """
    print(f"[run_tests] Tests requested for {project_dir}, but no runner is configured.")
    return []
