"""Unit tests for is_web_frontend_task helper."""

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "prompt,path,expected",
    [
        ("build a landing page in HTML and CSS", "", True),
        ("write javascript for the cart widget", "", True),
        ("create App.tsx for the auth flow", "src/App.tsx", True),
        ("style the navbar with CSS", "styles/navbar.css", True),
        ("add a React component for login", "src/Login.jsx", True),
        ("build a FastAPI backend", "app/main.py", False),
        ("create a Django model for users", "models.py", False),
        ("write unit tests in Python", "tests/test_auth.py", False),
        ("implement authentication with Flask", "app/auth.py", False),
        ("pure Python data pipeline", "pipeline.py", False),
        ("build a Vue dashboard", "src/Dashboard.vue", True),
        ("create a Svelte form", "src/Form.svelte", True),
    ],
)
def test_frontend_detection(prompt, path, expected):
    from orchestrator.design.frontend_detect import is_web_frontend_task

    assert is_web_frontend_task(prompt, path) == expected


@pytest.mark.unit
def test_empty_inputs_are_not_frontend():
    from orchestrator.design.frontend_detect import is_web_frontend_task

    assert is_web_frontend_task("", "") is False


@pytest.mark.unit
def test_fullstack_with_html_and_python_backend():
    from orchestrator.design.frontend_detect import is_web_frontend_task

    # FastAPI + HTML mention — backend keyword + .py path → not frontend
    assert is_web_frontend_task("build a FastAPI app that serves HTML", "app/main.py") is False
