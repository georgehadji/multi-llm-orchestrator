"""SEC-001 — one IDE server, one CORS policy, no wildcard-plus-credentials.

T1 hardened ``ide_backend/server.py``. It missed two sibling implementations:

* ``standalone_server.py`` — launched by ``start-ide.bat``, i.e. the path users
  actually run. Wildcard CORS, credentials on, bound to ``0.0.0.0``.
* ``ide_orchestrator_server.py`` — a second FastAPI app with the same three
  problems plus a ``/health`` route leaking the live session count.

``allow_origins=["*"]`` with ``allow_credentials=True`` makes every site the
user visits a same-origin client of the API. Browsers reject that combination,
but only browsers do: it is a policy statement the server should never make.

Scanned as text rather than AST on purpose — the generated-app templates in
``generators/multi_platform_generator.py`` are string literals, and shipping the
anti-pattern to every generated project counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "orchestrator"

WILDCARD_ORIGINS = re.compile(r"allow_origins\s*=\s*\[\s*[\"']\*[\"']\s*\]")

#: Modules allowed to name the anti-pattern rather than commit it:
#: `security.py` explains it in its docstring, and the output scanner carries
#: the detector regex for it (`cors-wildcard-credentials`, HIGH).
DOCUMENTED_EXCEPTIONS = {
    Path("orchestrator/ide_backend/security.py"),
    Path("orchestrator/safety/generated_output_scanner.py"),
}


def test_no_wildcard_cors_anywhere_in_the_package() -> None:
    offenders: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if rel in DOCUMENTED_EXCEPTIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # A comment naming the anti-pattern is documentation, not a policy.
            if line.lstrip().startswith("#"):
                continue
            if WILDCARD_ORIGINS.search(line):
                offenders.append(f"{rel}:{lineno}")

    assert not offenders, (
        'allow_origins=["*"] with credentials is banned (SEC-001) — use an '
        "explicit allowlist, and an env-driven one in generated templates:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "module",
    ["orchestrator.ide_backend.standalone_server"],
)
def test_duplicate_ide_servers_stay_deleted(module: str) -> None:
    """One IDE server. A second copy is a second security policy to forget."""
    import importlib.util

    assert importlib.util.find_spec(module) is None, (
        f"{module} is back — the hardened entry point is "
        "orchestrator.ide_backend.launch (server.py)."
    )


def test_legacy_module_exposes_no_server() -> None:
    """`ide_orchestrator_server` is library code now, not an entry point.

    Its generators and helpers are still under test, so the module stays; the
    FastAPI app, its CORS middleware, its routes and its ``__main__`` block do
    not.
    """
    import ast

    source = (PACKAGE_ROOT / "ide_backend" / "ide_orchestrator_server.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)

    # AST, not regex: the generated-app template this module emits is a string
    # literal containing its own `app = FastAPI(...)` and `if __name__ ==
    # "__main__"` at column 0. That is template text, not this module's code.
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            call = node.value
            is_fastapi = (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "FastAPI"
            )
            assert not (
                "app" in targets and is_fastapi
            ), f"line {node.lineno}: the legacy module builds a FastAPI app again"

        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            )
            assert not is_main_guard, f"line {node.lineno}: the legacy module is runnable again"


def test_launcher_uses_the_hardened_entry_point() -> None:
    launcher = REPO_ROOT / "start-ide.bat"
    text = launcher.read_text(encoding="utf-8", errors="replace")

    assert (
        "standalone_server" not in text
    ), "start-ide.bat still launches the deleted standalone server"
    assert (
        "orchestrator.ide_backend.launch" in text
    ), "start-ide.bat should launch the hardened entry point"
