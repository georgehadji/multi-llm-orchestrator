"""
T7 (execution & filesystem surface) proof-of-defect and no-regression tests.

One VERIFIED DEFECT from docs/hunts/t7-execution/inventory.md:

C1 — orchestrator/appbuilder/verifier.py had silently diverged from the
     canonical orchestrator/app_verifier.py (the module orchestrator/
     appbuilder/builder.py::AppBuilder actually constructs its verifier
     from). The appbuilder/ copy was missing a fix: verify_local() builds
     a `requirements.txt` path and passes it to `subprocess.run(...,
     cwd=output_dir)` for `pip install -r <path>`. If that path is
     relative, the subprocess resolves it a SECOND time against
     `cwd=output_dir`, double-nesting and failing to find the file. The
     canonical module fixes this with `str(req_file.resolve())`; the
     appbuilder/ copy still used the unresolved `str(req_file)`.

     Because orchestrator/appbuilder/__init__.py does
     `from .verifier import *` *after* `from .builder import *`, this
     divergence also meant `orchestrator.appbuilder.AppVerifier` — the
     package's own public name — resolved to the buggy class, even though
     `AppBuilder` itself (in builder.py) always constructed the correct,
     canonical one directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.app_detector import AppProfile

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_appbuilder_verifier_module_is_canonical():
    from orchestrator.app_verifier import AppVerifier as canonical
    from orchestrator.app_verifier import VerifyReport as canonical_report
    from orchestrator.appbuilder.verifier import AppVerifier as via_appbuilder
    from orchestrator.appbuilder.verifier import VerifyReport as via_appbuilder_report

    assert via_appbuilder is canonical
    assert via_appbuilder_report is canonical_report


@pytest.mark.unit
def test_c1_appbuilder_package_exposes_canonical_appverifier():
    """The core defect: orchestrator.appbuilder's own public AppVerifier name
    (what a caller doing `from orchestrator.appbuilder import AppVerifier`
    would get) must be the same class AppBuilder itself constructs."""
    import orchestrator.appbuilder as appbuilder_pkg
    from orchestrator.app_verifier import AppVerifier as canonical

    assert appbuilder_pkg.AppVerifier is canonical


@pytest.mark.unit
def test_c1_pip_install_uses_absolute_requirements_path(tmp_path, monkeypatch):
    """Real trigger, not simulated: verify_local() is called with a
    *relative* Path for output_dir (as a caller invoking it from a project
    root plausibly would), and the pip-install subprocess.run call it
    issues is inspected directly. Pre-fix, the appbuilder/ copy passed the
    *unresolved* relative path straight through — which, combined with
    cwd=output_dir in the same subprocess.run call, double-resolves and
    fails to open in a real subprocess. Post-fix, the path argument must be
    absolute regardless of how output_dir was spelled by the caller.
    """
    from orchestrator.appbuilder.verifier import AppVerifier

    monkeypatch.chdir(tmp_path)
    output_dir = Path("generated_app")
    (tmp_path / output_dir).mkdir()
    (tmp_path / output_dir / "requirements.txt").write_text("requests\n", encoding="utf-8")
    profile = AppProfile(app_type="script", test_command="pytest", run_command="")

    calls = []

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(args, **kwargs):
        calls.append(args)
        return _FakeCompleted()

    assert not output_dir.is_absolute(), "test setup must pass a relative output_dir"
    with patch("subprocess.run", side_effect=_fake_run):
        AppVerifier().verify_local(output_dir, profile)

    pip_calls = [c for c in calls if "pip" in c]
    assert pip_calls, f"expected a pip install subprocess call, got: {calls}"
    req_path_arg = pip_calls[0][-1]
    assert Path(req_path_arg).is_absolute(), (
        f"pip install requirements path must be absolute (cwd=output_dir "
        f"would double-resolve a relative one), got: {req_path_arg}"
    )
