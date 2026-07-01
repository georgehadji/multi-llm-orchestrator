"""
Unit tests for ArchitectureScorer — rigorous, deterministic architecture scoring
of a generated SaaS/micro-SaaS project directory (0-100, also exposed as /10).

The scorer must give a *meaningful* signal: a well-layered, tested, config-clean
project scores high (>9/10); a flat god-file with a hardcoded secret and no tests
scores low. Both must be derivable from static analysis only (no LLM, no network).
"""

import pytest


@pytest.fixture(scope="module")
def Scorer():
    from orchestrator.safety.architecture_scorer import ArchitectureScorer

    return ArchitectureScorer


def _write(root, rel, content=""):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_good_project(root):
    """A clean layered FastAPI-style micro-SaaS."""
    _write(
        root,
        "app/domain/models.py",
        "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    id: str\n    email: str\n",
    )
    _write(root, "app/domain/__init__.py")
    _write(
        root,
        "app/application/user_service.py",
        "from ..domain.models import User\n\n\ndef create_user(email: str) -> User:\n    return User(id='1', email=email)\n",
    )
    _write(root, "app/application/__init__.py")
    _write(
        root,
        "app/infrastructure/user_repository.py",
        "from ..domain.models import User\n\n\nclass UserRepository:\n    def save(self, user: User) -> None:\n        ...\n",
    )
    _write(root, "app/infrastructure/__init__.py")
    _write(
        root,
        "app/api/routes.py",
        "from ..application.user_service import create_user\n\n\ndef register(email: str):\n    try:\n        return create_user(email)\n    except ValueError as e:\n        raise e\n",
    )
    _write(root, "app/api/__init__.py")
    _write(root, "app/__init__.py")
    _write(
        root,
        "tests/test_user_service.py",
        "from app.application.user_service import create_user\n\n\ndef test_create_user():\n    assert create_user('a@b.com').email == 'a@b.com'\n",
    )
    _write(root, "tests/test_routes.py", "def test_register():\n    assert True\n")
    _write(
        root,
        "main.py",
        "from app.api.routes import register\n\nif __name__ == '__main__':\n    print('run')\n",
    )
    _write(root, "requirements.txt", "fastapi==0.110.0\nuvicorn==0.29.0\n")
    _write(root, ".env.example", "DATABASE_URL=\nAPI_KEY=\n")
    _write(root, "README.md", "# My SaaS\n\n" + ("A production-grade micro-SaaS. " * 20))
    _write(root, "config.py", "import os\n\nDATABASE_URL = os.environ['DATABASE_URL']\n")


@pytest.mark.unit
class TestArchitectureScorer:
    def test_good_project_scores_above_nine(self, Scorer, tmp_path):
        _make_good_project(tmp_path)
        result = Scorer().score(tmp_path)
        assert (
            result.total > 90
        ), f"expected >90, got {result.total}: {[(d.name, d.score) for d in result.dimensions]}"
        assert result.out_of_ten > 9.0
        assert abs(result.out_of_ten - result.total / 10) < 1e-6

    def test_poor_project_scores_low(self, Scorer, tmp_path):
        _write(tmp_path, "app.py", "x = 1\n" * 1200)  # one giant flat god-file
        result = Scorer().score(tmp_path)
        assert result.total < 50, f"expected <50, got {result.total}"
        assert result.out_of_ten < 5.0

    def test_hardcoded_secret_is_flagged_and_penalized(self, Scorer, tmp_path):
        _make_good_project(tmp_path)
        clean = Scorer().score(tmp_path).total
        _write(
            tmp_path,
            "app/leak.py",
            'OPENAI_API_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr"\n',
        )
        leaked = Scorer().score(tmp_path)
        assert leaked.total < clean
        assert any("secret" in w.lower() or "hardcoded" in w.lower() for w in leaked.weaknesses)

    def test_dependency_direction_violation_detected(self, Scorer, tmp_path):
        # domain importing infrastructure is an inward-dependency violation
        _write(tmp_path, "app/domain/models.py", "from app.infrastructure.db import conn\n")
        _write(tmp_path, "app/infrastructure/db.py", "conn = object()\n")
        result = Scorer().score(tmp_path)
        dep = next(d for d in result.dimensions if "depend" in d.name.lower())
        assert dep.score < dep.max_score
        assert any("depend" in w.lower() or "layer" in w.lower() for w in result.weaknesses)

    def test_missing_tests_lowers_score_and_recommends(self, Scorer, tmp_path):
        _make_good_project(tmp_path)
        # remove tests
        for f in (tmp_path / "tests").glob("*.py"):
            f.unlink()
        result = Scorer().score(tmp_path)
        tests_dim = next(d for d in result.dimensions if "test" in d.name.lower())
        assert tests_dim.score < tests_dim.max_score
        assert any("test" in r.lower() for r in result.recommendations)

    def test_empty_project_does_not_crash(self, Scorer, tmp_path):
        result = Scorer().score(tmp_path)
        assert 0 <= result.total <= 100
