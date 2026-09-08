"""
ArchitectureScorer — deterministic architecture quality scoring
================================================================

Scores a generated project directory (SaaS / micro-SaaS) on objective,
static-analysis criteria — no LLM, no network. Produces a 0-100 total (also
exposed as /10) with a per-dimension breakdown and concrete
strengths / weaknesses / recommendations.

Dimensions (weights sum to 100):

    layering              20   distinct architectural layers present
    dependency_direction  20   inner layers must not import outer (hexagonal)
    modularity            15   file (<800 LOC) and function (<50 LOC) discipline
    tests                 15   test files present, healthy test:source ratio
    config_secrets        10   env template, config separation, NO hardcoded secrets
    runnability           10   dependency manifest + entry point + pinned deps
    documentation          5   non-trivial README
    error_handling         5   error handling present across source files

A score > 90 (i.e. > 9/10) marks a project as architecturally production-grade.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Layer taxonomy (hexagonal: domain is innermost, adapters are outermost) ──
# rank: lower = more inner. A file may NOT import a layer of higher rank.
_LAYER_KEYWORDS: dict[str, tuple[int, tuple[str, ...]]] = {
    "domain": (0, ("domain", "models", "model", "entities", "entity", "schemas", "schema")),
    "application": (1, ("application", "services", "service", "usecases", "use_cases", "logic")),
    "presentation": (
        2,
        (
            "api",
            "routes",
            "controllers",
            "views",
            "pages",
            "handlers",
            "endpoints",
            "interfaces",
            "web",
            "ui",
        ),
    ),
    "infrastructure": (
        2,
        (
            "infrastructure",
            "infra",
            "repositories",
            "repository",
            "repos",
            "database",
            "adapters",
            "persistence",
            "dao",
        ),
    ),
}

_CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}
_IGNORE_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    "coverage_html",
    ".mypy_cache",
    ".pytest_cache",
}

_MAX_FILE_LOC = 800
_MAX_FUNC_LOC = 50

# A project at or above this /10 score is considered architecturally
# production-grade. Used as the quality-gate threshold by callers.
PRODUCTION_GRADE_THRESHOLD = 9.0

# Hardcoded-secret heuristics (avoid env reads / empty assignments).
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(
        r"\b(?:api[_-]?key|apikey|secret|password|passwd|token|access[_-]?key)\s*"
        r"[:=]\s*['\"][^'\"]{8,}['\"]",
        re.IGNORECASE,
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
]
# Allow obvious non-secrets so env reads / placeholders don't trip the scanner.
_SECRET_ALLOW = re.compile(
    r"os\.environ|getenv|process\.env|example|placeholder|your[_-]?key" r"|xxx|<.*>|\$\{",
    re.IGNORECASE,
)


@dataclass
class DimensionScore:
    """Score for a single architecture dimension."""

    name: str
    score: float
    max_score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ArchitectureScore:
    """Full architecture scoring result for a project directory."""

    total: float  # 0-100
    out_of_ten: float  # 0-10
    pattern: str
    dimensions: list[DimensionScore]
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_production_grade(self) -> bool:
        """True when the project meets the >9/10 production-grade bar."""
        return self.out_of_ten >= PRODUCTION_GRADE_THRESHOLD


class ArchitectureScorer:
    """Score a project directory's architecture deterministically."""

    def score(self, project_path: Path) -> ArchitectureScore:
        project_path = Path(project_path)
        files = self._collect_code_files(project_path)

        dims = [
            self._score_layering(project_path, files),
            self._score_dependency_direction(files),
            self._score_modularity(files),
            self._score_tests(project_path, files),
            self._score_config_secrets(project_path, files),
            self._score_runnability(project_path),
            self._score_documentation(project_path),
            self._score_error_handling(files),
        ]

        total = round(sum(d.score for d in dims), 1)
        strengths: list[str] = []
        weaknesses: list[str] = []
        recommendations: list[str] = []
        for d in dims:
            ratio = d.score / d.max_score if d.max_score else 0
            for note in d.notes:
                if note.startswith("+"):
                    strengths.append(note[1:].strip())
                elif note.startswith("-"):
                    weaknesses.append(note[1:].strip())
                elif note.startswith(">"):
                    recommendations.append(note[1:].strip())
            if ratio < 0.5 and not any(n.startswith(">") for n in d.notes):
                recommendations.append(f"Improve {d.name.replace('_', ' ')}")

        return ArchitectureScore(
            total=total,
            out_of_ten=round(total / 10, 2),
            pattern=self._detect_pattern(files),
            dimensions=dims,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    # ── File collection ──────────────────────────────────────────────────

    def _collect_code_files(self, root: Path) -> list[Path]:
        out: list[Path] = []
        if not root.exists():
            return out
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in _CODE_EXTS:
                continue
            if any(part in _IGNORE_DIRS for part in p.parts):
                continue
            out.append(p)
        return out

    @staticmethod
    def _is_test_file(p: Path) -> bool:
        name = p.name.lower()
        parts = {part.lower() for part in p.parts}
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith("_test.go")
            or re.search(r"\.(test|spec)\.[jt]sx?$", name) is not None
            or "tests" in parts
            or "__tests__" in parts
        )

    @classmethod
    def _classify_layer(cls, p: Path) -> str | None:
        parts = [part.lower() for part in p.parts] + [p.stem.lower()]
        for layer, (_rank, kws) in _LAYER_KEYWORDS.items():
            if any(any(kw == part or kw in part for kw in kws) for part in parts):
                return layer
        return None

    @staticmethod
    def _classify_module_string(module: str) -> str | None:
        low = module.lower()
        for layer, (_rank, kws) in _LAYER_KEYWORDS.items():
            if any(kw in low for kw in kws):
                return layer
        return None

    # ── Dimensions ───────────────────────────────────────────────────────

    def _score_layering(self, root: Path, files: list[Path]) -> DimensionScore:
        d = DimensionScore("layering", 0.0, 20.0)
        src = [f for f in files if not self._is_test_file(f)]
        layers = {self._classify_layer(f) for f in src}
        layers.discard(None)
        n = len(layers)
        d.score = min(n, 4) / 4 * 20.0
        nested = any(len(f.relative_to(root).parts) > 1 for f in src) if root.exists() else False
        if not nested and src:
            d.score *= 0.6
            d.notes.append("- Code is flat (no directory structure / module separation)")
            d.notes.append(
                "> Organise code into layered directories (domain, application, api, infrastructure)"
            )
        if n >= 3:
            d.notes.append(
                f"+ Clear layering: {n} architectural layers ({', '.join(sorted(layers))})"
            )
        elif n == 0:
            d.notes.append("- No recognizable architectural layers")
            d.notes.append(
                "> Separate concerns into domain / application / interface / infrastructure layers"
            )
        return d

    def _score_dependency_direction(self, files: list[Path]) -> DimensionScore:
        d = DimensionScore("dependency_direction", 0.0, 20.0)
        py = [f for f in files if f.suffix == ".py" and not self._is_test_file(f)]
        arch_imports = 0
        violations = 0
        for f in py:
            importer = self._classify_layer(f)
            if importer is None:
                continue
            importer_rank = _LAYER_KEYWORDS[importer][0]
            for mod in self._iter_imports(f):
                imported = self._classify_module_string(mod)
                if imported is None or imported == importer:
                    continue
                arch_imports += 1
                if _LAYER_KEYWORDS[imported][0] > importer_rank:
                    violations += 1
                    d.notes.append(
                        f"- {importer} layer imports outer {imported} layer " f"({f.name}: '{mod}')"
                    )
        if arch_imports == 0:
            # No cross-layer imports to verify — give partial credit, don't reward fully.
            d.score = 10.0
            d.notes.append(
                "> Add explicit layer boundaries so dependency direction can be verified"
            )
            return d
        clean = 1 - violations / arch_imports
        d.score = round(20.0 * max(0.0, clean), 1)
        if violations == 0:
            d.notes.append("+ Dependency direction respected (inner layers do not import outer)")
        else:
            d.notes.append(
                f"> Invert {violations} inward dependency violation(s) (use ports/interfaces)"
            )
        return d

    def _score_modularity(self, files: list[Path]) -> DimensionScore:
        d = DimensionScore("modularity", 0.0, 15.0)
        if not files:
            return d
        items = 0
        oversized = 0
        big_files: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            loc = text.count("\n") + 1
            items += 1
            if loc > _MAX_FILE_LOC:
                oversized += 1
                big_files.append(f"{f.name} ({loc} LOC)")
            if f.suffix == ".py":
                for fn_loc in self._iter_python_function_locs(text):
                    items += 1
                    if fn_loc > _MAX_FUNC_LOC:
                        oversized += 1
        frac_ok = 1 - oversized / items if items else 0
        d.score = round(15.0 * frac_ok, 1)
        if big_files:
            d.notes.append("- Oversized files (>800 LOC): " + ", ".join(big_files[:5]))
            d.notes.append("> Split oversized files/functions into focused modules")
        elif frac_ok >= 0.95:
            d.notes.append("+ Good modularity: files and functions within size limits")
        return d

    def _score_tests(self, root: Path, files: list[Path]) -> DimensionScore:
        d = DimensionScore("tests", 0.0, 15.0)
        tests = [f for f in files if self._is_test_file(f)]
        src = [f for f in files if not self._is_test_file(f)]
        if not tests:
            d.notes.append("- No test files detected")
            d.notes.append(
                "> Add unit and integration tests (aim for >= 1 test file per 4 source files)"
            )
            return d
        if not src:
            d.notes.append("- Test files present but no source files to test")
            return d
        ratio = len(tests) / len(src)
        d.score = round(min(1.0, ratio / 0.25) * 15.0, 1)
        if d.score >= 12:
            d.notes.append(f"+ Healthy test coverage surface ({len(tests)} test files)")
        else:
            d.notes.append(
                f"> Increase test count ({len(tests)} tests for {len(src)} source files)"
            )
        return d

    def _score_config_secrets(self, root: Path, files: list[Path]) -> DimensionScore:
        d = DimensionScore("config_secrets", 10.0, 10.0)
        has_env = (root / ".env.example").exists() or (root / ".env.sample").exists()
        has_cfg = any(
            (root / n).exists() for n in ("config.py", "settings.py", "config.js", "config.ts")
        ) or any("config" in f.stem.lower() or "settings" in f.stem.lower() for f in files)
        if not has_env:
            d.score -= 3
            d.notes.append("- No .env.example template")
            d.notes.append("> Add a .env.example documenting required environment variables")
        else:
            d.notes.append("+ Environment template (.env.example) present")
        if not has_cfg:
            d.score -= 2
            d.notes.append("> Centralise configuration in a config/settings module")
        secrets = self._find_hardcoded_secrets(files)
        if secrets:
            d.score -= 5 * len(secrets)
            d.notes.append(
                f"- Hardcoded secret(s) detected in {len(secrets)} file(s): "
                + ", ".join(sorted(secrets)[:3])
            )
            d.notes.append(
                "> Move secrets to environment variables / a secret manager and rotate them"
            )
        d.score = max(0.0, round(d.score, 1))
        return d

    def _score_runnability(self, root: Path) -> DimensionScore:
        d = DimensionScore("runnability", 0.0, 10.0)
        manifests = (
            "requirements.txt",
            "pyproject.toml",
            "package.json",
            "go.mod",
            "Cargo.toml",
            "Pipfile",
        )
        manifest = next((m for m in manifests if (root / m).exists()), None)
        if manifest:
            d.score += 5
            d.notes.append(f"+ Dependency manifest present ({manifest})")
        else:
            d.notes.append("- No dependency manifest")
            d.notes.append(
                "> Add a dependency manifest (requirements.txt / package.json / pyproject.toml)"
            )
        entry_points = (
            "main.py",
            "app.py",
            "manage.py",
            "wsgi.py",
            "asgi.py",
            "index.js",
            "index.ts",
            "server.js",
            "main.go",
        )
        has_entry = any((root / e).exists() for e in entry_points) or any(
            (root / "src" / e).exists() for e in entry_points
        )
        if has_entry:
            d.score += 3
            d.notes.append("+ Clear entry point present")
        else:
            d.notes.append(
                "> Provide a clear application entry point (main.py / index.ts / server.js)"
            )
        if self._has_pinned_deps(root, manifest):
            d.score += 2
            d.notes.append("+ Dependencies are version-pinned")
        d.score = round(d.score, 1)
        return d

    def _score_documentation(self, root: Path) -> DimensionScore:
        d = DimensionScore("documentation", 0.0, 5.0)
        readme = next(
            (root / n for n in ("README.md", "README.rst", "README.txt") if (root / n).exists()),
            None,
        )
        if readme is None:
            d.notes.append("- No README")
            d.notes.append("> Add a README documenting setup, usage, and architecture")
            return d
        try:
            length = len(readme.read_text(encoding="utf-8", errors="replace").strip())
        except OSError:
            length = 0
        if length > 200:
            d.score = 5.0
            d.notes.append("+ Substantive README present")
        else:
            d.score = 2.0
            d.notes.append("> Expand the README (setup, usage, architecture overview)")
        return d

    def _score_error_handling(self, files: list[Path]) -> DimensionScore:
        d = DimensionScore("error_handling", 0.0, 5.0)
        src = [f for f in files if not self._is_test_file(f)]
        if not src:
            return d
        with_eh = 0
        for f in src:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r"\btry\b|\.catch\(|except\b|rescue\b", text):
                with_eh += 1
        frac = with_eh / len(src)
        d.score = round(min(1.0, frac / 0.5) * 5.0, 1)
        if frac >= 0.5:
            d.notes.append("+ Error handling present across the codebase")
        else:
            d.notes.append("> Add explicit error handling at I/O and boundary points")
        return d

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _iter_imports(path: Path):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    yield node.module

    @staticmethod
    def _iter_python_function_locs(text: str):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", None)
                if end is not None:
                    yield end - node.lineno + 1

    def _find_hardcoded_secrets(self, files: list[Path]) -> set[str]:
        hits: set[str] = set()
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                # Check the allow-list against the *matched secret substring*,
                # not the whole line -- an unrelated "example"/"<...>" token
                # elsewhere on the same line (a comment, a doc URL) must not
                # exempt a genuine, co-located hardcoded secret.
                for pat in _SECRET_PATTERNS:
                    m = pat.search(line)
                    if m and not _SECRET_ALLOW.search(m.group(0)):
                        hits.add(f.name)
                        break
                else:
                    continue
                break
        return hits

    @staticmethod
    def _has_pinned_deps(root: Path, manifest: str | None) -> bool:
        if manifest == "requirements.txt":
            try:
                text = (root / manifest).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            deps = [
                ln.strip()
                for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            return bool(deps) and all("==" in dep for dep in deps)
        if manifest == "package.json":
            try:
                text = (root / manifest).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            # A caret/tilde-prefixed range ("^1.2.3"/"~1.2.3") is npm's
            # explicitly *unpinned* convention -- it must not count as pinned
            # merely because it also contains a "digit.digit" shape.
            return bool(re.search(r'"\d+\.\d+', text)) and not re.search(r'"[\^~]\d', text)
        if manifest in ("pyproject.toml", "go.mod", "Cargo.toml"):
            try:
                text = (root / manifest).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            # Exact-pin markers: pip/PEP 621 "==", Poetry/Cargo-style
            # 'name = "1.2.3"'. A manifest with only caret/tilde/range
            # operators (each ecosystem's default) is not considered pinned.
            return bool(re.search(r'==\s*\d|=\s*"\d+\.\d+\.\d+"', text))
        return False

    def _detect_pattern(self, files: list[Path]) -> str:
        layers = {self._classify_layer(f) for f in files}
        layers.discard(None)
        if {"domain", "application", "infrastructure"} <= layers:
            return "Hexagonal / Layered"
        if {"domain", "application"} <= layers or {"application", "infrastructure"} <= layers:
            return "Layered"
        if {"domain", "presentation"} <= layers:
            return "MVC/MVT"
        return "No clear pattern" if not layers else ", ".join(sorted(layers))
