# Dependency Pinning Policy

**Date:** 2026-05-25  
**Applies to:** All `requirements.txt`, `requirements-dev.txt`, and `pyproject.toml` dependency declarations

---

## Rationale

Exact dependency pins prevent supply-chain attacks where a malicious or compromised package is published to PyPI at a version that falls within a declared range. Without exact pins, a `pip install` (or `uv sync`) at any time may pull in an untrusted transitive or direct dependency without a code review on our side.

This policy was informed by industry incidents including:
- **litellm compromise** — malicious release captured by range-based pins
- **Mini Shai-Hulud worm** (May 2026) — `mistralai 2.4.6` on PyPI; every install with `mistralai>=2.3.0,<3` would have pulled it before quarantine

---

## Rules

### 1. Exact Pins (`==X.Y.Z`)

All direct dependencies in `requirements.txt` and `requirements-dev.txt` MUST use exact version pins:

```
httpx==0.28.1
pydantic==2.13.4
```

**No ranges.** No `>=X.Y.Z,<N` without a written justification appended to this document.

### 2. Upper Bounds for Optional Dependencies

Optional extras (in `pyproject.toml`) may use `>=floor,<next_major` when the dependency is not part of the core install:

```toml
anthropic = ["anthropic==0.86.0"]
```

All optional deps are also exact-pinned for consistency. Ranges are only acceptable with documented rationale.

### 3. Audit on Update

Every dependency version bump requires:
1. Version change in the pin file
2. `pip install -r requirements-dev.txt` to resolve transitives
3. Full test suite: `pytest`
4. Security scan: `safety check` and `pip-audit`

### 4. No Lazy Ranges

Dependencies that are always loaded at startup (core deps) MUST be exact-pinned. Dependencies loaded only when a specific backend is active (optional providers, memory backends, platform adapters) should live in extras and be lazy-installed at first use (see `tools/lazy_deps.py` pattern from Hermes Agent).

### 5. CI Enforcement

CI MUST run `pip install -r requirements-dev.txt` (not `pip install -e .`) at least once to verify the resolved dependency tree is consistent and installable.

---

## Future: Lazy Dependencies

When adding optional backends (Phase 2: plugin surfaces), follow this pattern:

```python
# orchestrator/plugins/discovery.py
_LAZY_DEPS: dict[str, str] = {
    "telegram": "python-telegram-bot[webhooks]==22.6",
    "discord": "discord.py[voice]==2.7.1",
}

def ensure_dep(name: str) -> None:
    """Install an optional dependency at first use if missing."""
    if name not in _LAZY_DEPS:
        return
    try:
        __import__(name.replace("-", "_"))
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", _LAZY_DEPS[name]]
        )
```

This keeps the core install lean and reduces the blast radius of supply-chain attacks.

---

## Audit Checklist

Before merging any dependency change:

- [ ] Every new pin uses `==X.Y.Z` (exact)
- [ ] No bare `>=X.Y.Z` without a ceiling in core deps
- [ ] `pip install -r requirements-dev.txt` succeeds
- [ ] `safety check` passes (no known vulnerabilities in resolved tree)
- [ ] Full test suite passes
