"""
LSP Validator — Post-generation code validation via language servers
=====================================================================
Author: Reasonix Code (CodeWhale Phase 1 implementation)

Runs pyright (Python), tsc (TypeScript/JavaScript) on generated code
and returns structured diagnostics. Follows the same subprocess-validator
pattern as validate_pytest / validate_ruff in quality/validators.py.

Architecture:
    Infrastructure layer adapter — satisfies domain.ports.LSPValidatorPort.
    Application layer (CritiqueCycle) imports only the port, never this file.

Design decisions:
    - Tempfile + file-based validation (not LSP protocol over stdin)
    - 30-second timeout per validation
    - Graceful degradation if server not installed (logs once, returns [])
    - Offloads to asyncio.to_thread() for non-blocking operation
    - Follows quality/validators.py subprocess pattern
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..domain.ports import LSPDiagnostic, LSPValidatorPort

logger = logging.getLogger(__name__)


# ── Server registry ──────────────────────────────────────────────────────────
# Each entry: (binary_name, args_fn(filepath) -> list[str], parser_fn(stdout) -> list[LSPDiagnostic])
# Extensible — add entries for gopls, rust-analyzer, etc.

ServerEntry = tuple[str, Callable[[str], list[str]], Callable[[bytes], list[LSPDiagnostic]]]


def _parse_pyright(stdout: bytes) -> list[LSPDiagnostic]:
    """Parse pyright --outputjson output."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    results: list[LSPDiagnostic] = []
    for diag in data.get("generalDiagnostics", []):
        sev = diag.get("severity", "error")
        severity_map = {
            0: "hint",
            1: "information",
            2: "warning",
            3: "error",
        }
        severity = severity_map.get(sev, "error") if isinstance(sev, int) else str(sev).lower()
        rng = diag.get("range", {})
        start = rng.get("start", {}) if isinstance(rng, dict) else {}
        results.append(
            LSPDiagnostic(
                severity=severity,
                message=diag.get("message", ""),
                line=(start.get("line", 0) or 0) + 1,
                column=(start.get("character", 0) or 0) + 1,
                source="pyright",
                code=diag.get("rule", ""),
            )
        )
    return results


def _parse_tsc(stdout: bytes) -> list[LSPDiagnostic]:
    """Parse tsc --noEmit --pretty false output."""
    results: list[LSPDiagnostic] = []
    text = stdout.decode("utf-8", errors="replace")
    for line in text.splitlines():
        m = re.match(
            r"^(.+)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)$",
            line,
        )
        if m:
            results.append(
                LSPDiagnostic(
                    severity=m.group(4),
                    message=m.group(6),
                    line=int(m.group(2)),
                    column=int(m.group(3)),
                    source="tsc",
                    code=m.group(5),
                )
            )
    return results


_SERVERS: dict[str, ServerEntry] = {
    "python": ("pyright", lambda fn: ["pyright", "--outputjson", fn], _parse_pyright),
    "typescript": (
        "tsc",
        lambda fn: ["tsc", "--noEmit", "--pretty", "false", "--lib", "es2020,dom", fn],
        _parse_tsc,
    ),
}


def _binary_exists(name: str) -> bool:
    """Check if a binary is on PATH."""
    try:
        proc = subprocess.run(
            ["which", name] if os.name != "nt" else ["where", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        return False


class LspValidator(LSPValidatorPort):
    """Validates code by writing to a tempfile and running a language server.

    Satisfies domain.ports.LSPValidatorPort — wired via ServiceContainer.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        temp_dir: str | None = None,
    ):
        self._timeout = timeout_seconds
        self._temp_dir = temp_dir
        self._existence_cache: dict[str, bool | None] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def available_servers(self) -> frozenset[str]:
        """Check which language servers are installed on PATH."""
        available: set[str] = set()
        for lang, (bin_name, _, _) in _SERVERS.items():
            cached = self._existence_cache.get(bin_name)
            if cached is True:
                available.add(lang)
            elif cached is None:
                exists = _binary_exists(bin_name)
                self._existence_cache[bin_name] = exists
                if exists:
                    available.add(lang)
        return frozenset(available)

    async def validate(
        self, code: str, language: str = "python", filename: str = ""
    ) -> list[LSPDiagnostic]:
        """Validate a code string via language server.

        Writes code to a tempfile, runs the server, parses output, cleans up.
        Returns [] gracefully if the server is not installed, times out, or errors.
        """
        entry = _SERVERS.get(language)
        if not entry:
            logger.debug(f"No LSP server registered for language '{language}'")
            return []

        bin_name, args_fn, parser = entry

        # Fast path: skip if binary not installed
        cached = self._existence_cache.get(bin_name)
        if cached is False:
            return []
        if cached is None:
            exists = _binary_exists(bin_name)
            self._existence_cache[bin_name] = exists
            if not exists:
                logger.info(f"LSP server '{bin_name}' not on PATH — skipping validation")
                return []

        # Language-specific extension for tempfile
        suffix_map = {"python": ".py", "typescript": ".ts"}
        suffix = suffix_map.get(language, ".txt")

        # Create tempdir or use configured dir
        if self._temp_dir:
            Path(self._temp_dir).mkdir(parents=True, exist_ok=True)

        tmpdir_context = tempfile.mkdtemp(prefix="orch_lsp_", dir=self._temp_dir)
        tmpfile = str(Path(tmpdir_context) / (filename or f"code{suffix}"))
        try:
            with open(tmpfile, "w", encoding="utf-8") as f:
                f.write(code)

            return await self.validate_file(tmpfile)

        except Exception as e:
            logger.warning(f"LSP validation failed for {language}: {e}")
            return []
        finally:
            # Cleanup tempfile
            try:
                os.unlink(tmpfile)
                os.rmdir(tmpdir_context)
            except OSError:
                pass

    async def validate_file(self, filepath: str) -> list[LSPDiagnostic]:
        """Validate a file already on disk. Non-blocking via asyncio.to_thread()."""
        path = Path(filepath)
        ext = path.suffix
        lang_map = {".py": "python", ".ts": "typescript", ".js": "typescript", ".tsx": "typescript"}
        language = lang_map.get(ext)
        if not language:
            return []

        entry = _SERVERS.get(language)
        if not entry:
            return []

        bin_name, args_fn, parser = entry

        # Check binary
        cached = self._existence_cache.get(bin_name)
        if cached is False:
            return []
        if cached is None:
            exists = _binary_exists(bin_name)
            self._existence_cache[bin_name] = exists
            if not exists:
                return []

        args = args_fn(filepath)

        try:
            # Offload to thread pool for non-blocking I/O (same pattern as
            # async_run_validators in quality/validators.py)
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self._timeout,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode not in (0, 1):
                # pyright/tsc return 1 when diagnostics are found, 0 on clean
                logger.debug(
                    "LSP '%s' exited code %d on %s: %s",
                    bin_name,
                    proc.returncode,
                    path.name,
                    stderr.decode("utf-8", errors="replace")[:200],
                )
            return parser(stdout)
        except asyncio.TimeoutError:
            logger.warning("LSP '%s' timed out (%d s) on %s", bin_name, self._timeout, path.name)
            return []
        except FileNotFoundError:
            self._existence_cache[bin_name] = False
            return []
        except Exception as e:
            logger.warning("LSP '%s' error on %s: %s", bin_name, path.name, e)
            return []

    # ── Diagnostics summary helpers (used by CritiqueCycle) ──────────────────

    @staticmethod
    def diagnostics_summary(diags: list[LSPDiagnostic]) -> str:
        """Build a human-readable summary of diagnostics for prompt injection."""
        if not diags:
            return ""

        errors = [d for d in diags if d.severity == "error"]
        warnings = [d for d in diags if d.severity == "warning"]
        infos = [d for d in diags if d.severity in ("information", "hint")]

        parts: list[str] = []
        if errors:
            parts.append(f"### {len(errors)} Error(s)")
            for e in errors[:10]:
                code_str = f" ({e.code})" if e.code else ""
                parts.append(f"- L{e.line}:{e.column} {e.message}{code_str}")
            if len(errors) > 10:
                parts.append(f"- ... and {len(errors) - 10} more errors")

        if warnings:
            parts.append(f"### {len(warnings)} Warning(s)")
            for w in warnings[:10]:
                code_str = f" ({w.code})" if w.code else ""
                parts.append(f"- L{w.line}:{w.column} {w.message}{code_str}")
            if len(warnings) > 10:
                parts.append(f"- ... and {len(warnings) - 10} more warnings")

        if infos:
            parts.append(f"*{len(infos)} information(s) (not shown)*")

        return "\n".join(parts)

    @staticmethod
    def inject_inline_diagnostics(code: str, diags: list[LSPDiagnostic]) -> str:
        """Inject diagnostics as inline comments above the flagged lines.

        Errors and warnings are injected as source-level comments so the LLM
        reviewer can see them in context during the critique cycle.
        """
        lines = code.splitlines()
        sorted_diags = sorted(diags, key=lambda d: d.line, reverse=True)

        for d in sorted_diags:
            if d.severity not in ("error", "warning"):
                continue
            idx = max(0, min(d.line - 1, len(lines) - 1))
            # Choose comment char based on crude language detection from first line
            prefix = "//" if lines and lines[0].strip().startswith(("import ", "const ", "let ", "var ", "function ")) else "#"
            code_str = f" ({d.code})" if d.code else ""
            comment = f"{prefix} LSP [{d.severity.upper()}]: {d.message}{code_str}"
            lines.insert(idx, comment)

        return "\n".join(lines)
