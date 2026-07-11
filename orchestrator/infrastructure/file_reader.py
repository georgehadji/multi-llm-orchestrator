"""
FileReader — Infrastructure adapter for FileReaderPort
========================================================
Author: Orchestrator core

Concrete adapter that satisfies ``FileReaderPort`` via
``aiofiles`` (async) or synchronous ``Path.read_text()`` fallback.

Keeps the application ``ingest`` layer pure by keeping ``open()``
calls in infrastructure.
"""

from __future__ import annotations

from pathlib import Path

from ..domain.ports import FileReaderPort


class FileReader(FileReaderPort):
    """Reads files from disk using the standard library.

    The ``FileReaderPort`` Protocol is satisfied structurally —
    no explicit registration needed.
    """

    async def read_text(self, path: str) -> str:
        """Return UTF-8 text at *path* or raise FileNotFoundError.

        Uses ``Path.read_text()`` (synchronous stdlib call in an async
        method — acceptable for short-lived local file reads that are
        not on a hot path).
        """
        return Path(path).read_text(encoding="utf-8")


class FileReaderAsync:
    """Async variant using ``aiofiles`` when available.

    Falls back to ``FileReader`` (sync stdlib) if ``aiofiles`` is
    not installed.
    """

    def __init__(self) -> None:
        self._reader = FileReader()

    async def read_text(self, path: str) -> str:
        try:
            import aiofiles  # type: ignore[import-untyped]  # noqa: F811

            async with aiofiles.open(path, encoding="utf-8") as f:
                return await f.read()
        except ImportError:
            return await self._reader.read_text(path)
