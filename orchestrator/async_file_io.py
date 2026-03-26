"""
Async File I/O Helpers
======================
Author: Georgios-Chrysovalantis Chatzivantsidis

Async file operations using aiofiles to prevent event loop blocking.
Use these instead of pathlib's sync methods in async code.

USAGE:
    from orchestrator.async_file_io import async_write_text, async_read_text
    
    # Write file asynchronously
    await async_write_text("output.txt", "Hello, World!")
    
    # Read file asynchronously
    content = await async_read_text("input.txt")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator.async_file_io")

# Try to import aiofiles, fall back to sync if not available
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    aiofiles = None
    logger.warning("aiofiles not installed, using synchronous file I/O")


async def async_write_text(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
    mkdir_parents: bool = True,
) -> None:
    """
    Write text to file asynchronously.
    
    Args:
        path: File path
        content: Content to write
        encoding: File encoding
        mkdir_parents: Create parent directories if needed
    """
    path = Path(path) if isinstance(path, str) else path
    
    if mkdir_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_AIOFILES and aiofiles:
        async with aiofiles.open(path, 'w', encoding=encoding) as f:
            await f.write(content)
    else:
        # Fallback to sync I/O
        path.write_text(content, encoding=encoding)
    
    logger.debug(f"Wrote {len(content)} chars to {path}")


async def async_read_text(
    path: Path | str,
    encoding: str = "utf-8",
) -> str:
    """
    Read text from file asynchronously.
    
    Args:
        path: File path
        encoding: File encoding
    
    Returns:
        File content
    """
    path = Path(path) if isinstance(path, str) else path
    
    if HAS_AIOFILES and aiofiles:
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            return await f.read()
    else:
        # Fallback to sync I/O
        return path.read_text(encoding=encoding)


async def async_write_json(
    path: Path | str,
    data: dict,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    mkdir_parents: bool = True,
) -> None:
    """
    Write JSON to file asynchronously.
    
    Args:
        path: File path
        data: Dictionary to serialize
        encoding: File encoding
        indent: JSON indentation
        ensure_ascii: Escape non-ASCII characters
        mkdir_parents: Create parent directories if needed
    """
    import json
    content = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=str,  # Handle datetime and other non-serializable types
    )
    await async_write_text(path, content, encoding, mkdir_parents)


async def async_read_json(
    path: Path | str,
    encoding: str = "utf-8",
) -> dict:
    """
    Read JSON from file asynchronously.
    
    Args:
        path: File path
        encoding: File encoding
    
    Returns:
        Parsed JSON as dictionary
    """
    import json
    content = await async_read_text(path, encoding)
    return json.loads(content)


async def async_write_lines(
    path: Path | str,
    lines: list[str],
    encoding: str = "utf-8",
    mkdir_parents: bool = True,
) -> None:
    """
    Write lines to file asynchronously (one line per row).
    
    Args:
        path: File path
        lines: List of lines to write
        encoding: File encoding
        mkdir_parents: Create parent directories if needed
    """
    content = '\n'.join(lines)
    await async_write_text(path, content, encoding, mkdir_parents)


async def async_append_text(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
    mkdir_parents: bool = True,
) -> None:
    """
    Append text to file asynchronously.
    
    Args:
        path: File path
        content: Content to append
        encoding: File encoding
        mkdir_parents: Create parent directories if needed
    """
    path = Path(path) if isinstance(path, str) else path
    
    if mkdir_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_AIOFILES and aiofiles:
        async with aiofiles.open(path, 'a', encoding=encoding) as f:
            await f.write(content)
    else:
        # Fallback to sync I/O
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)


async def async_file_exists(path: Path | str) -> bool:
    """
    Check if file exists asynchronously.
    
    Args:
        path: File path
    
    Returns:
        True if file exists
    """
    path = Path(path) if isinstance(path, str) else path
    return path.exists()


async def async_mkdir_parents(
    path: Path | str,
    exist_ok: bool = True,
) -> None:
    """
    Create directory with parents asynchronously.
    
    Args:
        path: Directory path
        exist_ok: Don't error if directory exists
    """
    path = Path(path) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=exist_ok)


# Convenience functions for common patterns

async def async_write_progress_line(
    path: Path | str,
    entry: dict,
    encoding: str = "utf-8",
) -> None:
    """
    Append a JSON line to progress file (for PROGRESS.jsonl pattern).
    
    Args:
        path: File path
        entry: Dictionary to serialize as JSON line
        encoding: File encoding
    """
    import json
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    await async_append_text(path, line, encoding)


# Lock for file operations that need atomicity
_file_locks: dict[str, asyncio.Lock] = {}


def get_file_lock(path: Path | str) -> asyncio.Lock:
    """
    Get or create a lock for a specific file path.
    
    Use this when multiple coroutines might write to the same file.
    
    Args:
        path: File path
    
    Returns:
        asyncio.Lock for the file
    """
    path_str = str(path)
    if path_str not in _file_locks:
        _file_locks[path_str] = asyncio.Lock()
    return _file_locks[path_str]


async def async_write_text_locked(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
    mkdir_parents: bool = True,
) -> None:
    """
    Write text to file with locking for atomic operations.
    
    Args:
        path: File path
        content: Content to write
        encoding: File encoding
        mkdir_parents: Create parent directories if needed
    """
    lock = get_file_lock(path)
    async with lock:
        await async_write_text(path, content, encoding, mkdir_parents)


async def async_append_text_locked(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
    mkdir_parents: bool = True,
) -> None:
    """
    Append text to file with locking for atomic operations.
    
    Args:
        path: File path
        content: Content to append
        encoding: File encoding
        mkdir_parents: Create parent directories if needed
    """
    lock = get_file_lock(path)
    async with lock:
        await async_append_text(path, content, encoding, mkdir_parents)
