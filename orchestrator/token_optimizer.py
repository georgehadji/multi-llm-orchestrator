"""
Token Optimizer — CLI Output Filtering & Compression
======================================================

Implements RTK (Rust Token Killer) functionality:
- Filters and compresses command outputs before they reach LLM context
- Achieves 60-90% token savings on common operations

Supported Commands:
- ls, tree, find → Directory listings
- cat, head, tail → File contents
- grep, rg → Search results
- git status, diff, log, add, commit, push → Git operations
- npm test, pip test, pytest, cargo test → Test outputs
- docker ps, docker images → Docker operations
- ruff, flake8, eslint → Linting

Usage:
    from orchestrator.token_optimizer import TokenOptimizer
    
    optimizer = TokenOptimizer()
    
    # Optimize command output
    optimized = optimizer.optimize("ls -la", output)
    optimized = optimizer.optimize("pytest", output)
    optimized = optimizer.optimize("git diff", output)
    
    # Get token savings stats
    stats = optimizer.get_stats()
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .log_config import get_logger

logger = get_logger(__name__)


class CommandCategory(Enum):
    """Categories of commands for optimization strategies."""
    DIRECTORY_LISTING = "directory_listing"
    FILE_CONTENT = "file_content"
    SEARCH = "search"
    GIT = "git"
    TEST = "test"
    DOCKER = "docker"
    LINT = "lint"
    BUILD = "build"
    UNKNOWN = "unknown"


@dataclass
class OptimizationResult:
    """Result of optimizing command output."""
    original: str
    optimized: str
    original_tokens: int
    optimized_tokens: int
    savings_percent: float
    command: str
    category: CommandCategory
    
    @property
    def was_optimized(self) -> bool:
        return self.optimized_tokens < self.original_tokens


@dataclass
class TokenStats:
    """Token usage statistics."""
    total_original: int = 0
    total_optimized: int = 0
    commands_optimized: int = 0
    commands_processed: int = 0
    
    @property
    def total_savings(self) -> int:
        return self.total_original - self.total_optimized
    
    @property
    def savings_percent(self) -> float:
        if self.total_original == 0:
            return 0.0
        return (self.total_savings / self.total_original) * 100


class TokenOptimizer:
    """
    Optimizes CLI command outputs to reduce token consumption.
    
    Implements multiple optimization strategies:
    1. Truncation - Keep first N lines
    2. Summary - Replace long outputs with summaries
    3. Deduplication - Remove repeated patterns
    4. Noise removal - Remove irrelevant lines
    5. Key extraction - Keep only important information
    """

    # Token estimation: ~4 characters per token on average
    CHARS_PER_TOKEN = 4
    
    # Line limits by category
    LINE_LIMITS = {
        CommandCategory.DIRECTORY_LISTING: 50,
        CommandCategory.FILE_CONTENT: 100,
        CommandCategory.SEARCH: 30,
        CommandCategory.GIT: 40,
        CommandCategory.TEST: 50,
        CommandCategory.DOCKER: 20,
        CommandCategory.LINT: 30,
        CommandCategory.BUILD: 30,
    }

    def __init__(
        self,
        max_lines: Optional[int] = None,
        enable_summary: bool = True,
        enable_dedup: bool = True,
    ):
        self.max_lines = max_lines
        self.enable_summary = enable_summary
        self.enable_dedup = enable_dedup
        self._stats = TokenStats()
        
        # Command patterns for categorization
        self._command_patterns: Dict[CommandCategory, List[str]] = {
            CommandCategory.DIRECTORY_LISTING: [
                r'^ls\b', r'^dir\b', r'^tree\b', r'^find\b',
            ],
            CommandCategory.FILE_CONTENT: [
                r'^cat\b', r'^head\b', r'^tail\b', r'^less\b', r'^more\b',
                r'^type\b', r'^wc\b',
            ],
            CommandCategory.SEARCH: [
                r'^grep\b', r'^rg\b', r'^ag\b', r'^ack\b',
                r'^find\b.*-name', r'^find\b.*-type',
            ],
            CommandCategory.GIT: [
                r'^git\s+status', r'^git\s+diff', r'^git\s+log',
                r'^git\s+add', r'^git\s+commit', r'^git\s+push',
                r'^git\s+pull', r'^git\s+branch', r'^git\s+show',
                r'^git\s+stash', r'^git\s+remote',
            ],
            CommandCategory.TEST: [
                r'^pytest\b', r'^npm\s+test', r'^npm\s+run\s+test',
                r'^cargo\s+test', r'^go\s+test', r'^python\s+-m\s+pytest',
                r'^pip\s+test', r'^make\s+test', r'^yarn\s+test',
            ],
            CommandCategory.DOCKER: [
                r'^docker\s+ps', r'^docker\s+images', r'^docker\s+logs',
                r'^docker\s+inspect', r'^docker\s+compose\s+ps',
            ],
            CommandCategory.LINT: [
                r'^ruff\b', r'^flake8\b', r'^eslint\b', r'^pylint\b',
                r'^mypy\b', r'^tslint\b', r'^prettier\b', r'^black\b',
                r'^golangci-lint\b', r'^hadolint\b',
            ],
            CommandCategory.BUILD: [
                r'^npm\s+build', r'^npm\s+run\s+build', r'^cargo\s+build',
                r'^go\s+build', r'^make\b', r'^cmake\b', r'^gradle\b',
                r'^webpack\b', r'^vite\b', r'^rollup\b',
            ],
        }

    def categorize_command(self, command: str) -> CommandCategory:
        """Categorize a command for optimization strategy selection."""
        for category, patterns in self._command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return category
        return CommandCategory.UNKNOWN

    def _count_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate output to max lines."""
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
        return '\n'.join(lines[:max_lines]) + f'\n... (+{len(lines) - max_lines} more lines)'

    def _optimize_directory_listing(self, output: str) -> str:
        """Optimize ls, tree, find outputs."""
        lines = output.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip permission strings in ls -la
            if re.match(r'^total\s+\d+', line):
                continue
            
            # Skip common non-essential lines
            if re.match(r'^d[rwx-]{9}', line):
                # Keep directory entries but simplify
                parts = line.split()
                if len(parts) >= 9:
                    optimized_lines.append(f"d {parts[-1]}")
                continue
            
            # Keep file entries
            if re.match(r'^[d-][rwx-]{9}', line):
                parts = line.split()
                if len(parts) >= 9:
                    optimized_lines.append(f"{parts[0][:1]} {parts[-1]}")
                continue
            
            # Keep other meaningful lines
            if line.strip():
                optimized_lines.append(line)
        
        # Limit lines
        limit = self.LINE_LIMITS.get(CommandCategory.DIRECTORY_LISTING, 50)
        if len(optimized_lines) > limit:
            optimized_lines = optimized_lines[:limit] + [f"... (+{len(optimized_lines) - limit} more)"]
        
        return '\n'.join(optimized_lines)

    def _optimize_file_content(self, output: str) -> str:
        """Optimize cat, head, tail outputs."""
        # For large files, keep only first portion
        lines = output.split('\n')
        
        # Check if it's a binary file
        if '\x00' in output[:100]:
            return "[Binary file - content not shown]"
        
        # Keep first N lines
        limit = self.LINE_LIMITS.get(CommandCategory.FILE_CONTENT, 100)
        if len(lines) > limit:
            return '\n'.join(lines[:limit]) + f'\n... (+{len(lines) - limit} lines)'
        
        return output

    def _optimize_search_output(self, output: str) -> str:
        """Optimize grep, rg, ag outputs."""
        lines = output.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Simplify file:line:content format
            # Keep just file:line:content
            optimized_lines.append(line)
        
        # Limit lines
        limit = self.LINE_LIMITS.get(CommandCategory.SEARCH, 30)
        if len(optimized_lines) > limit:
            optimized_lines = optimized_lines[:limit] + [f"... (+{len(optimized_lines) - limit} matches)"]
        
        return '\n'.join(optimized_lines)

    def _optimize_git_output(self, output: str, command: str) -> str:
        """Optimize git command outputs."""
        # git status --short is already optimized
        if 'git status' in command and '--short' in command:
            return output
        
        # git status
        if 'git status' in command:
            lines = output.split('\n')
            optimized = []
            for line in lines:
                if line.startswith('On branch'):
                    optimized.append(line)
                elif line.startswith('Changes'):
                    optimized.append(line)
                elif line.startswith('Untracked'):
                    optimized.append(line)
                elif line.startswith(('modified:', 'new file:', 'deleted:')):
                    optimized.append(line.strip())
            limit = self.LINE_LIMITS.get(CommandCategory.GIT, 40)
            if len(optimized) > limit:
                optimized = optimized[:limit]
            return '\n'.join(optimized) if optimized else "No changes"
        
        # git diff
        if 'git diff' in command:
            lines = output.split('\n')
            # Keep only diff headers and first few changes
            optimized = []
            for i, line in enumerate(lines):
                if line.startswith(('diff --git', 'index', '---', '+++', '@@')):
                    optimized.append(line)
                elif line.startswith('+') and len(optimized) < 20:
                    optimized.append(line[:100])  # Truncate long lines
                elif line.startswith('-') and len(optimized) < 20:
                    optimized.append(line[:100])
            limit = self.LINE_LIMITS.get(CommandCategory.GIT, 40)
            if len(optimized) > limit:
                optimized = optimized[:limit] + ["... (diff truncated)"]
            return '\n'.join(optimized) if optimized else "No changes"
        
        # git log
        if 'git log' in command:
            lines = output.split('\n')
            optimized = []
            for line in lines:
                if line.startswith('commit '):
                    optimized.append(line[:16])
                elif line.startswith('Author:') or line.startswith('Date:'):
                    optimized.append(line[:50])
                elif line.strip() and not line.startswith('    '):
                    optimized.append(line[:80])
            limit = self.LINE_LIMITS.get(CommandCategory.GIT, 40)
            if len(optimized) > limit:
                optimized = optimized[:limit] + ["... (log truncated)"]
            return '\n'.join(optimized)
        
        # Default git optimization
        return self._truncate_lines(output, self.LINE_LIMITS.get(CommandCategory.GIT, 40))

    def _optimize_test_output(self, output: str) -> str:
        """Optimize test command outputs."""
        lines = output.split('\n')
        optimized = []
        
        # Extract key information
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        
        for line in lines:
            # pytest summary
            if 'passed' in line.lower() or 'failed' in line.lower() or 'error' in line.lower():
                # Extract numbers
                nums = re.findall(r'\d+', line)
                if nums:
                    if 'passed' in line.lower():
                        passed = int(nums[0])
                    if 'failed' in line.lower():
                        failed = int(nums[0])
                    if 'error' in line.lower():
                        errors = int(nums[0])
                    if 'skipped' in line.lower():
                        skipped = int(nums[0])
                optimized.append(line.strip())
            elif line.startswith('=') or line.startswith('-'):
                continue  # Skip separators
            elif 'PASSED' in line or 'FAILED' in line or 'ERROR' in line:
                optimized.append(line.strip()[:100])
        
        # If no summary found, create one
        if not optimized and lines:
            # Just truncate
            limit = self.LINE_LIMITS.get(CommandCategory.TEST, 50)
            return self._truncate_lines(output, limit)
        
        # Add summary
        if passed or failed or errors:
            summary = f"Tests: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped"
            if optimized and summary not in optimized[-1]:
                optimized.append(summary)
        
        limit = self.LINE_LIMITS.get(CommandCategory.TEST, 50)
        if len(optimized) > limit:
            optimized = optimized[:limit] + ["... (output truncated)"]
        
        return '\n'.join(optimized)

    def _optimize_docker_output(self, output: str) -> str:
        """Optimize docker command outputs."""
        lines = output.split('\n')
        optimized = []
        
        # Keep header
        if lines:
            optimized.append(lines[0])
        
        # Keep container info (first few columns)
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if parts:
                # Keep first 4-5 fields
                optimized.append(' '.join(parts[:5]))
        
        limit = self.LINE_LIMITS.get(CommandCategory.DOCKER, 20)
        if len(optimized) > limit:
            optimized = optimized[:limit] + ["... (+ more containers)"]
        
        return '\n'.join(optimized)

    def _optimize_lint_output(self, output: str) -> str:
        """Optimize linting command outputs."""
        lines = output.split('\n')
        optimized = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip summary lines with counts
            if re.match(r'^\d+\s+(error|warning)', line, re.IGNORECASE):
                optimized.append(line.strip())
                continue
            
            # Keep error/warning lines
            if any(x in line.lower() for x in ['error', 'warning', 'warn']):
                optimized.append(line.strip()[:120])
        
        limit = self.LINE_LIMITS.get(CommandCategory.LINT, 30)
        if len(optimized) > limit:
            optimized = optimized[:limit] + [f"... (+{len(optimized) - limit} more issues)"]
        
        return '\n'.join(optimized) if optimized else "No issues found"

    def _deduplicate(self, text: str) -> str:
        """Remove duplicate lines."""
        lines = text.split('\n')
        seen = set()
        unique = []
        
        for line in lines:
            # Create a normalized version for comparison
            normalized = line.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(line)
            elif not normalized:
                unique.append(line)  # Keep empty lines
        
        return '\n'.join(unique)

    def _create_summary(self, output: str, category: CommandCategory) -> str:
        """Create a summary of the output."""
        lines = output.split('\n')
        non_empty = [l for l in lines if l.strip()]
        
        summary = f"[{category.value}] {len(non_empty)} lines"
        
        # Add category-specific info
        if category == CommandCategory.TEST:
            # Try to extract test counts
            for line in lines:
                if 'passed' in line.lower():
                    summary += f" - {line.strip()}"
                    break
        elif category == CommandCategory.GIT:
            for line in lines:
                if 'changed' in line.lower():
                    summary += f" - {line.strip()}"
                    break
        
        return summary

    def optimize(self, command: str, output: str) -> str:
        """
        Optimize command output for token efficiency.
        
        Args:
            command: The command that was executed
            output: The raw output from the command
            
        Returns:
            Optimized output string
        """
        # Update stats
        original_tokens = self._count_tokens(output)
        self._stats.total_original += original_tokens
        self._stats.commands_processed += 1
        
        # Handle empty output
        if not output or not output.strip():
            self._stats.total_optimized += 1
            return output
        
        # Categorize command
        category = self.categorize_command(command)
        
        # Apply category-specific optimization
        if category == CommandCategory.DIRECTORY_LISTING:
            optimized = self._optimize_directory_listing(output)
        elif category == CommandCategory.FILE_CONTENT:
            optimized = self._optimize_file_content(output)
        elif category == CommandCategory.SEARCH:
            optimized = self._optimize_search_output(output)
        elif category == CommandCategory.GIT:
            optimized = self._optimize_git_output(output, command)
        elif category == CommandCategory.TEST:
            optimized = self._optimize_test_output(output)
        elif category == CommandCategory.DOCKER:
            optimized = self._optimize_docker_output(output)
        elif category == CommandCategory.LINT:
            optimized = self._optimize_lint_output(output)
        else:
            # Default: just truncate
            limit = self.max_lines or 50
            optimized = self._truncate_lines(output, limit)
        
        # Apply deduplication if enabled
        if self.enable_dedup:
            optimized = self._deduplicate(optimized)
        
        # Calculate savings
        optimized_tokens = self._count_tokens(optimized)
        savings = original_tokens - optimized_tokens
        
        # Update stats
        self._stats.total_optimized += optimized_tokens
        if savings > 0:
            self._stats.commands_optimized += 1
        
        # Log significant savings
        if savings > 100:
            logger.debug(f"Token optimization: {command[:30]}... saved {savings} tokens ({savings/original_tokens*100:.1f}%)")
        
        return optimized

    def optimize_result(self, command: str, output: str) -> OptimizationResult:
        """Optimize and return detailed result."""
        original = output
        optimized = self.optimize(command, output)
        
        original_tokens = self._count_tokens(original)
        optimized_tokens = self._count_tokens(optimized)
        
        savings = 0
        if original_tokens > 0:
            savings = ((original_tokens - optimized_tokens) / original_tokens) * 100
        
        return OptimizationResult(
            original=original,
            optimized=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            savings_percent=savings,
            command=command,
            category=self.categorize_command(command),
        )

    def get_stats(self) -> TokenStats:
        """Get token usage statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset token statistics."""
        self._stats = TokenStats()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self._count_tokens(text)


# Global optimizer instance
_default_optimizer: Optional[TokenOptimizer] = None


def get_optimizer() -> TokenOptimizer:
    """Get the default optimizer instance."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = TokenOptimizer()
    return _default_optimizer


def optimize_command_output(command: str, output: str) -> str:
    """Convenience function to optimize command output."""
    return get_optimizer().optimize(command, output)