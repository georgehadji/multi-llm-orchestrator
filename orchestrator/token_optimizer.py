"""
TokenOptimizer — Command-specific token compression
=================================================
Module for applying domain-specific token compression strategies for various command outputs
like git logs, pytest results, ESLint output, etc.

Pattern: Strategy
Async: No — pure text processing
Layer: L2 Verification

Usage:
    from orchestrator.token_optimizer import TokenOptimizer
    optimizer = TokenOptimizer()
    compressed = optimizer.compress_command_output("git log", git_log_output)
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("orchestrator.token_optimizer")


class TokenOptimizer:
    """Applies domain-specific token compression strategies for various command outputs."""

    def __init__(self):
        """Initialize the token optimizer with command-specific strategies."""
        self.strategies = {
            "git_log": self._compress_git_log,
            "pytest": self._compress_pytest_output,
            "eslint": self._compress_eslint_output,
            "docker_ps": self._compress_docker_ps,
            "npm_ls": self._compress_npm_ls,
            "ps_aux": self._compress_ps_aux,
            "df_h": self._compress_df_h,
            "free_m": self._compress_free_m,
            "top": self._compress_top,
            "netstat": self._compress_netstat,
        }
    
    def compress_command_output(self, command: str, output: str, 
                              target_ratio: float = 0.5) -> str:
        """
        Compress command output using the appropriate strategy.
        
        Args:
            command: The command that generated the output
            output: The output to compress
            target_ratio: Target compression ratio (0.1 = 10% of original)
            
        Returns:
            str: Compressed output
        """
        # Normalize command name
        command_normalized = self._normalize_command(command)
        
        # Apply the appropriate strategy if available
        if command_normalized in self.strategies:
            strategy = self.strategies[command_normalized]
            try:
                compressed = strategy(output, target_ratio)
                logger.debug(f"Applied {command_normalized} compression strategy")
                return compressed
            except Exception as e:
                logger.warning(f"Compression strategy for {command_normalized} failed: {e}")
                # Fall back to generic compression
        
        # If no specific strategy or it failed, use generic compression
        return self._generic_compress(output, target_ratio)
    
    def _normalize_command(self, command: str) -> str:
        """Normalize command name to a standard form."""
        # Extract base command name, ignoring arguments
        base_cmd = command.split()[0].split('/')[-1].lower()
        
        # Map variations to standard names
        cmd_mapping = {
            "git": "git_log",  # Default to git_log for git commands
            "pytest": "pytest",
            "py.test": "pytest",
            "python -m pytest": "pytest",
            "eslint": "eslint",
            "docker": "docker_ps",  # Default to docker_ps for docker commands
            "npm": "npm_ls",  # Default to npm_ls for npm commands
            "ps": "ps_aux",
            "df": "df_h",
            "free": "free_m",
            "top": "top",
            "netstat": "netstat"
        }
        
        # Special handling for specific subcommands
        if command.startswith("git log"):
            return "git_log"
        elif command.startswith("docker ps"):
            return "docker_ps"
        elif command.startswith("npm ls") or command.startswith("npm list"):
            return "npm_ls"
        elif command.startswith("df -h"):
            return "df_h"
        elif command.startswith("free -m"):
            return "free_m"
        
        return cmd_mapping.get(base_cmd, "generic")
    
    def _compress_git_log(self, output: str, target_ratio: float) -> str:
        """Compress git log output."""
        lines = output.split('\n')
        
        # Extract commit hashes and messages
        commits = []
        current_commit = {}
        
        for line in lines:
            if line.startswith('commit '):
                if current_commit:
                    commits.append(current_commit)
                current_commit = {'hash': line.split()[1][:8], 'details': []}
            elif line.startswith('Author:') or line.startswith('Date:'):
                current_commit[line.split(':')[0].lower()] = line[len(line.split(':')[0])+2:]
            else:
                # Add other lines as details
                if 'details' not in current_commit:
                    current_commit['details'] = []
                current_commit['details'].append(line)
        
        # Add the last commit
        if current_commit:
            commits.append(current_commit)
        
        # Calculate target number of commits to keep
        target_commits = max(1, int(len(commits) * target_ratio))
        
        # Keep the most recent commits
        selected_commits = commits[:target_commits]
        
        # Reconstruct output with selected commits
        result_lines = []
        for commit in selected_commits:
            result_lines.append(f"commit {commit['hash']}")
            if 'author' in commit:
                result_lines.append(f"Author: {commit['author']}")
            if 'date' in commit:
                result_lines.append(f"Date: {commit['date']}")
            result_lines.extend(commit.get('details', []))
            result_lines.append("")  # Empty line between commits
        
        return '\n'.join(result_lines)
    
    def _compress_pytest_output(self, output: str, target_ratio: float) -> str:
        """Compress pytest output."""
        # Extract key information: passed, failed, skipped counts
        summary_match = re.search(r'(\d+) passed, (\d+) failed, (\d+) error, (\d+) skipped', output)
        if not summary_match:
            summary_match = re.search(r'(\d+) passed, (\d+) failed', output)
        
        if summary_match:
            # Start with the summary
            summary = f"PYTEST SUMMARY: {summary_match.group(0)}\n\n"
        else:
            summary = "PYTEST OUTPUT (summary not found)\n\n"
        
        # Extract failure details if any
        failure_sections = []
        lines = output.split('\n')
        in_failure = False
        current_failure = []
        
        for line in lines:
            if 'FAILED' in line or line.strip().startswith('_ test'):
                in_failure = True
                if current_failure:
                    failure_sections.append('\n'.join(current_failure))
                    if len(failure_sections) >= 5:  # Limit to top 5 failures
                        break
                current_failure = [line]
            elif in_failure:
                if line.strip() == '' and len(current_failure) > 5:  # End of failure block
                    # Check if we're moving to a new test or end of output
                    next_nonempty = next((l for l in lines[lines.index(line)+1:] if l.strip()), '')
                    if next_nonempty.strip().startswith('===') or 'passed' in next_nonempty.lower():
                        failure_sections.append('\n'.join(current_failure))
                        current_failure = []
                        in_failure = False
                    else:
                        current_failure.append(line)
                else:
                    current_failure.append(line)
        
        # Add the last failure if exists
        if current_failure and len(failure_sections) < 5:
            failure_sections.append('\n'.join(current_failure))
        
        # Combine summary and limited failures
        result = summary
        if failure_sections:
            result += "TOP FAILURES:\n\n"
            result += "\n\n".join(failure_sections)
        
        # Add final summary if not included in our extracted text
        if '=====' in output:
            # Extract the final summary section
            parts = output.split('=====') 
            if len(parts) > 1:
                final_summary = parts[-1]
                result += f"\n\n{final_summary}"
        
        return result
    
    def _compress_eslint_output(self, output: str, target_ratio: float) -> str:
        """Compress ESLint output."""
        lines = output.split('\n')
        
        # Extract summary information
        summary_lines = []
        detail_lines = []
        
        for line in lines:
            if re.match(r'^\s*\d+ problems? \(\d+ errors?, \d+ warnings?\)', line):
                summary_lines.append(line)
            elif re.match(r'^.*:\d+:\d+\s+', line):  # Lines with file:line:col
                detail_lines.append(line)
            elif '[Error]' in line or '[Warning]' in line:
                detail_lines.append(line)
        
        # Calculate how many details to keep based on target ratio
        target_details = max(1, int(len(detail_lines) * target_ratio))
        
        # Combine summary with limited details
        result_lines = summary_lines
        result_lines.extend(detail_lines[:target_details])
        
        return '\n'.join(result_lines)
    
    def _compress_docker_ps(self, output: str, target_ratio: float) -> str:
        """Compress docker ps output."""
        lines = output.split('\n')
        
        if not lines:
            return output
        
        # Keep header
        header = lines[0]
        containers = lines[1:]  # Skip header
        
        # Calculate how many containers to keep
        target_containers = max(1, int(len(containers) * target_ratio))
        
        # Keep header and top containers
        result_lines = [header]
        result_lines.extend(containers[:target_containers])
        
        return '\n'.join(result_lines)
    
    def _compress_npm_ls(self, output: str, target_ratio: float) -> str:
        """Compress npm ls output."""
        lines = output.split('\n')
        
        # Count depth levels to identify packages
        packages = []
        for line in lines:
            # Count leading spaces or tree characters to determine depth
            depth = 0
            for char in line:
                if char in [' ', '|', '+', '-']:
                    depth += 1
                else:
                    break
            packages.append((depth, line))
        
        # Calculate how many packages to keep
        target_packages = max(1, int(len(packages) * target_ratio))
        
        # Keep top-level packages and some nested ones
        result_lines = []
        current_depth = -1
        kept_count = 0
        
        for depth, line in packages:
            if kept_count >= target_packages:
                break
                
            # If we're going deeper, we might skip some
            if depth > current_depth:
                result_lines.append(line)
                kept_count += 1
                current_depth = depth
            # If we're at the same or shallower level, include it
            elif depth <= current_depth:
                result_lines.append(line)
                kept_count += 1
                current_depth = depth
        
        return '\n'.join(result_lines)
    
    def _compress_ps_aux(self, output: str, target_ratio: float) -> str:
        """Compress ps aux output."""
        lines = output.split('\n')
        
        if not lines:
            return output
        
        # Keep header
        header = lines[0]
        processes = lines[1:]  # Skip header
        
        # Calculate how many processes to keep
        target_processes = max(1, int(len(processes) * target_ratio))
        
        # Keep header and top processes (maybe sort by CPU or memory usage)
        # For now, just take the first N
        result_lines = [header]
        result_lines.extend(processes[:target_processes])
        
        return '\n'.join(result_lines)
    
    def _compress_df_h(self, output: str, target_ratio: float) -> str:
        """Compress df -h output."""
        lines = output.split('\n')
        
        if not lines:
            return output
        
        # Keep header
        header = lines[0]
        filesystems = lines[1:]  # Skip header
        
        # Calculate how many filesystems to keep
        target_filesystems = max(1, int(len(filesystems) * target_ratio))
        
        # Keep header and top filesystems (perhaps filter out tmpfs, etc.)
        result_lines = [header]
        
        # Optionally filter out less important filesystems
        important_fs = []
        for fs in filesystems:
            if fs.strip() and not any(skip in fs for skip in ['tmpfs', 'devtmpfs', 'overlay']):
                important_fs.append(fs)
        
        result_lines.extend(important_fs[:target_filesystems])
        
        return '\n'.join(result_lines)
    
    def _compress_free_m(self, output: str, target_ratio: float) -> str:
        """Compress free -m output."""
        # Free output is usually just a few lines, so just return as is if small
        # Otherwise, return just the main memory info
        lines = output.split('\n')
        
        # If already small, return as is
        if len(lines) <= 5:
            return output
        
        # Otherwise, return just the main memory line
        for line in lines:
            if line.startswith('Mem:') or line.startswith('Swap:'):
                return f"MEMORY USAGE:\n{line}"
        
        # If no Mem: or Swap: line found, return first few lines
        return '\n'.join(lines[:3])
    
    def _compress_top(self, output: str, target_ratio: float) -> str:
        """Compress top output."""
        lines = output.split('\n')
        
        # Keep header lines (first few lines with system info)
        header_lines = []
        process_lines = []
        
        in_processes = False
        for line in lines:
            if not in_processes and ('PID' in line and 'USER' in line):
                in_processes = True
                process_lines.append(line)
            elif in_processes:
                process_lines.append(line)
            else:
                header_lines.append(line)
        
        # Calculate how many processes to keep
        target_processes = max(1, int(len(process_lines) * target_ratio))
        
        # Combine headers with top processes
        result_lines = header_lines
        result_lines.extend(process_lines[:target_processes])
        
        return '\n'.join(result_lines)
    
    def _compress_netstat(self, output: str, target_ratio: float) -> str:
        """Compress netstat output."""
        lines = output.split('\n')
        
        if not lines:
            return output
        
        # Keep header
        header = lines[0]
        connections = lines[1:]  # Skip header
        
        # Calculate how many connections to keep
        target_connections = max(1, int(len(connections) * target_ratio))
        
        # Keep header and top connections
        result_lines = [header]
        result_lines.extend(connections[:target_connections])
        
        return '\n'.join(result_lines)
    
    def _generic_compress(self, output: str, target_ratio: float) -> str:
        """Generic compression when no specific strategy applies."""
        # Simple approach: keep the first and last portions, drop the middle
        lines = output.split('\n')
        total_lines = len(lines)
        
        if total_lines <= 10:  # Already small
            return output
        
        # Calculate how many lines to keep
        target_lines = max(10, int(total_lines * target_ratio))  # Keep at least 10 lines
        
        if target_lines >= total_lines:  # No compression needed
            return output
        
        # Keep header portion and tail portion
        header_lines = max(5, target_lines // 2)
        tail_lines = target_lines - header_lines
        
        result_lines = lines[:header_lines]
        if tail_lines > 0 and len(lines) > header_lines:
            result_lines.extend(lines[-tail_lines:])
        
        return '\n'.join(result_lines)
    
    def get_compression_ratio(self, original: str, compressed: str) -> float:
        """
        Calculate the compression ratio.
        
        Args:
            original: Original text
            compressed: Compressed text
            
        Returns:
            float: Compression ratio (compressed/original)
        """
        if len(original) == 0:
            return 0.0
        return len(compressed) / len(original)
    
    def add_strategy(self, command_type: str, strategy_func):
        """
        Add a custom compression strategy for a command type.
        
        Args:
            command_type: Type of command (e.g., "custom_tool")
            strategy_func: Function that takes (output, target_ratio) and returns compressed output
        """
        self.strategies[command_type] = strategy_func
        logger.info(f"Added custom compression strategy for {command_type}")
    
    def list_strategies(self) -> List[str]:
        """
        List all available compression strategies.
        
        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())


# Global instance for convenience
_global_token_optimizer = TokenOptimizer()


def get_global_token_optimizer() -> TokenOptimizer:
    """
    Get the global token optimizer instance.

    Returns:
        TokenOptimizer instance
    """
    return _global_token_optimizer


# Alias used by engine.py import
get_optimizer = get_global_token_optimizer