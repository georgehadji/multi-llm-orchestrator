"""
Code Post-Processor
===================
Author: Georgios-Chrysovalantis Chatzivantsidis

Fixes common LLM-generated code mistakes before writing to disk:
1. Removes JavaScript-style comments (/** */) from Python files
2. Fixes indentation errors (method order issues)
3. Fixes broken imports (non-existent modules)
4. Removes duplicate class definitions
5. Validates and fixes syntax errors

Usage:
    from orchestrator.code_post_processor import CodePostProcessor
    
    processor = CodePostProcessor()
    fixed_code = processor.process(code, filename="task_001.py")
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Pattern

logger = logging.getLogger("orchestrator.code_post_processor")


class CodePostProcessor:
    """
    Post-processes LLM-generated code to fix common mistakes.
    """

    # JavaScript-style comment patterns
    JS_COMMENT_PATTERNS: list[Pattern] = [
        re.compile(r"/\*\*.*?\*/", re.DOTALL),  # /** ... */
        re.compile(r"/\*.*?\*/", re.DOTALL),    # /* ... */
        re.compile(r"^\s*\*\s+.*$", re.MULTILINE),  # * line (JSDoc continuation)
    ]

    # Common fake module names that LLMs hallucinate
    FAKE_MODULE_PATTERNS: list[tuple[Pattern, str]] = [
        (re.compile(r"^from\s+task_\d+\s+import", re.MULTILINE), "# FIXED: Removed fake task_N import"),
        (re.compile(r"^import\s+task_\d+\b", re.MULTILINE), "# FIXED: Removed fake task_N import"),
        (re.compile(r"^from\s+memory_system\s+import", re.MULTILINE), "# FIXED: Removed fake memory_system import"),
        (re.compile(r"^import\s+memory_system$", re.MULTILINE), "# FIXED: Removed fake memory_system import"),
        # NEW: Catch other common hallucinations
        (re.compile(r"^from\s+existing_module\s+import", re.MULTILINE), "# FIXME: Module not found - from existing_module import ..."),
        (re.compile(r"^import\s+existing_module\b", re.MULTILINE), "# FIXME: Module not found - import existing_module"),
        (re.compile(r"^from\s+your_module\s+import", re.MULTILINE), "# FIXME: Module not found - from your_module import ..."),
        (re.compile(r"^from\s+my_module\s+import", re.MULTILINE), "# FIXME: Module not found - from my_module import ..."),
        (re.compile(r"^from\s+some_module\s+import", re.MULTILINE), "# FIXME: Module not found - from some_module import ..."),
    ]

    def __init__(self):
        self.fixes_applied: list[str] = []

    def process(self, code: str, filename: str = "") -> str:
        """
        Process code and apply all fixes.
        
        Args:
            code: The LLM-generated code
            filename: Optional filename for context
            
        Returns:
            Fixed code
        """
        self.fixes_applied = []
        original_code = code
        
        # Apply fixes in order
        code = self._remove_js_comments(code, filename)
        code = self._fix_fake_imports(code)
        code = self._fix_method_order(code)
        code = self._fix_unterminated_strings(code)
        code = self._strip_prose_after_code(code)  # NEW: Strip trailing prose
        
        # Validate syntax
        is_valid, error = self._validate_syntax(code)
        if not is_valid:
            logger.warning(f"Syntax error in {filename}: {error}")
            # Try to extract just the valid parts
            code = self._extract_valid_code(code, error)
            self.fixes_applied.append("syntax_error_truncated")
        
        # Final validation - warn if still broken
        is_valid, final_error = self._validate_syntax(code)
        if not is_valid:
            logger.error(f"CRITICAL: {filename} still has syntax errors after post-processing: {final_error}")
        
        if self.fixes_applied:
            logger.info(f"Applied {len(self.fixes_applied)} fixes to {filename}: {', '.join(self.fixes_applied)}")
        
        return code

    def _remove_js_comments(self, code: str, filename: str) -> str:
        """Remove JavaScript-style comments from Python files."""
        if not filename.endswith(".py"):
            return code
        
        original = code
        for pattern in self.JS_COMMENT_PATTERNS:
            code = pattern.sub("", code)
        
        if code != original:
            self.fixes_applied.append("removed_js_comments")
            # Clean up empty lines left by comment removal
            code = re.sub(r"\n\s*\n\s*\n", "\n\n", code)
        
        return code

    def _fix_fake_imports(self, code: str) -> str:
        """Remove imports from non-existent modules."""
        original = code
        
        for pattern, replacement in self.FAKE_MODULE_PATTERNS:
            code = pattern.sub(replacement, code)
        
        if code != original:
            self.fixes_applied.append("fixed_fake_imports")
        
        return code

    def _fix_method_order(self, code: str) -> str:
        """
        Fix common indentation/order issues in class methods.
        Specifically fixes the case where docstring appears before __init__.
        """
        lines = code.split("\n")
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for method with docstring before __init__
            # Pattern: def method_name(self, ...): followed by """ on next lines
            # but __init__ comes after the docstring
            if re.match(r"^\s+def\s+\w+", line) and "__init__" not in line:
                # Look ahead for docstring
                j = i + 1
                docstring_start = -1
                docstring_end = -1
                
                while j < len(lines):
                    if docstring_start == -1:
                        if '"""' in lines[j] or "'''" in lines[j]:
                            docstring_start = j
                    else:
                        if '"""' in lines[j] or "'''" in lines[j]:
                            docstring_end = j
                            break
                    j += 1
                
                # Check if __init__ appears after the docstring
                if docstring_start != -1 and docstring_end != -1:
                    for k in range(docstring_end + 1, min(docstring_end + 10, len(lines))):
                        if re.match(r"^\s+def\s+__init__", lines[k]):
                            # Found the problem! Move __init__ before this method
                            init_lines = []
                            k_end = k + 1
                            # Find end of __init__ method
                            init_indent = len(lines[k]) - len(lines[k].lstrip())
                            while k_end < len(lines):
                                if lines[k_end].strip() and not lines[k_end].strip().startswith("#"):
                                    current_indent = len(lines[k_end]) - len(lines[k_end].lstrip())
                                    if current_indent <= init_indent and re.match(r"^\s+def\s+", lines[k_end]):
                                        break
                                k_end += 1
                            
                            init_lines = lines[k:k_end]
                            # Remove __init__ from original position
                            lines = lines[:k] + lines[k_end:]
                            # Insert before current method
                            lines = lines[:i] + init_lines + lines[i:]
                            
                            self.fixes_applied.append("fixed_method_order")
                            return "\n".join(lines)
            
            fixed_lines.append(line)
            i += 1
        
        return "\n".join(fixed_lines)

    def _fix_unterminated_strings(self, code: str) -> str:
        """Fix unterminated string literals in the code."""
        # Count quotes to detect unterminated strings
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        triple_single = code.count("'''")
        triple_double = code.count('"""')
        
        # If odd number of non-triple quotes, try to fix
        if (single_quotes - triple_single * 3) % 2 == 1:
            # Find the last single quote and add closing
            last_quote = code.rfind("'")
            if last_quote != -1:
                code = code[:last_quote+1] + "'" + code[last_quote+1:]
                self.fixes_applied.append("fixed_unterminated_string")
        
        if (double_quotes - triple_double * 3) % 2 == 1:
            # Find the last double quote and add closing
            last_quote = code.rfind('"')
            if last_quote != -1:
                code = code[:last_quote+1] + '"' + code[last_quote+1:]
                self.fixes_applied.append("fixed_unterminated_string")
        
        return code

    def _strip_prose_after_code(self, code: str) -> str:
        """
        Strip prose/explanation text that appears after valid Python code.
        LLMs often add 'This code does...' or 'Example usage:' explanations at the end.
        """
        lines = code.split('\n')
        
        # Common prose markers that indicate non-code content
        prose_markers = [
            "this code",
            "this module",
            "this class",
            "this function",
            "this implementation",
            "the above",
            "the implementation",
            "in summary",
            "to summarize",
            "in conclusion",
        ]
        
        # Find the first prose marker after line 50 (avoid cutting early comments)
        prose_start = None
        for i, line in enumerate(lines[50:], start=50):
            stripped = line.strip().lower()
            # Skip empty lines
            if not stripped:
                continue
            # Check if line starts with prose marker
            for marker in prose_markers:
                if stripped.startswith(marker):
                    prose_start = i
                    break
            if prose_start:
                break
        
        if prose_start:
            # Check if this is actually valid Python (might be a docstring)
            test_code = '\n'.join(lines[:prose_start])
            try:
                ast.parse(test_code)
                # Valid parse - truncate at prose
                self.fixes_applied.append("stripped_trailing_prose")
                return test_code
            except SyntaxError:
                # Might need more context, try including more lines
                pass
        
        # Alternative: Find last line that ends a valid Python block
        # Work backwards from end to find last valid parse point
        for end_line in range(len(lines), max(len(lines) - 20, 0), -1):
            test_code = '\n'.join(lines[:end_line])
            try:
                ast.parse(test_code)
                # Found valid parse point
                if end_line < len(lines):
                    # Check if removed content looks like prose
                    removed = '\n'.join(lines[end_line:])
                    if any(marker in removed.lower() for marker in prose_markers):
                        self.fixes_applied.append("stripped_trailing_prose")
                        return test_code
                break
            except SyntaxError:
                continue
        
        return code

    def _validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _extract_valid_code(self, code: str, error: str) -> str:
        """Try to extract valid portions of code when there are syntax errors."""
        lines = code.split("\n")
        
        # Try to find the problematic line
        match = re.search(r"Line (\d+):", error)
        if match:
            error_line = int(match.group(1))
            
            # Special handling for unterminated strings - find the start of the string
            if "unterminated" in error.lower() or "string literal" in error.lower():
                # Work backwards to find a good truncation point
                # Look for class or function definition before the error
                for truncate_at in range(error_line - 1, 0, -1):
                    line = lines[truncate_at - 1].strip()
                    # Good truncation points: end of a function/method
                    if line == "" or line.startswith("def ") or line.startswith("class "):
                        truncated = "\n".join(lines[:truncate_at])
                        # Verify this parses
                        try:
                            ast.parse(truncated)
                            return truncated
                        except SyntaxError:
                            continue
            
            # Default: Keep code up to the line before the error
            if error_line > 1:
                truncated = "\n".join(lines[:error_line-1])
                try:
                    ast.parse(truncated)
                    return truncated
                except SyntaxError:
                    # Error is earlier, use more aggressive truncation
                    for truncate_at in range(error_line - 2, 0, -1):
                        truncated = "\n".join(lines[:truncate_at])
                        try:
                            ast.parse(truncated)
                            return truncated
                        except SyntaxError:
                            continue
        
        # Fallback: find last valid parse point by binary search
        left, right = 1, len(lines)
        last_valid = 0
        while left <= right:
            mid = (left + right) // 2
            test_code = "\n".join(lines[:mid])
            try:
                ast.parse(test_code)
                last_valid = mid
                left = mid + 1
            except SyntaxError:
                right = mid - 1
        
        if last_valid > 0:
            return "\n".join(lines[:last_valid])
        
        # Worst case: return original code
        return code


def post_process_code(code: str, filename: str = "") -> str:
    """
    Convenience function to post-process code.
    
    Args:
        code: The LLM-generated code
        filename: Optional filename for context
        
    Returns:
        Fixed code
    """
    processor = CodePostProcessor()
    return processor.process(code, filename)
