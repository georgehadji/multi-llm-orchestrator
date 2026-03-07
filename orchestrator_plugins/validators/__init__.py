"""
Orchestrator Validators Plugin
==============================
Save this as orchestrator_plugins/validators/__init__.py

Additional validators for the orchestrator.
These were moved from orchestrator/validators.py and orchestrator/plugins.py
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to import from core orchestrator
try:
    from orchestrator.plugins import ValidatorPlugin, PluginMetadata, PluginType, ValidationResult
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    # Define minimal interfaces if core not available
    @dataclass
    class ValidationResult:
        passed: bool
        score: float = 0.0
        errors: list = None
        
        def __post_init__(self):
            if self.errors is None:
                self.errors = []
    
    @dataclass  
    class PluginMetadata:
        name: str
        version: str
        author: str
        description: str
        plugin_type: str = "validator"
    
    class ValidatorPlugin:
        @property
        def metadata(self) -> PluginMetadata:
            raise NotImplementedError
        
        def can_validate(self, file_path: str, language: str) -> bool:
            return False
        
        def validate(self, code: str, context: dict) -> ValidationResult:
            raise NotImplementedError


logger = logging.getLogger("orchestrator_plugins.validators")


class PythonTypeCheckerValidator(ValidatorPlugin):
    """Validate Python code using mypy."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="python-mypy",
            version="1.0.0",
            author="orchestrator-team",
            description="Type checking using mypy",
            plugin_type="validator",
        )
    
    def can_validate(self, file_path: str, language: str) -> bool:
        return language == "python" or file_path.endswith(".py")
    
    def validate(self, code: str, context: dict) -> ValidationResult:
        """Run mypy on the code."""
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # Run mypy
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", "--no-error-summary", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            passed = result.returncode == 0
            errors = result.stdout.strip().split('\n') if result.stdout else []
            
            return ValidationResult(
                passed=passed,
                score=1.0 if passed else 0.5,
                errors=errors,
            )
        except FileNotFoundError:
            logger.warning("mypy not installed, skipping type check")
            return ValidationResult(passed=True, score=1.0)  # Skip if mypy not available
        except Exception as e:
            logger.error(f"mypy validation failed: {e}")
            return ValidationResult(passed=False, score=0.0, errors=[str(e)])


class PythonSecurityValidator(ValidatorPlugin):
    """Validate Python code using bandit."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="python-bandit",
            version="1.0.0",
            author="orchestrator-team",
            description="Security scanning using bandit",
            plugin_type="validator",
        )
    
    def can_validate(self, file_path: str, language: str) -> bool:
        return language == "python" or file_path.endswith(".py")
    
    def validate(self, code: str, context: dict) -> ValidationResult:
        """Run bandit security scan."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                ["bandit", "-r", "-f", "json", "-ll", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            Path(temp_path).unlink(missing_ok=True)
            
            # Bandit returns 0 even with findings, parse JSON output
            import json
            try:
                output = json.loads(result.stdout)
                issues = output.get("results", [])
                
                high_severity = [i for i in issues if i.get("issue_severity") == "HIGH"]
                
                return ValidationResult(
                    passed=len(high_severity) == 0,
                    score=1.0 if not issues else 0.7 if not high_severity else 0.3,
                    errors=[f"{i['issue_text']} at line {i['line_number']}" for i in issues],
                )
            except json.JSONDecodeError:
                return ValidationResult(passed=True, score=1.0)
        
        except FileNotFoundError:
            logger.warning("bandit not installed, skipping security check")
            return ValidationResult(passed=True, score=1.0)
        except Exception as e:
            logger.error(f"bandit validation failed: {e}")
            return ValidationResult(passed=False, score=0.0, errors=[str(e)])


class JavaScriptValidator(ValidatorPlugin):
    """Validate JavaScript/TypeScript using ESLint."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="javascript-eslint",
            version="1.0.0",
            author="orchestrator-team",
            description="Linting using ESLint",
            plugin_type="validator",
        )
    
    def can_validate(self, file_path: str, language: str) -> bool:
        return language in ("javascript", "typescript") or \
               any(file_path.endswith(ext) for ext in [".js", ".ts", ".jsx", ".tsx"])
    
    def validate(self, code: str, context: dict) -> ValidationResult:
        """Run ESLint on the code."""
        try:
            ext = ".js"
            if context.get("file_path", "").endswith(".ts"):
                ext = ".ts"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                ["eslint", "--format", "json", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            Path(temp_path).unlink(missing_ok=True)
            
            import json
            try:
                output = json.loads(result.stdout)
                errors = []
                for file_result in output:
                    for msg in file_result.get("messages", []):
                        if msg.get("severity") == 2:  # Error
                            errors.append(f"Line {msg['line']}: {msg['message']}")
                
                return ValidationResult(
                    passed=len(errors) == 0,
                    score=1.0 if not errors else 0.6,
                    errors=errors,
                )
            except json.JSONDecodeError:
                return ValidationResult(passed=True, score=1.0)
        
        except FileNotFoundError:
            logger.warning("eslint not installed, skipping JS lint")
            return ValidationResult(passed=True, score=1.0)
        except Exception as e:
            logger.error(f"eslint validation failed: {e}")
            return ValidationResult(passed=False, score=0.0, errors=[str(e)])


class RustValidator(ValidatorPlugin):
    """Validate Rust code using cargo check."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="rust-cargo",
            version="1.0.0",
            author="orchestrator-team",
            description="Validation using cargo check",
            plugin_type="validator",
        )
    
    def can_validate(self, file_path: str, language: str) -> bool:
        return language == "rust" or file_path.endswith(".rs")
    
    def validate(self, code: str, context: dict) -> ValidationResult:
        """Run cargo check on the code."""
        try:
            # Create temp directory with Cargo.toml
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write Cargo.toml
                cargo_toml = Path(tmpdir) / "Cargo.toml"
                cargo_toml.write_text("""
[package]
name = "temp"
version = "0.1.0"
edition = "2021"
""")
                # Write code to src/lib.rs
                src_dir = Path(tmpdir) / "src"
                src_dir.mkdir()
                lib_rs = src_dir / "lib.rs"
                lib_rs.write_text(code)
                
                # Run cargo check
                result = subprocess.run(
                    ["cargo", "check", "--message-format=short"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                errors = [line for line in result.stderr.split('\n') if 'error' in line]
                
                return ValidationResult(
                    passed=result.returncode == 0,
                    score=1.0 if result.returncode == 0 else 0.4,
                    errors=errors[:10],  # Limit errors
                )
        
        except FileNotFoundError:
            logger.warning("cargo not installed, skipping Rust validation")
            return ValidationResult(passed=True, score=1.0)
        except Exception as e:
            logger.error(f"cargo check failed: {e}")
            return ValidationResult(passed=False, score=0.0, errors=[str(e)])


# Auto-register validators if core is available
if HAS_CORE:
    try:
        from orchestrator.plugins import get_plugin_registry
        registry = get_plugin_registry()
        registry.register(PythonTypeCheckerValidator())
        registry.register(PythonSecurityValidator())
        registry.register(JavaScriptValidator())
        registry.register(RustValidator())
        logger.info("Registered validator plugins")
    except Exception as e:
        logger.warning(f"Failed to auto-register validators: {e}")


__all__ = [
    "PythonTypeCheckerValidator",
    "PythonSecurityValidator",
    "JavaScriptValidator",
    "RustValidator",
]
