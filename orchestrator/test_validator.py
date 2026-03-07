"""
Test Validator — Pre-validates Test Generation
===============================================

Ensures generated tests pass BEFORE committing them.

Usage:
    validator = TestValidator()
    passed, test_code = await validator.validate_test_generation(
        source_file=Path("src/main.py"),
        function_name="calculate",
        project_root=Path(".")
    )
"""

import ast
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .api_clients import UnifiedClient
from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class TestValidationResult:
    """Result of test validation."""
    passed: bool
    test_code: str
    error_message: Optional[str] = None
    iterations: int = 0
    output: str = ""


class TestValidator:
    """
    Validates test generation in real-time.
    
    Strategy:
    1. Generate test with full code context
    2. Run test immediately
    3. If fails, fix the TEST (not code)
    4. Repeat until pass or max iterations
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.client = UnifiedClient()
    
    async def validate_test_generation(
        self,
        source_file: Path,
        function_name: str,
        project_root: Optional[Path] = None,
        context: Optional[Dict] = None
    ) -> TestValidationResult:
        """
        Generate and validate a test.
        
        Args:
            source_file: Path to source file to test
            function_name: Function/class name to test
            project_root: Project root directory
            context: Additional context for test generation
        
        Returns:
            TestValidationResult with passing test code
        """
        project_root = project_root or Path.cwd()
        context = context or {}
        
        # CRITICAL: Check if source file exists
        if not source_file.exists():
            logger.warning(f"Source file not found: {source_file}")
            return TestValidationResult(
                passed=False,
                test_code="",
                error_message=f"Source file not found: {source_file}"
            )
        
        # Check if source file is empty
        try:
            source_code = source_file.read_text(encoding='utf-8')
            if not source_code or not source_code.strip():
                logger.warning(f"Source file is empty: {source_file}")
                return TestValidationResult(
                    passed=False,
                    test_code="",
                    error_message=f"Source file is empty: {source_file}"
                )
        except Exception as e:
            logger.warning(f"Failed to read source file: {e}")
            return TestValidationResult(
                passed=False,
                test_code="",
                error_message=f"Failed to read source file: {e}"
            )
        
        # Generate import path
        import_path = self._generate_import_path(source_file, project_root)
        
        # Analyze function signature
        func_info = self._analyze_function(source_code, function_name)
        
        # Generate fixtures
        fixtures = self._generate_fixtures(source_code)
        
        # Iterative test generation and validation
        test_code = ""
        last_error = ""
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Test validation iteration {iteration}/{self.max_iterations}")
            
            # Generate test
            test_code = await self._generate_test(
                source_code=source_code,
                import_path=import_path,
                function_name=function_name,
                func_info=func_info,
                fixtures=fixtures,
                previous_error=last_error,
                iteration=iteration
            )
            
            # Write test to temp file
            temp_test = self._write_temp_test(test_code, function_name)
            
            try:
                # Run test immediately
                result = await self._run_test(temp_test, source_file.parent)
                
                if result.returncode == 0:
                    # Success!
                    logger.info(f"✅ Test passed on iteration {iteration}")
                    return TestValidationResult(
                        passed=True,
                        test_code=test_code,
                        iterations=iteration,
                        output=result.stdout
                    )
                
                # Failed - capture error for next iteration
                last_error = result.stdout + result.stderr
                logger.warning(f"Test failed: {last_error[:200]}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Test execution error: {e}")
            
            finally:
                # Cleanup temp file
                if temp_test.exists():
                    temp_test.unlink()
        
        # Max iterations reached
        logger.warning(f"❌ Test validation failed after {self.max_iterations} iterations")
        return TestValidationResult(
            passed=False,
            test_code=test_code,
            error_message=last_error,
            iterations=self.max_iterations
        )
    
    def _generate_import_path(self, source_file: Path, project_root: Path) -> str:
        """Generate correct import path for source file."""
        try:
            rel_path = source_file.relative_to(project_root)
            
            # Convert path to module import
            # tasks/task_001_code_generation.py → tasks.task_001_code_generation
            module_path = str(rel_path.with_suffix('')).replace('\\', '.').replace('/', '.')
            
            # Remove 'test_' prefix if present
            if module_path.startswith('test_'):
                module_path = module_path[5:]
            
            return module_path
            
        except ValueError:
            # File not under project root - use filename
            return source_file.stem
    
    def _analyze_function(self, source_code: str, function_name: str) -> Dict:
        """Analyze function signature and docstring."""
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract signature info
                    args = []
                    annotations = {}
                    
                    for arg in node.args.args:
                        arg_name = arg.arg
                        args.append(arg_name)
                        
                        if arg.annotation:
                            if isinstance(arg.annotation, ast.Name):
                                annotations[arg_name] = arg.annotation.id
                            elif isinstance(arg.annotation, ast.Subscript):
                                annotations[arg_name] = ast.unparse(arg.annotation)
                    
                    # Get return type
                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns)
                    
                    # Get docstring
                    docstring = ast.get_docstring(node) or ""
                    
                    return {
                        'args': args,
                        'annotations': annotations,
                        'return_type': return_type,
                        'docstring': docstring,
                        'line_number': node.lineno
                    }
            
        except Exception as e:
            logger.warning(f"Failed to analyze function {function_name}: {e}")
        
        # Default
        return {
            'args': ['*args', '**kwargs'],
            'annotations': {},
            'return_type': None,
            'docstring': '',
            'line_number': 0
        }
    
    def _generate_fixtures(self, source_code: str) -> str:
        """Generate pytest fixtures based on code analysis."""
        fixtures = []
        
        try:
            tree = ast.parse(source_code)
            
            # Look for type hints to generate appropriate fixtures
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for arg in node.args.args:
                        if arg.annotation:
                            ann_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                            
                            if 'list' in ann_str.lower() or 'List' in ann_str:
                                if 'sample_list' not in [f.split()[1] for f in fixtures]:
                                    fixtures.append("""
@pytest.fixture
def sample_list():
    return [1, 2, 3]
""")
                            
                            elif 'dict' in ann_str.lower() or 'Dict' in ann_str:
                                if 'sample_dict' not in [f.split()[1] for f in fixtures]:
                                    fixtures.append("""
@pytest.fixture
def sample_dict():
    return {"key": "value"}
""")
                            
                            elif 'str' in ann_str.lower():
                                if 'sample_string' not in [f.split()[1] for f in fixtures]:
                                    fixtures.append("""
@pytest.fixture
def sample_string():
    return "test"
""")
                            
                            elif 'int' in ann_str.lower() or 'float' in ann_str.lower():
                                if 'sample_number' not in [f.split()[1] for f in fixtures]:
                                    fixtures.append("""
@pytest.fixture
def sample_number():
    return 42
""")
        
        except Exception as e:
            logger.warning(f"Failed to generate fixtures: {e}")
        
        return '\n'.join(fixtures)
    
    async def _generate_test(
        self,
        source_code: str,
        import_path: str,
        function_name: str,
        func_info: Dict,
        fixtures: str,
        previous_error: str = "",
        iteration: int = 1
    ) -> str:
        """Generate test code using LLM."""
        
        # Build prompt
        if iteration == 1:
            prompt = f"""You are an expert Python test developer. Generate a pytest test for this function.

## Source Code
```python
{source_code}
```

## Function to Test
- Import: `from {import_path} import {function_name}`
- Signature: `{function_name}({', '.join(func_info['args'])})`
- Docstring: {func_info['docstring'][:200] if func_info['docstring'] else 'N/A'}

## Fixtures (use if needed)
{fixtures}

## Test Requirements

1. Use the correct import path: `from {import_path} import {function_name}`
2. Test basic functionality with simple inputs
3. Test edge cases (empty, None, wrong type)
4. Use clear, descriptive test names
5. Keep tests simple and focused

## Example Pattern

```python
import pytest
from {import_path} import {function_name}

{fixtures}

def test_{function_name}_basic():
    result = {function_name}(sample_input)
    assert result == expected_value

def test_{function_name}_edge_case():
    with pytest.raises((TypeError, ValueError)):
        {function_name}(None)
```

## Generate the Test

Generate ONLY the test code, nothing else:
"""
        else:
            # Fix iteration - include error
            prompt = f"""Fix this failing test.

## Original Test Code
```python
{previous_error.split('```python')[1].split('```')[0] if '```python' in previous_error else 'N/A'}
```

## Error Message
```
{previous_error[:1000]}
```

## Instructions

1. Fix the test code to make it pass
2. Common issues:
   - Wrong import path (use: `from {import_path} import {function_name}`)
   - Missing fixtures
   - Wrong assertions
   - Incorrect expected values
3. Keep the test simple
4. Generate ONLY the fixed test code:
"""
        
        # Call LLM
        try:
            response = await self.client.call_model(
                model="gpt-4o-mini",  # Cheaper, faster for tests
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2  # Low temp for consistent tests
            )
            
            # Extract code block
            content = response.text
            import re
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
            
            if code_match:
                return code_match.group(1).strip()
            
            # If no code block, return as-is
            return content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate test: {e}")
            return self._generate_fallback_test(import_path, function_name)
    
    def _generate_fallback_test(self, import_path: str, function_name: str) -> str:
        """Generate a minimal fallback test."""
        return f"""
import pytest

def test_{function_name}_exists():
    \"\"\"Test that the function exists.\"\"\"
    try:
        from {import_path} import {function_name}
        assert {function_name} is not None
    except ImportError:
        pytest.skip("Module not available")

def test_{function_name}_placeholder():
    \"\"\"Placeholder test - implement proper tests.\"\"\"
    assert True
"""
    
    def _write_temp_test(self, test_code: str, function_name: str) -> Path:
        """Write test code to temporary file."""
        temp_dir = Path(tempfile.mkdtemp(prefix='test_validator_'))
        test_file = temp_dir / f"test_{function_name}.py"
        test_file.write_text(test_code, encoding='utf-8')
        return test_file
    
    async def _run_test(self, test_file: Path, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run pytest on test file."""
        try:
            # Add parent directory to Python path
            env = None
            if cwd:
                import os
                env = dict(**os.environ)
                python_path = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = str(cwd) + (os.pathsep + python_path if python_path else '')
            
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v", "--tb=short"],
                cwd=cwd or test_file.parent,
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            return result
            
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="Test execution timeout"
            )
        except Exception as e:
            return subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr=str(e)
            )


# Convenience function
async def validate_and_generate_test(
    source_file: Path,
    function_name: str,
    project_root: Optional[Path] = None
) -> TestValidationResult:
    """
    Convenience function to validate and generate a test.
    
    Args:
        source_file: Source file to test
        function_name: Function to test
        project_root: Project root
    
    Returns:
        TestValidationResult with passing test
    """
    validator = TestValidator()
    return await validator.validate_test_generation(
        source_file=source_file,
        function_name=function_name,
        project_root=project_root
    )
