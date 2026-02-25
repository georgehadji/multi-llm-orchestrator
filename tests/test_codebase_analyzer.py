import pytest
import tempfile
from pathlib import Path
from orchestrator.codebase_analyzer import CodebaseAnalyzer, CodebaseMap


class TestCodebaseAnalyzerFileScan:
    """Test basic file scanning capabilities"""

    def test_scan_simple_project_structure(self):
        """Scan a simple project and extract file counts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create test structure
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("# main")
            (root / "src" / "utils.py").write_text("# utils")
            (root / "tests").mkdir()
            (root / "tests" / "test_main.py").write_text("# test")
            (root / "README.md").write_text("# Project")
            (root / "requirements.txt").write_text("fastapi\n")

            analyzer = CodebaseAnalyzer()
            result = analyzer.scan(str(root))

            # Assertions
            assert result.total_files == 5
            assert result.files_by_language[".py"] == 3
            assert result.files_by_language[".md"] == 1
            assert result.files_by_language[".txt"] == 1

    def test_scan_empty_directory(self):
        """Scanning empty directory should return zeros"""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer()
            result = analyzer.scan(tmpdir)

            assert result.total_files == 0
            assert result.files_by_language == {}
            assert result.total_lines_of_code == 0

    def test_count_lines_of_code(self):
        """Count total lines across text files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file1.py").write_text("line1\nline2\nline3\n")
            (root / "file2.py").write_text("line1\n")

            analyzer = CodebaseAnalyzer()
            result = analyzer.scan(str(root))

            assert result.total_lines_of_code == 4
