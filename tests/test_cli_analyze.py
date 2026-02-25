import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner


class TestCLIAnalyzeCommand:
    """Test --analyze-codebase CLI command"""

    def test_analyze_codebase_cli(self):
        """CLI should accept --analyze-codebase flag"""
        from orchestrator.cli import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("# python")

            runner = CliRunner()
            result = runner.invoke(cli, ['--analyze-codebase', str(root)])

            assert result.exit_code == 0
            assert "CodebaseProfile" in result.output or "Purpose" in result.output
