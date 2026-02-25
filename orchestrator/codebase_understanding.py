"""LLM-powered semantic understanding of codebases"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

from orchestrator.codebase_analyzer import CodebaseAnalyzer
from orchestrator.codebase_profile import CodebaseProfile
from orchestrator.engine import Orchestrator, Budget
from orchestrator.models import TaskType


class CodebaseUnderstanding:
    """Analyze codebase semantically using LLM"""

    def __init__(self, llm_provider: str = "deepseek"):
        self.analyzer = CodebaseAnalyzer()
        self.llm_provider = llm_provider

    async def analyze(self, codebase_path: str) -> CodebaseProfile:
        """
        Analyze codebase and generate understanding profile.

        Args:
            codebase_path: Path to codebase root

        Returns:
            CodebaseProfile with semantic understanding
        """
        # Static analysis first
        codebase_map = self.analyzer.scan(codebase_path)

        # Read key files for LLM context
        key_file_contents = self._read_key_files(codebase_path)

        # Prepare LLM prompt
        prompt = self._build_analysis_prompt(codebase_map, key_file_contents)

        # Call LLM for semantic analysis
        llm_response = await self._call_llm_async(prompt)

        # Create profile from LLM response
        profile = CodebaseProfile(
            purpose=llm_response.get("purpose", "Unknown"),
            primary_patterns=llm_response.get("patterns", []),
            anti_patterns=llm_response.get("anti_patterns", []),
            test_coverage=llm_response.get("test_coverage", "unknown"),
            documentation=llm_response.get("documentation", "unknown"),
            primary_language=codebase_map.primary_language or "unknown",
            project_type=codebase_map.project_type,
        )

        return profile

    def _read_key_files(self, codebase_path: str) -> Dict[str, str]:
        """Read contents of key files for LLM analysis"""
        root = Path(codebase_path)
        key_files = {
            "README.md": root / "README.md",
            "main.py": root / "main.py",
            "app.py": root / "app.py",
            "index.js": root / "index.js",
            "package.json": root / "package.json",
        }

        contents = {}
        for name, path in key_files.items():
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    # Limit to first 500 lines to save tokens
                    lines = text.split("\n")[:500]
                    contents[name] = "\n".join(lines)
                except (UnicodeDecodeError, OSError):
                    pass

        return contents

    def _build_analysis_prompt(
        self, codebase_map, key_file_contents: Dict[str, str]
    ) -> str:
        """Build prompt for LLM analysis"""

        file_section = "\n\n".join([
            f"### {name}\n```\n{content}\n```"
            for name, content in key_file_contents.items()
        ])

        prompt = f"""Analyze this codebase and provide semantic understanding:

## Static Analysis
- Total files: {codebase_map.total_files}
- Total LOC: {codebase_map.total_lines_of_code}
- Languages: {codebase_map.primary_language}
- Project type: {codebase_map.project_type}
- Has tests: {codebase_map.has_tests}
- Has docs: {codebase_map.has_docs}

## Key Files
{file_section}

Based on this, provide:
1. A 1-2 sentence description of what this project does (purpose)
2. List 3-5 architectural patterns you observe
3. List any anti-patterns or code issues you detect
4. Estimate test coverage (low/moderate/good/excellent)
5. Estimate documentation quality (minimal/good/excellent)

Return as JSON:
{{
    "purpose": "...",
    "patterns": ["pattern1", "pattern2"],
    "anti_patterns": ["issue1", "issue2"],
    "test_coverage": "low|moderate|good|excellent",
    "documentation": "minimal|good|excellent"
}}
"""
        return prompt

    async def _call_llm_async(self, prompt: str) -> Dict[str, Any]:
        """
        Call DeepSeek Reasoner for semantic analysis.
        Uses orchestrator to route the analysis task.
        """
        try:
            # Create minimal orchestrator for analysis task
            orch = Orchestrator(budget=Budget(max_usd=1.0))

            # Run analysis as a reasoning task
            result = await orch.run_task(
                task_type=TaskType.REASONING,
                task_description=prompt,
                expected_output_format="json",
            )

            # Parse JSON response
            if result.score >= 0.75:
                try:
                    response_text = result.output
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Fallback if LLM call fails
            return self._default_analysis()

        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
            return self._default_analysis()

    def _default_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        return {
            "purpose": "Project (analysis unavailable)",
            "patterns": [],
            "anti_patterns": [],
            "test_coverage": "unknown",
            "documentation": "unknown",
        }
