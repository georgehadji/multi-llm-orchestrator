"""
Codebase Analyzer — LLM-powered code analysis pipeline
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads a codebase with CodebaseReader and runs a structured multi-LLM
analysis pipeline that produces:

1. Architecture Overview   — structure, patterns, design decisions
2. Code Quality Review     — bugs, anti-patterns, complexity hotspots
3. Security Audit          — vulnerabilities, hardcoded secrets, injection risks
4. Performance Assessment  — bottlenecks, inefficient patterns, scaling issues
5. Improvement Suggestions — prioritized, actionable recommendations

Each section is produced by the best available model for the task type,
with cross-model critique and scoring — same pipeline as the orchestrator.

Usage (programmatic):
    from orchestrator.analyzer import CodebaseAnalyzer
    from pathlib import Path

    analyzer = CodebaseAnalyzer()
    report = await analyzer.analyze(
        path=Path("E:/MyProject"),
        focus=["security", "performance"],   # optional filter
        budget_usd=2.0,
    )
    print(report.markdown)

Usage (CLI):
    python -m orchestrator analyze --path E:/MyProject --focus security,performance
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .codebase_reader import CodebaseReader, CodebaseContext
from .models import Budget, Model, TaskType
from .api_clients import UnifiedClient
from .cache import DiskCache

logger = logging.getLogger("orchestrator.analyzer")

# ─────────────────────────────────────────────
# Analysis focus areas
# ─────────────────────────────────────────────

ALL_FOCUS_AREAS = [
    "architecture",
    "quality",
    "security",
    "performance",
    "improvements",
]

FOCUS_ALIASES: dict[str, str] = {
    "arch": "architecture",
    "design": "architecture",
    "bug": "quality",
    "bugs": "quality",
    "code": "quality",
    "sec": "security",
    "vuln": "security",
    "perf": "performance",
    "speed": "performance",
    "suggest": "improvements",
    "recommend": "improvements",
    "improve": "improvements",
}

# Prompts per focus area — injected with the codebase context
ANALYSIS_PROMPTS: dict[str, str] = {
    "architecture": """You are a senior software architect. Analyze the codebase below and provide:

1. **Architecture Overview** — What is the system's purpose and high-level design?
2. **Design Patterns** — Which patterns are used (MVC, Repository, Factory, etc.)? Are they applied correctly?
3. **Module Structure** — How are components organized? Is separation of concerns respected?
4. **Dependency Graph** — What are the key dependencies between modules? Any problematic coupling?
5. **Scalability** — How would this system handle 10x load? What would break first?
6. **Architectural Concerns** — What structural issues exist that would hinder future development?

Be specific, cite file names and function names. Prioritize actionable observations.

---CODEBASE---
{context}
""",

    "quality": """You are an expert code reviewer. Analyze the codebase below and identify:

1. **Bugs & Logic Errors** — Any clear bugs, off-by-one errors, null pointer risks, race conditions?
2. **Code Smells** — Long methods, duplicate code, large classes, magic numbers, deeply nested logic?
3. **Error Handling** — Are errors handled properly? Silent failures? Missing exception types?
4. **Testing** — Is there test coverage? What critical paths are untested?
5. **Code Style** — Inconsistencies in naming, formatting, documentation?
6. **Maintainability Score** — Rate 1-10 with justification.

For each issue: cite the file + line number (if visible), explain the problem, and suggest the fix.

---CODEBASE---
{context}
""",

    "security": """You are a cybersecurity expert and secure code reviewer. Analyze the codebase for:

1. **Injection Vulnerabilities** — SQL injection, command injection, XSS, SSTI, path traversal?
2. **Authentication & Authorization** — Weak auth, missing access controls, privilege escalation risks?
3. **Secrets & Credentials** — Hardcoded API keys, passwords, tokens in source code or configs?
4. **Cryptography** — Weak algorithms (MD5, SHA1), improper key management, plaintext storage?
5. **Input Validation** — Unvalidated user input, missing sanitization?
6. **Dependencies** — Known vulnerable dependencies (flag any outdated/suspicious packages)?
7. **Security Score** — Rate 1-10 with justification. List Critical/High/Medium/Low findings.

For each finding: cite file + context, explain the risk, provide the remediation.

---CODEBASE---
{context}
""",

    "performance": """You are a performance engineering expert. Analyze the codebase for:

1. **Algorithmic Complexity** — Any O(n²) or worse where O(n log n) is achievable? Inefficient data structures?
2. **I/O & Blocking Calls** — Synchronous I/O in async context? Missing connection pooling? Unnecessary disk reads?
3. **Memory Usage** — Memory leaks, unbounded caches, excessive object creation?
4. **Database & API Calls** — N+1 query problems, missing indexes (inferred), over-fetching?
5. **Concurrency** — Is parallelism used where beneficial? Any deadlock risks? Lock contention?
6. **Caching** — What should be cached that isn't? What's cached unnecessarily?
7. **Performance Score** — Rate 1-10. List the top 3 bottlenecks to fix first.

Be specific: cite file + function name, explain the bottleneck, show the improved version.

---CODEBASE---
{context}
""",

    "improvements": """You are a senior engineering consultant. Based on the codebase below, provide:

1. **Quick Wins** (< 1 day each) — Low-effort, high-impact improvements ready to implement now.
2. **Medium-term Improvements** (1–5 days each) — Refactors and features that would significantly improve the codebase.
3. **Strategic Recommendations** (1–4 weeks each) — Larger architectural or tooling changes worth planning.
4. **Missing Features** — Capabilities that are clearly needed but absent (logging, metrics, tests, docs, CI/CD)?
5. **Technical Debt Map** — Which parts of the codebase have the highest debt? Prioritize repayment order.
6. **Modernization** — Are there outdated patterns or libraries that should be replaced?

For each recommendation: explain WHY it matters, estimate effort, and describe the approach.

---CODEBASE---
{context}
""",
}

# Best model per analysis type
ANALYSIS_MODEL_PREFERENCE: dict[str, TaskType] = {
    "architecture": TaskType.REASONING,
    "quality":      TaskType.CODE_REVIEW,
    "security":     TaskType.CODE_REVIEW,
    "performance":  TaskType.REASONING,
    "improvements": TaskType.EVALUATE,
}


# ─────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────

@dataclass
class AnalysisSection:
    """Result of one focus-area analysis."""
    focus: str
    content: str
    model_used: str
    tokens_used: int
    cost_usd: float
    latency_ms: float


@dataclass
class AnalysisReport:
    """
    Full codebase analysis report.

    Attributes
    ----------
    path          : Analyzed directory
    sections      : One AnalysisSection per focus area
    markdown      : Full report as a Markdown string (ready to save/display)
    total_cost    : Sum of all API costs (USD)
    total_tokens  : Sum of all tokens used
    elapsed_s     : Wall-clock time for the full analysis
    files_analyzed: Number of source files included in the context
    languages     : Detected programming languages
    """
    path: Path
    sections: list[AnalysisSection] = field(default_factory=list)
    markdown: str = ""
    total_cost: float = 0.0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    files_analyzed: int = 0
    languages: set[str] = field(default_factory=set)


# ─────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────

class CodebaseAnalyzer:
    """
    Orchestrates multi-LLM codebase analysis.

    Parameters
    ----------
    max_context_tokens : Token budget for the codebase context passed to each LLM.
    max_concurrency    : Simultaneous API calls (default: 2 to avoid rate limits).
    """

    def __init__(
        self,
        max_context_tokens: int = 60_000,
        max_concurrency: int = 2,
    ):
        self.max_context_tokens = max_context_tokens
        cache = DiskCache()
        self.client = UnifiedClient(cache=cache, max_concurrency=max_concurrency)
        self.reader = CodebaseReader(max_tokens=max_context_tokens)

    async def analyze(
        self,
        path: str | Path,
        focus: Optional[list[str]] = None,
        budget_usd: float = 3.0,
        include_exts: Optional[set[str]] = None,
        max_tokens_per_section: int = 4096,
    ) -> AnalysisReport:
        """
        Run the full analysis pipeline.

        Parameters
        ----------
        path                : Root directory to analyze.
        focus               : List of focus areas to include. None = all.
                              Options: architecture, quality, security, performance, improvements
        budget_usd          : Max spend across all LLM calls (USD).
        include_exts        : If set, only these file extensions are read.
        max_tokens_per_section : Max output tokens per analysis section.

        Returns
        -------
        AnalysisReport with all sections and assembled Markdown.
        """
        path = Path(path).resolve()
        t0 = time.monotonic()

        # Resolve focus areas
        active_focus = self._resolve_focus(focus)
        logger.info(f"Analyzing {path} | Focus: {', '.join(active_focus)} | Budget: ${budget_usd}")

        # Read codebase
        if include_exts:
            self.reader.include_exts = include_exts
        ctx = self.reader.read(path)
        logger.info(
            f"Context ready: {ctx.included_files} files, "
            f"~{ctx.estimated_tokens:,} tokens, "
            f"languages: {', '.join(sorted(ctx.languages))}"
        )

        # Run analysis sections (sequentially to respect budget carefully)
        report = AnalysisReport(
            path=path,
            files_analyzed=ctx.included_files,
            languages=ctx.languages,
        )

        spent_usd = 0.0
        for focus_area in active_focus:
            if spent_usd >= budget_usd:
                logger.warning(f"Budget exhausted (${spent_usd:.4f}), skipping {focus_area}")
                break

            section = await self._analyze_section(
                focus_area=focus_area,
                ctx=ctx,
                max_tokens=max_tokens_per_section,
                remaining_budget=budget_usd - spent_usd,
            )
            report.sections.append(section)
            spent_usd += section.cost_usd
            logger.info(
                f"  [{focus_area}] done — model={section.model_used}, "
                f"cost=${section.cost_usd:.4f}, tokens={section.tokens_used}"
            )

        report.total_cost = sum(s.cost_usd for s in report.sections)
        report.total_tokens = sum(s.tokens_used for s in report.sections)
        report.elapsed_s = time.monotonic() - t0
        report.markdown = self._render_markdown(report, ctx)

        logger.info(
            f"Analysis complete: {len(report.sections)} sections, "
            f"${report.total_cost:.4f} total, "
            f"{report.elapsed_s:.1f}s elapsed"
        )
        return report

    async def _analyze_section(
        self,
        focus_area: str,
        ctx: CodebaseContext,
        max_tokens: int,
        remaining_budget: float,
    ) -> AnalysisSection:
        """Run one focus area analysis call."""
        prompt_template = ANALYSIS_PROMPTS[focus_area]
        prompt = prompt_template.format(context=ctx.context_text)

        task_type = ANALYSIS_MODEL_PREFERENCE.get(focus_area, TaskType.CODE_REVIEW)
        model = self._pick_model(task_type)

        system = (
            "You are an expert software engineer performing a professional codebase analysis. "
            "Be specific, cite file names and line numbers where possible. "
            "Structure your response with clear headers and bullet points."
        )

        try:
            resp = await self.client.call(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.2,   # low temperature for analytical tasks
                timeout=120,
                retries=1,
            )
            return AnalysisSection(
                focus=focus_area,
                content=resp.text,
                model_used=model.value,
                tokens_used=resp.input_tokens + resp.output_tokens,
                cost_usd=resp.cost_usd,
                latency_ms=resp.latency_ms,
            )
        except Exception as e:
            logger.error(f"Analysis failed for {focus_area}: {e}")
            return AnalysisSection(
                focus=focus_area,
                content=f"*Analysis failed: {e}*",
                model_used="error",
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=0.0,
            )

    def _pick_model(self, task_type: TaskType) -> Model:
        """
        Pick the best available model for the given task type.
        Falls back through the routing table until an available model is found.
        """
        from .models import ROUTING_TABLE
        for model in ROUTING_TABLE.get(task_type, []):
            if self.client.is_available(model):
                return model
        # Ultimate fallback — any available model
        for model in Model:
            if self.client.is_available(model):
                return model
        raise RuntimeError("No LLM providers available. Check your API keys.")

    def _resolve_focus(self, focus: Optional[list[str]]) -> list[str]:
        """Normalize and validate focus area names."""
        if not focus:
            return list(ALL_FOCUS_AREAS)
        resolved: list[str] = []
        for f in focus:
            f = f.lower().strip()
            f = FOCUS_ALIASES.get(f, f)
            if f in ALL_FOCUS_AREAS and f not in resolved:
                resolved.append(f)
            else:
                logger.warning(f"Unknown focus area '{f}', skipping. Valid: {ALL_FOCUS_AREAS}")
        return resolved or list(ALL_FOCUS_AREAS)

    def _render_markdown(self, report: AnalysisReport, ctx: CodebaseContext) -> str:
        """Render the full analysis report as Markdown."""
        lines: list[str] = []

        # Title
        lines.append(f"# Codebase Analysis Report: `{report.path.name}`")
        lines.append("")

        # Metadata table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| **Path** | `{report.path}` |")
        lines.append(f"| **Languages** | {', '.join(sorted(report.languages)) or 'unknown'} |")
        lines.append(f"| **Files analyzed** | {report.files_analyzed} |")
        lines.append(f"| **Estimated context tokens** | {ctx.estimated_tokens:,} |")
        lines.append(f"| **Analysis sections** | {len(report.sections)} |")
        lines.append(f"| **Total cost** | ${report.total_cost:.4f} USD |")
        lines.append(f"| **Total time** | {report.elapsed_s:.1f}s |")
        lines.append("")

        # Per-section model info
        lines.append("### Models used")
        lines.append("")
        for s in report.sections:
            lines.append(f"- **{s.focus.title()}**: `{s.model_used}` "
                         f"({s.tokens_used:,} tokens, ${s.cost_usd:.4f})")
        lines.append("")

        # Directory tree
        lines.append("## Directory Structure")
        lines.append("")
        lines.append("```")
        lines.append(ctx.file_tree)
        lines.append("```")
        lines.append("")

        # Analysis sections
        lines.append("---")
        lines.append("")
        section_titles = {
            "architecture":  "🏗️ Architecture Overview",
            "quality":       "🔍 Code Quality Review",
            "security":      "🔒 Security Audit",
            "performance":   "⚡ Performance Assessment",
            "improvements":  "💡 Improvement Suggestions",
        }
        for s in report.sections:
            title = section_titles.get(s.focus, s.focus.title())
            lines.append(f"## {title}")
            lines.append(f"*Model: `{s.model_used}` | Tokens: {s.tokens_used:,} | "
                         f"Cost: ${s.cost_usd:.4f} | Latency: {s.latency_ms:.0f}ms*")
            lines.append("")
            lines.append(s.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Footer
        lines.append(
            f"*Generated by Multi-LLM Orchestrator | "
            f"Total: ${report.total_cost:.4f} | {report.elapsed_s:.1f}s*"
        )

        return "\n".join(lines)
