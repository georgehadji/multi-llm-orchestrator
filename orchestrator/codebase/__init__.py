"""Codebase analysis package — extracted from flat codebase_*.py files."""

from .analyzer import CodebaseAnalyzer
from .context import CodebaseContext, RelevanceRanker, QualityAnalyzer
from .decomposer import CodebaseDecomposer
from .profile import CodebaseProfile
from .reader import CodebaseReader, FileSystemWalker, ASTIndexer, DependencyGraph
from .understanding import CodebaseUnderstanding
from .writer import CodebaseWriter

__all__ = [
    "CodebaseAnalyzer", "CodebaseContext", "RelevanceRanker", "QualityAnalyzer",
    "CodebaseDecomposer", "CodebaseProfile", "CodebaseReader",
    "FileSystemWalker", "ASTIndexer", "DependencyGraph",
    "CodebaseUnderstanding", "CodebaseWriter",
]
