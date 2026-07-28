"""Codebase analysis package — extracted from flat codebase_*.py files."""

from .analyzer import CodebaseAnalyzer
from .context import (
    CodebaseContext,
    ContextSlicingStrategy,
    DeleteFileSlicingStrategy,
    InstallDepSlicingStrategy,
    ModifyFileSlicingStrategy,
    QualityAnalyzer,
    RelevanceRanker,
    get_slicing_strategy,
)
from .decomposer import CodebaseDecomposer
from .profile import CodebaseProfile
from .reader import ASTIndexer, CodebaseReader, DependencyGraph, FileNode, FileSystemWalker
from .understanding import CodebaseUnderstanding
from .writer import CodebaseWriter

__all__ = [
    "CodebaseAnalyzer",
    "CodebaseContext",
    "CodebaseDecomposer",
    "CodebaseProfile",
    "CodebaseReader",
    "CodebaseUnderstanding",
    "CodebaseWriter",
    "ContextSlicingStrategy",
    "DeleteFileSlicingStrategy",
    "DependencyGraph",
    "FileNode",
    "FileSystemWalker",
    "ASTIndexer",
    "InstallDepSlicingStrategy",
    "ModifyFileSlicingStrategy",
    "QualityAnalyzer",
    "RelevanceRanker",
    "get_slicing_strategy",
]
