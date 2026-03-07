"""
Adaptive Prompt Template System
===============================

Self-improving prompt templates with A/B testing and EMA-based convergence.
Automatically discovers optimal prompt variants per (model, task_type, context).

Key Features:
- Template variant registry with versioning
- Automatic A/B testing with statistical significance
- EMA-based score tracking for convergence
- Context-aware template selection
- Exploration vs exploitation balance

Usage:
    from orchestrator.adaptive_templates import AdaptiveTemplateSystem
    
    ats = AdaptiveTemplateSystem()
    
    # Register template variants
    ats.register_template(
        task_type=TaskType.CODE_GEN,
        variants=[
            TemplateVariant(name="concise", template="Write {lang} code: {task}"),
            TemplateVariant(name="structured", template=STRUCTURED_TEMPLATE),
        ]
    )
    
    # Get best template (automatically handles exploration)
    template = await ats.select_template(
        task_type=TaskType.CODE_GEN,
        model=Model.DEEPSEEK_CHAT,
        context={"language": "python", "complexity": "high"}
    )
"""

from __future__ import annotations

import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import statistics

from .log_config import get_logger
from .models import Model, TaskType

logger = get_logger(__name__)


class TemplateStyle(Enum):
    """Template style categories."""
    CONCISE = "concise"
    STRUCTURED = "structured"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    XML_TAGGED = "xml_tagged"


@dataclass
class TemplateVariant:
    """A single template variant."""
    name: str
    template: str
    style: TemplateStyle = TemplateStyle.STRUCTURED
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return self.template
    
    def get_hash(self) -> str:
        """Get hash of template for versioning."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]


@dataclass
class TemplatePerformance:
    """Performance tracking for a template variant."""
    variant_name: str
    task_type: TaskType
    model: Model
    
    # EMA tracking
    ema_score: float = 0.5
    ema_alpha: float = 0.1
    
    # Statistics
    total_uses: int = 0
    total_successes: int = 0
    total_failures: int = 0
    
    # Score history (last 100)
    scores: List[float] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0
    
    # Timestamps
    first_used: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    
    def update_score(self, score: float, success: bool = True) -> None:
        """Update EMA score with new result."""
        # Update EMA
        self.ema_score = (1 - self.ema_alpha) * self.ema_score + self.ema_alpha * score
        
        # Update counters
        self.total_uses += 1
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
        
        # Update history
        self.scores.append(score)
        if len(self.scores) > 100:
            self.scores.pop(0)
        
        # Update confidence (more uses = higher confidence, max at 50 uses)
        self.confidence = min(1.0, self.total_uses / 50)
        self.last_used = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.5
        return self.total_successes / self.total_uses
    
    @property
    def score_variance(self) -> float:
        """Calculate score variance for uncertainty quantification."""
        if len(self.scores) < 2:
            return 1.0
        return statistics.variance(self.scores)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "task_type": self.task_type.value,
            "model": self.model.value,
            "ema_score": self.ema_score,
            "total_uses": self.total_uses,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "first_used": self.first_used.isoformat(),
            "last_used": self.last_used.isoformat(),
        }


@dataclass
class ContextProfile:
    """Profile of a context for similarity matching."""
    language: Optional[str] = None
    framework: Optional[str] = None
    complexity: str = "medium"  # low, medium, high
    domain: Optional[str] = None  # web, ml, data, etc.
    code_size: str = "medium"  # small, medium, large
    
    def to_key(self) -> str:
        """Convert to cache key."""
        parts = [
            self.language or "any",
            self.framework or "any",
            self.complexity,
            self.domain or "any",
            self.code_size,
        ]
        return "|".join(parts)
    
    def similarity(self, other: ContextProfile) -> float:
        """Calculate similarity to another context."""
        score = 0.0
        weights = {
            "language": 0.3,
            "framework": 0.25,
            "complexity": 0.2,
            "domain": 0.15,
            "code_size": 0.1,
        }
        
        if self.language and other.language:
            score += weights["language"] if self.language == other.language else 0
        
        if self.framework and other.framework:
            score += weights["framework"] if self.framework == other.framework else 0
        
        if self.complexity == other.complexity:
            score += weights["complexity"]
        elif abs(["low", "medium", "high"].index(self.complexity) - 
                 ["low", "medium", "high"].index(other.complexity)) == 1:
            score += weights["complexity"] * 0.5
        
        if self.domain and other.domain:
            score += weights["domain"] if self.domain == other.domain else 0
        
        if self.code_size == other.code_size:
            score += weights["code_size"]
        
        return score


class AdaptiveTemplateSystem:
    """
    Self-improving prompt template system.
    
    Optimized for:
    - Fast template selection (cached performance data)
    - Automatic A/B testing (epsilon-greedy exploration)
    - Context-aware recommendations (similarity matching)
    - Statistical significance (confidence intervals)
    """
    
    # Configuration
    EXPLORATION_RATE = 0.15  # 15% exploration
    MIN_SAMPLES_FOR_CONFIDENCE = 10
    CONVERGENCE_THRESHOLD = 0.95  # Stop exploring after this confidence
    MAX_VARIANTS_PER_TASK = 6
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".adaptive_templates")
        self.storage_path.mkdir(exist_ok=True)
        
        # Template registry: (task_type, variant_name) -> TemplateVariant
        self._templates: Dict[Tuple[TaskType, str], TemplateVariant] = {}
        
        # Performance tracking: (task_type, model, variant_name) -> TemplatePerformance
        self._performance: Dict[Tuple[TaskType, Model, str], TemplatePerformance] = {}
        
        # Context-based performance: (task_type, context_key, variant_name) -> score
        self._context_performance: Dict[Tuple[TaskType, str, str], float] = {}
        
        # Current A/B test assignments
        self._ab_test_assignments: Dict[str, str] = {}
        
        # Load existing data
        self._load_data()
        
        # Initialize default templates
        self._init_default_templates()
    
    def _load_data(self) -> None:
        """Load performance data from disk."""
        perf_file = self.storage_path / "performance.json"
        if perf_file.exists():
            try:
                data = json.loads(perf_file.read_text())
                for key, perf_data in data.items():
                    parts = key.split(":")
                    if len(parts) == 3:
                        task_type = TaskType(parts[0])
                        model = Model(parts[1])
                        variant_name = parts[2]
                        
                        perf = TemplatePerformance(
                            variant_name=variant_name,
                            task_type=task_type,
                            model=model,
                            ema_score=perf_data.get("ema_score", 0.5),
                            total_uses=perf_data.get("total_uses", 0),
                            total_successes=perf_data.get("total_successes", 0),
                            total_failures=perf_data.get("total_failures", 0),
                            confidence=perf_data.get("confidence", 0),
                        )
                        self._performance[(task_type, model, variant_name)] = perf
                
                logger.info(f"Loaded {len(self._performance)} performance records")
            except Exception as e:
                logger.error(f"Failed to load performance data: {e}")
    
    def _save_data(self) -> None:
        """Save performance data to disk."""
        try:
            data = {}
            for key, perf in self._performance.items():
                data[":".join([k.value for k in key])] = perf.to_dict()
            
            perf_file = self.storage_path / "performance.json"
            perf_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def _init_default_templates(self) -> None:
        """Initialize default template variants."""
        # CODE_GEN templates
        self.register_variants(
            TaskType.CODE_GEN,
            [
                TemplateVariant(
                    name="concise",
                    template="Write {language} code for: {task}",
                    style=TemplateStyle.CONCISE,
                    description="Minimal, direct prompt",
                ),
                TemplateVariant(
                    name="structured",
                    template="""Task: {task}
Language: {language}
Requirements: {criteria}

Provide clean, well-documented code with:
- Type hints where appropriate
- Error handling
- Docstrings for public functions""",
                    style=TemplateStyle.STRUCTURED,
                    description="Structured with clear requirements",
                ),
                TemplateVariant(
                    name="role_based",
                    template="""You are an expert {language} developer.

Task: {task}

Requirements:
{criteria}

Write production-ready code following best practices.""",
                    style=TemplateStyle.ROLE_BASED,
                    description="Role-based prompting",
                ),
                TemplateVariant(
                    name="chain_of_thought",
                    template="""Task: {task}
Language: {language}

Think through this step by step:
1. What are the key requirements?
2. What approach should I take?
3. What edge cases should I handle?

Now write the code:
{criteria}""",
                    style=TemplateStyle.CHAIN_OF_THOUGHT,
                    description="Chain-of-thought reasoning",
                ),
            ]
        )
        
        # CODE_REVIEW templates
        self.register_variants(
            TaskType.CODE_REVIEW,
            [
                TemplateVariant(
                    name="concise",
                    template="Review this {language} code:\n\n```{language}\n{code}\n```",
                    style=TemplateStyle.CONCISE,
                ),
                TemplateVariant(
                    name="structured",
                    template="""Review the following {language} code for:
1. Bugs and errors
2. Performance issues
3. Security concerns
4. Code style improvements

```{language}
{code}
```

Provide specific line-by-line feedback.""",
                    style=TemplateStyle.STRUCTURED,
                ),
            ]
        )
        
        # REASONING templates
        self.register_variants(
            TaskType.REASONING,
            [
                TemplateVariant(
                    name="structured",
                    template="""Analyze the following problem:

{task}

Consider:
- Technical constraints
- Trade-offs between approaches
- Scalability implications

Provide a well-reasoned recommendation.""",
                    style=TemplateStyle.STRUCTURED,
                ),
                TemplateVariant(
                    name="chain_of_thought",
                    template="""Problem: {task}

Let's think through this systematically:

Step 1: Understand the requirements
Step 2: Identify constraints
Step 3: Evaluate options
Step 4: Make recommendation

Provide your analysis.""",
                    style=TemplateStyle.CHAIN_OF_THOUGHT,
                ),
            ]
        )
    
    def register_variants(
        self,
        task_type: TaskType,
        variants: List[TemplateVariant],
    ) -> None:
        """Register template variants for a task type."""
        for variant in variants:
            key = (task_type, variant.name)
            self._templates[key] = variant
            logger.debug(f"Registered template: {task_type.value}/{variant.name}")
    
    def register_variant(
        self,
        task_type: TaskType,
        variant: TemplateVariant,
    ) -> None:
        """Register a single template variant."""
        self._templates[(task_type, variant.name)] = variant
    
    async def select_template(
        self,
        task_type: TaskType,
        model: Model,
        context: Optional[Dict[str, Any]] = None,
        force_exploration: bool = False,
    ) -> Tuple[TemplateVariant, Dict[str, Any]]:
        """
        Select the best template variant using epsilon-greedy strategy.
        
        Returns:
            (selected_variant, metadata)
        """
        context = context or {}
        context_profile = self._build_context_profile(context)
        
        # Get available variants
        variants = self._get_variants_for_task(task_type)
        if not variants:
            logger.warning(f"No templates registered for {task_type.value}")
            # Return a default
            return (
                TemplateVariant(name="default", template="{task}"),
                {"strategy": "fallback", "reason": "no_variants"}
            )
        
        # Check for exploration
        if force_exploration or random.random() < self.EXPLORATION_RATE:
            explore_variant = self._select_exploration_variant(
                variants, task_type, model, context_profile
            )
            if explore_variant:
                return (
                    explore_variant,
                    {
                        "strategy": "exploration",
                        "variant": explore_variant.name,
                        "reason": "epsilon_greedy",
                    }
                )
        
        # Exploitation: select best performing variant
        best_variant, selection_metadata = self._select_best_variant(
            variants, task_type, model, context_profile
        )
        
        return best_variant, selection_metadata
    
    def _get_variants_for_task(self, task_type: TaskType) -> List[TemplateVariant]:
        """Get all variants for a task type."""
        return [
            variant for (tt, _), variant in self._templates.items()
            if tt == task_type
        ]
    
    def _build_context_profile(self, context: Dict[str, Any]) -> ContextProfile:
        """Build context profile from context dict."""
        return ContextProfile(
            language=context.get("language"),
            framework=context.get("framework"),
            complexity=context.get("complexity", "medium"),
            domain=context.get("domain"),
            code_size=context.get("code_size", "medium"),
        )
    
    def _select_exploration_variant(
        self,
        variants: List[TemplateVariant],
        task_type: TaskType,
        model: Model,
        context_profile: ContextProfile,
    ) -> Optional[TemplateVariant]:
        """Select a variant for exploration."""
        # Find under-sampled variants
        candidates = []
        for variant in variants:
            perf = self._performance.get((task_type, model, variant.name))
            if perf is None or perf.total_uses < self.MIN_SAMPLES_FOR_CONFIDENCE:
                # Weight by inverse sample size
                uses = perf.total_uses if perf else 0
                candidates.append((variant, 1.0 / (1 + uses)))
        
        if not candidates:
            return None
        
        # Weighted random selection
        total_weight = sum(w for _, w in candidates)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for variant, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                return variant
        
        return candidates[-1][0]
    
    def _select_best_variant(
        self,
        variants: List[TemplateVariant],
        task_type: TaskType,
        model: Model,
        context_profile: ContextProfile,
    ) -> Tuple[TemplateVariant, Dict[str, Any]]:
        """Select the best variant based on performance data."""
        scores = []
        
        for variant in variants:
            score = self._calculate_variant_score(
                variant, task_type, model, context_profile
            )
            scores.append((variant, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1]["composite"], reverse=True)
        
        best_variant, best_score = scores[0]
        
        # Build metadata
        all_scores = {
            variant.name: {
                "composite": score["composite"],
                "ema": score["ema"],
                "context": score["context"],
                "confidence": score["confidence"],
            }
            for variant, score in scores
        }
        
        metadata = {
            "strategy": "exploitation",
            "variant": best_variant.name,
            "composite_score": best_score["composite"],
            "ema_score": best_score["ema"],
            "confidence": best_score["confidence"],
            "all_scores": all_scores,
        }
        
        return best_variant, metadata
    
    def _calculate_variant_score(
        self,
        variant: TemplateVariant,
        task_type: TaskType,
        model: Model,
        context_profile: ContextProfile,
    ) -> Dict[str, float]:
        """Calculate composite score for a variant."""
        # Get performance record
        perf = self._performance.get((task_type, model, variant.name))
        
        if perf is None:
            # No data yet - neutral score with low confidence
            return {
                "composite": 0.5,
                "ema": 0.5,
                "context": 0.5,
                "confidence": 0.0,
                "samples": 0,
            }
        
        # Base EMA score
        ema_score = perf.ema_score
        
        # Context-adjusted score
        context_key = context_profile.to_key()
        context_score = self._context_performance.get(
            (task_type, context_key, variant.name),
            ema_score  # Fall back to global EMA
        )
        
        # Composite score with confidence weighting
        confidence = perf.confidence
        composite = (
            0.6 * ema_score +
            0.3 * context_score +
            0.1 * perf.success_rate
        ) * (0.5 + 0.5 * confidence)  # Penalize low confidence
        
        return {
            "composite": composite,
            "ema": ema_score,
            "context": context_score,
            "confidence": confidence,
            "samples": perf.total_uses,
        }
    
    async def report_result(
        self,
        task_type: TaskType,
        model: Model,
        variant_name: str,
        score: float,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Report the result of using a template variant.
        
        This updates the EMA scores and triggers re-evaluation.
        """
        # Update global performance
        key = (task_type, model, variant_name)
        if key not in self._performance:
            self._performance[key] = TemplatePerformance(
                variant_name=variant_name,
                task_type=task_type,
                model=model,
            )
        
        self._performance[key].update_score(score, success)
        
        # Update context-specific performance
        if context:
            context_profile = self._build_context_profile(context)
            context_key = context_profile.to_key()
            
            # EMA update for context
            current = self._context_performance.get((task_type, context_key, variant_name), 0.5)
            self._context_performance[(task_type, context_key, variant_name)] = (
                0.9 * current + 0.1 * score
            )
        
        # Persist
        self._save_data()
        
        logger.debug(
            f"Updated {task_type.value}/{model.value}/{variant_name}: "
            f"score={score:.3f}, ema={self._performance[key].ema_score:.3f}"
        )
    
    def get_template_stats(
        self,
        task_type: Optional[TaskType] = None,
        model: Optional[Model] = None,
    ) -> Dict[str, Any]:
        """Get statistics about template performance."""
        stats = {
            "total_variants": len(self._templates),
            "total_performance_records": len(self._performance),
            "by_task_type": defaultdict(int),
            "by_model": defaultdict(int),
            "top_performers": [],
        }
        
        # Count by task type and model
        for (tt, m, variant), perf in self._performance.items():
            if task_type is None or tt == task_type:
                if model is None or m == model:
                    stats["by_task_type"][tt.value] += 1
                    stats["by_model"][m.value] += 1
        
        # Find top performers
        performers = [
            {
                "task_type": key[0].value,
                "model": key[1].value,
                "variant": key[2],
                "ema_score": perf.ema_score,
                "total_uses": perf.total_uses,
                "confidence": perf.confidence,
            }
            for key, perf in self._performance.items()
            if (task_type is None or key[0] == task_type) and
               (model is None or key[1] == model)
        ]
        
        performers.sort(key=lambda x: x["ema_score"], reverse=True)
        stats["top_performers"] = performers[:10]
        
        # Convert defaultdicts to regular dicts
        stats["by_task_type"] = dict(stats["by_task_type"])
        stats["by_model"] = dict(stats["by_model"])
        
        return stats
    
    def get_best_template_for_context(
        self,
        task_type: TaskType,
        context: Dict[str, Any],
    ) -> Optional[TemplateVariant]:
        """Get the best template for a specific context (for analysis)."""
        context_profile = self._build_context_profile(context)
        variants = self._get_variants_for_task(task_type)
        
        if not variants:
            return None
        
        best_variant = None
        best_score = -1
        
        for variant in variants:
            score = self._calculate_variant_score(
                variant, task_type, Model.DEEPSEEK_CHAT, context_profile
            )
            if score["composite"] > best_score:
                best_score = score["composite"]
                best_variant = variant
        
        return best_variant


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-built Template Libraries
# ═══════════════════════════════════════════════════════════════════════════════

# Python-specific templates
PYTHON_TEMPLATES = [
    TemplateVariant(
        name="python_type_hints",
        template="""Write Python code with type hints for:

Task: {task}

Requirements:
- Use type hints for all function signatures
- Use dataclasses or TypedDict where appropriate
- Handle edge cases with Optional and Union types
- Follow PEP 484 style

{criteria}""",
        style=TemplateStyle.STRUCTURED,
    ),
    TemplateVariant(
        name="python_async",
        template="""Write async Python code for:

Task: {task}

Requirements:
- Use async/await patterns
- Handle concurrent operations properly
- Include proper error handling
- Use asyncio primitives where appropriate

{criteria}""",
        style=TemplateStyle.STRUCTURED,
    ),
]

# Web development templates
WEB_TEMPLATES = [
    TemplateVariant(
        name="react_component",
        template="""Create a React component:

Component: {task}

Requirements:
- Use TypeScript
- Follow React hooks patterns
- Include PropTypes or TypeScript interfaces
- Handle loading and error states
- Make it accessible (ARIA labels)

{criteria}""",
        style=TemplateStyle.STRUCTURED,
    ),
    TemplateVariant(
        name="api_endpoint",
        template="""Create a REST API endpoint:

Endpoint: {task}

Requirements:
- Include proper HTTP status codes
- Validate input parameters
- Return consistent error responses
- Include rate limiting considerations
- Document with OpenAPI comments

{criteria}""",
        style=TemplateStyle.STRUCTURED,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_ats: Optional[AdaptiveTemplateSystem] = None


def get_adaptive_template_system() -> AdaptiveTemplateSystem:
    """Get global adaptive template system."""
    global _ats
    if _ats is None:
        _ats = AdaptiveTemplateSystem()
    return _ats


def reset_adaptive_template_system() -> None:
    """Reset global adaptive template system (for testing)."""
    global _ats
    _ats = None
