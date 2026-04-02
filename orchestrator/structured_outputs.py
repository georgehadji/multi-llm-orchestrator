"""
Structured LLM Outputs with Instructor
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Provides type-safe structured outputs for LLM responses using Instructor + Pydantic.
Automatically validates, retries on failure, and ensures schema compliance.

Usage:
    from orchestrator.structured_outputs import TaskDecomposer, StructuredResponse

    decomposer = TaskDecomposer()
    result = await decomposer.decompose(project_desc, criteria)
    # result is validated TaskDecomposition object, no JSON parsing needed
"""

from __future__ import annotations

import asyncio
import warnings

# instructor 1.14.x imports google.generativeai (deprecated); suppress until
# instructor ships a release targeting google.genai.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import instructor
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime, timezone

# Import orchestrator models for type compatibility
from .models import TaskType, Task

# ═══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ═══════════════════════════════════════════════════════════════════


class TaskInput(BaseModel):
    """Input task structure for validation"""

    id: str = Field(description="Unique task identifier (e.g., task_001)")
    type: Literal[
        "code_generation",
        "code_review",
        "complex_reasoning",
        "creative_writing",
        "data_extraction",
        "evaluation",
    ]
    prompt: str = Field(description="Task prompt/instructions")
    context: str = Field(default="", description="Additional context")
    dependencies: list[str] = Field(
        default_factory=list, description="List of task IDs this task depends on"
    )
    acceptance_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum acceptance score"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task ID cannot be empty")
        return v.strip()

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task prompt cannot be empty")
        return v.strip()

    @field_validator("acceptance_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        # Ensure reasonable threshold
        return max(0.5, min(0.95, v))

    def to_task(self) -> Task:
        """Convert to orchestrator Task object"""
        return Task(
            id=self.id,
            type=TaskType(self.type),
            prompt=self.prompt,
            context=self.context,
            dependencies=self.dependencies,
            acceptance_threshold=self.acceptance_threshold,
        )


class TaskDecomposition(BaseModel):
    """Structured decomposition output with validation"""

    tasks: list[TaskInput] = Field(min_length=1, description="List of decomposed tasks")
    execution_order: list[str] = Field(description="Ordered list of task IDs for execution")

    @field_validator("execution_order")
    @classmethod
    def validate_execution_order(cls, v: list[str], info) -> list[str]:
        if not v:
            raise ValueError("Execution order cannot be empty")
        return v

    def validate_consistency(self) -> bool:
        """Validate that execution_order matches task IDs"""
        task_ids = {task.id for task in self.tasks}
        order_ids = set(self.execution_order)

        # All tasks must be in execution order
        if task_ids != order_ids:
            missing = task_ids - order_ids
            extra = order_ids - task_ids
            if missing:
                raise ValueError(f"Tasks missing from execution order: {missing}")
            if extra:
                raise ValueError(f"Extra IDs in execution order: {extra}")

        return True

    def to_tasks(self) -> list[Task]:
        """Convert all tasks to orchestrator Task objects"""
        self.validate_consistency()
        return [task.to_task() for task in self.tasks]


class CodeReview(BaseModel):
    """Structured code review output"""

    score: float = Field(ge=0.0, le=1.0, description="Overall code quality score (0-1)")
    issues: list[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: list[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    critical_issues: list[str] = Field(
        default_factory=list, description="Critical issues that must be fixed"
    )
    passed: bool = Field(description="Whether code passes review (score >= threshold)")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ArchitectureDecision(BaseModel):
    """Structured architecture decision output"""

    architecture: Literal[
        "monolithic", "modular_monolith", "microservices", "serverless", "jamstack", "headless_cms"
    ]
    paradigm: Literal[
        "procedural", "object_oriented", "functional", "event_driven", "reactive", "declarative"
    ]
    api_style: Literal["rest", "graphql", "grpc", "websocket", "rpc"]
    database: Literal["relational", "document", "key_value", "graph", "time_series", "none"]
    technology_stack: dict[str, list[str]] = Field(
        description="Technology choices by category (e.g., {'frontend': ['React'], 'backend': ['FastAPI']})"
    )
    key_constraints: list[str] = Field(
        default_factory=list, description="Key architectural constraints"
    )
    recommended_patterns: list[str] = Field(
        default_factory=list, description="Recommended design patterns"
    )
    rationale: str = Field(description="Rationale for architectural decisions")


# ═══════════════════════════════════════════════════════════════════
# INSTRUCTOR CLIENTS WITH RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════


class StructuredClient:
    """Base class for Instructor clients with retry logic"""

    def __init__(self, api_client=None):
        """
        Initialize structured client.

        Args:
            api_client: Existing API client to wrap (optional)
        """
        self._client = None
        self._api_client = api_client
        self._mode = instructor.Mode.MD_JSON  # MD_JSON handles markdown-fenced responses

    def get_client(self, model: str):
        """
        Get Instructor client for specific model.

        Args:
            model: Model ID (e.g., "qwen/qwen3-coder-30b-a3b-instruct:free")

        Returns:
            Instructor client configured for the model
        """
        # Determine provider from model ID
        provider = model.split("/")[0] if "/" in model else "openai"

        # Create Instructor client based on provider
        if self._api_client:
            # UnifiedClient wraps OpenRouter internally; build a direct AsyncOpenAI
            # for instructor since UnifiedClient doesn't expose a .client attribute.
            import openai

            self._client = instructor.from_openai(
                openai.AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self._get_api_key(),
                ),
                mode=self._mode,
            )
        else:
            # Create new client from environment
            if provider in ["anthropic", "claude"]:
                import anthropic

                self._client = instructor.from_anthropic(anthropic.Anthropic(), mode=self._mode)
            elif provider in ["google", "gemini"]:
                import google.generativeai as genai

                self._client = instructor.from_gemini(genai, mode=self._mode)
            else:
                # Default to OpenAI-compatible (includes OpenRouter)
                import openai

                self._client = instructor.from_openai(
                    openai.AsyncOpenAI(
                        base_url="https://openrouter.ai/api/v1", api_key=self._get_api_key()
                    ),
                    mode=self._mode,
                )

        return self._client

    def _get_api_key(self) -> str:
        """Get API key from environment"""
        import os

        return os.getenv("OPENROUTER_API_KEY", "")


class TaskDecomposer(StructuredClient):
    """
    Task decomposition with structured outputs.

    Usage:
        decomposer = TaskDecomposer()
        result = await decomposer.decompose(project_desc, criteria)
        tasks = result.to_tasks()
        execution_order = result.execution_order
    """

    async def decompose(
        self,
        project_description: str,
        success_criteria: str,
        model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        max_retries: int = 3,
    ) -> TaskDecomposition:
        """
        Decompose project into structured tasks.

        Args:
            project_description: Project description
            success_criteria: Success criteria
            model: Model to use (default: FREE Nemotron for SEO/agents)
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            Validated TaskDecomposition object
        """
        client = self.get_client(model)

        # Create decomposition prompt
        prompt = self._create_decomposition_prompt(project_description, success_criteria)

        # Call with Instructor (automatic validation + retries)
        result = await client.chat.completions.create(
            model=model,
            response_model=TaskDecomposition,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software architect. Decompose projects into clear, actionable tasks with proper dependencies. Output MUST be valid JSON matching the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            max_retries=max_retries,
        )

        # Result is already validated TaskDecomposition object
        return result

    def _create_decomposition_prompt(self, project_desc: str, criteria: str) -> str:
        """Create structured decomposition prompt"""
        return f"""
PROJECT DESCRIPTION:
{project_desc}

SUCCESS CRITERIA:
{criteria}

TASK: Decompose this project into clear, actionable tasks.

REQUIREMENTS:
1. Each task must have a unique ID (format: task_001, task_002, etc.)
2. Task types must be one of: code_generation, code_review, complex_reasoning, creative_writing, data_extraction, evaluation
3. Dependencies must reference other task IDs
4. Acceptance threshold between 0.5 and 0.95 (default: 0.85)
5. Execution order must include ALL task IDs exactly once

OUTPUT FORMAT:
Return a JSON object with:
- tasks: Array of task objects
- execution_order: Array of task IDs in execution order

EXAMPLE:
{{
  "tasks": [
    {{
      "id": "task_001",
      "type": "code_generation",
      "prompt": "Create project structure...",
      "dependencies": [],
      "acceptance_threshold": 0.85
    }}
  ],
  "execution_order": ["task_001"]
}}

DECOMPOSE THE PROJECT NOW:
"""


class CodeReviewer(StructuredClient):
    """
    Code review with structured outputs.

    Usage:
        reviewer = CodeReviewer()
        review = await reviewer.review(code, criteria)
        if review.passed:
            # Code passed review
        else:
            # Fix critical_issues first
    """

    async def review(
        self,
        code: str,
        criteria: list[str],
        model: str = "x-ai/grok-4.1-fast",
        threshold: float = 0.80,
        max_retries: int = 2,
    ) -> CodeReview:
        """
        Review code with structured output.

        Args:
            code: Code to review
            criteria: Review criteria
            model: Model to use
            threshold: Passing threshold
            max_retries: Maximum retry attempts

        Returns:
            Validated CodeReview object
        """
        client = self.get_client(model)

        prompt = self._create_review_prompt(code, criteria)

        result = await client.chat.completions.create(
            model=model,
            response_model=CodeReview,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a senior code reviewer. Review code against criteria. Score objectively (0-1). List specific issues and suggestions. Code passes if score >= {threshold}.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            max_retries=max_retries,
        )

        # Set passed based on threshold
        result.passed = result.score >= threshold

        return result

    def _create_review_prompt(self, code: str, criteria: list[str]) -> str:
        """Create review prompt"""
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        return f"""
CODE TO REVIEW:
```
{code}
```

REVIEW CRITERIA:
{criteria_text}

TASK: Review this code against the criteria.

OUTPUT:
- score: Overall quality (0-1)
- issues: Specific issues found
- suggestions: Improvement suggestions
- critical_issues: Must-fix issues (security, bugs, etc.)
- passed: true if score >= threshold
"""


class ArchitecturePlanner(StructuredClient):
    """
    Architecture planning with structured outputs.

    Usage:
        planner = ArchitecturePlanner()
        decision = await planner.plan(project_desc, requirements)
        # decision.architecture, decision.paradigm, etc.
    """

    async def plan(
        self,
        project_description: str,
        requirements: list[str],
        model: str = "xiaomi/mimo-v2-pro",
        max_retries: int = 2,
    ) -> ArchitectureDecision:
        """
        Create architecture decision with structured output.

        Args:
            project_description: Project description
            requirements: Technical requirements
            model: Model to use
            max_retries: Maximum retry attempts

        Returns:
            Validated ArchitectureDecision object
        """
        client = self.get_client(model)

        prompt = self._create_planning_prompt(project_description, requirements)

        result = await client.chat.completions.create(
            model=model,
            response_model=ArchitectureDecision,
            messages=[
                {
                    "role": "system",
                    "content": "You are a principal software architect. Choose optimal architecture, paradigm, and technology stack based on project requirements. Provide clear rationale.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            max_retries=max_retries,
        )

        return result

    def _create_planning_prompt(self, project_desc: str, requirements: list[str]) -> str:
        """Create planning prompt"""
        req_text = "\n".join(f"- {r}" for r in requirements)

        return f"""
PROJECT:
{project_desc}

REQUIREMENTS:
{req_text}

TASK: Choose optimal architecture and technology stack.

DECISIONS:
1. Architecture: monolithic, modular_monolith, microservices, serverless, jamstack, or headless_cms
2. Paradigm: procedural, object_oriented, functional, event_driven, reactive, or declarative
3. API Style: rest, graphql, grpc, websocket, or rpc
4. Database: relational, document, key_value, graph, time_series, or none
5. Technology Stack: Specific technologies for frontend, backend, database, etc.

Provide clear rationale for each decision based on project requirements.
"""


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


async def decompose_project(
    project_desc: str, criteria: str, model: str = "nvidia/nemotron-3-super-120b-a12b:free"
) -> tuple[list[Task], list[str]]:
    """
    Convenience function to decompose project.

    Args:
        project_desc: Project description
        criteria: Success criteria
        model: Model to use (default: FREE Nemotron)

    Returns:
        Tuple of (tasks, execution_order)
    """
    decomposer = TaskDecomposer()
    result = await decomposer.decompose(project_desc, criteria, model)
    return result.to_tasks(), result.execution_order


async def review_code(
    code: str, criteria: list[str], model: str = "x-ai/grok-4.1-fast", threshold: float = 0.80
) -> CodeReview:
    """
    Convenience function to review code.

    Args:
        code: Code to review
        criteria: Review criteria
        model: Model to use
        threshold: Passing threshold

    Returns:
        CodeReview object
    """
    reviewer = CodeReviewer()
    return await reviewer.review(code, criteria, model, threshold)


async def plan_architecture(
    project_desc: str, requirements: list[str], model: str = "xiaomi/mimo-v2-pro"
) -> ArchitectureDecision:
    """
    Convenience function to plan architecture.

    Args:
        project_desc: Project description
        requirements: Technical requirements
        model: Model to use

    Returns:
        ArchitectureDecision object
    """
    planner = ArchitecturePlanner()
    return await planner.plan(project_desc, requirements, model)
