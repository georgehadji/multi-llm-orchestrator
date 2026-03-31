"""
Skills — Claude skills system
============================
Module for implementing the Claude skills system that allows the orchestrator to
perform specific tasks through defined skills.

Pattern: Command
Async: Yes — for I/O-bound operations
Layer: L3 Agents

Usage:
    from orchestrator.skills import SkillManager
    skill_manager = SkillManager()
    skill_manager.register_skill("calculate_sum", calculate_sum, ["numbers"])
    result = await skill_manager.execute_skill("calculate_sum", numbers=[1, 2, 3, 4])
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("orchestrator.skills")


@dataclass
class SkillDefinition:
    """Definition of a skill."""

    name: str
    description: str
    function: Callable
    parameters: list[str]  # Parameter names
    required_params: list[str]  # Required parameter names
    examples: list[dict[str, Any]]  # Example usages


class SkillManager:
    """Manages Claude skills that can be executed by the orchestrator."""

    def __init__(self):
        """Initialize the skill manager."""
        self.skills: dict[str, SkillDefinition] = {}
        self.skill_history: list[dict[str, Any]] = []
        self.max_history = 100

    def register_skill(
        self,
        name: str,
        func: Callable,
        parameters: list[str],
        required_params: list[str] = None,
        description: str = "",
        examples: list[dict[str, Any]] = None,
    ):
        """
        Register a new skill.

        Args:
            name: Name of the skill
            func: Function to execute when skill is called
            parameters: List of parameter names
            required_params: List of required parameter names (defaults to all)
            description: Description of what the skill does
            examples: Example usages of the skill
        """
        if required_params is None:
            required_params = parameters[:]

        if examples is None:
            examples = []

        # Validate function signature matches parameters
        sig = inspect.signature(func)
        func_params = list(sig.parameters.keys())

        # Check if all required parameters are in the function
        for param in required_params:
            if param not in func_params:
                raise ValueError(f"Required parameter '{param}' not in function signature")

        # Check if all provided parameters are in the function
        for param in parameters:
            if param not in func_params:
                raise ValueError(f"Parameter '{param}' not in function signature")

        skill_def = SkillDefinition(
            name=name,
            description=description,
            function=func,
            parameters=parameters,
            required_params=required_params,
            examples=examples,
        )

        self.skills[name] = skill_def
        logger.info(f"Registered skill: {name}")

    def register_async_skill(
        self,
        name: str,
        func: Callable,
        parameters: list[str],
        required_params: list[str] = None,
        description: str = "",
        examples: list[dict[str, Any]] = None,
    ):
        """
        Register a new asynchronous skill.

        Args:
            name: Name of the skill
            func: Async function to execute when skill is called
            parameters: List of parameter names
            required_params: List of required parameter names (defaults to all)
            description: Description of what the skill does
            examples: Example usages of the skill
        """
        # Validate that the function is indeed a coroutine
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Function must be a coroutine function for async skills")

        self.register_skill(name, func, parameters, required_params, description, examples)

    async def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """
        Execute a registered skill with the provided arguments.

        Args:
            skill_name: Name of the skill to execute
            **kwargs: Arguments to pass to the skill

        Returns:
            Result of the skill execution
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not found")

        skill_def = self.skills[skill_name]

        # Validate required parameters
        for param in skill_def.required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter '{param}' for skill '{skill_name}'")

        # Filter kwargs to only include valid parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in skill_def.parameters}

        # Log skill execution
        execution_record = {
            "skill_name": skill_name,
            "arguments": kwargs,
            "timestamp": asyncio.get_event_loop().time(),
            "status": "started",
        }

        try:
            # Execute the skill function
            result = await self._execute_function(skill_def.function, **valid_kwargs)

            execution_record["status"] = "completed"
            execution_record["result"] = result

            logger.info(f"Executed skill '{skill_name}' successfully")
            return result
        except Exception as e:
            execution_record["status"] = "failed"
            execution_record["error"] = str(e)

            logger.error(f"Skill '{skill_name}' execution failed: {e}")
            raise
        finally:
            # Add to history
            self.skill_history.append(execution_record)

            # Trim history if it gets too long
            if len(self.skill_history) > self.max_history:
                self.skill_history = self.skill_history[-self.max_history :]

    async def _execute_function(self, func: Callable, **kwargs) -> Any:
        """Execute a function, handling both sync and async cases."""
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            # For synchronous functions, run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**kwargs))

    def get_skill_definition(self, skill_name: str) -> SkillDefinition | None:
        """Get the definition of a skill."""
        return self.skills.get(skill_name)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "parameters": skill.parameters,
                "required_params": skill.required_params,
                "examples": skill.examples,
            }
            for skill in self.skills.values()
        ]

    def skill_exists(self, skill_name: str) -> bool:
        """Check if a skill exists."""
        return skill_name in self.skills

    async def execute_skills_batch(self, skills_data: list[dict[str, Any]]) -> list[Any]:
        """
        Execute multiple skills in parallel.

        Args:
            skills_data: List of dicts with 'skill_name' and 'arguments'

        Returns:
            List of results from skill executions
        """
        tasks = []
        for skill_data in skills_data:
            skill_name = skill_data["skill_name"]
            arguments = skill_data.get("arguments", {})

            task = self.execute_skill(skill_name, **arguments)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during execution
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    def get_skill_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent skill execution history."""
        return self.skill_history[-limit:]

    def clear_skill_history(self):
        """Clear the skill execution history."""
        self.skill_history.clear()
        logger.info("Cleared skill execution history")

    def get_skill_stats(self) -> dict[str, Any]:
        """
        Get statistics about registered skills.

        Returns:
            Dict with skill statistics
        """
        total_executions = len(self.skill_history)
        successful_executions = sum(
            1 for record in self.skill_history if record.get("status") == "completed"
        )
        failed_executions = total_executions - successful_executions

        # Count executions by skill
        skill_counts = {}
        for record in self.skill_history:
            skill_name = record["skill_name"]
            skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1

        return {
            "total_skills_registered": len(self.skills),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "skill_execution_counts": skill_counts,
            "history_size": len(self.skill_history),
        }

    def create_skill_from_function(
        self, func: Callable, name: str = None, description: str = ""
    ) -> str:
        """
        Automatically create and register a skill from a function.

        Args:
            func: Function to create skill from
            name: Name for the skill (defaults to function name)
            description: Description of the skill

        Returns:
            Name of the registered skill
        """
        skill_name = name or func.__name__

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = list(sig.parameters.keys())

        # For this simple implementation, assume all parameters are required
        # In a more advanced implementation, we could detect optional parameters
        required_params = []
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                parameters.append(param_name)

        self.register_skill(
            name=skill_name,
            func=func,
            parameters=parameters,
            required_params=required_params,
            description=description,
        )

        return skill_name


# Predefined utility skills
async def search_web(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """
    Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results
    """
    # In a real implementation, this would call a search API
    # For now, return mock results
    logger.info(f"Searching web for: {query}")
    return [
        {
            "title": f"Result {i} for '{query}'",
            "url": f"https://example.com/result{i}",
            "snippet": f"This is a mock snippet for result {i} of the search query '{query}'",
        }
        for i in range(1, num_results + 1)
    ]


async def calculate_math(expression: str) -> float:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to calculate

    Returns:
        Result of the calculation
    """
    # In a real implementation, this would safely evaluate the expression
    # For security, we won't actually eval() anything in this example
    logger.info(f"Calculating: {expression}")

    # Mock implementation - in reality, we'd use a safe math evaluation library
    if expression == "2 + 2":
        return 4.0
    elif expression == "10 * 5":
        return 50.0
    elif expression == "100 / 4":
        return 25.0
    else:
        # For other expressions, return a mock result
        return hash(expression) % 1000


async def get_current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.

    Args:
        format_str: Format string for the date/time

    Returns:
        Current date and time as a string
    """
    from datetime import datetime

    current_time = datetime.now()
    return current_time.strftime(format_str)


# Initialize a global skill manager with some default skills
global_skill_manager = SkillManager()

# Register default skills
global_skill_manager.register_async_skill(
    name="search_web",
    func=search_web,
    parameters=["query", "num_results"],
    required_params=["query"],
    description="Search the web for information",
    examples=[
        {"query": "AI developments 2023", "num_results": 3},
        {"query": "Python programming tutorials"},
    ],
)

global_skill_manager.register_async_skill(
    name="calculate_math",
    func=calculate_math,
    parameters=["expression"],
    required_params=["expression"],
    description="Calculate a mathematical expression",
    examples=[{"expression": "2 + 2"}, {"expression": "10 * 5"}],
)

global_skill_manager.register_async_skill(
    name="get_current_datetime",
    func=get_current_datetime,
    parameters=["format_str"],
    required_params=[],
    description="Get the current date and time",
    examples=[{"format_str": "%Y-%m-%d"}, {"format_str": "%H:%M:%S"}],
)


def get_global_skill_manager() -> SkillManager:
    """
    Get the global skill manager instance.

    Returns:
        Global SkillManager instance
    """
    return global_skill_manager
