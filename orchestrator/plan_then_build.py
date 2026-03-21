"""
PlanThenBuild — Plan-first, then execute pattern
===============================================
Module for implementing the plan-then-build pattern where tasks are planned first
before execution to improve outcomes.

Pattern: Template Method
Async: Yes — for I/O-bound planning and execution
Layer: L4 Supervisor

Usage:
    from orchestrator.plan_then_build import PlanThenBuilder
    builder = PlanThenBuilder()
    result = await builder.execute_planned_task(description="Build a REST API")
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import Model, Task, TaskType

logger = logging.getLogger("orchestrator.plan_then_build")


@dataclass
class PlanStep:
    """Represents a single step in a plan."""
    
    id: str
    description: str
    dependencies: List[str]  # IDs of steps this step depends on
    estimated_duration: float  # Estimated time in seconds
    resources_needed: List[str]  # Resources required for this step


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan."""
    
    task_description: str
    steps: List[PlanStep]
    estimated_total_duration: float
    critical_path: List[str]  # Steps on the critical path
    risks: List[str]  # Identified risks
    success_criteria: List[str]  # Criteria for success


class PlanThenBuilder:
    """Implements the plan-then-build pattern for improved task execution."""

    def __init__(self, planner_model: Model = Model.DEEPSEEK_REASONER, 
                 executor_model: Model = Model.DEEPSEEK_CHAT):
        """Initialize the plan-then-builder."""
        self.planner_model = planner_model
        self.executor_model = executor_model
    
    async def execute_planned_task(self, description: str, 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task using the plan-then-build pattern.
        
        Args:
            description: Description of the task to execute
            context: Additional context for the task
            
        Returns:
            Dict[str, Any]: Result of the execution
        """
        # Step 1: Create a plan
        plan = await self.create_plan(description, context)
        
        # Step 2: Execute the plan
        result = await self.execute_plan(plan, context)
        
        return result
    
    async def create_plan(self, task_description: str, 
                          context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create an execution plan for the given task.
        
        Args:
            task_description: Description of the task
            context: Additional context
            
        Returns:
            ExecutionPlan: The created execution plan
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Build the planning prompt
        context_str = f"\nAdditional Context: {context}" if context else ""
        
        planning_prompt = f"""
        Create a detailed execution plan for the following task:
        
        TASK: {task_description}
        {context_str}
        
        The plan should include:
        1. A sequence of steps required to complete the task
        2. Dependencies between steps
        3. Estimated duration for each step
        4. Resources needed for each step
        5. Potential risks
        6. Success criteria
        
        Format the response as:
        STEPS:
        1. [Step ID]: [Description] (Depends: [comma-separated dependencies], Duration: [seconds], Resources: [comma-separated])
        2. [Step ID]: [Description] (Depends: [dependencies], Duration: [seconds], Resources: [resources])
        (Continue for each step)
        
        RISKS:
        - [Risk 1]
        - [Risk 2]
        (Continue for each risk)
        
        SUCCESS CRITERIA:
        - [Criterion 1]
        - [Criterion 2]
        (Continue for each criterion)
        """
        
        try:
            response = await client.acomplete(
                model=self.planner_model,
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            return self._parse_execution_plan(response.content)
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Return a basic plan if planning fails
            return ExecutionPlan(
                task_description=task_description,
                steps=[
                    PlanStep(
                        id="fallback_step_1",
                        description=f"Execute the task: {task_description}",
                        dependencies=[],
                        estimated_duration=30.0,
                        resources_needed=["general_computation"]
                    )
                ],
                estimated_total_duration=30.0,
                critical_path=["fallback_step_1"],
                risks=["Planning failed, using fallback execution"],
                success_criteria=[f"Task '{task_description}' completed"]
            )
    
    def _parse_execution_plan(self, plan_text: str) -> ExecutionPlan:
        """Parse the execution plan from the LLM response."""
        lines = plan_text.split('\n')
        
        steps = []
        risks = []
        success_criteria = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('STEPS:'):
                current_section = 'steps'
            elif line.startswith('RISKS:'):
                current_section = 'risks'
            elif line.startswith('SUCCESS CRITERIA:'):
                current_section = 'success_criteria'
            elif current_section == 'steps' and line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # Parse step information
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    step_info = parts[1].strip()
                    
                    # Extract dependencies, duration, and resources
                    deps = []
                    duration = 30.0  # Default duration
                    resources = []
                    
                    if '(Depends:' in step_info:
                        # Extract dependencies, duration, and resources
                        paren_part = step_info[step_info.find('(Depends:'):]
                        step_desc = step_info[:step_info.find('(Depends:')].strip()
                        
                        # Parse dependencies
                        if 'Depends:' in paren_part:
                            deps_part = paren_part.split('Depends:')[1].split(',')[0].split(')', 1)[0].strip()
                            deps = [d.strip() for d in deps_part.split(',') if d.strip()]
                        
                        # Parse duration
                        if 'Duration:' in paren_part:
                            dur_part = paren_part.split('Duration:')[1].split(',')[0].split(')', 1)[0].strip()
                            try:
                                duration = float(dur_part)
                            except ValueError:
                                duration = 30.0  # Default if parsing fails
                        
                        # Parse resources
                        if 'Resources:' in paren_part:
                            res_part = paren_part.split('Resources:')[1].split(')')[0].strip()
                            resources = [r.strip() for r in res_part.split(',') if r.strip()]
                    else:
                        # Simple format without parentheses
                        step_desc = step_info
                    
                    # Extract step ID from the beginning
                    step_num = parts[0].split('.')[0].strip()
                    step_id = f"step_{step_num}"
                    
                    step = PlanStep(
                        id=step_id,
                        description=step_desc,
                        dependencies=deps,
                        estimated_duration=duration,
                        resources_needed=resources
                    )
                    steps.append(step)
            elif current_section == 'risks' and line.startswith('- '):
                risk = line[2:].strip()
                risks.append(risk)
            elif current_section == 'success_criteria' and line.startswith('- '):
                criterion = line[2:].strip()
                success_criteria.append(criterion)
        
        # Calculate total duration and critical path
        total_duration = sum(step.estimated_duration for step in steps)
        critical_path = [step.id for step in steps]  # Simplified: assume all steps are critical
        
        return ExecutionPlan(
            task_description="",
            steps=steps,
            estimated_total_duration=total_duration,
            critical_path=critical_path,
            risks=risks,
            success_criteria=success_criteria
        )
    
    async def execute_plan(self, plan: ExecutionPlan, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the given execution plan.
        
        Args:
            plan: The plan to execute
            context: Additional context
            
        Returns:
            Dict[str, Any]: Result of the execution
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Execute steps in dependency order
        completed_steps = set()
        results = {}
        
        # Keep executing until all steps are completed
        while len(completed_steps) < len(plan.steps):
            # Find steps whose dependencies are all completed
            ready_steps = []
            for step in plan.steps:
                if step.id not in completed_steps:
                    # Check if all dependencies are completed
                    deps_met = all(dep in completed_steps for dep in step.dependencies)
                    if deps_met:
                        ready_steps.append(step)
            
            if not ready_steps:
                # No progress can be made, probably a circular dependency
                logger.error("Cannot proceed with plan execution due to unmet dependencies")
                break
            
            # Execute ready steps in parallel
            execution_tasks = []
            for step in ready_steps:
                task = self._execute_step(step, context, results)
                execution_tasks.append(asyncio.create_task(task))
            
            # Wait for all ready steps to complete
            step_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            for i, step in enumerate(ready_steps):
                result = step_results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Step {step.id} failed: {result}")
                    results[step.id] = {"status": "failed", "error": str(result)}
                else:
                    results[step.id] = {"status": "completed", "output": result}
                    completed_steps.add(step.id)
        
        # Compile final result
        final_result = {
            "plan_executed": plan.task_description,
            "steps_completed": len(completed_steps),
            "total_steps": len(plan.steps),
            "step_results": results,
            "risks_encountered": [],
            "success_criteria_met": len(results) == len(plan.steps)  # Simplified check
        }
        
        return final_result
    
    async def _execute_step(self, step: PlanStep, context: Optional[Dict[str, Any]], 
                            previous_results: Dict[str, Any]) -> str:
        """Execute a single step in the plan."""
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Build context for this step
        step_context = f"Step: {step.description}\n"
        if previous_results:
            step_context += f"Previous results: {previous_results}\n"
        if context:
            step_context += f"Global context: {context}\n"
        
        execution_prompt = f"""
        Execute the following step:
        
        {step_context}
        
        Provide the output of this step.
        """
        
        try:
            response = await client.acomplete(
                model=self.executor_model,
                messages=[{"role": "user", "content": execution_prompt}]
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise
    
    async def validate_plan_feasibility(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Validate if the plan is feasible given constraints.
        
        Args:
            plan: The plan to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Check for circular dependencies
        dependency_graph = {step.id: set(step.dependencies) for step in plan.steps}
        
        # Simple cycle detection using DFS
        visiting = set()
        visited = set()
        
        def has_cycle(node):
            if node in visited:
                return False
            if node in visiting:
                return True
            
            visiting.add(node)
            for neighbor in dependency_graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False
        
        cycles = []
        for step_id in dependency_graph:
            if has_cycle(step_id):
                cycles.append(step_id)
        
        # Check resource availability (simplified check)
        all_resources = set()
        for step in plan.steps:
            all_resources.update(step.resources_needed)
        
        # Compile validation results
        validation_result = {
            "is_feasible": len(cycles) == 0,
            "cycles_detected": cycles,
            "resource_availability": {resource: True for resource in all_resources},  # Simplified
            "estimated_duration": plan.estimated_total_duration,
            "recommendations": []
        }
        
        if cycles:
            validation_result["recommendations"].append(
                f"Resolve circular dependencies in steps: {cycles}"
            )
        
        if plan.estimated_total_duration > 3600:  # More than 1 hour
            validation_result["recommendations"].append(
                "Consider breaking down the plan into smaller chunks for better manageability"
            )
        
        return validation_result
    
    async def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Optimize the plan for better execution.
        
        Args:
            plan: The plan to optimize
            
        Returns:
            ExecutionPlan: The optimized plan
        """
        # For now, we'll just return the plan as is
        # In a more advanced implementation, we could:
        # - Parallelize independent steps
        # - Optimize step order based on dependencies
        # - Adjust durations based on historical data
        return plan