"""
Brain — AI reasoning and cognitive layer
========================================
The Brain module implements an AI reasoning layer that can understand context,
make decisions, and coordinate between different orchestrator components.

Pattern: Cognitive Architecture
Async: Yes — for I/O-bound reasoning tasks
Layer: L3 Agents

Usage:
    from orchestrator.brain import Brain
    brain = Brain(model="deepseek/deepseek-chat")
    decision = await brain.reason(context="...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Model

logger = logging.getLogger("orchestrator.brain")


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning process."""

    step_number: int
    thought: str
    action: str
    result: str | None = None


@dataclass
class CognitiveState:
    """Represents the current cognitive state of the brain."""

    context: str
    reasoning_history: list[ReasoningStep]
    current_goal: str
    confidence_score: float


class Brain:
    """AI reasoning and cognitive layer for the orchestrator."""

    def __init__(self, model: Model = Model.DEEPSEEK_CHAT):
        """Initialize the brain with a reasoning model."""
        self.model = model
        self._cognitive_state: CognitiveState | None = None

    async def reason(self, context: str, goal: str = "") -> CognitiveState:
        """
        Perform reasoning based on the provided context and goal.

        Args:
            context: The context to reason about
            goal: The goal to achieve

        Returns:
            CognitiveState: The resulting cognitive state after reasoning
        """
        from .api_clients import UnifiedClient

        client = UnifiedClient()

        # Create a structured prompt for reasoning
        prompt = f"""
        You are an advanced reasoning system. Given the following context:

        CONTEXT: {context}

        GOAL: {goal or "Analyze and provide insights"}

        Think step-by-step and provide your reasoning. Follow this format:
        1. Thought: [Your thought process]
        2. Action: [What you're doing]
        3. Result: [What you concluded]

        Be thorough but concise.
        """

        try:
            response = await client.acomplete(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response into reasoning steps
            reasoning_steps = self._parse_reasoning_response(response.content)

            # Calculate confidence score based on response quality
            confidence = self._calculate_confidence(response.content)

            self._cognitive_state = CognitiveState(
                context=context,
                reasoning_history=reasoning_steps,
                current_goal=goal,
                confidence_score=confidence,
            )

            return self._cognitive_state

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            # Return a default cognitive state with the context
            self._cognitive_state = CognitiveState(
                context=context, reasoning_history=[], current_goal=goal, confidence_score=0.0
            )
            return self._cognitive_state

    def _parse_reasoning_response(self, response: str) -> list[ReasoningStep]:
        """Parse the reasoning response into structured steps."""
        steps = []
        lines = response.split("\n")

        current_step = None

        for line in lines:
            line = line.strip()

            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                # Extract step number
                step_num = int(line[0])

                if "Thought:" in line:
                    thought = line.split("Thought:", 1)[1].strip()
                    current_step = ReasoningStep(step_number=step_num, thought=thought, action="")

                elif "Action:" in line:
                    action = line.split("Action:", 1)[1].strip()
                    if current_step:
                        current_step.action = action
                    else:
                        current_step = ReasoningStep(
                            step_number=step_num, thought="", action=action
                        )

                elif "Result:" in line:
                    result = line.split("Result:", 1)[1].strip()
                    if current_step:
                        current_step.result = result
                        steps.append(current_step)
                        current_step = None
                    else:
                        current_step = ReasoningStep(
                            step_number=step_num, thought="", action="", result=result
                        )
                        steps.append(current_step)
                        current_step = None

        # Handle any remaining step
        if current_step:
            steps.append(current_step)

        return steps

    def _calculate_confidence(self, response: str) -> float:
        """Calculate a confidence score based on the response quality."""
        # Simple heuristic: longer, more structured responses get higher scores
        if len(response) < 50:
            return 0.3  # Low confidence for very short responses

        # Check for presence of reasoning keywords
        keywords = ["because", "therefore", "thus", "consequently", "hence"]
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in response.lower())

        # Base confidence on keyword count and response length
        base_confidence = min(0.5 + (keyword_count * 0.15), 1.0)

        # Adjust based on response length (longer responses tend to be more thoughtful)
        length_factor = min(len(response) / 500, 0.3)  # Up to 30% bonus for length

        return min(base_confidence + length_factor, 1.0)

    async def make_decision(self, options: list[str], context: str) -> str:
        """
        Make a decision among multiple options based on context.

        Args:
            options: List of possible options to choose from
            context: Context to inform the decision

        Returns:
            str: The selected option
        """
        prompt = f"""
        You are making a decision based on the following context:

        CONTEXT: {context}

        OPTIONS:
        {chr(10).join([f"- {option}" for option in options])}

        Which option is the best choice given the context? Explain your reasoning briefly,
        then state your final choice as "FINAL CHOICE: [selected option]"
        """

        from .api_clients import UnifiedClient

        client = UnifiedClient()

        try:
            response = await client.acomplete(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )

            # Extract the final choice from the response
            lines = response.content.split("\n")
            for line in lines:
                if line.startswith("FINAL CHOICE:"):
                    choice = line.split("FINAL CHOICE:", 1)[1].strip()
                    return choice

            # If no explicit final choice, return the first option
            return options[0] if options else ""

        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            # Default to first option if decision making fails
            return options[0] if options else ""

    async def summarize_context(self, context: str, max_length: int = 200) -> str:
        """
        Summarize a long context to a shorter form.

        Args:
            context: The context to summarize
            max_length: Maximum length of the summary

        Returns:
            str: The summarized context
        """
        prompt = f"""
        Please summarize the following context in {max_length} characters or less:

        {context}

        SUMMARY:
        """

        from .api_clients import UnifiedClient

        client = UnifiedClient()

        try:
            response = await client.acomplete(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Context summarization failed: {e}")
            # Return truncated version if summarization fails
            return context[:max_length] + "..." if len(context) > max_length else context
