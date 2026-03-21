"""
Evaluation — LLM-based evaluation scoring
==========================================
Module for evaluating the quality of generated content using LLM-based scoring.

Pattern: Strategy
Async: Yes — for I/O-bound evaluation tasks
Layer: L2 Verification

Usage:
    from orchestrator.evaluation import Evaluator
    evaluator = Evaluator()
    score = await evaluator.evaluate(content="...", criteria="accuracy, relevance")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .models import Model

logger = logging.getLogger("orchestrator.evaluation")


@dataclass
class EvaluationResult:
    """Represents the result of an evaluation."""
    
    score: float  # Normalized score between 0.0 and 1.0
    breakdown: Dict[str, float]  # Scores for individual criteria
    feedback: str  # Human-readable feedback
    reasoning: str  # Explanation of the scoring


class Evaluator:
    """LLM-based evaluation scorer for content quality assessment."""

    def __init__(self, model: Model = Model.DEEPSEEK_REASONER):
        """Initialize the evaluator with a reasoning model."""
        self.model = model
    
    async def evaluate(
        self, 
        content: str, 
        criteria: Union[str, List[str]], 
        reference: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate content against specified criteria.
        
        Args:
            content: The content to evaluate
            criteria: Evaluation criteria (either a description or list of criteria)
            reference: Optional reference content for comparison
            
        Returns:
            EvaluationResult: The evaluation result with score and feedback
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Format criteria for the prompt
        if isinstance(criteria, list):
            criteria_str = ", ".join(criteria)
            criteria_detail = "\n".join([f"- {criterion}" for criterion in criteria])
        else:
            criteria_str = criteria
            criteria_detail = f"- {criteria}"
        
        # Build the evaluation prompt
        prompt = f"""
        Evaluate the following content based on these criteria:
        
        EVALUATION CRITERIA:
        {criteria_detail}
        
        CONTENT TO EVALUATE:
        {content}
        """
        
        if reference:
            prompt += f"""
        REFERENCE CONTENT (for comparison):
        {reference}
        """
        
        prompt += """
        Provide your evaluation in the following format:
        
        SCORE: [A number between 0.0 and 1.0 representing overall quality]
        
        BREAKDOWN:
        - Criterion 1: [Score for first criterion between 0.0 and 1.0]
        - Criterion 2: [Score for second criterion between 0.0 and 1.0]
        (Continue for each criterion)
        
        FEEDBACK: [Brief feedback on strengths and weaknesses]
        
        REASONING: [Detailed explanation of your scoring]
        
        Be objective and consistent in your evaluations.
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_evaluation_response(response.content)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return a default evaluation result in case of failure
            return EvaluationResult(
                score=0.5,
                breakdown={str(criteria): 0.5},
                feedback="Evaluation failed due to an error",
                reasoning="Default score assigned due to evaluation failure"
            )
    
    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """Parse the evaluation response into an EvaluationResult object."""
        lines = response.split('\n')
        
        score = 0.5
        breakdown = {}
        feedback = ""
        reasoning = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score_str = line.split('SCORE:', 1)[1].strip()
                    score = float(score_str)
                except (ValueError, IndexError):
                    pass  # Keep default score
            
            elif line.startswith('BREAKDOWN:'):
                current_section = 'breakdown'
            
            elif line.startswith('FEEDBACK:'):
                current_section = 'feedback'
                feedback = line.split('FEEDBACK:', 1)[1].strip()
            
            elif line.startswith('REASONING:'):
                current_section = 'reasoning'
                reasoning = line.split('REASONING:', 1)[1].strip()
            
            elif current_section == 'breakdown' and line.startswith('- '):
                try:
                    parts = line[2:].split(':', 1)  # Remove '- ' prefix
                    if len(parts) == 2:
                        criterion = parts[0].strip()
                        try:
                            criterion_score = float(parts[1].strip())
                            breakdown[criterion] = criterion_score
                        except ValueError:
                            # If we can't parse the score, use a default
                            breakdown[criterion] = 0.5
                except IndexError:
                    pass  # Skip malformed lines
            
            elif current_section == 'feedback' and not line.startswith('REASONING:'):
                feedback += ' ' + line
            
            elif current_section == 'reasoning':
                reasoning += ' ' + line
        
        # If score is still default and we have breakdown, calculate average
        if score == 0.5 and breakdown:
            score = sum(breakdown.values()) / len(breakdown)
        
        return EvaluationResult(
            score=min(max(score, 0.0), 1.0),  # Normalize to 0.0-1.0 range
            breakdown=breakdown,
            feedback=feedback.strip(),
            reasoning=reasoning.strip()
        )
    
    async def compare_content(self, content1: str, content2: str, criteria: str) -> EvaluationResult:
        """
        Compare two pieces of content based on specified criteria.
        
        Args:
            content1: First piece of content
            content2: Second piece of content
            criteria: Criteria for comparison
            
        Returns:
            EvaluationResult: Evaluation of the comparison
        """
        prompt = f"""
        Compare the following two pieces of content based on: {criteria}
        
        CONTENT 1:
        {content1}
        
        CONTENT 2:
        {content2}
        
        Determine which is better according to the criteria, or if they are equivalent.
        Provide your evaluation in the standard format:
        
        SCORE: [0.0-0.5 for content1 being better, 0.5 for equivalent, 0.5-1.0 for content2 being better]
        
        BREAKDOWN:
        - {criteria}: [Score for how well each meets the criteria]
        
        FEEDBACK: [Comparison feedback]
        
        REASONING: [Reasoning behind the comparison]
        """
        
        from .api_clients import UnifiedClient
        client = UnifiedClient()
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_evaluation_response(response.content)
            
        except Exception as e:
            logger.error(f"Content comparison failed: {e}")
            return EvaluationResult(
                score=0.5,
                breakdown={criteria: 0.5},
                feedback="Comparison failed due to an error",
                reasoning="Default score assigned due to comparison failure"
            )
    
    async def evaluate_task_result(self, content: str, task_instruction: str) -> EvaluationResult:
        """
        Evaluate a task result against its original instruction.
        
        Args:
            content: The result of the task
            task_instruction: The original instruction for the task
            
        Returns:
            EvaluationResult: Evaluation of how well the result matches the instruction
        """
        criteria = f"How well does the content address: {task_instruction}"
        return await self.evaluate(content=content, criteria=criteria)