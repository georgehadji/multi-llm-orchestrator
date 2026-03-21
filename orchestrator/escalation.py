"""
Escalation — Automatic escalation to higher-capability models
=============================================================
Module for automatically escalating to higher-capability models when quality thresholds
are not met or when specific conditions are triggered.

Pattern: Chain of Responsibility
Async: Yes — for I/O-bound model calls
Layer: L2 Verification

Usage:
    from orchestrator.escalation import EscalationHandler
    handler = EscalationHandler()
    result = await handler.process_with_escalation(content="...", criteria="quality")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .models import Model
from .evaluation import Evaluator, EvaluationResult

logger = logging.getLogger("orchestrator.escalation")


@dataclass
class EscalationRule:
    """Defines a rule for when to escalate to a higher-capability model."""
    
    condition: str  # Condition that triggers escalation (e.g., "quality_threshold", "error_retry")
    threshold: float  # Threshold value for numeric conditions
    target_model: Model  # Model to escalate to
    max_escalations: int = 3  # Maximum number of escalations allowed


@dataclass
class EscalationResult:
    """Represents the result of an escalation process."""
    
    final_content: str
    final_model: Model
    escalation_path: List[Model]  # Path of models used during escalation
    evaluation_result: Optional[EvaluationResult] = None
    was_escalated: bool = False


class EscalationHandler:
    """Handles automatic escalation to higher-capability models."""

    def __init__(self, evaluator: Optional[Evaluator] = None):
        """Initialize the escalation handler."""
        self.evaluator = evaluator or Evaluator()
        self.rules: List[EscalationRule] = []
        self.max_escalations = 3
    
    def add_rule(self, rule: EscalationRule):
        """Add an escalation rule."""
        self.rules.append(rule)
    
    def add_quality_threshold_rule(
        self, 
        threshold: float, 
        target_model: Model,
        max_escalations: int = 3
    ):
        """Add a rule based on quality threshold."""
        rule = EscalationRule(
            condition="quality_threshold",
            threshold=threshold,
            target_model=target_model,
            max_escalations=max_escalations
        )
        self.add_rule(rule)
    
    def add_error_retry_rule(self, target_model: Model, max_escalations: int = 3):
        """Add a rule for escalating on errors."""
        rule = EscalationRule(
            condition="error_retry",
            threshold=0.0,  # Not used for error retries
            target_model=target_model,
            max_escalations=max_escalations
        )
        self.add_rule(rule)
    
    async def process_with_escalation(
        self,
        content: str,
        criteria: Union[str, List[str]],
        initial_model: Model,
        max_escalations: Optional[int] = None
    ) -> EscalationResult:
        """
        Process content with potential escalation based on rules.
        
        Args:
            content: The content to process
            criteria: Criteria for evaluation
            initial_model: Starting model for processing
            max_escalations: Maximum number of escalations allowed
            
        Returns:
            EscalationResult: Result of the processing with escalation info
        """
        current_model = initial_model
        current_content = content
        escalation_path = [initial_model]
        escalation_count = 0
        max_escalations = max_escalations or self.max_escalations
        evaluation_result = None
        
        # First, try with the initial model
        try:
            current_content = await self._process_with_model(current_content, current_model)
            evaluation_result = await self.evaluator.evaluate(current_content, criteria)
        except Exception as e:
            logger.warning(f"Initial model {current_model} failed: {e}")
            # Trigger escalation due to error
            if escalation_count < max_escalations:
                escalated = await self._apply_escalation_rule(
                    "error_retry", 
                    current_content, 
                    current_model, 
                    criteria
                )
                if escalated:
                    current_model = escalated.target_model
                    current_content = await self._process_with_model(current_content, current_model)
                    evaluation_result = await self.evaluator.evaluate(current_content, criteria)
                    escalation_path.append(current_model)
                    escalation_count += 1
        
        # Check if escalation is needed based on quality
        while (escalation_count < max_escalations and 
               evaluation_result and 
               evaluation_result.score < 0.7):  # Default threshold
            
            # Find applicable escalation rules
            applicable_rule = self._find_applicable_rule(evaluation_result.score)
            
            if applicable_rule:
                current_model = applicable_rule.target_model
                current_content = await self._process_with_model(current_content, current_model)
                evaluation_result = await self.evaluator.evaluate(current_content, criteria)
                escalation_path.append(current_model)
                escalation_count += 1
            else:
                # No applicable rule, break the loop
                break
        
        return EscalationResult(
            final_content=current_content,
            final_model=current_model,
            escalation_path=escalation_path,
            evaluation_result=evaluation_result,
            was_escalated=len(escalation_path) > 1
        )
    
    async def _process_with_model(self, content: str, model: Model) -> str:
        """Process content with a specific model."""
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        prompt = f"""
        Improve or refine the following content:
        
        {content}
        
        Please provide an improved version.
        """
        
        try:
            response = await client.acomplete(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content
        except Exception as e:
            logger.error(f"Processing with model {model} failed: {e}")
            raise
    
    def _find_applicable_rule(self, score: float) -> Optional[EscalationRule]:
        """Find an applicable escalation rule based on the current score."""
        for rule in self.rules:
            if rule.condition == "quality_threshold" and score < rule.threshold:
                return rule
        return None
    
    async def _apply_escalation_rule(
        self, 
        condition: str, 
        content: str, 
        current_model: Model, 
        criteria: Union[str, List[str]]
    ) -> Optional[EscalationRule]:
        """Apply an escalation rule based on the condition."""
        for rule in self.rules:
            if rule.condition == condition:
                return rule
        return None
    
    async def escalate_based_on_complexity(
        self, 
        content: str, 
        complexity_threshold: float = 0.7
    ) -> EscalationResult:
        """
        Automatically escalate based on content complexity.
        
        Args:
            content: Content to assess for complexity
            complexity_threshold: Threshold for complexity-based escalation
            
        Returns:
            EscalationResult: Result of processing with potential escalation
        """
        # Assess complexity by analyzing the content
        complexity_score = await self._assess_complexity(content)
        
        if complexity_score > complexity_threshold:
            # Use a more capable model for complex content
            target_model = Model.DEEPSEEK_REASONER  # Assuming this is a high-capability model
        else:
            # Use the default model for simpler content
            target_model = Model.DEEPSEEK_CHAT
        
        try:
            processed_content = await self._process_with_model(content, target_model)
            evaluation_result = await self.evaluator.evaluate(processed_content, "relevance and accuracy")
            
            return EscalationResult(
                final_content=processed_content,
                final_model=target_model,
                escalation_path=[target_model],
                evaluation_result=evaluation_result,
                was_escalated=complexity_score > complexity_threshold
            )
        except Exception as e:
            logger.error(f"Complexity-based escalation failed: {e}")
            # Fallback to initial model
            fallback_content = await self._process_with_model(content, Model.DEEPSEEK_CHAT)
            evaluation_result = await self.evaluator.evaluate(fallback_content, "relevance and accuracy")
            
            return EscalationResult(
                final_content=fallback_content,
                final_model=Model.DEEPSEEK_CHAT,
                escalation_path=[Model.DEEPSEEK_CHAT],
                evaluation_result=evaluation_result,
                was_escalated=False
            )
    
    async def _assess_complexity(self, content: str) -> float:
        """Assess the complexity of content on a scale of 0.0 to 1.0."""
        # This is a simplified complexity assessment
        # In a real implementation, this could use more sophisticated NLP techniques
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count sentences to assess complexity
        sentences = content.split('.')
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Count unique words to assess vocabulary diversity
        unique_words = set(word.lower() for word in words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Combine metrics into a complexity score (normalized to 0.0-1.0)
        # Higher values indicate higher complexity
        complexity_score = (
            (avg_word_length / 10) * 0.3 +  # Average word length contributes 30%
            (min(avg_sentence_length / 20, 1.0)) * 0.4 +  # Sentence length contributes 40%
            vocabulary_diversity * 0.3  # Vocabulary diversity contributes 30%
        )
        
        return min(complexity_score, 1.0)  # Ensure score is between 0.0 and 1.0