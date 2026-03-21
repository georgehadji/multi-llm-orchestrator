"""
PromptEnhancer — Prompt enhancement and optimization
===================================================
Module for enhancing and optimizing prompts to improve model performance.

Pattern: Decorator
Async: Yes — for I/O-bound enhancement operations
Layer: L2 Verification

Usage:
    from orchestrator.prompt_enhancer import PromptEnhancer
    enhancer = PromptEnhancer()
    enhanced_prompt = await enhancer.enhance(prompt="...", context="...")
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .models import Model

logger = logging.getLogger("orchestrator.prompt_enhancer")


class PromptEnhancer:
    """Enhances and optimizes prompts to improve model performance."""

    def __init__(self, model: Model = Model.DEEPSEEK_REASONER):
        """Initialize the prompt enhancer."""
        self.model = model
        self.enhancement_templates = self._load_enhancement_templates()
    
    def _load_enhancement_templates(self) -> Dict[str, str]:
        """Load enhancement templates for different types of prompts."""
        return {
            "instruction": (
                "You are an expert assistant. {instruction}\n\n"
                "Follow these guidelines:\n"
                "- Be accurate and factual\n"
                "- Be concise but thorough\n"
                "- Structure your response clearly\n"
                "- Use examples where helpful\n\n"
                "Response:"
            ),
            "question_answering": (
                "Based on the following context, answer the question:\n\n"
                "CONTEXT: {context}\n\n"
                "QUESTION: {question}\n\n"
                "Provide a clear, accurate answer based solely on the context."
            ),
            "creative_writing": (
                "Write creatively on the following topic:\n\n"
                "TOPIC: {topic}\n\n"
                "Requirements:\n"
                "- Be imaginative and engaging\n"
                "- Use vivid descriptions\n"
                "- Maintain coherence\n\n"
                "Creative Output:"
            ),
            "analysis": (
                "Analyze the following information:\n\n"
                "INFORMATION: {info}\n\n"
                "Provide a structured analysis covering:\n"
                "- Key points\n"
                "- Implications\n"
                "- Recommendations\n\n"
                "Analysis:"
            )
        }
    
    async def enhance(self, prompt: str, context: Optional[str] = None, 
                      enhancement_type: Optional[str] = None) -> str:
        """
        Enhance a prompt using LLM-based optimization.
        
        Args:
            prompt: The original prompt to enhance
            context: Additional context for enhancement
            enhancement_type: Type of enhancement to apply (instruction, question_answering, etc.)
            
        Returns:
            str: The enhanced prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Select enhancement template based on type
        template = self.enhancement_templates.get(enhancement_type or "instruction", 
                                                 self.enhancement_templates["instruction"])
        
        # Format the template with the provided prompt
        formatted_prompt = template.format(instruction=prompt, 
                                          context=context or "", 
                                          question=prompt, 
                                          topic=prompt, 
                                          info=prompt)
        
        # Create an enhancement request
        enhancement_request = f"""
        The following is a prompt that needs enhancement:
        
        ORIGINAL PROMPT:
        {formatted_prompt}
        
        Please enhance this prompt to make it more effective. Consider:
        1. Clarity and specificity
        2. Structure and organization
        3. Expected output format
        4. Constraints and requirements
        
        ENHANCED PROMPT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": enhancement_request}]
            )
            
            enhanced_prompt = response.content.strip()
            
            # Log the enhancement for debugging
            logger.debug(f"Prompt enhanced:\nOriginal: {prompt}\nEnhanced: {enhanced_prompt}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            # Return original prompt if enhancement fails
            return prompt
    
    async def optimize_for_model(self, prompt: str, target_model: Model) -> str:
        """
        Optimize a prompt specifically for a target model.
        
        Args:
            prompt: The prompt to optimize
            target_model: The model to optimize for
            
        Returns:
            str: The model-optimized prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Create a model-specific optimization request
        optimization_request = f"""
        Optimize the following prompt for the model: {target_model.value}
        
        Different models have different strengths:
        - Reasoning models: Benefit from step-by-step instructions
        - Chat models: Work well with conversational framing
        - Coding models: Need clear specification of programming tasks
        
        PROMPT TO OPTIMIZE:
        {prompt}
        
        OPTIMIZED PROMPT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": optimization_request}]
            )
            
            optimized_prompt = response.content.strip()
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Model-specific prompt optimization failed: {e}")
            # Return original prompt if optimization fails
            return prompt
    
    async def add_context_awareness(self, prompt: str, context: str) -> str:
        """
        Add context awareness to a prompt.
        
        Args:
            prompt: The original prompt
            context: Context to incorporate
            
        Returns:
            str: The context-aware prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Create a context incorporation request
        context_request = f"""
        Incorporate the following context into the prompt in a way that enhances its effectiveness:
        
        ORIGINAL PROMPT:
        {prompt}
        
        CONTEXT:
        {context}
        
        PROMPT WITH CONTEXT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": context_request}]
            )
            
            context_aware_prompt = response.content.strip()
            return context_aware_prompt
            
        except Exception as e:
            logger.error(f"Context-aware prompt enhancement failed: {e}")
            # Return original prompt with context appended if enhancement fails
            return f"{prompt}\n\nAdditional Context: {context}"
    
    async def apply_role_playing(self, prompt: str, role: str) -> str:
        """
        Apply role-playing to a prompt to improve performance.
        
        Args:
            prompt: The original prompt
            role: The role to assign to the model
            
        Returns:
            str: The role-enhanced prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Create a role-playing enhancement request
        role_request = f"""
        Transform the following prompt to use role-playing for better results:
        
        ORIGINAL PROMPT:
        {prompt}
        
        ROLE: {role}
        
        Please rewrite the prompt to frame the task as if the model is playing the role of {role}.
        This often improves performance by giving the model a clearer perspective and expertise.
        
        ROLE-PLAYING PROMPT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": role_request}]
            )
            
            role_prompt = response.content.strip()
            return role_prompt
            
        except Exception as e:
            logger.error(f"Role-playing prompt enhancement failed: {e}")
            # Return original prompt with role prepended if enhancement fails
            return f"You are acting as a {role}. {prompt}"
    
    async def improve_clarity(self, prompt: str) -> str:
        """
        Improve the clarity of a prompt.
        
        Args:
            prompt: The original prompt
            
        Returns:
            str: The clarity-improved prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Create a clarity improvement request
        clarity_request = f"""
        Improve the clarity of the following prompt. Make it more specific, remove ambiguity,
        and structure it in a way that clearly communicates expectations:
        
        ORIGINAL PROMPT:
        {prompt}
        
        CLARITY-IMPROVED PROMPT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": clarity_request}]
            )
            
            clear_prompt = response.content.strip()
            return clear_prompt
            
        except Exception as e:
            logger.error(f"Clarity improvement failed: {e}")
            # Return original prompt if improvement fails
            return prompt
    
    async def format_for_output(self, prompt: str, output_format: str) -> str:
        """
        Format a prompt to specify the desired output format.
        
        Args:
            prompt: The original prompt
            output_format: The desired output format (e.g., "JSON", "Markdown", "Bullet Points")
            
        Returns:
            str: The format-specific prompt
        """
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Create a format specification request
        format_request = f"""
        Modify the following prompt to explicitly specify the output format:
        
        ORIGINAL PROMPT:
        {prompt}
        
        DESIRED OUTPUT FORMAT: {output_format}
        
        Please rewrite the prompt to clearly instruct the model to respond in {output_format} format.
        
        FORMATTED PROMPT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": format_request}]
            )
            
            formatted_prompt = response.content.strip()
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Output format specification failed: {e}")
            # Return original prompt with format instruction appended if fails
            return f"{prompt}\n\nPlease respond in {output_format} format."
    
    async def batch_enhance(self, prompts: List[str], context: Optional[str] = None) -> List[str]:
        """
        Enhance multiple prompts in a batch.
        
        Args:
            prompts: List of prompts to enhance
            context: Additional context for all enhancements
            
        Returns:
            List[str]: List of enhanced prompts
        """
        enhanced_prompts = []
        
        for prompt in prompts:
            enhanced = await self.enhance(prompt, context)
            enhanced_prompts.append(enhanced)
        
        return enhanced_prompts