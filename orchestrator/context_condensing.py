"""
ContextCondensing — Context compression for long runs
==================================================
Module for compressing and condensing context to manage token usage in long-running processes.

Pattern: Strategy
Async: Yes — for I/O-bound compression operations
Layer: L2 Verification

Usage:
    from orchestrator.context_condensing import ContextCondenser
    condenser = ContextCondenser()
    compressed_context = await condenser.condense(context="...", target_ratio=0.5)
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

from .models import Model

logger = logging.getLogger("orchestrator.context_condensing")


class ContextCondenser:
    """Compresses and condenses context to manage token usage in long-running processes."""

    def __init__(self, model: Model = Model.DEEPSEEK_REASONER):
        """Initialize the context condenser."""
        self.model = model
    
    async def condense(self, context: str, target_ratio: float = 0.5, 
                       preserve_formatting: bool = True) -> str:
        """
        Condense the context to a target ratio of its original size.
        
        Args:
            context: The context to condense
            target_ratio: Target size as a ratio of original (0.1 = 10% of original)
            preserve_formatting: Whether to preserve basic formatting
            
        Returns:
            str: The condensed context
        """
        if len(context) == 0:
            return context
        
        if target_ratio >= 1.0:
            return context  # No need to condense
        
        # Try different condensing strategies
        strategies = [
            self._semantic_condense,
            self._summarize_condense,
            self._keyword_extract_condense
        ]
        
        for strategy in strategies:
            try:
                condensed = await strategy(context, target_ratio, preserve_formatting)
                if len(condensed) <= len(context) * target_ratio:
                    return condensed
            except Exception as e:
                logger.warning(f"Condensing strategy {strategy.__name__} failed: {e}")
                continue
        
        # Fallback: simple truncation if all strategies fail
        target_length = int(len(context) * target_ratio)
        return context[:target_length]
    
    async def _semantic_condense(self, context: str, target_ratio: float, 
                                preserve_formatting: bool) -> str:
        """Condense context semantically using LLM."""
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        target_length = int(len(context) * target_ratio)
        
        # Create a condensing prompt
        condense_prompt = f"""
        Please condense the following text to approximately {target_length} characters 
        while preserving the most important information:
        
        {context}
        
        CONDENSED TEXT:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": condense_prompt}]
            )
            
            condensed = response.content.strip()
            
            # If the result is still too long, recursively condense
            if len(condensed) > target_length:
                new_target_ratio = target_ratio * (target_length / len(condensed))
                return await self.condense(condensed, new_target_ratio, preserve_formatting)
            
            return condensed
        except Exception as e:
            logger.error(f"Semantic condensing failed: {e}")
            raise
    
    async def _summarize_condense(self, context: str, target_ratio: float, 
                                  preserve_formatting: bool) -> str:
        """Condense context by summarization."""
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # Estimate target length
        target_length = int(len(context) * target_ratio)
        
        # Create a summarization prompt
        summary_prompt = f"""
        Provide a concise summary of the following text in approximately {target_length} characters:
        
        {context}
        
        SUMMARY:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            
            summary = response.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Summarization condensing failed: {e}")
            raise
    
    async def _keyword_extract_condense(self, context: str, target_ratio: float, 
                                       preserve_formatting: bool) -> str:
        """Condense context by extracting key sentences based on keywords."""
        from .api_clients import UnifiedClient
        
        client = UnifiedClient()
        
        # First, identify key topics/themes in the context
        topic_prompt = f"""
        Identify the main topics/themes in the following text. 
        Respond with a list of 3-5 key topics:
        
        {context}
        
        KEY TOPICS:
        """
        
        try:
            response = await client.acomplete(
                model=self.model,
                messages=[{"role": "user", "content": topic_prompt}]
            )
            
            topics = [topic.strip() for topic in response.content.split('\n') if topic.strip()]
            
            # Split context into sentences
            sentences = re.split(r'[.!?]+', context)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Score sentences based on keyword presence
            scored_sentences = []
            for sentence in sentences:
                score = 0
                for topic in topics:
                    if topic.lower() in sentence.lower():
                        score += 1
                
                # Also score based on sentence position (beginning and end often contain important info)
                if sentences.index(sentence) < len(sentences) * 0.1:  # First 10%
                    score += 0.5
                elif sentences.index(sentence) > len(sentences) * 0.9:  # Last 10%
                    score += 0.5
                
                scored_sentences.append((sentence, score))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate how many sentences we need to reach the target ratio
            target_char_count = int(len(context) * target_ratio)
            selected_sentences = []
            current_chars = 0
            
            for sentence, score in scored_sentences:
                if current_chars + len(sentence) > target_char_count:
                    break
                selected_sentences.append(sentence)
                current_chars += len(sentence)
            
            # Reconstruct the condensed context
            if preserve_formatting:
                # Try to maintain some structure by joining with original separators
                original_parts = re.split(r'([.!?]+)', context)
                condensed_parts = []
                
                for part in original_parts:
                    if any(s in part for s in selected_sentences):
                        condensed_parts.append(part)
                
                result = ''.join(condensed_parts)
            else:
                result = '. '.join(selected_sentences) + '.'
            
            return result
        except Exception as e:
            logger.error(f"Keyword extraction condensing failed: {e}")
            raise
    
    async def condense_dialogue(self, dialogue: List[Dict[str, str]], 
                               target_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Condense a dialogue by reducing the number of turns while preserving key information.
        
        Args:
            dialogue: List of dialogue turns [{"role": "user", "content": "..."}, ...]
            target_turns: Target number of turns (if None, reduces by 50%)
            
        Returns:
            List[Dict[str, str]]: Condensed dialogue
        """
        if not dialogue:
            return dialogue
        
        if target_turns is None:
            target_turns = max(1, len(dialogue) // 2)
        
        if len(dialogue) <= target_turns:
            return dialogue
        
        # Use LLM to summarize intermediate turns
        condensed_dialogue = []
        
        # Always keep the first turn
        condensed_dialogue.append(dialogue[0])
        
        # Identify turns to summarize
        turns_to_summarize = dialogue[1:-1]  # Exclude first and last
        turns_per_summary = max(1, len(turns_to_summarize) // max(1, target_turns - 2))
        
        i = 1  # Start after the first turn we kept
        while i < len(dialogue) - 1:  # Stop before the last turn
            if len(condensed_dialogue) >= target_turns - 1:  # Leave space for the last turn
                break
            
            # Collect turns to summarize
            summary_batch = []
            batch_end = min(i + turns_per_summary, len(dialogue) - 1)  # Don't go past the last turn
            
            for j in range(i, batch_end):
                summary_batch.append(dialogue[j])
            
            if len(summary_batch) == 1:
                # Just add the single turn if there's nothing to summarize
                condensed_dialogue.append(summary_batch[0])
            else:
                # Summarize the batch
                summary_content = "\n".join([f"{turn['role']}: {turn['content']}" for turn in summary_batch])
                
                summary = await self.condense(
                    f"Summarize this dialogue segment: {summary_content}",
                    target_ratio=0.3  # Aggressive summary
                )
                
                condensed_dialogue.append({
                    "role": "system",
                    "content": f"Summary of previous conversation: {summary}"
                })
            
            i = batch_end
        
        # Always keep the last turn if there is one and we haven't reached the target
        if len(dialogue) > 1 and len(condensed_dialogue) < target_turns:
            condensed_dialogue.append(dialogue[-1])
        
        return condensed_dialogue
    
    async def condense_with_preservation(self, context: str, important_phrases: List[str], 
                                         target_ratio: float) -> str:
        """
        Condense context while preserving specified important phrases.
        
        Args:
            context: The context to condense
            important_phrases: Phrases that must be preserved
            target_ratio: Target size as a ratio of original
            
        Returns:
            str: The condensed context with important phrases preserved
        """
        # First, identify positions of important phrases
        phrase_positions = []
        for phrase in important_phrases:
            start_idx = 0
            while True:
                pos = context.lower().find(phrase.lower(), start_idx)
                if pos == -1:
                    break
                phrase_positions.append((pos, pos + len(phrase), phrase))
                start_idx = pos + 1
        
        # Sort positions by start index
        phrase_positions.sort(key=lambda x: x[0])
        
        # If no important phrases, just use regular condensation
        if not phrase_positions:
            return await self.condense(context, target_ratio)
        
        # Create segments that must be preserved
        segments_to_preserve = []
        last_end = 0
        
        for start, end, phrase in phrase_positions:
            # Add context around the phrase
            segment_start = max(0, start - 50)  # 50 chars before
            segment_end = min(len(context), end + 50)  # 50 chars after
            
            # Avoid overlapping segments
            if segment_start < last_end:
                segment_start = last_end
            
            if segment_start < segment_end:
                segments_to_preserve.append(context[segment_start:segment_end])
                last_end = segment_end
        
        # Calculate how much room is left for other content
        preserved_length = sum(len(segment) for segment in segments_to_preserve)
        remaining_chars = max(0, int(len(context) * target_ratio) - preserved_length)
        
        if remaining_chars <= 0:
            # If no room left, just join preserved segments
            return "".join(segments_to_preserve)
        
        # Identify non-preserved sections
        non_preserved_sections = []
        last_end = 0
        
        for start, end, phrase in phrase_positions:
            if start > last_end:
                non_preserved_sections.append(context[last_end:start])
            last_end = end
        
        if last_end < len(context):
            non_preserved_sections.append(context[last_end:])
        
        # Condense non-preserved sections proportionally
        total_non_preserved = sum(len(section) for section in non_preserved_sections)
        if total_non_preserved == 0:
            return "".join(segments_to_preserve)
        
        condensed_non_preserved = []
        for section in non_preserved_sections:
            if total_non_preserved > 0:
                section_ratio = len(section) / total_non_preserved
                target_section_chars = int(remaining_chars * section_ratio)
                if len(section) > target_section_chars:
                    condensed_section = await self.condense(section, 
                                                           target_ratio=target_section_chars/len(section))
                    condensed_non_preserved.append(condensed_section)
                else:
                    condensed_non_preserved.append(section)
            else:
                condensed_non_preserved.append(section)
        
        # Interleave preserved and condensed non-preserved sections
        result_parts = []
        non_preserved_idx = 0
        
        for i, original_pos in enumerate(phrase_positions):
            # Add preserved segment
            if i < len(segments_to_preserve):
                result_parts.append(segments_to_preserve[i])
            
            # Add condensed non-preserved section that comes after this preserved segment
            if non_preserved_idx < len(condensed_non_preserved):
                result_parts.append(condensed_non_preserved[non_preserved_idx])
                non_preserved_idx += 1
        
        # Add any remaining non-preserved content
        while non_preserved_idx < len(condensed_non_preserved):
            result_parts.append(condensed_non_preserved[non_preserved_idx])
            non_preserved_idx += 1
        
        return "".join(result_parts)
    
    async def estimate_token_reduction(self, original_context: str, 
                                      target_ratio: float) -> Dict[str, float]:
        """
        Estimate the token reduction achievable with condensation.
        
        Args:
            original_context: The original context
            target_ratio: Target size ratio
            
        Returns:
            Dict with reduction estimates
        """
        # Simple estimation based on character count
        # In practice, tokenizers vary, but this gives a rough estimate
        original_tokens = len(original_context) / 4  # Rough estimate: 1 token ~ 4 chars
        target_tokens = original_tokens * target_ratio
        
        return {
            "original_tokens": original_tokens,
            "estimated_target_tokens": target_tokens,
            "estimated_reduction_percentage": (1 - target_ratio) * 100,
            "estimated_savings_tokens": original_tokens - target_tokens
        }