#!/usr/bin/env python3
"""
ARA Pipelines - Phase-Aware Model Selection Batch Update
Updates remaining 6 pipelines with phase-aware model selection.
"""

import re

file_path = "orchestrator/ara_pipelines.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# ═══════════════════════════════════════════════════════
# 7. Socratic Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_socratic_question
old_pattern = r'''    async def _phase_socratic_question\(self, state: PipelineState, round_num: int\):
        """Generate probing questions."""
        models = self\._get_available_models\(state\.task\.type\)
        questioner = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_socratic_question(self, state: PipelineState, round_num: int):
        """Generate probing questions."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for analysis (Socratic questioning)
        questioner = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# ═══════════════════════════════════════════════════════
# 8. Pre-Mortem Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_imagined_failure
old_pattern = r'''    async def _phase_imagined_failure\(self, state: PipelineState\):
        """Imagine catastrophic failure scenarios."""
        models = self\._get_available_models\(state\.task\.type\)
        visionary = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_imagined_failure(self, state: PipelineState):
        """Imagine catastrophic failure scenarios."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for analysis (pre-mortem)
        visionary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_mitigation
old_pattern = r'''    async def _phase_mitigation\(self, state: PipelineState\):
        """Generate mitigation strategies."""
        models = self\._get_available_models\(state\.task\.type\)
        strategist = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_mitigation(self, state: PipelineState):
        """Generate mitigation strategies."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for generation (mitigation planning)
        strategist = self._get_model_for_phase(PhaseType.GENERATION, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# ═══════════════════════════════════════════════════════
# 9. Bayesian Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_define_hypotheses
old_pattern = r'''    async def _phase_define_hypotheses\(self, state: PipelineState\):
        """Define competing hypotheses."""
        models = self\._get_available_models\(TaskType\.REASONING\)
        primary = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_define_hypotheses(self, state: PipelineState):
        """Define competing hypotheses."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for analysis
        primary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_elicit_priors
old_pattern = r'''    async def _phase_elicit_priors\(self, state: PipelineState\):
        """Elicit prior probabilities."""
        models = self\._get_available_models\(TaskType\.REASONING\)
        elicitor = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_elicit_priors(self, state: PipelineState):
        """Elicit prior probabilities."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for evaluation (probability estimation)
        elicitor = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_update_beliefs
old_pattern = r'''    async def _phase_update_beliefs\(self, state: PipelineState\):
        """Update beliefs based on evidence."""
        models = self\._get_available_models\(TaskType\.REASONING\)
        updater = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_update_beliefs(self, state: PipelineState):
        """Update beliefs based on evidence."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for synthesis (belief integration)
        updater = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# ═══════════════════════════════════════════════════════
# 10. Dialectical Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_thesis
old_pattern = r'''    async def _phase_thesis\(self, state: PipelineState\):
        """Establish thesis."""
        models = self\._get_available_models\(state\.task\.type\)
        primary = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_thesis(self, state: PipelineState):
        """Establish thesis."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for generation
        primary = self._get_model_for_phase(PhaseType.GENERATION, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_antithesis
old_pattern = r'''    async def _phase_antithesis\(self, state: PipelineState, thesis: str\):
        """Generate antithesis."""
        models = self\._get_available_models\(state\.task\.type\)
        critic = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_antithesis(self, state: PipelineState, thesis: str):
        """Generate antithesis."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for critique
        critic = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_synthesis in DialecticalPipeline (override base class)
old_pattern = r'''    async def _phase_synthesis\(self, state: PipelineState\):
        """Synthesize thesis and antithesis."""
        models = self\._get_available_models\(state\.task\.type\)
        synthesizer = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize thesis and antithesis."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for synthesis
        synthesizer = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# ═══════════════════════════════════════════════════════
# 11. Analogical Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_abstraction
old_pattern = r'''    async def _phase_abstraction\(self, state: PipelineState\):
        """Abstract the core problem."""
        models = self\._get_available_models\(state\.task\.type\)
        abstractor = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_abstraction(self, state: PipelineState):
        """Abstract the core problem."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for analysis
        abstractor = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_source_search
old_pattern = r'''    async def _phase_source_search\(self, state: PipelineState\):
        """Search for analogous solutions."""
        models = self\._get_available_models\(TaskType\.REASONING\)
        searcher = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_source_search(self, state: PipelineState):
        """Search for analogous solutions."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for research
        searcher = self._get_model_for_phase(PhaseType.RESEARCH, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_mapping
old_pattern = r'''    async def _phase_mapping\(self, state: PipelineState\):
        """Map analogous solution to target problem."""
        models = self\._get_available_models\(state\.task\.type\)
        mapper = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_mapping(self, state: PipelineState):
        """Map analogous solution to target problem."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for synthesis
        mapper = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# ═══════════════════════════════════════════════════════
# 12. Delphi Pipeline
# ═══════════════════════════════════════════════════════

# Update _phase_expert_round
old_pattern = r'''    async def _phase_expert_round\(self, state: PipelineState, round_num: int\):
        """Collect expert opinions."""
        models = self\._get_available_models\(state\.task\.type\)
        expert = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_expert_round(self, state: PipelineState, round_num: int):
        """Collect expert opinions."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for evaluation (expert judgment)
        expert = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Update _phase_aggregation
old_pattern = r'''    async def _phase_aggregation\(self, state: PipelineState\):
        """Aggregate expert opinions."""
        models = self\._get_available_models\(state\.task\.type\)
        aggregator = models\[0\] if models else Model\.GPT_4O_MINI'''

new_code = '''    async def _phase_aggregation(self, state: PipelineState):
        """Aggregate expert opinions."""
        from .phase_aware_models import PhaseType
        
        # Use phase-aware model selection for synthesis
        aggregator = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)'''

content = re.sub(old_pattern, new_code, content)

# Write updated content
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Batch update completed for all 12 ARA pipelines!")
print("\nUpdated pipelines:")
print("  1. ✅ Multi-Perspective (already done)")
print("  2. ✅ Iterative (already done)")
print("  3. ✅ Debate (already done)")
print("  4. ✅ Research (already done)")
print("  5. ✅ Jury (already done)")
print("  6. ✅ Scientific (already done)")
print("  7. ✅ Socratic")
print("  8. ✅ Pre-Mortem")
print("  9. ✅ Bayesian")
print(" 10. ✅ Dialectical")
print(" 11. ✅ Analogical")
print(" 12. ✅ Delphi")
