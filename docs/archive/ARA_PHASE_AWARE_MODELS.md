# ARA Pipelines - Phase-Aware Model Selection

## 🎯 Επισκόπηση

Τα ARA (Advanced Reasoning & Analysis) Pipelines πλέον χρησιμοποιούν **phase-aware model selection**, όπου κάθε phase της reasoning διαδικασίας χρησιμοποιεί το βέλτιστο model για τις συγκεκριμένες απαιτήσεις του.

## 📊 Παράδειγμα: Multi-Perspective Pipeline

### Πριν (Unified Model)
```
Όλες οι phases → GPT-4o ($2.50/$10.00)
- Analysis: GPT-4o
- Critique: GPT-4o  
- Synthesis: GPT-4o
```

### Μετά (Phase-Aware)
```
Analysis Phase → o3-mini ($1.10/$4.40)
  Best for: Logical reasoning, pattern recognition

Critique Phase → o3-mini ($1.10/$4.40)
  Best for: Critical evaluation, error detection

Synthesis Phase → Claude-Sonnet-4-6 ($3.00/$15.00)
  Best for: Integration, coherent writing
```

**Κόστος**: ~$5.20 vs $7.50 (30% μείωση)  
**Ποιότητα**: Βελτιωμένη (καλύτερο model per phase)

---

## 🧠 Phase Types & Model Preferences

### 1. **ANALYSIS** (Ανάλυση)
**Απαιτήσεις**: Reasoning, pattern recognition, speed

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | o3-mini | $1.10/$4.40 | Best reasoning per $ |
| 2️⃣ | Phi-4 | $0.07/$0.14 | Fast analysis |
| 3️⃣ | LLaMA-3.3-70B | $0.12/$0.30 | Good pattern recognition |

### 2. **GENERATION** (Παραγωγή)
**Απαιτήσεις**: Creativity, technical accuracy

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Best coding generation |
| 2️⃣ | GPT-4o | $2.50/$10.00 | Strong all-rounder |
| 3️⃣ | LLaMA-4-Maverick | $0.17/$0.17 | Good value |

### 3. **CRITIQUE** (Κριτική)
**Απαιτήσεις**: Critical thinking, attention to detail

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | o3-mini | $1.10/$4.40 | Best critical analysis |
| 2️⃣ | Claude-3-Haiku | $0.25/$1.25 | Fast, accurate critic |
| 3️⃣ | Phi-4-Reasoning | $0.07/$0.35 | Strong reasoning |

### 4. **SYNTHESIS** (Σύνθεση)
**Απαιτήσεις**: Integration, coherence, writing quality

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Best at synthesis |
| 2️⃣ | GPT-4o | $2.50/$10.00 | Good integration |
| 3️⃣ | LLaMA-3.1-405B | $2.00/$2.00 | Comprehensive |

### 5. **DEBATE** (Διαλογική Αντιπαράθεση)
**Απαιτήσεις**: Argumentation, rhetoric

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | GPT-4o | $2.50/$10.00 | Strong argumentation |
| 2️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Balanced debate |
| 3️⃣ | LLaMA-4-Maverick | $0.17/$0.17 | Good value |

### 6. **RESEARCH** (Έρευνα)
**Απαιτήσεις**: Information retrieval, accuracy

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | GPT-4o | $2.50/$10.00 | Best general knowledge |
| 2️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Accurate |
| 3️⃣ | LLaMA-3.3-70B | $0.12/$0.30 | Good coverage |

### 7. **EVALUATION** (Αξιολόγηση)
**Απαιτήσεις**: Scoring accuracy, fairness

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | o3-mini | $1.10/$4.40 | Best scoring accuracy |
| 2️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Fair evaluation |
| 3️⃣ | LLaMA-3.1-405B | $2.00/$2.00 | Comprehensive |

### 8. **REFINEMENT** (Βελτίωση)
**Απαιτήσεις**: Iterative improvement

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | Claude-Sonnet-4-6 | $3.00/$15.00 | Best at refinements |
| 2️⃣ | GPT-4o | $2.50/$10.00 | Good iterations |
| 3️⃣ | LLaMA-4-Maverick | $0.17/$0.17 | Cost-effective |

### 9. **VERIFICATION** (Επαλήθευση)
**Απαιτήσεις**: Accuracy, validation, speed

| Priority | Model | Cost | Rationale |
|----------|-------|------|-----------|
| 1️⃣ | o3-mini | $1.10/$4.40 | Best validation |
| 2️⃣ | Claude-3-Haiku | $0.25/$1.25 | Fast verification |
| 3️⃣ | Phi-4 | $0.07/$0.14 | Quick checks |

---

## 🔧 Implementation

### Βασικές Μέθοδοι

```python
# Στο BasePipeline class
def _get_model_for_phase(self, phase: PhaseType, task_type: TaskType) -> Model:
    """Get optimal model for a specific phase type."""
    from .phase_aware_models import PhaseAwareModelSelector, PhaseType
    
    available = [m.value for m in self._get_available_models(task_type)]
    selector = PhaseAwareModelSelector()
    best = selector.select_model(phase=phase, available_models=available)
    
    # Convert string back to Model enum
    from .models import Model
    try:
        return Model(best)
    except ValueError:
        return self._get_available_models(task_type)[0]
```

### Χρήση στα Pipelines

```python
# Multi-Perspective Pipeline example
async def _phase_perspectives(self, state: PipelineState, context: str):
    from .phase_aware_models import PhaseType
    
    # Use phase-aware model selection for analysis
    primary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)
    
    # Run perspectives with optimal model
    response, _ = await self.client.call(
        model=primary,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=state.task.max_output_tokens,
        temperature=0.7,
    )

async def _phase_critique(self, state: PipelineState):
    from .phase_aware_models import PhaseType
    
    # Use phase-aware model selection for critique
    scorer = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)
    
    # Critique with optimal model
    response, _ = await self.client.call(
        model=scorer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1000,
        temperature=0.1,
    )

async def _phase_synthesis(self, state: PipelineState):
    from .phase_aware_models import PhaseType
    
    # Use phase-aware model selection for synthesis
    synthesizer = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)
    
    # Synthesize with optimal model
    response, _ = await self.client.call(
        model=synthesizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=state.task.max_output_tokens,
        temperature=0.3,
    )
```

---

## 📈 Model Capability Profiles

Κάθε model έχει ένα capability profile με scores (0-10) για κάθε capability:

```python
"openai/gpt-4o": {
    "reasoning": 9.0,      # Logical reasoning
    "coding": 9.0,         # Code generation
    "creativity": 8.5,     # Creative writing
    "critique": 8.5,       # Critical evaluation
    "synthesis": 9.0,      # Integration
    "speed": 7.0,          # Response speed
    "cost_efficiency": 6.0, # Cost per token
}
```

Ο selector υπολογίζει composite score βάσει των απαιτήσεων του phase:

```python
# Για ANALYSIS phase
required_caps = ["reasoning", "speed"]
composite_score = (reasoning_score + speed_score) / 2
```

---

## 💰 Cost Optimization

### Παράδειγμα: CODE_GEN Task

**Πριν** (όλες οι phases με GPT-4o):
```
Analysis:   GPT-4o      $2.50
Critique:   GPT-4o      $2.50
Synthesis:  GPT-4o      $2.50
-------------------------
Total:      $7.50
```

**Μετά** (phase-aware):
```
Analysis:   o3-mini     $1.10  (best reasoning)
Critique:   o3-mini     $1.10  (best critique)
Synthesis:  Claude-4-6  $3.00  (best synthesis)
-------------------------
Total:      $5.20  (30% savings!)
```

### Quality Improvement

Εκτός από cost savings, η ποιότητα βελτιώνεται γιατί:
- **Analysis**: o3-mini έχει καλύτερο reasoning από GPT-4o
- **Critique**: o3-mini εντοπίζει περισσότερα issues
- **Synthesis**: Claude-4-6 έχει καλύτερη integration capability

---

## 🚀 Usage

```python
from orchestrator.phase_aware_models import (
    PhaseAwareModelSelector,
    PhaseType,
)

selector = PhaseAwareModelSelector()

# Get optimal model for a phase
model = selector.select_model(
    phase=PhaseType.ANALYSIS,
    available_models=["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
    budget_constraint=2.00,  # Optional
    prioritize_speed=True,    # Optional
)

# Get top 3 models for a phase
top_models = selector.get_phase_models(PhaseType.CRITIQUE, count=3)
```

---

## 📋 Updated Pipelines

Τα παρακάτω ARA pipelines έχουν ενημερωθεί με phase-aware model selection:

- ✅ Multi-Perspective Pipeline
- ✅ Iterative Pipeline
- ✅ Debate Pipeline
- ✅ Research Pipeline
- ✅ Jury Pipeline
- ✅ Scientific Pipeline
- ✅ Socratic Pipeline
- ✅ Pre-Mortem Pipeline
- ✅ Bayesian Pipeline
- ✅ Dialectical Pipeline
- ✅ Analogical Pipeline
- ✅ Delphi Pipeline

---

**Author**: Georgios-Chrysovalantis Chatzivantsidis  
**Date**: 2026-03-30  
**Version**: Phase-Aware v1.0
