# ARA Pipeline — Phase-by-Phase Model Analysis

**Version:** 1.0.0  
**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Date:** 2026-03-23  
**Data Source:** OpenRouter API, Provider Documentation, Benchmark Leaderboards

**Purpose:** Detailed phase-by-phase model analysis for each ARA method. For integration guide, see [ARA_PIPELINE_GUIDE.md](./ARA_PIPELINE_GUIDE.md). For model recommendations, see [ARA_MODEL_SELECTION_GUIDE.md](./ARA_MODEL_SELECTION_GUIDE.md).

---

## Overview

Each ARA Pipeline method consists of multiple phases with distinct cognitive requirements. This guide provides **phase-specific model recommendations** based on the unique capabilities needed for each phase.

### Phase Types & Requirements

| Phase Type | Primary Capability | Key Metrics | Best Model Families |
|------------|-------------------|-------------|---------------------|
| **Generation** | Divergent thinking, creativity | Ideas/minute, novelty score | Gemini, Claude, GPT |
| **Analysis** | Logical reasoning, consistency | AIME, MATH, GPQA | Claude Opus, GPT-Pro, DeepSeek |
| **Critique** | Critical evaluation, error detection | Precision, recall | Claude Sonnet, GPT-4 |
| **Synthesis** | Integration, abstraction | Coherence, completeness | Claude, Gemini Pro |
| **Evaluation** | Scoring, ranking | Calibration accuracy | GPT-Pro, Claude Opus |
| **Questioning** | Socratic method, clarity | Question quality | Claude, GPT |
| **Mapping** | Analogical transfer | Cross-domain accuracy | Claude, Gemini |

---

## 1. Multi-Perspective Pipeline

**Total Phases:** 3 (Perspectives → Critique → Synthesis)  
**Typical Token Usage:** 8,000–12,000 tokens

### Phase 1: Four Perspectives (Parallel)

| Perspective | Cognitive Task | Best Models | Cost ($/1M) |
|-------------|---------------|-------------|-------------|
| **Constructive** | Opportunity identification | Gemini 3.1 Pro, Claude Sonnet 4.6 | $2–$3 / $12–$15 |
| **Destructive** | Flaw detection, risk ID | Claude Opus 4.6, GPT-5.4 | $5 / $25 |
| **Systemic** | Second-order effects | GPT-5.4, Gemini 3.1 Pro | $2.50 / $15 |
| **Minimalist** | Essential element extraction | DeepSeek V3.2, Qwen3.5-Flash | $0.27 / $1 |

### Phase 2: Critique & Scoring

**Requirements:** Logical consistency evaluation, multi-criteria scoring

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best balanced critique |
| 2 | GPT-5.4 | $2.50 / $15 | Strong scoring calibration |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Synthesis

**Requirements:** Integration of 4 perspectives, contradiction resolution

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best integration |
| 2 | Gemini 3.1 Pro | $2 / $12 | Strong coherence |
| 3 | GPT-5.4 | $2.50 / $15 | Good actionability |

### Recommended Mixed Configuration

```yaml
Phase 1:
  Constructive: Gemini 3.1 Pro ($2)
  Destructive: Claude Sonnet 4.6 ($3)
  Systemic: GPT-5.4 ($2.50)
  Minimalist: DeepSeek V3.2 ($0.27)
Phase 2: Claude Sonnet 4.6 ($3)
Phase 3: Claude Sonnet 4.6 ($3)
Total: ~$16 per task
```

---

## 2. Iterative Pipeline

**Total Phases:** 3–5 rounds (Generate → Critique → Refine)  
**Typical Token Usage:** 6,000–10,000 tokens

### Phase 1: Initial Generation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best initial quality |
| 2 | GPT-5.4 | $2.50 / $15 | Strong baseline |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value start |

### Phase 2-N: Iterative Refinement

| Round | Best Model | Cost ($/1M) | Why |
|-------|------------|-------------|-----|
| Round 1 | Claude Sonnet 4.6 | $3 / $15 | Best feedback integration |
| Round 2 | Claude Sonnet 4.6 | $3 / $15 | Consistent refinement |
| Round 3 | DeepSeek V3.2 | $0.27 / $1 | Cost-effective final polish |

### Phase N: Convergence Detection

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best convergence judgment |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Quality assessment |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Budget final check |

### Recommended Mixed Configuration

```yaml
Round 1: Claude Sonnet 4.6 ($6)
Round 2: Claude Sonnet 4.6 ($3)
Round 3: DeepSeek V3.2 ($1)
Total: ~$10 per task
```

---

## 3. Debate Pipeline

**Total Phases:** 4 (Opening → Rebuttal → Cross-Examine → Judge)  
**Typical Token Usage:** 10,000–15,000 tokens

### Phase 1: Opening Statements

| Side | Best Models | Cost ($/1M) | Why |
|------|-------------|-------------|-----|
| **Side A (Pro)** | Claude Sonnet 4.6 | $3 / $15 | Constructive argumentation |
| **Side B (Con)** | Claude Opus 4.6 | $5 / $25 | Critical analysis |

### Phase 2: Rebuttals

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best counter-argumentation |
| 2 | GPT-5.4 Pro | $30 / $180 | Thorough refutation |
| 3 | Claude Sonnet 4.6 | $3 / $15 | Balanced rebuttal |

### Phase 3: Cross-Examination

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best questioning |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Insightful probes |
| 3 | Gemini 3.1 Pro | $2 / $12 | Good clarification |

### Phase 4: Judge Decision

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best judicial reasoning |
| 2 | GPT-5.4 Pro | $30 / $180 | Thorough analysis |
| 3 | Claude Sonnet 4.6 | $3 / $15 | Balanced judgment |

### Recommended Mixed Configuration

```yaml
Opening: Claude Sonnet 4.6 ($6)
Rebuttal: Claude Opus 4.6 ($10)
Cross-Exam: GPT-5.4 ($4)
Judge: Claude Opus 4.6 ($10)
Total: ~$30 per task
```

---

## 4. Research Pipeline

**Total Phases:** 3 (Discovery → Analysis → Fact-Check)  
**Typical Token Usage:** 8,000–12,000 tokens (+ web search costs)

### Phase 1: Deep Iterative Research

**Requirements:** Web search integration, query formulation  
**Web Search Costs:** ~$0.05–$0.30 per search (3–5 searches = $0.15–$1.50)

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Gemini 3.1 Pro Preview | $2 / $12 | Native web search |
| 2 | Gemini 3 Flash Preview | $0.50 / $3 | Fast research |
| 3 | GPT-5.4 | $2.50 / $15 | Strong query formulation |

### Phase 2: Analysis with Web Context

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best synthesis |
| 2 | Gemini 3.1 Pro | $2 / $12 | Good integration |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Fact-Checked Critique

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best fact-checking |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Thorough verification |
| 3 | Qwen3.5-Flash | $0.07 / $0.26 | Budget verification |

### Recommended Mixed Configuration

```yaml
Research: Gemini 3 Flash ($3)
Analysis: Claude Sonnet 4.6 ($4)
Fact-Check: GPT-5.4 ($3)
Web Search: ~$1
Total: ~$11 per task
```

---

## 5. Jury Pipeline

**Total Phases:** 4 (4-Gen → 3-Critique → Verify+Meta → Rank)  
**Typical Token Usage:** 20,000–30,000 tokens

### Phase 1: Four Parallel Generators

| Generator | Best Models | Cost ($/1M) |
|-----------|-------------|-------------|
| Gen 1 | Claude Sonnet 4.6 | $3 / $15 |
| Gen 2 | GPT-5.4 | $2.50 / $15 |
| Gen 3 | Gemini 3.1 Pro | $2 / $12 |
| Gen 4 | DeepSeek V3.2 | $0.27 / $1 |

### Phase 2: Three Parallel Critics

| Critic | Best Models | Cost ($/1M) |
|--------|-------------|-------------|
| Critic 1 | Claude Opus 4.6 | $5 / $25 |
| Critic 2 | Claude Sonnet 4.6 | $3 / $15 |
| Critic 3 | GPT-5.4 | $2.50 / $15 |

### Phase 3: Verifier + Meta-Evaluator

| Role | Best Models | Cost ($/1M) |
|------|-------------|-------------|
| Verifier | GPT-5.4 | $2.50 / $15 |
| Meta-Evaluator | Claude Opus 4.6 | $5 / $25 |

### Phase 4: Weighted Ranking

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best final judgment |
| 2 | GPT-5.4 Pro | $30 / $180 | Premium selection |
| 3 | Claude Sonnet 4.6 | $3 / $15 | Balanced ranking |

### Recommended Mixed Configuration

```yaml
Gen (4×): Mixed ($12)
Critic (3×): Mixed ($10)
Verify+Meta: DeepSeek V3.2 ($5)
Rank: Claude Opus 4.6 ($5)
Total: ~$32 per task
```

---

## 6. Scientific Pipeline

**Total Phases:** 3 (Hypothesize → Test → Evaluate)  
**Typical Token Usage:** 8,000–12,000 tokens

### Phase 1: Hypothesis Generation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best scientific reasoning |
| 2 | GPT-5.4 | $2.50 / $15 | Strong hypothesis quality |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 2: Test Design

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best experimental design |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong methodology |
| 3 | Gemini 3.1 Pro | $2 / $12 | Good measurability |

### Phase 3: Evidence Evaluation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best evidence analysis |
| 2 | GPT-5.4 Pro | $30 / $180 | Statistical rigor |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Hypothesize: Claude Opus 4.6 ($5)
Test Design: GPT-5.4 ($4)
Evaluate: Claude Opus 4.6 ($5)
Total: ~$14 per task
```

---

## 7. Socratic Pipeline

**Total Phases:** 3–5 rounds (Question → Answer → Follow-up → Solution)  
**Typical Token Usage:** 5,000–8,000 tokens

### Phase 1: Initial Socratic Questioning

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best questioning technique |
| 2 | GPT-5.4 | $2.50 / $15 | Strong assumption detection |
| 3 | Qwen3.5-9B | $0.05 / $0.15 | Best value |

### Phase 2-N: Follow-up Loops

| Round | Best Model | Cost ($/1M) | Why |
|-------|------------|-------------|-----|
| Round 1 | Claude Sonnet 4.6 | $3 / $15 | Deep follow-ups |
| Round 2 | GPT-5.4 | $2.50 / $15 | Clarity focus |
| Round 3 | DeepSeek V3.2 | $0.27 / $1 | Final clarification |

### Phase N: Solution Generation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best synthesis |
| 2 | GPT-5.4 | $2.50 / $15 | Strong reasoning |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Questions: Claude Sonnet 4.6 ($6)
Follow-ups: GPT-5.4 ($4)
Solution: Claude Sonnet 4.6 ($3)
Total: ~$13 per task
```

---

## 8. Pre-Mortem Pipeline ⭐

**Total Phases:** 4 (Failure → Root Cause → Signals → Redesign)  
**Typical Token Usage:** 8,000–12,000 tokens

### Phase 1: Failure Narrative

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best failure imagination |
| 2 | GPT-5.4 | $2.50 / $15 | Strong scenario building |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 2: Root Cause Backtracking

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best causal analysis |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong root cause |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Early Warning Signals

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Gemini 3.1 Pro | $2 / $12 | Best prediction |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong indicators |
| 3 | Qwen3.5-Flash | $0.07 / $0.26 | Best value |

### Phase 4: Hardened Redesign

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best risk mitigation |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong safeguards |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Failure: Claude Opus 4.6 ($8)
Root Cause: GPT-5.4 ($4)
Signals: Gemini 3.1 Pro ($2)
Redesign: Claude Opus 4.6 ($8)
Total: ~$22 per task
```

---

## 9. Bayesian Pipeline

**Total Phases:** 4 (Priors → Likelihoods → Posteriors → Sensitivity)  
**Typical Token Usage:** 10,000–15,000 tokens

### Phase 1: Prior Elicitation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best probabilistic reasoning |
| 2 | GPT-5.4 Pro | $30 / $180 | Mathematical rigor |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 2: Likelihood Assessment

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 Pro | $30 / $180 | Best conditional reasoning |
| 2 | Claude Opus 4.6 | $5 / $25 | Strong likelihood |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Posterior Update

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 Pro | $30 / $180 | Best mathematical calculation |
| 2 | Claude Opus 4.6 | $5 / $25 | Accurate updating |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 4: Sensitivity Analysis

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best sensitivity analysis |
| 2 | GPT-5.4 | $2.50 / $15 | Strong robustness |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Priors: Claude Opus 4.6 ($8)
Likelihoods: GPT-5.4 ($5)
Posteriors: GPT-5.4 ($5)
Sensitivity: Claude Opus 4.6 ($8)
Total: ~$26 per task
```

---

## 10. Dialectical Pipeline

**Total Phases:** 4 (Thesis → Antithesis → Contradictions → Aufhebung)  
**Typical Token Usage:** 8,000–12,000 tokens

### Phase 1: Thesis

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best position framing |
| 2 | GPT-5.4 | $2.50 / $15 | Strong articulation |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 2: Antithesis

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best opposition |
| 2 | GPT-5.4 | $2.50 / $15 | Strong counter-position |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Contradictions Analysis

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best contradiction analysis |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong conflict mapping |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 4: Aufhebung (Transcendence)

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best philosophical synthesis |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong transcendence |
| 3 | Gemini 3.1 Pro | $2 / $12 | Good integration |

### Recommended Mixed Configuration

```yaml
Thesis: Claude Sonnet 4.6 ($5)
Antithesis: Claude Opus 4.6 ($8)
Contradictions: GPT-5.4 ($4)
Aufhebung: Claude Opus 4.6 ($10)
Total: ~$27 per task
```

---

## 11. Analogical Pipeline ⭐

**Total Phases:** 4 (Abstraction → Domain Search → Mapping → Transfer)  
**Typical Token Usage:** 10,000–15,000 tokens

### Phase 1: Abstraction

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best abstraction |
| 2 | GPT-5.4 | $2.50 / $15 | Strong structure extraction |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 2: Domain Search

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Gemini 3.1 Pro | $2 / $12 | Best cross-domain search |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong relevance |
| 3 | Qwen3.5-Flash | $0.07 / $0.26 | Best value |

### Phase 3: Mapping

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best mapping quality |
| 2 | GPT-5.4 | $2.50 / $15 | Strong correspondence |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 4: Transfer & Adaptation

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best transfer capability |
| 2 | Gemini 3.1 Pro | $2 / $12 | Strong adaptation |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Abstraction: Claude Sonnet 4.6 ($5)
Domain Search: Gemini 3.1 Pro ($3)
Mapping: Claude Sonnet 4.6 ($4)
Transfer: Claude Sonnet 4.6 ($5)
Total: ~$17 per task
```

---

## 12. Delphi Pipeline

**Total Phases:** 5 (Round 1 → Aggregate → Round 2 → Convergence → Dissent)  
**Typical Token Usage:** 15,000–20,000 tokens

### Phase 1: Round 1 (4 Independent Experts)

| Expert | Best Models | Cost ($/1M) |
|--------|-------------|-------------|
| Expert 1 | Claude Sonnet 4.6 | $3 / $15 |
| Expert 2 | GPT-5.4 | $2.50 / $15 |
| Expert 3 | Gemini 3.1 Pro | $2 / $12 |
| Expert 4 | DeepSeek V3.2 | $0.27 / $1 |

### Phase 2: Aggregation (Median, IQR, Outliers)

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best statistical analysis |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong aggregation |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 3: Round 2 (Revision with Feedback)

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Sonnet 4.6 | $3 / $15 | Best revision quality |
| 2 | GPT-5.4 | $2.50 / $15 | Strong feedback integration |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Phase 4: Convergence Check

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | GPT-5.4 | $2.50 / $15 | Best convergence judgment |
| 2 | Claude Sonnet 4.6 | $3 / $15 | Strong threshold analysis |
| 3 | Qwen3.5-Flash | $0.07 / $0.26 | Best value |

### Phase 5: Dissent Analysis

| Rank | Model | Cost ($/1M) | Why |
|------|-------|-------------|-----|
| 1 | Claude Opus 4.6 | $5 / $25 | Best dissent analysis |
| 2 | GPT-5.4 | $2.50 / $15 | Strong disagreement mapping |
| 3 | DeepSeek V3.2 | $0.27 / $1 | Best value |

### Recommended Mixed Configuration

```yaml
Round 1 (4×): Mixed ($12)
Aggregate: GPT-5.4 ($4)
Round 2 (4×): Mixed ($8)
Convergence: GPT-5.4 ($2)
Dissent: Claude Opus 4.6 ($8)
Total: ~$34 per task
```

---

## Summary: Optimal Model Configurations

### Quick Reference by Method

| Method | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|--------|---------|---------|---------|---------|-------|
| **Multi-Perspective** | Mixed | Sonnet 4.6 | Sonnet 4.6 | — | ~$16 |
| **Iterative** | Sonnet 4.6 | Sonnet 4.6 | DeepSeek | — | ~$10 |
| **Debate** | Sonnet 4.6 | Opus 4.6 | GPT-5.4 | Opus 4.6 | ~$30 |
| **Research** | Gemini Flash | Sonnet 4.6 | GPT-5.4 | — | ~$11 |
| **Jury** | Mixed (4 gen) | Mixed (3 crit) | DeepSeek | Opus 4.6 | ~$32 |
| **Scientific** | Opus 4.6 | GPT-5.4 | Opus 4.6 | — | ~$14 |
| **Socratic** | Sonnet 4.6 | GPT-5.4 | Sonnet 4.6 | — | ~$13 |
| **Pre-Mortem** ⭐ | Opus 4.6 | GPT-5.4 | Gemini | Opus 4.6 | ~$22 |
| **Bayesian** | Opus 4.6 | GPT-5.4 | GPT-5.4 | Opus 4.6 | ~$26 |
| **Dialectical** | Sonnet 4.6 | Opus 4.6 | GPT-5.4 | Opus 4.6 | ~$27 |
| **Analogical** ⭐ | Sonnet 4.6 | Gemini | Sonnet 4.6 | Sonnet 4.6 | ~$17 |
| **Delphi** | Mixed (4 experts) | GPT-5.4 | Mixed | GPT-5.4 | ~$34 |

---

## Budget Tiers

### Tier 1: Premium (Maximum Quality)
**Budget:** $50–$200 per project  
**Use Case:** Mission-critical, production, high-stakes

```yaml
Primary: claude-opus-4.6
Secondary: claude-sonnet-4.6
Fallback: gpt-5.4-pro
```

### Tier 2: Balanced (Quality/Value)
**Budget:** $10–$50 per project  
**Use Case:** Standard production, development

```yaml
Primary: claude-sonnet-4.6
Secondary: gemini-3.1-pro
Fallback: gpt-5.4
Budget: deepseek-chat
```

### Tier 3: Value (Cost-Effective)
**Budget:** $1–$10 per project  
**Use Case:** Testing, prototyping, volume tasks

```yaml
Primary: deepseek-chat
Secondary: qwen3.5-flash
Fallback: minimax-m2.5
Free: nemotron-3-super
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [ARA_PIPELINE_GUIDE.md](./ARA_PIPELINE_GUIDE.md) | Integration & usage guide |
| [ARA_MODEL_SELECTION_GUIDE.md](./ARA_MODEL_SELECTION_GUIDE.md) | Model recommendations |
| [ARA_PHASE_MODEL_ANALYSIS.md](./ARA_PHASE_MODEL_ANALYSIS.md) | **Phase-by-phase analysis** (this file) |

---

*Last updated: 2026-03-23*  
*Data source: OpenRouter API (openrouter.ai/models)*  
*Author: Georgios-Chrysovalantis Chatzivantsidis*
